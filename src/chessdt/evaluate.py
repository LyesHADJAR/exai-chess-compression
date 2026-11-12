"""
Scoring a model on held-out positions.

One protocol, used by training and by the evaluation script alike: each
held-out game is tiled with consecutive windows from its first ply, the model
predicts every ply, and only positions carrying the split's own label count.
Positions dropped as leaked still serve as history but are never scored.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .constants import ELO_CENTRE, ELO_SCALE, N_MOVES, PUZZLE_ELO, PUZZLE_OUTCOME
from .data import Batch, Corpus, decode_records
from .encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, unpack_board

#: Windows scored per forward pass. Lower it (CHESSDT_EVAL_BATCH=64) on a GPU
#: with little free memory: it changes the memory used, never the result.
EVAL_BATCH = int(os.environ.get("CHESSDT_EVAL_BATCH", "512"))


def autocast(device: str):
    """Mixed precision where it helps, a no-op elsewhere."""
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def topk_hits(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    """Per row, whether the target is among the k highest-scoring moves.

    Top-1 asks whether the model's first choice is the move that was played.
    Top-3 asks whether the played move is one of its three favourites, which
    is the fairer measure of behaviour cloning: in most positions a human
    chooses between a few reasonable moves, and a model that ranks the played
    one second has still understood the position.
    """
    return (logits.topk(k, dim=-1).indices == target.unsqueeze(-1)).any(dim=-1)


@torch.no_grad()
def predict(model, batch: Batch, device: str) -> torch.Tensor:
    """Logits for every ply of a batch, as float32."""
    with autocast(device):
        return model(*batch.inputs()).float()


@torch.no_grad()
def score(
    model,
    corpus: Corpus,
    starts: torch.Tensor,
    plies: int,
    split: int,
    batch_size: int = EVAL_BATCH,
    mutate=None,
) -> dict:
    """Loss, top-1 and top-3 accuracy over the positions of `split` in the windows.

    `mutate` optionally rewrites each batch before the forward pass; the
    ablations are built on it.
    """
    model.eval()
    loss_sum, hits, hits3, count = 0.0, 0, 0, 0

    for i in range(0, len(starts), batch_size):
        batch = corpus.window(starts[i : i + batch_size], plies)
        if mutate is not None:
            batch = mutate(batch)
        keep = batch.valid & (batch.labels == split)

        logits = predict(model, batch, corpus.device)[keep]
        target = batch.moves[keep]
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        hits += topk_hits(logits, target, 1).sum().item()
        hits3 += topk_hits(logits, target, 3).sum().item()
        count += int(keep.sum())

    return {
        "loss": loss_sum / max(count, 1),
        "top1": hits / max(count, 1),
        "top3": hits3 / max(count, 1),
        "positions": count,
    }


@torch.no_grad()
def compare(
    model,
    corpus: Corpus,
    starts: torch.Tensor,
    plies: int,
    split: int,
    mutate,
    batch_size: int = EVAL_BATCH,
) -> dict:
    """How much an intervention on the inputs moves the predictions.

    Runs every window twice, as it is and after `mutate`, and reports the
    share of positions whose chosen move changes. A channel the model ignores
    scores zero here however the input is mangled, which is the whole point:
    this is the measurement that caught the previous model copying its input.
    """
    model.eval()
    flips, hits, count = 0, 0, 0

    for i in range(0, len(starts), batch_size):
        batch = corpus.window(starts[i : i + batch_size], plies)
        keep = batch.valid & (batch.labels == split)

        before = predict(model, batch, corpus.device).argmax(-1)
        after = predict(model, mutate(batch), corpus.device).argmax(-1).reshape(before.shape)

        flips += (before[keep] != after[keep]).sum().item()
        hits += (after[keep] == batch.moves[keep]).sum().item()
        count += int(keep.sum())

    return {"changed": flips / max(count, 1), "top1": hits / max(count, 1), "positions": count}


def alone(batch: Batch) -> Batch:
    """Every ply as its own one-ply window: the same positions, no history."""
    return Batch(
        codes=batch.codes.flatten(0, 1).unsqueeze(1),
        side=batch.side.flatten().unsqueeze(1),
        cond=batch.cond.flatten(0, 1).unsqueeze(1),
        moves=batch.moves.flatten().unsqueeze(1),
        valid=batch.valid.flatten().unsqueeze(1),
        labels=batch.labels.flatten().unsqueeze(1),
    )


def normalised_rating(elo: float) -> float:
    return (elo - ELO_CENTRE) / ELO_SCALE


#: Puzzle themes worth a line of their own: the motifs the tactical arms train on,
#: and the common ones around them.
REPORTED_THEMES = (
    "mateIn1",
    "mateIn2",
    "fork",
    "hangingPiece",
    "pin",
    "skewer",
    "discoveredAttack",
    "backRankMate",
)


@dataclass
class PuzzleSet:
    """A sample of the puzzle database, resident on one device.

    A puzzle has exactly one move that works, so it is the sharp half of the
    fidelity measure: a model can agree with a human on the bulk of ordinary
    positions and still have lost the tactic. Moves are scored with illegal
    ones masked out, the same way for every model.
    """

    codes: torch.Tensor  # (N, 64) long
    side: torch.Tensor  # (N,) long
    target: torch.Tensor  # (N,) long
    legal: torch.Tensor  # (N, N_MOVES) bool
    rating: np.ndarray
    mate_in_one: np.ndarray
    device: str
    themes: np.ndarray | None = None  # the puzzle database's theme string, one per puzzle

    @classmethod
    def load(cls, directory: Path, sample: int, device: str, seed: int = 0) -> "PuzzleSet":
        with np.load(directory / "puzzles.npz") as meta:
            rating, themes = meta["rating"], meta["themes"]
        records = np.memmap(directory / "positions.bin", dtype=np.uint8, mode="r")
        records = records.reshape(-1, BOARD_RECORD_BYTES)
        solutions = np.memmap(directory / "moves.bin", dtype=np.uint16, mode="r")

        pick = np.sort(np.random.default_rng(seed).choice(len(records), sample, replace=False))
        chosen = np.ascontiguousarray(records[pick])

        legal = torch.zeros(sample, N_MOVES, dtype=torch.bool)
        for row, record in enumerate(chosen):
            board = unpack_board(record.tobytes())
            legal[row, [MOVE_TO_INDEX[m.uci()] for m in board.legal_moves]] = True

        codes, side = decode_records(torch.from_numpy(chosen))
        return cls(
            codes=codes.to(device),
            side=side.to(device),
            target=torch.from_numpy(solutions[pick].astype(np.int64)).to(device),
            legal=legal.to(device),
            rating=rating[pick],
            mate_in_one=np.asarray([b"mateIn1" in t for t in themes[pick]]),
            device=device,
            themes=themes[pick],
        )

    def __len__(self) -> int:
        return len(self.target)

    @torch.no_grad()
    def solve(
        self,
        model,
        outcome: float = PUZZLE_OUTCOME,
        elo: float = PUZZLE_ELO,
        chunk: int = 2 * EVAL_BATCH,
    ) -> dict:
        """Solve rates, given the conditioning the puzzle is posed under.

        A puzzle has no game around it, so the player to move is described by
        `outcome` and `elo`. The defaults say the solver is winning and strong.
        """
        model.eval()
        cond = torch.tensor([outcome, normalised_rating(elo)], device=self.device)
        raw = torch.zeros(len(self), dtype=torch.bool, device=self.device)
        top1 = torch.zeros_like(raw)
        top3 = torch.zeros_like(raw)

        for i in range(0, len(self), chunk):
            n = len(self.codes[i : i + chunk])
            batch = Batch(
                codes=self.codes[i : i + n, None],
                side=self.side[i : i + n, None],
                cond=cond.expand(n, 1, 2),
                moves=torch.zeros(n, 1, dtype=torch.long, device=self.device),
                valid=torch.ones(n, 1, dtype=torch.bool, device=self.device),
                labels=torch.zeros(n, 1, dtype=torch.uint8, device=self.device),
            )
            logits = predict(model, batch, self.device)[:, 0]
            target = self.target[i : i + n]
            raw[i : i + n] = topk_hits(logits, target, 1)
            masked = logits.masked_fill(~self.legal[i : i + n], float("-inf"))
            top1[i : i + n] = topk_hits(masked, target, 1)
            top3[i : i + n] = topk_hits(masked, target, 3)

        solved = top1.cpu().numpy()
        bands = {}
        for low, high in ((0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 9999)):
            inside = (self.rating >= low) & (self.rating < high)
            if inside.any():
                bands[f"{low}-{high}"] = {
                    "solved": float(solved[inside].mean()),
                    "puzzles": int(inside.sum()),
                }

        by_theme = {}
        if self.themes is not None:
            tokens = [set(t.split()) for t in self.themes]
            for theme in REPORTED_THEMES:
                inside = np.asarray([theme.encode() in t for t in tokens])
                if inside.any():
                    by_theme[theme] = {
                        "solved": float(solved[inside].mean()),
                        "puzzles": int(inside.sum()),
                    }

        return {
            "puzzles": len(self),
            "solved": float(solved.mean()),
            "solution_in_top3": float(top3.float().mean().item()),
            "solved_without_legal_mask": float(raw.float().mean().item()),
            "mate_in_one": float(solved[self.mate_in_one].mean()),
            "mate_in_one_puzzles": int(self.mate_in_one.sum()),
            "by_rating": bands,
            "by_theme": by_theme,
        }
