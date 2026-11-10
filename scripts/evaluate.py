"""
Evaluate a trained checkpoint, and decide whether it is fit to release.

    python scripts/evaluate.py runs/reference/model_best.pt --puzzles data/puzzles

Three things are measured.

**Accuracy** on the held-out games: how often the model's first choice is the
move the human played, with history and without.

**Ablations**, which are the release gate. Each input channel is damaged in
turn and the share of predictions that change is reported. The board must
dominate. A channel that can be damaged without changing anything is a channel
the model does not use, and the write-up may not claim otherwise.

**Puzzles**, the sharp test: positions with exactly one good move, scored with
illegal moves masked out.

Writes `evaluation.json` beside the checkpoint.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import (  # noqa: E402
    ELO_CENTRE,
    ELO_SCALE,
    N_MOVES,
    PUZZLE_ELO,
    PUZZLE_OUTCOME,
)
from chessdt.data import Batch, Corpus, decode_records  # noqa: E402
from chessdt.encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, unpack_board  # noqa: E402
from chessdt.evaluate import alone, compare, predict, score, topk_hits  # noqa: E402
from chessdt.model import load_checkpoint  # noqa: E402
from chessdt.splits import TEST, VAL  # noqa: E402


def rating(elo: float) -> float:
    return (elo - ELO_CENTRE) / ELO_SCALE


def with_cond(batch: Batch, channel: int, fn) -> Batch:
    cond = batch.cond.clone()
    cond[..., channel] = fn(cond[..., channel])
    return replace(batch, cond=cond)


def scrambled_moves(batch: Batch) -> Batch:
    generator = torch.Generator(device=batch.moves.device).manual_seed(0)
    noise = torch.randint(
        N_MOVES, batch.moves.shape, generator=generator, device=batch.moves.device
    )
    return replace(batch, moves=noise)


#: name -> (what is done to the input, the function that does it)
ABLATIONS = {
    "board": ("every square emptied", lambda b: replace(b, codes=torch.zeros_like(b.codes))),
    "side to move": ("flipped", lambda b: replace(b, side=1 - b.side)),
    "outcome": ("win and loss swapped", lambda b: with_cond(b, 0, lambda x: -x)),
    "rating": (
        "set to 1800 for everyone",
        lambda b: with_cond(b, 1, lambda x: torch.full_like(x, rating(1800))),
    ),
    "earlier moves": ("replaced with random moves", scrambled_moves),
    "history": ("removed, each ply scored alone", alone),
}


def score_puzzles(model, directory: Path, sample: int, device: str, seed: int = 0) -> dict:
    """Solve rate on a seeded sample of the puzzle set."""
    with np.load(directory / "puzzles.npz") as meta:
        ratings, themes = meta["rating"], meta["themes"]
    records = np.memmap(directory / "positions.bin", dtype=np.uint8, mode="r")
    records = records.reshape(-1, BOARD_RECORD_BYTES)
    solutions = np.memmap(directory / "moves.bin", dtype=np.uint16, mode="r")

    pick = np.sort(np.random.default_rng(seed).choice(len(records), sample, replace=False))
    chosen = np.ascontiguousarray(records[pick])
    target = torch.from_numpy(solutions[pick].astype(np.int64)).to(device)

    legal = torch.zeros(sample, N_MOVES, dtype=torch.bool)
    for row, record in enumerate(chosen):
        moves = [MOVE_TO_INDEX[m.uci()] for m in unpack_board(record.tobytes()).legal_moves]
        legal[row, moves] = True
    legal = legal.to(device)

    cond = torch.tensor([PUZZLE_OUTCOME, rating(PUZZLE_ELO)], device=device)
    raw_hits = torch.zeros(sample, dtype=torch.bool, device=device)
    legal_hits = torch.zeros(sample, dtype=torch.bool, device=device)
    legal_hits3 = torch.zeros(sample, dtype=torch.bool, device=device)

    for i in range(0, sample, 1024):
        codes, side = decode_records(torch.from_numpy(chosen[i : i + 1024]).to(device))
        n = len(codes)
        batch = Batch(
            codes=codes[:, None],
            side=side[:, None],
            cond=cond.expand(n, 1, 2),
            moves=torch.zeros(n, 1, dtype=torch.long, device=device),
            valid=torch.ones(n, 1, dtype=torch.bool, device=device),
            labels=torch.zeros(n, 1, dtype=torch.uint8, device=device),
        )
        logits = predict(model, batch, device)[:, 0]
        raw_hits[i : i + n] = logits.argmax(-1) == target[i : i + n]
        masked = logits.masked_fill(~legal[i : i + n], float("-inf"))
        legal_hits[i : i + n] = topk_hits(masked, target[i : i + n], 1)
        legal_hits3[i : i + n] = topk_hits(masked, target[i : i + n], 3)

    solved = legal_hits.cpu().numpy()
    picked_ratings = ratings[pick]
    mate_in_one = np.asarray([b"mateIn1" in t for t in themes[pick]])

    bands = {}
    for low, high in ((0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 9999)):
        inside = (picked_ratings >= low) & (picked_ratings < high)
        if inside.any():
            bands[f"{low}-{high}"] = {
                "solved": float(solved[inside].mean()),
                "puzzles": int(inside.sum()),
            }

    return {
        "puzzles": sample,
        "solved": float(solved.mean()),
        "solution_in_top3": float(legal_hits3.float().mean().item()),
        "solved_without_legal_mask": float(raw_hits.float().mean().item()),
        "mate_in_one": float(solved[mate_in_one].mean()),
        "mate_in_one_puzzles": int(mate_in_one.sum()),
        "by_rating": bands,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--puzzles", type=Path, default=None)
    ap.add_argument("--puzzle-sample", type=int, default=50_000)
    ap.add_argument("--windows", type=int, default=None, help="cap held-out windows per split")
    ap.add_argument("--ablation-windows", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    model, payload = load_checkpoint(args.checkpoint, device)
    plies = model.cfg.context_plies
    corpus = Corpus(args.data, device)
    report = {"checkpoint": str(args.checkpoint), "step": payload.get("step")}

    print(f"\nCheckpoint {args.checkpoint} (step {payload.get('step')})\n")
    print("ACCURACY — the move played is the first choice (top-1) or among three (top-3)")
    for name, split in (("val", VAL), ("test", TEST)):
        starts = corpus.split_starts(int(split), plies, limit=args.windows)
        report[name] = score(model, corpus, starts, plies, int(split))
        report[f"{name}_no_history"] = score(model, corpus, starts, plies, int(split), mutate=alone)
        r, bare = report[name], report[f"{name}_no_history"]
        print(
            f"  {name:<5} top-1 {r['top1'] * 100:6.2f}%   top-3 {r['top3'] * 100:6.2f}%"
            f"   ({r['positions']:,} positions)"
        )
        print(
            f"        top-1 {bare['top1'] * 100:6.2f}%   top-3 {bare['top3'] * 100:6.2f}%"
            f"   without history"
        )

    print("\nABLATIONS — share of predictions that change when one input is damaged")
    starts = corpus.split_starts(int(VAL), plies, limit=args.ablation_windows)
    report["ablations"] = {}
    for name, (what, mutate) in ABLATIONS.items():
        result = compare(model, corpus, starts, plies, int(VAL), mutate)
        report["ablations"][name] = result
        print(
            f"  {name:<14} {result['changed'] * 100:6.2f}% changed"
            f"   accuracy after {result['top1'] * 100:6.2f}%   ({what})"
        )

    changed = {k: v["changed"] for k, v in report["ablations"].items()}
    board_ok = changed["board"] > 0.5
    inert = [k for k in ("outcome", "rating", "history") if changed[k] < 0.01]
    report["gate"] = {"board_dominates": board_ok, "inert_channels": inert}
    print(f"\n  board dominates: {'yes' if board_ok else 'NO — do not release this model'}")
    print(f"  inert channels : {', '.join(inert) if inert else 'none'}")

    if args.puzzles is not None:
        print(f"\nPUZZLES — {args.puzzle_sample:,} sampled, illegal moves masked")
        report["puzzle_set"] = p = score_puzzles(model, args.puzzles, args.puzzle_sample, device)
        print(f"  solved        {p['solved'] * 100:6.2f}%")
        print(f"  in top three  {p['solution_in_top3'] * 100:6.2f}%")
        print(f"  mate in one   {p['mate_in_one'] * 100:6.2f}%   ({p['mate_in_one_puzzles']:,})")
        for band, r in p["by_rating"].items():
            print(f"  rated {band:<10} {r['solved'] * 100:6.2f}%   ({r['puzzles']:,})")

    out = args.checkpoint.parent / "evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
