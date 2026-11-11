"""
Measure how well a model finds mate in one, against the humans it imitates.

    python scripts/build_tactics_set.py            # once
    python scripts/probe_tactics.py runs/reference/model_best.pt

Uses the fixed set of held-out positions with a mate in one built by
`build_tactics_set.py`, so every model is measured on the same thousands of
positions. The human rate is the ceiling; the gap to it is the weakness.

The model is scored twice: with the history it would have in a game, and with
none, as it is posed a puzzle. The difference shows how much of its tactical
ability comes from the plies before the position.

Writes `tactics.json` beside the checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.data import Corpus  # noqa: E402
from chessdt.evaluate import predict  # noqa: E402
from chessdt.model import load_checkpoint  # noqa: E402
from chessdt.splits import TEST, VAL  # noqa: E402


def first_choices(model, corpus: Corpus, starts: torch.Tensor, ply: torch.Tensor, plies: int):
    """The model's first choice at one chosen ply of each window, in batches."""
    chosen = []
    for i in range(0, len(starts), 256):
        batch = corpus.window(starts[i : i + 256], plies)
        logits = predict(model, batch, corpus.device)
        rows = torch.arange(len(logits), device=corpus.device)
        chosen.append(logits[rows, ply[i : i + 256]].argmax(-1))
    return torch.cat(chosen).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--set", type=Path, default=Path("data/tactics/mate_in_one.npz"))
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.set.exists():
        raise SystemExit(f"{args.set} not found; run scripts/build_tactics_set.py first")

    model, _ = load_checkpoint(args.checkpoint, args.device)
    corpus = Corpus(args.data, args.device)
    plies = model.cfg.context_plies

    tactics = np.load(args.set)
    label = int(TEST if args.split == "test" else VAL)
    chosen = np.flatnonzero(tactics["split"] == label)
    position = tactics["position"][chosen]
    mates = [
        set(tactics["moves"][tactics["offset"][k] : tactics["offset"][k + 1]].tolist())
        for k in chosen
    ]

    # A game's first position is where its window has to start; everything the
    # model is allowed to see before the position is the plies after that.
    games = np.load(args.data / "games.npz")["offset"]
    game_start = games[np.searchsorted(games, position, side="right") - 1]
    start = np.maximum(game_start, position - (plies - 1))
    ply = torch.from_numpy(position - start).to(args.device)

    with_history = first_choices(model, corpus, torch.from_numpy(start).to(args.device), ply, plies)
    alone = first_choices(
        model, corpus, torch.from_numpy(position).to(args.device), torch.zeros_like(ply), 1
    )
    human = corpus.moves[torch.from_numpy(position).to(args.device)].cpu().numpy()

    def rate(moves) -> float:
        return float(np.mean([int(m) in s for m, s in zip(moves, mates, strict=True)]))

    n = len(position)
    result = {
        "split": args.split,
        "positions": n,
        "human_plays_mate": rate(human),
        "model_plays_mate": rate(with_history),
        "model_plays_mate_without_history": rate(alone),
        "standard_error": (0.25 / n) ** 0.5,
    }
    (args.checkpoint.parent / "tactics.json").write_text(json.dumps(result, indent=2))

    print(f"{n:,} {args.split} positions with a mate in one")
    print(f"  a human played the mate          {result['human_plays_mate'] * 100:5.1f}%")
    print(f"  the model, with its history      {result['model_plays_mate'] * 100:5.1f}%")
    alone_rate = result["model_plays_mate_without_history"]
    print(f"  the model, no history            {alone_rate * 100:5.1f}%")
    print(f"  (standard error about {result['standard_error'] * 100:.1f} points)")


if __name__ == "__main__":
    main()
