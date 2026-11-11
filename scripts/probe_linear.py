"""
Gate 1: can a linear model do this task?

    python scripts/probe_linear.py --data data/encoded

Fits one linear layer from the one-hot board to the move, the strongest thing
a model with no depth can do, and scores it on held-out positions. If this
comes anywhere near the transformer, the transformer is not doing anything a
table could not, and it is not worth compressing.

The previous iteration of this project skipped this check. A linear probe on
that task scored R-squared 0.998, which is why this one exists.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import N_MOVES, SQUARE_CODES  # noqa: E402
from chessdt.data import Corpus  # noqa: E402
from chessdt.splits import VAL  # noqa: E402


def features(batch) -> torch.Tensor:
    """One-hot squares plus side to move: everything the board encoder is given."""
    squares = F.one_hot(batch.codes[:, 0], SQUARE_CODES).flatten(1).float()
    return torch.cat([squares, batch.side[:, :1].float()], dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--eval-positions", type=int, default=200_000)
    ap.add_argument("--output", type=Path, default=Path("runs/probe_linear.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    corpus = Corpus(args.data, args.device)
    generator = torch.Generator(device=args.device).manual_seed(0)
    probe = torch.nn.Linear(64 * SQUARE_CODES + 1, N_MOVES).to(args.device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        batch = corpus.sample_train(args.batch, 1, generator)
        loss = F.cross_entropy(probe(features(batch)), batch.moves[:, 0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"  step {step:>5}  loss {loss.item():.4f}", flush=True)

    held_out = torch.nonzero(corpus.labels == VAL).flatten()
    pick = torch.randperm(len(held_out), generator=generator, device=args.device)
    held_out = held_out[pick[: args.eval_positions]]

    hits = 0
    with torch.no_grad():
        for i in range(0, len(held_out), 8192):
            batch = corpus.window(held_out[i : i + 8192], 1)
            hits += (probe(features(batch)).argmax(-1) == batch.moves[:, 0]).sum().item()

    top1 = hits / len(held_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "top1": top1,
                "positions": len(held_out),
                "training_positions": args.steps * args.batch,
            },
            indent=2,
        )
    )
    print(f"\nLinear probe top-1 on {len(held_out):,} held-out positions: {top1 * 100:.2f}%")
    print(f"Seen {args.steps * args.batch:,} training positions.")
    print("The transformer has to beat this by a wide margin to be worth compressing.")


if __name__ == "__main__":
    main()
