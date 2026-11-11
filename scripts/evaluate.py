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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import (  # noqa: E402
    ELO_CENTRE,
    ELO_SCALE,
    N_MOVES,
)
from chessdt.data import Batch, Corpus  # noqa: E402
from chessdt.evaluate import PuzzleSet, alone, compare, score  # noqa: E402
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
        puzzles = PuzzleSet.load(args.puzzles, args.puzzle_sample, device)
        report["puzzle_set"] = p = puzzles.solve(model)
        print(f"  solved        {p['solved'] * 100:6.2f}%")
        print(f"  in top three  {p['solution_in_top3'] * 100:6.2f}%")
        print(f"  mate in one   {p['mate_in_one'] * 100:6.2f}%   ({p['mate_in_one_puzzles']:,})")
        for band, r in p["by_rating"].items():
            print(f"  rated {band:<10} {r['solved'] * 100:6.2f}%   ({r['puzzles']:,})")

        print("\nSTEERING — the same puzzles, posed to different players")
        print(f"  {'conditioned as':<26}{'solved':>9}{'top-3':>9}")
        report["steering"] = {}
        for label, outcome, elo in (
            ("winning, rated 2400", 1.0, 2400.0),
            ("winning, rated 1200", 1.0, 1200.0),
            ("neutral, rated 2000", 0.0, 2000.0),
            ("losing, rated 2400", -1.0, 2400.0),
            ("losing, rated 1200", -1.0, 1200.0),
        ):
            r = puzzles.solve(model, outcome=outcome, elo=elo)
            report["steering"][label] = r
            print(f"  {label:<26}{r['solved'] * 100:>8.2f}%{r['solution_in_top3'] * 100:>8.2f}%")
        top, bottom = (
            report["steering"]["winning, rated 2400"],
            report["steering"]["losing, rated 1200"],
        )
        spread = (top["solved"] - bottom["solved"]) * 100
        print(f"  strongest minus weakest: {spread:+.2f} points")

    out = args.checkpoint.parent / "evaluation.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
