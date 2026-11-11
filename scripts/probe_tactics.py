"""
Measure how well a model finds mate in one, against the humans it imitates.

    python scripts/probe_tactics.py runs/reference/model_best.pt

Scans held-out game positions for ones where a mate in one exists, and reports
how often the human played it and how often the model's first choice is a
mating move. The human rate is the ceiling; the gap to it is the weakness.

Writes `tactics.json` beside the checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.data import Corpus  # noqa: E402
from chessdt.encoding import MOVE_VOCABULARY, unpack_board  # noqa: E402
from chessdt.evaluate import predict  # noqa: E402
from chessdt.model import load_checkpoint  # noqa: E402
from chessdt.splits import TEST  # noqa: E402
from chessdt.tactics import mating_moves  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/encoded"))
    ap.add_argument("--windows", type=int, default=6000, help="test windows to scan")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, _ = load_checkpoint(args.checkpoint, args.device)
    corpus = Corpus(args.data, args.device)
    plies = model.cfg.context_plies
    starts = corpus.split_starts(int(TEST), plies, limit=args.windows)

    scanned = found = human = machine = 0
    for i in range(0, len(starts), 512):
        chunk = starts[i : i + 512]
        batch = corpus.window(chunk, plies)
        keep = batch.valid & (batch.labels == int(TEST))
        first_choice = predict(model, batch, args.device).argmax(-1)

        index = (chunk[:, None] + torch.arange(plies, device=args.device)).clamp(
            max=len(corpus) - 1
        )
        records = corpus.records[index][keep].cpu().numpy()
        played = batch.moves[keep].cpu().numpy()
        chosen = first_choice[keep].cpu().numpy()

        for record, h, m in zip(records, played, chosen, strict=True):
            scanned += 1
            mates = mating_moves(unpack_board(record.tobytes()))
            if mates:
                found += 1
                human += MOVE_VOCABULARY[h] in mates
                machine += MOVE_VOCABULARY[m] in mates

    result = {
        "positions_scanned": scanned,
        "positions_with_mate_in_one": found,
        "human_plays_mate": human / found,
        "model_plays_mate": machine / found,
        "standard_error": (0.25 / found) ** 0.5,
    }
    (args.checkpoint.parent / "tactics.json").write_text(json.dumps(result, indent=2))

    print(f"{scanned:,} held-out positions, {found:,} with a mate in one ({found / scanned:.2%})")
    print(f"  a human played the mate     {result['human_plays_mate'] * 100:5.1f}%")
    print(f"  the model's first choice    {result['model_plays_mate'] * 100:5.1f}%")
    print(f"  (standard error about {result['standard_error'] * 100:.1f} points)")


if __name__ == "__main__":
    main()
