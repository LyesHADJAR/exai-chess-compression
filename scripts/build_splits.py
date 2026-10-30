"""
Split an encoded corpus into train, validation and test.

Runs over the output of `extract.py`. No move replay and no network: this is a
metadata pass over `games.npz` plus one hash of each position, so it is fast
even on the full corpus.

    python scripts/build_splits.py --corpus data/encoded

Writes `split_labels.npy` beside the corpus: one uint8 per position, where
0 is train, 1 validation, 2 test and 3 dropped as leaked. Training code selects
a split with `np.flatnonzero(labels == TRAIN)`.

The run fails loudly if any held-out position still occurs in training. That
check is the point of the script; a deduplication that quietly did nothing
looks exactly like one that worked.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.encoding import BOARD_RECORD_BYTES  # noqa: E402
from chessdt.splits import (  # noqa: E402
    assign_games,
    drop_leaked,
    expand_to_positions,
    position_keys,
    summarise,
    verify,
)


def load_corpus(corpus: Path) -> tuple[np.ndarray, np.lib.npyio.NpzFile]:
    """Memory-map the positions and read the game index."""
    meta = np.load(corpus / "games.npz")
    positions = np.memmap(corpus / "positions.bin", dtype=np.uint8, mode="r")
    positions = positions.reshape(-1, BOARD_RECORD_BYTES)

    total = int(meta["offset"][-1] + meta["length"][-1])
    if len(positions) != total:
        raise ValueError(
            f"positions.bin holds {len(positions):,} records but games.npz "
            f"accounts for {total:,}; the corpus is inconsistent"
        )
    return positions, meta


def report(counts: dict, dropped: int, elapsed: float) -> None:
    held = counts["val"]["positions"] + counts["test"]["positions"] + dropped
    leak_rate = dropped / held * 100 if held else 0.0

    print(f"\nSplit {sum(c['games'] for c in counts.values()):,} games in {elapsed:.1f}s\n")
    print(f"  {'split':<10}{'games':>12}{'positions':>16}")
    for name in ("train", "val", "test", "dropped"):
        counts_for = counts[name]
        print(f"  {name:<10}{counts_for['games']:>12,}{counts_for['positions']:>16,}")

    print(f"\n  {dropped:,} held-out positions dropped as leaked ({leak_rate:.1f}% of held-out)")
    print("  Mostly opening theory, which repeats across games by nature.")
    print("  What remains is weighted toward middlegames and endgames, which is")
    print("  where compression damage actually shows.")
    print("\n  Verified: no held-out position occurs in train.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/encoded"))
    ap.add_argument("--out", type=Path, default=None, help="defaults to the corpus directory")
    args = ap.parse_args()

    out_dir = args.out or args.corpus
    started = time.monotonic()

    positions, meta = load_corpus(args.corpus)
    print(f"Corpus: {len(meta['game_id']):,} games, {len(positions):,} positions")

    print("Hashing positions ...", flush=True)
    keys = position_keys(positions)

    print("Assigning games to splits ...", flush=True)
    game_labels = assign_games(meta["game_id"])
    labels = expand_to_positions(game_labels, meta["offset"], meta["length"])

    print("Dropping held-out positions that occur in train ...", flush=True)
    dropped = drop_leaked(labels, keys)

    print("Verifying ...", flush=True)
    verify(labels, keys)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "split_labels.npy", labels)
    np.save(out_dir / "game_splits.npy", game_labels)

    report(summarise(labels, game_labels), dropped, time.monotonic() - started)
    print(f"\nWrote {out_dir / 'split_labels.npy'}")


if __name__ == "__main__":
    main()
