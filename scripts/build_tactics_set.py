"""
Find every mate-in-one position in the held-out splits, once.

    python scripts/build_tactics_set.py --corpus data/encoded

Mate in one is rare, about 0.4% of positions, so sampling a few thousand
positions finds only a few hundred and leaves the measurement uncertain by
several points. Scanning all of validation and test finds several thousand and
brings that down to under one. This does the scan once, in parallel, and saves
the positions and their mating moves so every later model is measured on the
same fixed set.

Writes `mate_in_one.npz` into `--out`:

    position   global index of each position in the corpus
    split      its split label, so validation and test stay separate
    offset     where each position's mating moves start in `moves`
    moves      move indices of every mating move, concatenated
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.encoding import BOARD_RECORD_BYTES  # noqa: E402
from chessdt.splits import TEST, TRAIN, VAL  # noqa: E402
from chessdt.tactics import scan_mates  # noqa: E402

BLOCK = 4000  # positions per task; small enough to balance, large enough to amortise

#: In a container, os.cpu_count() reports the host's cores, not the share the
#: container is allowed, and a pool that size can exhaust memory. Twelve is
#: what the cluster's notebook servers are given; --workers overrides it.
DEFAULT_WORKERS = 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/encoded"))
    ap.add_argument("--out", type=Path, default=Path("data/tactics"))
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, DEFAULT_WORKERS))
    ap.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["val", "test"],
        help="which splits to scan; train is needed to emphasise tactics while training",
    )
    args = ap.parse_args()

    records = np.memmap(args.corpus / "positions.bin", dtype=np.uint8, mode="r")
    records = records.reshape(-1, BOARD_RECORD_BYTES)
    labels = np.load(args.corpus / "split_labels.npy")
    chosen = {"train": int(TRAIN), "val": int(VAL), "test": int(TEST)}
    wanted = np.flatnonzero(np.isin(labels, [chosen[name] for name in args.splits]))
    print(
        f"Scanning {len(wanted):,} positions ({', '.join(args.splits)}) for a mate in one ...",
        flush=True,
    )

    blocks = [wanted[i : i + BLOCK] for i in range(0, len(wanted), BLOCK)]
    payloads = [np.ascontiguousarray(records[b]).tobytes() for b in blocks]

    started = time.monotonic()
    positions, split, offsets, moves = [], [], [0], []
    with Pool(args.workers) as pool:
        for done, (block, found) in enumerate(
            zip(blocks, pool.imap(scan_mates, payloads), strict=True), start=1
        ):
            for row, mates in found:
                positions.append(int(block[row]))
                split.append(int(labels[block[row]]))
                moves.extend(mates)
                offsets.append(len(moves))
            if done % 50 == 0 or done == len(blocks):
                print(f"  {done}/{len(blocks)} blocks, {len(positions):,} found, "
                      f"{time.monotonic() - started:.0f}s", flush=True)  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "mate_in_one.npz",
        position=np.asarray(positions, dtype=np.int64),
        split=np.asarray(split, dtype=np.uint8),
        offset=np.asarray(offsets, dtype=np.int64),
        moves=np.asarray(moves, dtype=np.uint16),
    )
    split = np.asarray(split)
    for name in args.splits:
        n = int((split == chosen[name]).sum())
        print(
            f"  {name:<6}{n:>8,} positions with a mate in one "
            f"(standard error about {(0.25 / max(n, 1)) ** 0.5 * 100:.1f} points)"
        )
    print(f"Wrote {args.out / 'mate_in_one.npz'}")


if __name__ == "__main__":
    main()
