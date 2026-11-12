"""
Find every position with a given tactic in the held-out splits, once.

    python scripts/build_tactics_set.py --corpus data/encoded
    python scripts/build_tactics_set.py --corpus data/encoded --motif fork --splits train val test

Mate in one is rare, about 0.4% of positions, so sampling a few thousand
positions finds only a few hundred and leaves the measurement uncertain by
several points. Scanning all of validation and test finds several thousand and
brings that down to under one. This does the scan once, in parallel, and saves
the positions and their mating moves so every later model is measured on the
same fixed set.

The motifs are mate_in_one (the default), mate_in_two (a forced mate that begins
with a check), free_piece (a capture that wins a piece) and fork; see
`chessdt.tactics` for exact definitions. Writes `<motif>.npz` into `--out`:

    position   global index of each position in the corpus
    split      its split label, so validation and test stay separate
    offset     where each position's mating moves start in `moves`
    moves      move indices of every mating move, concatenated
"""

import argparse
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.encoding import BOARD_RECORD_BYTES  # noqa: E402
from chessdt.splits import TEST, TRAIN, VAL  # noqa: E402
from chessdt.tactics import MOTIFS, MOVE_TO_SQUARE, NO_PREVIOUS, scan_block  # noqa: E402

BLOCK = 4000  # positions per task; small enough to balance, large enough to amortise

#: In a container, os.cpu_count() reports the host's cores, not the share the
#: container is allowed, and a pool that size can exhaust memory. Twelve is
#: what the cluster's notebook servers are given; --workers overrides it.
DEFAULT_WORKERS = 12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/encoded"))
    ap.add_argument("--out", type=Path, default=Path("data/tactics"))
    ap.add_argument("--motif", choices=sorted(MOTIFS), default="mate_in_one")
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
        f"Scanning {len(wanted):,} positions ({', '.join(args.splits)}) for {args.motif} ...",
        flush=True,
    )

    # Where the previous move of each position's game landed, so that a recapture
    # can be told from a capture. A game's first position has none.
    played = np.fromfile(args.corpus / "moves.bin", dtype=np.uint16)
    first = np.zeros(len(records), dtype=bool)
    first[np.load(args.corpus / "games.npz")["offset"]] = True
    previous = np.full(len(wanted), NO_PREVIOUS, dtype=np.uint8)
    follows = ~first[wanted]
    previous[follows] = MOVE_TO_SQUARE[played[wanted[follows] - 1]]

    blocks = [wanted[i : i + BLOCK] for i in range(0, len(wanted), BLOCK)]
    payloads = [
        (np.ascontiguousarray(records[b]).tobytes(), previous[i : i + BLOCK].tobytes())
        for i, b in zip(range(0, len(wanted), BLOCK), blocks, strict=True)
    ]

    started = time.monotonic()
    positions, split, offsets, moves = [], [], [0], []
    with Pool(args.workers) as pool:
        for done, (block, found) in enumerate(
            zip(blocks, pool.imap(partial(scan_block, motif=args.motif), payloads), strict=True),
            start=1,
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
        args.out / f"{args.motif}.npz",
        position=np.asarray(positions, dtype=np.int64),
        split=np.asarray(split, dtype=np.uint8),
        offset=np.asarray(offsets, dtype=np.int64),
        moves=np.asarray(moves, dtype=np.uint16),
    )
    split = np.asarray(split)
    for name in args.splits:
        n = int((split == chosen[name]).sum())
        print(
            f"  {name:<6}{n:>8,} positions with {args.motif} "
            f"(standard error about {(0.25 / max(n, 1)) ** 0.5 * 100:.1f} points)"
        )
    print(f"Wrote {args.out / (args.motif + '.npz')}")


if __name__ == "__main__":
    main()
