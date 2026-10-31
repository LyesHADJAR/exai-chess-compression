"""
Build the puzzle evaluation set.

Streams the Lichess puzzle database, resolves each puzzle to the position the
solver is actually shown, and drops any whose position also occurs in the
training corpus.

    python scripts/build_puzzles.py --corpus data/encoded --out data/puzzles

That deduplication is mandatory, not a precaution. Lichess puzzles are mined
from Lichess games, so overlap with a corpus drawn from the same source is
likely by construction rather than by accident. Without it the gate measures
recall of positions the model was trained on.

Writes, in the same layout as the game corpus so loaders are shared:

    positions.bin   one packed record per puzzle, the position after the
                    opponent's setup move
    moves.bin       uint16 index of the move that solves it
    puzzles.npz     id, rating, themes, line length, and whether the set was
                    deduplicated
"""

import argparse
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.encoding import BOARD_RECORD_BYTES, pack_board  # noqa: E402
from chessdt.lichess import CountingReader  # noqa: E402
from chessdt.puzzles import REQUIRED_COLUMNS, THEME_BYTES, parse_puzzle  # noqa: E402
from chessdt.splits import TRAIN, leaked_mask, position_keys  # noqa: E402

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def open_puzzles(url: str) -> tuple[csv.DictReader, CountingReader]:
    """Open the remote puzzle CSV as rows, without touching disk."""
    response = urllib.request.urlopen(url)  # noqa: S310 - fixed https URL
    counter = CountingReader(response)
    reader = zstandard.ZstdDecompressor().stream_reader(counter)
    handle = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

    rows = csv.DictReader(handle)
    missing = [c for c in REQUIRED_COLUMNS if c not in (rows.fieldnames or ())]
    if missing:
        raise ValueError(f"puzzle CSV is missing {missing}; got {rows.fieldnames}")
    return rows, counter


def read_puzzles(url: str, limit: int | None) -> tuple[list, int, CountingReader]:
    """Parse puzzles until the limit, returning them and how many were rejected."""
    rows, counter = open_puzzles(url)
    puzzles, rejected = [], 0

    for row in rows:
        puzzle = parse_puzzle(row)
        if puzzle is None:
            rejected += 1
            continue
        puzzles.append(puzzle)

        if len(puzzles) % 250_000 == 0:
            print(f"  {len(puzzles):,} parsed", flush=True)
        if limit and len(puzzles) >= limit:
            break

    return puzzles, rejected, counter


def train_keys_of(corpus: Path) -> np.ndarray:
    """Identity keys of every training position in an encoded corpus."""
    labels = np.load(corpus / "split_labels.npy")
    positions = np.memmap(corpus / "positions.bin", dtype=np.uint8, mode="r")
    positions = positions.reshape(-1, BOARD_RECORD_BYTES)

    if len(labels) != len(positions):
        raise ValueError(
            f"{len(labels):,} split labels against {len(positions):,} positions; "
            "run build_splits.py on this corpus first"
        )
    return position_keys(positions)[labels == TRAIN]


def write(out: Path, puzzles: list, keep: np.ndarray, deduped: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    kept = [p for p, k in zip(puzzles, keep, strict=True) if k]

    with (out / "positions.bin").open("wb") as handle:
        handle.write(b"".join(pack_board(p.board) for p in kept))
    with (out / "moves.bin").open("wb") as handle:
        handle.write(np.asarray([p.solution_index for p in kept], dtype=np.uint16).tobytes())

    np.savez(
        out / "puzzles.npz",
        puzzle_id=np.asarray([p.puzzle_id for p in kept], dtype="S8"),
        rating=np.asarray([p.rating for p in kept], dtype=np.int16),
        themes=np.asarray([p.themes[:THEME_BYTES] for p in kept], dtype=f"S{THEME_BYTES}"),
        line_plies=np.asarray([p.line_plies for p in kept], dtype=np.int8),
        deduped=np.bool_(deduped),
    )


def report(puzzles, rejected, leaked, counter, elapsed, deduped) -> None:
    kept = len(puzzles) - leaked
    print(
        f"\nParsed {len(puzzles):,} puzzles in {elapsed / 60:.1f} min "
        f"({counter.bytes_read / 1024**2:.0f} MB streamed)"
    )
    print(f"  rejected as malformed  {rejected:>12,}")

    if deduped:
        share = leaked / len(puzzles) * 100 if puzzles else 0.0
        print(f"  dropped, seen in train {leaked:>12,}  ({share:.2f}%)")
    else:
        print("  dropped, seen in train           skipped")
    print(f"  kept                   {kept:>12,}")

    if puzzles:
        ratings = np.asarray([p.rating for p in puzzles])
        print(f"\n  rating range {ratings.min()}-{ratings.max()}, median {int(np.median(ratings))}")
        mate_in_one = sum(1 for p in puzzles if "mateIn1" in p.themes)
        print(f"  mate-in-one puzzles {mate_in_one:,}")

    if not deduped:
        print("\n  WARNING: this set was NOT deduplicated against the training")
        print("  corpus. Any gate computed from it measures recall, not skill.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=PUZZLE_URL)
    ap.add_argument("--corpus", type=Path, default=Path("data/encoded"))
    ap.add_argument("--out", type=Path, default=Path("data/puzzles"))
    ap.add_argument("--limit", type=int, default=None, help="stop after this many puzzles")
    ap.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="encode without checking against the corpus; the output records this",
    )
    args = ap.parse_args()

    started = time.monotonic()
    print(f"Streaming {args.url}")
    puzzles, rejected, counter = read_puzzles(args.url, args.limit)
    if not puzzles:
        raise SystemExit("no puzzles parsed; check the URL and the CSV columns")

    leaked = 0
    keep = np.ones(len(puzzles), dtype=bool)
    if not args.skip_dedupe:
        print(f"Hashing the training corpus at {args.corpus} ...", flush=True)
        train = train_keys_of(args.corpus)
        keys = position_keys(
            np.frombuffer(b"".join(pack_board(p.board) for p in puzzles), dtype=np.uint8).reshape(
                -1, BOARD_RECORD_BYTES
            )
        )
        keep = ~leaked_mask(keys, train)
        leaked = int((~keep).sum())

    write(args.out, puzzles, keep, deduped=not args.skip_dedupe)
    report(
        puzzles, rejected, leaked, counter, time.monotonic() - started, deduped=not args.skip_dedupe
    )
    print(f"\nWrote {args.out}/")


if __name__ == "__main__":
    main()
