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
from dataclasses import dataclass, field
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


@dataclass
class Collected:
    """Parsed puzzles in their stored form.

    Boards are packed to their 36 bytes as each row is read rather than kept
    as python-chess objects. A Board is on the order of a kilobyte live, so
    holding six million of them costs several gigabytes and swaps the machine;
    packing on the way past costs about 550 MB for the whole database.
    """

    packed: bytearray = field(default_factory=bytearray)
    solution: list[int] = field(default_factory=list)
    puzzle_id: list[str] = field(default_factory=list)
    rating: list[int] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    line_plies: list[int] = field(default_factory=list)
    rejected: int = 0

    def __len__(self) -> int:
        return len(self.solution)

    def add(self, puzzle) -> None:
        self.packed += pack_board(puzzle.board)
        self.solution.append(puzzle.solution_index)
        self.puzzle_id.append(puzzle.puzzle_id)
        self.rating.append(puzzle.rating)
        self.themes.append(puzzle.themes)
        self.line_plies.append(puzzle.line_plies)

    def positions(self) -> np.ndarray:
        return np.frombuffer(bytes(self.packed), dtype=np.uint8).reshape(-1, BOARD_RECORD_BYTES)


def read_puzzles(url: str, limit: int | None) -> tuple[Collected, CountingReader]:
    """Parse puzzles until the limit, packing each board as it is read."""
    rows, counter = open_puzzles(url)
    collected = Collected()

    for row in rows:
        puzzle = parse_puzzle(row)
        if puzzle is None:
            collected.rejected += 1
            continue
        collected.add(puzzle)

        if len(collected) % 500_000 == 0:
            print(f"  {len(collected):,} parsed", flush=True)
        if limit and len(collected) >= limit:
            break

    return collected, counter


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


def write(out: Path, collected: Collected, keep: np.ndarray, deduped: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)

    (out / "positions.bin").write_bytes(collected.positions()[keep].tobytes())
    (out / "moves.bin").write_bytes(np.asarray(collected.solution, dtype=np.uint16)[keep].tobytes())

    # Compressed: the themes column is fixed-width and about 70% null padding,
    # which costs 750 MB uncompressed and almost nothing zipped. Metadata is
    # read once, so decompression cost does not matter.
    np.savez_compressed(
        out / "puzzles.npz",
        puzzle_id=np.asarray(collected.puzzle_id, dtype="S8")[keep],
        rating=np.asarray(collected.rating, dtype=np.int16)[keep],
        themes=np.asarray([t[:THEME_BYTES] for t in collected.themes], dtype=f"S{THEME_BYTES}")[
            keep
        ],
        line_plies=np.asarray(collected.line_plies, dtype=np.int8)[keep],
        deduped=np.bool_(deduped),
    )


def report(collected: Collected, leaked: int, counter, elapsed: float, deduped: bool) -> None:
    total = len(collected)
    kept = total - leaked

    print(
        f"\nParsed {total:,} puzzles in {elapsed / 60:.1f} min "
        f"({counter.bytes_read / 1024**2:.0f} MB streamed)"
    )
    print(f"  rejected as malformed  {collected.rejected:>12,}")

    if deduped:
        share = leaked / total * 100 if total else 0.0
        print(f"  dropped, seen in train {leaked:>12,}  ({share:.2f}%)")
    else:
        print("  dropped, seen in train           skipped")
    print(f"  kept                   {kept:>12,}")

    if total:
        ratings = np.asarray(collected.rating)
        print(f"\n  rating range {ratings.min()}-{ratings.max()}, median {int(np.median(ratings))}")
        mate_in_one = sum(1 for t in collected.themes if "mateIn1" in t)
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
    collected, counter = read_puzzles(args.url, args.limit)
    if not len(collected):
        raise SystemExit("no puzzles parsed; check the URL and the CSV columns")

    leaked = 0
    keep = np.ones(len(collected), dtype=bool)
    if not args.skip_dedupe:
        print(f"Hashing the training corpus at {args.corpus} ...", flush=True)
        train = train_keys_of(args.corpus)
        keep = ~leaked_mask(position_keys(collected.positions()), train)
        leaked = int((~keep).sum())

    write(args.out, collected, keep, deduped=not args.skip_dedupe)
    report(collected, leaked, counter, time.monotonic() - started, deduped=not args.skip_dedupe)
    print(f"\nWrote {args.out}/")


if __name__ == "__main__":
    main()
