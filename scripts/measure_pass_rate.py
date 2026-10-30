"""
Measure what share of Lichess games survive our collection filters.

The pass rate is the one number that sizes the whole extraction job. It decides
how many raw games we read, how many bytes we stream, and how long that takes.
None of the rest of collection can be planned until it is known.

This streams a monthly dump over HTTP, decompresses on the fly, and reads only
the PGN headers, which is far cheaper than replaying moves. Nothing is written
to disk and the stream is abandoned once the sample is in.

    python scripts/measure_pass_rate.py --limit 200000

Pass --parse-moves to also measure ply counts and move-replay throughput on a
smaller sample. That second number is what sizes the real encoding job, since
encoding has to replay every game to reach a position per ply.
"""

import argparse
import sys
import time
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import (  # noqa: E402
    DUMP_GAMES,
    DUMP_SIZE_GB,
    DUMP_URL,
    MIN_CLOCK_SECONDS,
    MIN_ELO,
    MIN_PLIES,
    TARGET_GAMES,
)
from chessdt.lichess import CountingReader, Tally, classify, open_dump  # noqa: E402


def report(tally: Tally, counter: CountingReader, elapsed: float) -> None:
    mb = counter.bytes_read / 1024 / 1024
    rate = tally.scanned / elapsed if elapsed else 0.0

    print(
        f"\nScanned {tally.scanned:,} games in {elapsed:.1f}s "
        f"({rate:,.0f} games/s, {mb:.1f} MB streamed)\n"
    )

    print("FILTER BREAKDOWN (cumulative survivors)")
    rows = [
        ("total scanned", tally.scanned),
        ("both players rated", tally.rated),
        (f"both >= {MIN_ELO}", tally.rating_ok),
        (f"clock >= {MIN_CLOCK_SECONDS}s", tally.clock_ok),
        ("termination usable", tally.termination_ok),
    ]
    for label, count in rows:
        share = count / tally.scanned * 100 if tally.scanned else 0.0
        print(f"  {label:<28}{count:>12,}  {share:>6.2f}%")

    if tally.plies:
        ordered = sorted(tally.plies)
        median = ordered[len(ordered) // 2]
        short = sum(1 for p in ordered if p < MIN_PLIES) / len(ordered) * 100
        print(f"\n  median plies per game{median:>19,}")
        print(f"  under {MIN_PLIES} plies{' ' * 19}{short:>6.2f}%  (removed by the length filter)")

    if not tally.pass_rate:
        print("\nNo games passed. Check the filters before projecting anything.")
        return

    raw_needed = TARGET_GAMES / tally.pass_rate
    share_of_dump = raw_needed / DUMP_GAMES
    gb = share_of_dump * DUMP_SIZE_GB
    minutes = raw_needed / rate / 60 if rate else 0.0

    print(f"\nPASS RATE: {tally.pass_rate * 100:.2f}%\n")
    print(f"PROJECTION for {TARGET_GAMES:,} filtered games")
    print(f"  raw games to read         {raw_needed:>14,.0f}")
    print(f"  share of the dump         {share_of_dump * 100:>13.2f}%")
    print(f"  bytes to stream           {gb:>13.2f} GB")
    print(f"  header-scan wall time     {minutes:>13.1f} min")
    print("\n  Encoding is slower than this: it has to replay every game to")
    print("  reach a position per ply. Use --parse-moves to size that.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DUMP_URL)
    ap.add_argument("--limit", type=int, default=200_000, help="games to scan")
    ap.add_argument(
        "--parse-moves",
        action="store_true",
        help="also replay moves, to measure ply counts and encoding throughput",
    )
    args = ap.parse_args()

    print(f"Streaming {args.url}")
    print(
        f"Sampling {args.limit:,} games, headers only"
        f"{' plus move replay' if args.parse_moves else ''}.\n"
    )

    handle, counter = open_dump(args.url)
    tally = Tally()
    started = time.monotonic()

    try:
        while tally.scanned < args.limit:
            if args.parse_moves:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                if classify(game.headers, tally):
                    tally.plies.append(sum(1 for _ in game.mainline_moves()))
            else:
                headers = chess.pgn.read_headers(handle)
                if headers is None:
                    break
                classify(headers, tally)

            if tally.scanned % 25_000 == 0:
                print(f"  {tally.scanned:,} scanned, {tally.termination_ok:,} passing", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — reporting on what was scanned so far.")
    finally:
        handle.close()

    report(tally, counter, time.monotonic() - started)


if __name__ == "__main__":
    main()
