"""
Extract and encode a training corpus from a Lichess dump.

Streams the dump, keeps the games that pass the collection filters, replays
each one and writes a packed record per position. The dump is never written to
disk; only the encoded output is.

    python scripts/extract.py --out data/encoded --games 300000

Writes three files:

    positions.bin   N records of BOARD_RECORD_BYTES, one per ply
    moves.bin       N uint16 move indices, aligned with positions.bin
    games.npz       per-game metadata and the offset of each game's first ply

The per-game offsets are what make the Decision Transformer cheap to sample:
a K-ply window is a slice, with no rescanning and no game-boundary search.

Splitting happens later, in a separate pass over games.npz, so that the
expensive replay runs exactly once.
"""

import argparse
import sys
import time
from pathlib import Path

import chess.pgn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessdt.constants import DUMP_URL, MIN_PLIES, TARGET_GAMES  # noqa: E402
from chessdt.encoding import BOARD_RECORD_BYTES, MOVE_TO_INDEX, pack_board  # noqa: E402
from chessdt.lichess import RESULTS, Tally, classify, game_id, open_dump, parse_elo  # noqa: E402


class Corpus:
    """Accumulates encoded positions, appending as games arrive.

    Positions and moves stream straight to disk because the corpus is far too
    large to hold in memory. Game metadata is small enough to keep and is
    written once at the end.
    """

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self._positions = (out_dir / "positions.bin").open("wb")
        self._moves = (out_dir / "moves.bin").open("wb")

        self.ids: list[str] = []
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.results: list[int] = []
        self.white_elos: list[int] = []
        self.black_elos: list[int] = []
        self.total_positions = 0

    def add(self, game: chess.pgn.Game, identifier: str) -> int:
        """Replay a game and append one record per ply. Returns plies written.

        A position is paired with the move played *from* it, so the final
        position of a game is not stored: nothing was played from it.
        """
        board = game.board()
        packed = bytearray()
        indices: list[int] = []

        for move in game.mainline_moves():
            index = MOVE_TO_INDEX.get(move.uci())
            if index is None:
                # Should be unreachable: the vocabulary covers legal chess.
                # If it fires, the vocabulary is wrong and we want to know.
                raise KeyError(f"move {move.uci()} is outside the vocabulary")
            packed += pack_board(board)
            indices.append(index)
            board.push(move)

        if len(indices) < MIN_PLIES:
            return 0

        self._positions.write(packed)
        self._moves.write(np.asarray(indices, dtype=np.uint16).tobytes())

        self.ids.append(identifier)
        self.offsets.append(self.total_positions)
        self.lengths.append(len(indices))
        self.results.append(RESULTS[game.headers["Result"]])
        self.white_elos.append(parse_elo(game.headers.get("WhiteElo")) or 0)
        self.black_elos.append(parse_elo(game.headers.get("BlackElo")) or 0)
        self.total_positions += len(indices)
        return len(indices)

    def close(self) -> None:
        self._positions.close()
        self._moves.close()
        np.savez(
            self.out_dir / "games.npz",
            game_id=np.asarray(self.ids, dtype="S12"),
            offset=np.asarray(self.offsets, dtype=np.int64),
            length=np.asarray(self.lengths, dtype=np.int32),
            result=np.asarray(self.results, dtype=np.int8),
            white_elo=np.asarray(self.white_elos, dtype=np.int16),
            black_elo=np.asarray(self.black_elos, dtype=np.int16),
            record_bytes=np.int32(BOARD_RECORD_BYTES),
        )


def extract(url: str, out_dir: Path, target_games: int, report_every: int) -> None:
    handle, counter = open_dump(url)
    corpus = Corpus(out_dir)
    tally = Tally()
    skipped = 0
    started = time.monotonic()

    try:
        while len(corpus.ids) < target_games:
            try:
                game = chess.pgn.read_game(handle)
            except (ValueError, UnicodeDecodeError):
                skipped += 1
                continue
            if game is None:
                print("\nReached the end of the dump before hitting the target.")
                break

            if not classify(game.headers, tally):
                continue
            if game.headers.get("Result") not in RESULTS:
                continue
            identifier = game_id(game.headers.get("Site"))
            if identifier is None:
                skipped += 1
                continue

            try:
                corpus.add(game, identifier)
            except KeyError:
                raise
            except ValueError:
                skipped += 1

            if len(corpus.ids) and len(corpus.ids) % report_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"  {len(corpus.ids):>8,} games  "
                    f"{corpus.total_positions:>12,} positions  "
                    f"{corpus.total_positions / elapsed:>9,.0f} pos/s  "
                    f"{counter.bytes_read / 1024**3:>5.2f} GB read",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nInterrupted — closing the corpus cleanly so far.")
    finally:
        corpus.close()
        handle.close()

    summarise(corpus, tally, counter, skipped, time.monotonic() - started)


def summarise(corpus, tally, counter, skipped, elapsed: float) -> None:
    positions = corpus.total_positions
    size_gb = positions * BOARD_RECORD_BYTES / 1024**3

    print(f"\nWrote {len(corpus.ids):,} games, {positions:,} positions in {elapsed / 60:.1f} min")
    print(f"  scanned             {tally.scanned:>14,}")
    print(f"  pass rate           {tally.pass_rate * 100:>13.2f}%")
    print(f"  skipped (malformed) {skipped:>14,}")
    print(f"  mean plies per game {positions / max(len(corpus.ids), 1):>14.1f}")
    print(f"  streamed            {counter.bytes_read / 1024**3:>13.2f} GB")
    print(f"  encoded on disk     {size_gb:>13.2f} GB")
    print(f"  throughput          {positions / max(elapsed, 1e-9):>13,.0f} positions/s")
    print(f"\nOutput in {corpus.out_dir}/ — next step is the split pass.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DUMP_URL)
    ap.add_argument("--out", type=Path, default=Path("data/encoded"))
    ap.add_argument("--games", type=int, default=TARGET_GAMES)
    ap.add_argument("--report-every", type=int, default=10_000)
    args = ap.parse_args()

    print(f"Streaming {args.url}")
    print(f"Target {args.games:,} games into {args.out}/\n")
    extract(args.url, args.out, args.games, args.report_every)


if __name__ == "__main__":
    main()
