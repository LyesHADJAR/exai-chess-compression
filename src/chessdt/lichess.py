"""
Reading Lichess monthly dumps.

The dumps are ~29 GB of zstd-compressed PGN served over HTTP. Nothing here
writes to disk: the stream is decompressed in memory and abandoned once enough
games have been read.
"""

import io
import urllib.request
from dataclasses import dataclass, field

import zstandard

from .constants import MIN_CLOCK_SECONDS, MIN_ELO

#: Terminations we keep. An abandoned game teaches nothing; a time forfeit is
#: a real game that happened to end on the clock.
GOOD_TERMINATIONS = frozenset({"Normal", "Time forfeit"})

#: Results that encode a finished game, mapped to white's score.
RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def initial_clock(time_control: str | None) -> int | None:
    """Base seconds from a PGN TimeControl header, or None if it has none.

    Headers look like "600+0" or "180+2". Correspondence games carry "-".
    """
    if not time_control or time_control == "-":
        return None
    base = time_control.split("+")[0]
    return int(base) if base.isdigit() else None


def parse_elo(value: str | None) -> int | None:
    """Rating from a PGN Elo header. Unrated and provisional show as '?'."""
    return int(value) if value and value.isdigit() else None


def game_id(site: str | None) -> str | None:
    """The Lichess game ID from a Site header such as https://lichess.org/abcd1234.

    This is the key splits are assigned from, so a game without one cannot be
    placed deterministically and has to be dropped.
    """
    if not site:
        return None
    tail = site.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


@dataclass
class Tally:
    """Cumulative survivors at each filter stage, in the order applied."""

    scanned: int = 0
    rated: int = 0
    rating_ok: int = 0
    clock_ok: int = 0
    termination_ok: int = 0
    plies: list[int] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.termination_ok / self.scanned if self.scanned else 0.0


def classify(headers, tally: Tally) -> bool:
    """Apply the collection filters in order, recording survivors at each stage."""
    tally.scanned += 1

    white = parse_elo(headers.get("WhiteElo"))
    black = parse_elo(headers.get("BlackElo"))
    if white is None or black is None:
        return False
    tally.rated += 1

    if white < MIN_ELO or black < MIN_ELO:
        return False
    tally.rating_ok += 1

    clock = initial_clock(headers.get("TimeControl"))
    if clock is None or clock < MIN_CLOCK_SECONDS:
        return False
    tally.clock_ok += 1

    if headers.get("Termination") not in GOOD_TERMINATIONS:
        return False
    tally.termination_ok += 1
    return True


class CountingReader(io.RawIOBase):
    """Wraps a byte stream and records how many compressed bytes were pulled."""

    def __init__(self, stream):
        self._stream = stream
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self._stream.read(len(buffer))
        if not chunk:
            return 0
        buffer[: len(chunk)] = chunk
        self.bytes_read += len(chunk)
        return len(chunk)


def open_dump(url: str) -> tuple[io.TextIOBase, CountingReader]:
    """Open a remote .pgn.zst as decompressed text, without touching disk.

    Returns the text handle and the counter, so callers can report how many
    compressed bytes a run actually cost.
    """
    response = urllib.request.urlopen(url)  # noqa: S310 - fixed https URL
    counter = CountingReader(response)
    reader = zstandard.ZstdDecompressor().stream_reader(counter)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), counter
