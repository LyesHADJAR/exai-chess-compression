"""
Reading Lichess monthly dumps.

The dumps are ~29 GB of zstd-compressed PGN served over HTTP. Nothing here
writes to disk: the stream is decompressed in memory and abandoned once enough
games have been read.
"""

import http.client
import io
import time
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


class ResumableStream(io.RawIOBase):
    """An HTTP download that survives dropped connections.

    A multi-gigabyte stream over a long-lived connection will be cut sooner or
    later, and a cut connection reads exactly like the end of the file. Taken
    at face value that truncates the corpus silently, which is how a run once
    stopped at 161,062 games of 300,000 and reported success.

    So the stream knows how long the file is. If the data stops before that,
    it reconnects and asks for the file from the byte it had reached. The
    consumer sees one unbroken sequence of bytes, so a decompressor reading
    from this never learns anything happened.

    It is patient by default: forty consecutive failed attempts, backing off
    to a minute apart, rides out an outage of over half an hour before giving
    up. Any successful read resets the count.
    """

    def __init__(
        self,
        url: str,
        opener=urllib.request.urlopen,
        retries: int = 40,
        timeout: float = 60.0,
        pause=time.sleep,
    ):
        self.url = url
        self.bytes_read = 0
        self.total: int | None = None
        self.reconnects = 0
        self._opener, self._retries, self._timeout, self._pause = opener, retries, timeout, pause
        self._response = None
        self._connect()

    def _connect(self) -> None:
        headers = {"Range": f"bytes={self.bytes_read}-"} if self.bytes_read else {}
        request = urllib.request.Request(self.url, headers=headers)  # noqa: S310 - fixed https URL
        response = self._opener(request, timeout=self._timeout)

        if self.bytes_read and getattr(response, "status", 206) != 206:
            raise OSError("server ignored the Range header; cannot resume the download")
        if self.total is None:
            length = response.headers.get("Content-Length")
            self.total = int(length) if length else None
        self._response = response

    def _finished(self) -> bool:
        return self.total is None or self.bytes_read >= self.total

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        failures = 0
        while True:
            try:
                chunk = self._response.read(len(buffer))
            except (OSError, http.client.HTTPException):
                chunk = b""

            if chunk:
                buffer[: len(chunk)] = chunk
                self.bytes_read += len(chunk)
                return len(chunk)
            if self._finished():
                return 0

            # The data stopped short of the file's length: the connection was
            # cut. Back off, then pick up from the byte we had reached.
            while True:
                failures += 1
                if failures > self._retries:
                    raise OSError(
                        f"download of {self.url} kept failing at byte {self.bytes_read:,} "
                        f"of {self.total:,}"
                    )
                self._pause(min(2.0**failures, 60.0))
                try:
                    self._connect()
                    self.reconnects += 1
                    break
                except (OSError, http.client.HTTPException):
                    continue


def open_dump(url: str) -> tuple[io.TextIOBase, ResumableStream]:
    """Open a remote .pgn.zst as decompressed text, without touching disk.

    Returns the text handle and the stream, so callers can report how many
    compressed bytes a run cost and whether the connection had to be resumed.
    """
    stream = ResumableStream(url)
    reader = zstandard.ZstdDecompressor().stream_reader(stream)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), stream
