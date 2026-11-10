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
from pathlib import Path

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


class RangeNotSupported(RuntimeError):
    """The server would not resume a download from a byte offset."""


class ResumableStream(io.RawIOBase):
    """An HTTP download that survives failed and dropped connections.

    A multi-gigabyte stream over a long-lived connection will be cut sooner or
    later, and a cut connection reads exactly like the end of the file. Taken
    at face value that truncates the corpus silently, which is how a run once
    stopped at 161,062 games of 300,000 and reported success.

    So the stream knows how long the file is. If the data stops before that,
    it reconnects and asks for the file from the byte it had reached. The
    consumer sees one unbroken sequence of bytes, so a decompressor reading
    from this never learns anything happened. The first connection is retried
    on the same terms as any later one.

    It is patient by default: forty consecutive failures, backing off to a
    minute apart, rides out an outage of over half an hour before giving up.
    Any successful read resets the count.
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
        self._failures = 0
        self._connect()

    def _request(self) -> None:
        headers = {"Range": f"bytes={self.bytes_read}-"} if self.bytes_read else {}
        request = urllib.request.Request(self.url, headers=headers)  # noqa: S310 - fixed https URL
        response = self._opener(request, timeout=self._timeout)

        if self.bytes_read and getattr(response, "status", 206) != 206:
            raise RangeNotSupported(f"{self.url} ignored the Range header; cannot resume")
        if self.total is None:
            length = response.headers.get("Content-Length")
            self.total = int(length) if length else None
        self._response = response

    def _failed(self, cause: BaseException | None = None) -> None:
        """Count one failure, and either wait before the next attempt or give up."""
        self._failures += 1
        if self._failures > self._retries:
            total = f"{self.total:,}" if self.total is not None else "an unknown number"
            raise OSError(
                f"download of {self.url} kept failing at byte {self.bytes_read:,} of {total}"
            ) from cause
        self._pause(min(2.0**self._failures, 60.0))

    def _connect(self) -> None:
        """Open a connection at the current offset, retrying until it holds."""
        while True:
            try:
                self._request()
                return
            except (OSError, http.client.HTTPException) as error:
                self._failed(error)

    def _finished(self) -> bool:
        return self.total is None or self.bytes_read >= self.total

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            try:
                chunk = self._response.read(len(buffer))
            except (OSError, http.client.HTTPException):
                chunk = b""

            if chunk:
                self._failures = 0
                buffer[: len(chunk)] = chunk
                self.bytes_read += len(chunk)
                return len(chunk)
            if self._finished():
                return 0

            # The data stopped short of the file's length: the connection was
            # cut. Wait, then pick up from the byte we had reached.
            self._failed()
            self._connect()
            self.reconnects += 1


class LocalStream(io.RawIOBase):
    """A file on disk, presented with the same counters as a download."""

    def __init__(self, path: Path, reconnects: int = 0):
        self._file = path.open("rb")
        self.total = path.stat().st_size
        self.bytes_read = 0
        self.reconnects = reconnects

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        count = self._file.readinto(buffer)
        self.bytes_read += count
        return count

    def close(self) -> None:
        self._file.close()
        super().close()


def fetch_head(url: str, path: Path, size: int, report=print) -> int:
    """Download the first `size` bytes of a remote file to disk. Returns reconnects.

    Parsing a dump is far slower than downloading it, so streaming it holds one
    connection open for as long as the parse takes. On a network that will not
    keep a connection alive that long, fetching the needed prefix in one quick
    burst and parsing it from disk takes the network out of the slow part.

    A file already on disk at the full size is kept, so a rerun does not
    download again.
    """
    if path.exists() and path.stat().st_size >= size:
        report(f"Using {path}, already {path.stat().st_size / 1024**3:.2f} GB on disk")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = ResumableStream(url)
    size = min(size, stream.total or size)
    partial = path.with_suffix(path.suffix + ".part")
    written, last = 0, 0
    with partial.open("wb") as out:
        while written < size:
            chunk = stream.read(min(1 << 20, size - written))
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
            if written - last >= 100 << 20:
                report(f"  downloaded {written / 1024**3:.2f} of {size / 1024**3:.2f} GB")
                last = written
    stream.close()

    if written < size:
        raise OSError(f"download ended at {written:,} of {size:,} bytes")
    partial.replace(path)
    report(f"Downloaded {written / 1024**3:.2f} GB to {path}, resumed {stream.reconnects} times")
    return stream.reconnects


def open_dump(source: str | Path) -> tuple[io.TextIOBase, ResumableStream | LocalStream]:
    """Open a .pgn.zst as decompressed text, from a URL or from a file on disk.

    Returns the text handle and the stream, so callers can report how many
    compressed bytes a run cost and whether the connection had to be resumed.
    """
    remote = isinstance(source, str) and source.startswith(("http://", "https://"))
    stream = ResumableStream(source) if remote else LocalStream(Path(source))
    reader = zstandard.ZstdDecompressor().stream_reader(stream)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), stream
