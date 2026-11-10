"""The resumable download.

A dropped connection reads exactly like the end of a file. These tests cut the
connection in the middle of a download, in both of the ways it can fail, and
check that every byte still arrives exactly once and in order.
"""

import http.client
import io
import random

import pytest
import zstandard

from chessdt.lichess import ResumableStream

PAYLOAD = bytes(range(256)) * 400  # 102,400 distinct-ish bytes


class FakeServer:
    """Serves PAYLOAD, honouring Range, and cuts each connection after `cut` bytes."""

    def __init__(self, cut: int | None, fail_with=None, refuse_first: int = 0, ranges: bool = True):
        self.cut, self.fail_with, self.ranges = cut, fail_with, ranges
        self.refuse = refuse_first
        self.requested: list[str | None] = []

    def __call__(self, request, timeout=None):
        if self.refuse and self.requested:
            self.refuse -= 1
            raise OSError("connection refused")

        header = request.get_header("Range")
        self.requested.append(header)
        start = int(header.removeprefix("bytes=").rstrip("-")) if header and self.ranges else 0
        end = len(PAYLOAD) if self.cut is None else min(start + self.cut, len(PAYLOAD))
        return FakeResponse(
            PAYLOAD[start:end],
            206 if header and self.ranges else 200,
            len(PAYLOAD) - start,
            self.fail_with if end < len(PAYLOAD) else None,
        )


class FakeResponse:
    def __init__(self, body: bytes, status: int, length: int, fail_with):
        self._body, self.status, self._fail = io.BytesIO(body), status, fail_with
        self.headers = {"Content-Length": str(length)}

    def read(self, n: int) -> bytes:
        chunk = self._body.read(n)
        if not chunk and self._fail is not None:
            raise self._fail
        return chunk


def stream(server, **kwargs) -> ResumableStream:
    return ResumableStream(
        "https://example.test/file", opener=server, pause=lambda _: None, **kwargs
    )


def test_an_uninterrupted_download_arrives_whole():
    server = FakeServer(cut=None)
    s = stream(server)
    assert s.read() == PAYLOAD
    assert s.reconnects == 0 and server.requested == [None]


def test_a_connection_that_closes_quietly_is_resumed():
    """The failure seen in practice: the read just returns nothing."""
    server = FakeServer(cut=30_000)
    s = stream(server)
    assert s.read() == PAYLOAD
    assert s.reconnects == 3
    assert server.requested == [None, "bytes=30000-", "bytes=60000-", "bytes=90000-"]


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), TimeoutError("stalled"), http.client.IncompleteRead(b"")],
)
def test_a_connection_that_fails_loudly_is_resumed(error):
    s = stream(FakeServer(cut=40_000, fail_with=error))
    assert s.read() == PAYLOAD
    assert s.reconnects == 2


def test_a_reconnect_that_is_refused_is_tried_again():
    server = FakeServer(cut=50_000, refuse_first=3)
    s = stream(server)
    assert s.read() == PAYLOAD


def test_it_gives_up_loudly_rather_than_returning_a_short_file():
    server = FakeServer(cut=50_000, refuse_first=99)
    s = stream(server, retries=4)
    with pytest.raises(OSError, match="kept failing at byte 50,000"):
        s.read()


def test_a_server_that_ignores_range_is_not_trusted():
    """Restarting from zero while believing we resumed would corrupt the stream."""
    s = stream(FakeServer(cut=50_000, ranges=False), retries=2)
    with pytest.raises(OSError):
        s.read()


def test_a_decompressor_reading_across_a_cut_sees_nothing_unusual():
    # Random bytes barely compress, so the compressed stream is long enough to cut.
    original = random.Random(0).randbytes(20_000)
    compressed = zstandard.ZstdCompressor().compress(original)

    class Server(FakeServer):
        def __call__(self, request, timeout=None):
            header = request.get_header("Range")
            start = int(header.removeprefix("bytes=").rstrip("-")) if header else 0
            end = min(start + 97, len(compressed))
            return FakeResponse(
                compressed[start:end], 206 if header else 200, len(compressed) - start, None
            )

    s = stream(Server(cut=97), retries=10_000)
    assert zstandard.ZstdDecompressor().stream_reader(s).read() == original
    assert s.reconnects > 0
