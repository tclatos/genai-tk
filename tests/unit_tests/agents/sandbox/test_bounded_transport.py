"""Exercise hostile command bodies at the external HTTP transport boundary."""

from collections.abc import AsyncIterator

import httpx
import pytest

pytest.importorskip("opensandbox")

from opensandbox.adapters.sse import aiter_sse_events  # noqa: E402

from genai_tk.agents.sandbox.transport import BoundedCommandTransport, CommandResponseLimitError  # noqa: E402

pytestmark = pytest.mark.unit


class IncomingBody(httpx.AsyncByteStream):
    """A response body that records how much was consumed and whether it closed."""

    def __init__(self, pieces: tuple[bytes, ...]) -> None:
        self.pieces = pieces
        self.chunks_read = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            self.chunks_read += 1
            yield piece

    async def aclose(self) -> None:
        self.closed = True


def response_transport(body: IncomingBody, encoding: str = "identity") -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream", "Content-Encoding": encoding},
            stream=body,
        )
    )


@pytest.mark.parametrize("prefix", [b"{", b"data: {"])
async def test_unterminated_sse_frame_is_bounded_before_parser(prefix: bytes) -> None:
    body = IncomingBody((prefix, b"x" * 128, b"x" * 128, b"must not be consumed"))
    transport = BoundedCommandTransport(200, response_transport(body))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(CommandResponseLimitError, match="200 wire bytes"):
            async with client.stream("POST", "http://sandbox.test/command") as response:
                async for _event in aiter_sse_events(response):
                    pass
    assert body.closed and body.chunks_read == 3


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "gzip, identity"])
async def test_compression_is_rejected_before_reading_body(encoding: str) -> None:
    body = IncomingBody((b"not consumed",))
    transport = BoundedCommandTransport(200, response_transport(body, encoding))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(CommandResponseLimitError, match="identity content encoding"):
            await client.post("http://sandbox.test/command")
    assert body.closed and body.chunks_read == 0


async def test_response_at_exact_limit_is_accepted() -> None:
    body = IncomingBody((b"40", b"+2"))
    transport = BoundedCommandTransport(4, response_transport(body))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.post("http://sandbox.test/command")
    assert response.content == b"40+2" and body.closed


async def test_early_consumer_exit_closes_body() -> None:
    body = IncomingBody((b"first", b"second"))
    transport = BoundedCommandTransport(200, response_transport(body))
    async with httpx.AsyncClient(transport=transport) as client:
        async with client.stream("POST", "http://sandbox.test/command") as response:
            async for _chunk in response.aiter_bytes():
                break
    assert body.closed and body.chunks_read == 1


async def test_file_transfer_is_not_subject_to_command_limit() -> None:
    body = IncomingBody((b"file bytes" * 100,))
    transport = BoundedCommandTransport(4, response_transport(body))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("http://sandbox.test/files/download")
    assert len(response.content) == 1000 and body.closed
