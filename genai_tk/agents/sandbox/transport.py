"""Bound untrusted command responses before the OpenSandbox SSE parser."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import httpx
from opensandbox.exceptions import SandboxException


class CommandResponseLimitError(SandboxException):
    """An incoming command response exceeded the host's transport policy."""


class _BoundedCommandStream(httpx.AsyncByteStream):
    """Count bytes before handing them to HTTP decoding or SSE framing."""

    def __init__(self, stream: httpx.AsyncByteStream, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        received = 0
        try:
            async for chunk in self._stream:
                received += len(chunk)
                if received > self._max_bytes:
                    raise CommandResponseLimitError(f"Sandbox command response exceeded {self._max_bytes} wire bytes.")
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._stream.aclose()


class BoundedCommandTransport(httpx.AsyncBaseTransport):
    """Apply a wire-byte limit to command responses, preserving file transfers."""

    def __init__(self, max_response_bytes: int, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._max_response_bytes = max_response_bytes
        self._transport = transport if transport is not None else httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)
        if request.url.path.rstrip("/").endswith("/command"):
            # Reject compression before httpx can expand a small wire payload
            # into arbitrarily large data ahead of the SSE/event callbacks.
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if encoding != "identity":
                await response.aclose()
                raise CommandResponseLimitError("Sandbox command responses require identity content encoding.")
            response.stream = _BoundedCommandStream(
                cast(httpx.AsyncByteStream, response.stream), self._max_response_bytes
            )
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()
