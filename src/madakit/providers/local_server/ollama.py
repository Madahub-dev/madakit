"""Ollama local server provider for mada-modelkit.

Provides OllamaClient — an OpenAICompatMixin + HttpAgentClient subclass that
targets a locally-running Ollama server. No API key required. TLS is not
enforced (local server). Uses the OpenAI-compatible wire format via
OpenAICompatMixin. The ``health_check`` method queries ``/api/tags`` (Ollama's
model-list endpoint) instead of the default ``GET /``.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from madakit._errors import ProviderError
from madakit._types import AgentRequest, StreamChunk
from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin

__all__ = ["OllamaClient"]


class OllamaClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for a locally-running Ollama server.

    Thin subclass of ``OpenAICompatMixin`` + ``HttpAgentClient``. Targets the
    Ollama OpenAI-compatible endpoint at ``http://localhost:11434/v1`` by
    default. No API key required; TLS is not enforced. All payload building,
    response parsing, and endpoint routing are inherited from the mixin.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        **kwargs: object,
    ) -> None:
        """Initialise the Ollama client.

        Args:
            model: Ollama model tag. Defaults to ``"llama3.2"``.
            base_url: Base URL of the Ollama server. Defaults to
                ``"http://localhost:11434/v1"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._base_url = base_url
        super().__init__(base_url=base_url, **kwargs)  # type: ignore[arg-type]

    async def health_check(self) -> bool:
        """Return True if the Ollama server is reachable, False otherwise.

        Queries ``GET /api/tags`` (Ollama's model-list endpoint) rather than
        the default ``GET /`` used by ``HttpAgentClient``. Returns ``True`` for
        any HTTP response; ``False`` on ``ConnectError`` or ``TimeoutException``.
        """
        try:
            await self._http_client.get("/api/tags")
            return True
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream the Ollama response as SSE StreamChunks.

        Adds ``stream: true`` to the OpenAI-compat payload and issues a
        streaming POST via httpx. Yields one ``StreamChunk`` per ``data:`` SSE
        line that contains a delta. Emits a final
        ``StreamChunk(delta="", is_final=True)`` when the ``[DONE]`` sentinel
        is received. ``ConnectError`` and ``TimeoutException`` are wrapped as
        ``ProviderError``. Non-2xx responses raise ``ProviderError`` with the
        HTTP status code.
        """
        payload = self._build_payload(request)
        payload["stream"] = True
        try:
            async with self._http_client.stream(
                "POST", self._endpoint(), json=payload
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise ProviderError(
                        f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        yield StreamChunk(delta="", is_final=True)
                        return
                    data = json.loads(raw)
                    content = (
                        data["choices"][0].get("delta", {}).get("content") or ""
                    )
                    yield StreamChunk(delta=content)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Request timed out: {exc}") from exc
        except httpx.ConnectError as exc:
            raise ProviderError(f"Connection failed: {exc}") from exc

    def __repr__(self) -> str:
        """Return a repr showing the model tag and server base URL."""
        return f"OllamaClient(model={self._model!r}, base_url={self._base_url!r})"
