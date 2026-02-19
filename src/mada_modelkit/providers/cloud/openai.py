"""OpenAI cloud provider for mada-modelkit.

Provides OpenAIClient — an HttpAgentClient + OpenAICompatMixin subclass that
targets the OpenAI chat-completions API. Requires the ``httpx`` optional extra.
TLS is enforced; the API key is passed as a Bearer token and redacted in repr.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, StreamChunk
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["OpenAIClient"]


class OpenAIClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for the OpenAI chat-completions API.

    Combines ``OpenAICompatMixin`` for OpenAI wire-format handling with
    ``HttpAgentClient`` for the shared async HTTP pipeline. TLS is always
    enforced; pass ``connect_timeout``, ``read_timeout``, or
    ``max_concurrent`` as keyword arguments to tune behaviour.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        **kwargs: object,
    ) -> None:
        """Initialise the OpenAI client.

        Args:
            api_key: OpenAI API key; placed in ``Authorization: Bearer`` header.
            model: Chat model identifier. Defaults to ``"gpt-4o-mini"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._api_key = api_key
        super().__init__(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted."""
        return f"OpenAIClient(model={self._model!r}, api_key=***)"

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream the OpenAI response as SSE StreamChunks.

        Adds ``stream: true`` to the payload and issues a streaming POST via
        httpx.  Yields one ``StreamChunk`` per ``data:`` SSE line that contains
        a delta.  Emits a final ``StreamChunk(delta="", is_final=True)`` when
        the ``[DONE]`` sentinel is received.  ``ConnectError`` and
        ``TimeoutException`` are wrapped as ``ProviderError``.  Non-2xx
        responses raise ``ProviderError`` with the HTTP status code.
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
