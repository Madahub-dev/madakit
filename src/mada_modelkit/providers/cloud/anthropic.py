"""Anthropic cloud provider for mada-modelkit.

Provides AnthropicClient — an HttpAgentClient subclass that targets the
Anthropic Messages API. Requires the ``httpx`` optional extra. TLS is
enforced; the API key is passed as an x-api-key header and redacted in repr.
Does NOT use OpenAICompatMixin — Anthropic uses its own wire format.
"""

from __future__ import annotations

import base64
from typing import Any

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient

__all__ = ["AnthropicClient"]


class AnthropicClient(HttpAgentClient):
    """HTTP client for the Anthropic Messages API.

    Subclasses ``HttpAgentClient`` directly (no OpenAICompatMixin) because
    Anthropic uses its own request/response format. TLS is always enforced;
    pass ``connect_timeout``, ``read_timeout``, or ``max_concurrent`` as
    keyword arguments to tune behaviour.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        **kwargs: object,
    ) -> None:
        """Initialise the Anthropic client.

        Args:
            api_key: Anthropic API key; placed in ``x-api-key`` header.
            model: Chat model identifier. Defaults to ``"claude-sonnet-4-6"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._api_key = api_key
        super().__init__(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted."""
        return f"AnthropicClient(model={self._model!r}, api_key=***)"

    def _endpoint(self) -> str:
        """Return the Anthropic messages endpoint path."""
        return "/messages"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build the Anthropic Messages API request payload.

        Formats the request using Anthropic's wire format: the system prompt is
        a top-level ``system`` field (omitted when absent), and ``messages``
        contains only the user turn.  Stop sequences are mapped to
        ``stop_sequences`` (Anthropic's name).  ``temperature`` and
        ``max_tokens`` are always included.

        When ``request.attachments`` is non-empty, the user message ``content``
        becomes a list of Anthropic source blocks: one ``image`` block per
        attachment (base64-encoded bytes + media_type), followed by a ``text``
        block for the prompt.  Without attachments, ``content`` is a plain
        string.
        """
        if request.attachments:
            content: str | list[dict[str, Any]] = [
                *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": att.media_type,
                            "data": base64.b64encode(att.content).decode("ascii"),
                        },
                    }
                    for att in request.attachments
                ],
                {"type": "text", "text": request.prompt},
            ]
        else:
            content = request.prompt

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": content},
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.stop:
            payload["stop_sequences"] = request.stop
        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse an Anthropic Messages API response into an AgentResponse.

        Extracts ``content[0].text`` for the response text, reads
        ``usage.input_tokens`` and ``usage.output_tokens`` (defaulting to 0
        when absent), and uses the response ``model`` field with a fallback to
        ``self._model``.
        """
        usage = data.get("usage", {})
        return AgentResponse(
            content=data["content"][0]["text"],
            model=data.get("model", self._model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
