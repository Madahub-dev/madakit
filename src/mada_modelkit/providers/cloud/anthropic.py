"""Anthropic cloud provider for mada-modelkit.

Provides AnthropicClient — an HttpAgentClient subclass that targets the
Anthropic Messages API. Requires the ``httpx`` optional extra. TLS is
enforced; the API key is passed as an x-api-key header and redacted in repr.
Does NOT use OpenAICompatMixin — Anthropic uses its own wire format.
"""

from __future__ import annotations

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

    def _endpoint(self) -> str:
        """Return the Anthropic messages endpoint path."""
        return "/messages"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build the Anthropic request payload. Implemented in task 4.2.2."""
        raise NotImplementedError

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse the Anthropic API response. Implemented in task 4.2.3."""
        raise NotImplementedError
