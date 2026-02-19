"""Shared HTTP base client for mada-modelkit providers.

Provides HttpAgentClient — an abstract BaseAgentClient subclass that owns
an httpx.AsyncClient and implements the common request/response pipeline for
all cloud and local-server providers. Requires the optional ``httpx`` extra.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import httpx

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse

__all__ = ["HttpAgentClient"]


class HttpAgentClient(BaseAgentClient):
    """Abstract base for all HTTP-based providers (cloud and local-server).

    Subclasses must implement ``_build_payload``, ``_parse_response``, and
    ``_endpoint``. Cloud providers should set ``_require_tls = True`` to
    enforce HTTPS at construction time.
    """

    _require_tls: bool = False

    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialise the shared HTTP client.

        Args:
            base_url: Root URL for all requests (e.g. ``https://api.openai.com/v1``).
            connect_timeout: Seconds to wait for a TCP connection. Defaults to 5.0.
            read_timeout: Seconds to wait for the first response byte. Defaults to 60.0.
            headers: Extra HTTP headers merged into every request. Defaults to none.
            max_concurrent: If set, caps parallel requests via an asyncio.Semaphore.
        """
        super().__init__(max_concurrent=max_concurrent)
        if self._require_tls and base_url.startswith("http://"):
            raise ValueError(f"TLS required: {base_url!r} must use https://")
        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=connect_timeout,
                pool=connect_timeout,
            ),
            headers=headers or {},
        )

    @abstractmethod
    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build the provider-specific JSON request body from an AgentRequest."""

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse a successful JSON response dict into an AgentResponse."""

    @abstractmethod
    def _endpoint(self) -> str:
        """Return the endpoint path relative to base_url (e.g. '/chat/completions')."""

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """POST to the provider endpoint (stub; full pipeline added in task 3.1.4)."""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check provider availability (stub; implemented in task 3.1.5)."""
        return True

    async def close(self) -> None:
        """Close the underlying httpx client (stub; implemented in task 3.1.6)."""
