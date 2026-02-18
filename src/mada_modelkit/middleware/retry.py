"""Retry middleware for mada-modelkit.

Wraps any BaseAgentClient and transparently retries failed requests with
configurable exponential backoff. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["RetryMiddleware"]


class RetryMiddleware(BaseAgentClient):
    """Middleware that retries failed requests with exponential backoff."""

    def __init__(
        self,
        client: BaseAgentClient,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        is_retryable: Callable[[Exception], bool] | None = None,
    ) -> None:
        """Initialise with a wrapped client and retry configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            max_retries: Maximum number of retry attempts after the first failure.
            backoff_base: Base delay in seconds for exponential backoff
                (sleep = backoff_base * 2 ** attempt).
            is_retryable: Optional predicate deciding if an exception should
                trigger a retry. Defaults to _default_is_retryable when None.
        """
        super().__init__()
        self._client = client
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._is_retryable = is_retryable

    @staticmethod
    def _default_is_retryable(exc: Exception) -> bool:
        """Return True if the exception should trigger a retry attempt.

        Rules:
        - ProviderError with status_code=None (unknown server error): retryable.
        - ProviderError with status_code=429 (rate limited): retryable.
        - ProviderError with status_code>=500 (server error): retryable.
        - ProviderError with any other 4xx (client error): not retryable.
        - Any non-ProviderError exception: not retryable.
        """
        if isinstance(exc, ProviderError):
            if exc.status_code is None:
                return True
            return exc.status_code == 429 or exc.status_code >= 500
        return False

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Delegate to the wrapped client (full retry logic added in task 2.1.3)."""
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate streaming to the wrapped client (retry logic added in task 2.1.4)."""
        async for chunk in self._client.send_request_stream(request):
            yield chunk
