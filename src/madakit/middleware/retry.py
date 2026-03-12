"""Retry middleware for mada-modelkit.

Wraps any BaseAgentClient and transparently retries failed requests with
configurable exponential backoff. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import ProviderError, RetryExhaustedError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

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
        """Execute the request, retrying on retryable failures with exponential backoff.

        Attempts the request up to max_retries + 1 times total. Between each
        retry, sleeps for backoff_base * 2 ** attempt seconds. Non-retryable
        exceptions are re-raised immediately. When all retries are exhausted,
        raises RetryExhaustedError with the last caught exception as last_error.
        """
        is_retryable = self._is_retryable if self._is_retryable is not None else self._default_is_retryable
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.send_request(request)
            except Exception as exc:
                last_exc = exc
                if not is_retryable(exc):
                    raise
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base * 2**attempt)
        assert last_exc is not None
        raise RetryExhaustedError(
            f"Request failed after {self._max_retries + 1} attempt(s): {last_exc}",
            last_error=last_exc,
        ) from last_exc

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks, retrying failures before the first chunk is yielded.

        The retry/backoff logic (same rules as send_request) applies only while no
        chunk has been yielded. Once the first chunk reaches the consumer, any
        subsequent streaming failure propagates directly without retry.
        """
        is_retryable = self._is_retryable if self._is_retryable is not None else self._default_is_retryable
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            first_yielded = False
            try:
                async for chunk in self._client.send_request_stream(request):
                    first_yielded = True
                    yield chunk
                return
            except Exception as exc:
                if first_yielded:
                    raise  # Post-first-chunk: propagate directly to consumer
                last_exc = exc
                if not is_retryable(exc):
                    raise
                if attempt < self._max_retries:
                    await asyncio.sleep(self._backoff_base * 2**attempt)
        assert last_exc is not None
        raise RetryExhaustedError(
            f"Stream failed after {self._max_retries + 1} attempt(s): {last_exc}",
            last_error=last_exc,
        ) from last_exc
