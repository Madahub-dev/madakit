"""Testing utilities for mada-modelkit.

Enhanced MockProvider and assertion helpers for testing middleware behavior.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = [
    "MockProvider",
    "assert_cache_hit",
    "assert_cache_miss",
    "assert_retry_count",
    "assert_response_time",
]


class MockProvider(BaseAgentClient):
    """Enhanced mock provider for testing.

    Supports configurable responses, errors, latency, streaming behavior,
    and call tracking for assertions.
    """

    def __init__(
        self,
        responses: list[AgentResponse] | None = None,
        errors: list[Exception] | None = None,
        latency: float = 0.0,
        stream_chunks: list[str] | None = None,
        fail_on_request: bool = False,
        fail_on_stream: bool = False,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialize mock provider.

        Args:
            responses: Pre-loaded responses to return in order.
            errors: Pre-loaded errors to raise in order.
            latency: Simulated latency in seconds.
            stream_chunks: Chunks to yield in streaming mode.
            fail_on_request: Always fail send_request calls.
            fail_on_stream: Always fail send_request_stream calls.
            max_concurrent: Maximum concurrent requests.
        """
        super().__init__(max_concurrent=max_concurrent)
        self._responses: list[AgentResponse] = responses or []
        self._errors: list[Exception] = errors or []
        self._latency = latency
        self._stream_chunks = stream_chunks or ["Mock ", "stream ", "response"]
        self._fail_on_request = fail_on_request
        self._fail_on_stream = fail_on_stream

        # Tracking
        self.call_count = 0
        self.stream_count = 0
        self.last_request: AgentRequest | None = None
        self._response_content = "mock"  # For customization

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Return next pre-loaded response or error.

        Args:
            request: The request to process.

        Returns:
            AgentResponse from queue or default.

        Raises:
            Exception: If fail_on_request is True or errors are queued.
        """
        self.call_count += 1
        self.last_request = request

        if self._latency:
            await asyncio.sleep(self._latency)

        if self._fail_on_request:
            raise RuntimeError("Mock provider error")

        if self._errors:
            raise self._errors.pop(0)

        if self._responses:
            return self._responses.pop(0)

        return AgentResponse(
            content=self._response_content,
            model="mock",
            input_tokens=10,
            output_tokens=5,
        )

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Yield stream chunks.

        Args:
            request: The request to process.

        Yields:
            StreamChunk instances.

        Raises:
            Exception: If fail_on_stream is True.
        """
        self.stream_count += 1
        self.last_request = request

        if self._fail_on_stream:
            raise RuntimeError("Mock stream error")

        for i, chunk_text in enumerate(self._stream_chunks):
            if self._latency:
                await asyncio.sleep(self._latency / len(self._stream_chunks))

            is_final = i == len(self._stream_chunks) - 1
            metadata_dict = {"index": i} if is_final else {}
            yield StreamChunk(
                delta=chunk_text,
                is_final=is_final,
                metadata=metadata_dict,
            )


def assert_cache_hit(response: AgentResponse) -> None:
    """Assert that response came from cache.

    Args:
        response: The response to check.

    Raises:
        AssertionError: If response did not come from cache.

    Example:
        >>> response = await cache_middleware.send_request(request)
        >>> assert_cache_hit(response)
    """
    if not hasattr(response, "metadata") or response.metadata is None:
        raise AssertionError("Response has no metadata")

    if response.metadata.get("cache_hit") is not True:
        raise AssertionError(
            f"Expected cache hit, but cache_hit={response.metadata.get('cache_hit')}"
        )


def assert_cache_miss(response: AgentResponse) -> None:
    """Assert that response did not come from cache.

    Args:
        response: The response to check.

    Raises:
        AssertionError: If response came from cache.

    Example:
        >>> response = await cache_middleware.send_request(request)
        >>> assert_cache_miss(response)
    """
    if hasattr(response, "metadata") and response.metadata is not None:
        if response.metadata.get("cache_hit") is True:
            raise AssertionError("Expected cache miss, but got cache hit")


def assert_retry_count(response: AgentResponse, expected: int) -> None:
    """Assert that request was retried expected number of times.

    Args:
        response: The response to check.
        expected: Expected retry count.

    Raises:
        AssertionError: If retry count doesn't match.

    Example:
        >>> response = await retry_middleware.send_request(request)
        >>> assert_retry_count(response, 2)
    """
    if not hasattr(response, "metadata") or response.metadata is None:
        raise AssertionError("Response has no metadata")

    actual = response.metadata.get("retry_count", 0)
    if actual != expected:
        raise AssertionError(
            f"Expected {expected} retries, but got {actual}"
        )


def assert_response_time(duration: float, max_seconds: float) -> None:
    """Assert that operation completed within time limit.

    Args:
        duration: Actual duration in seconds.
        max_seconds: Maximum allowed duration.

    Raises:
        AssertionError: If duration exceeds limit.

    Example:
        >>> start = time.perf_counter()
        >>> response = await client.send_request(request)
        >>> duration = time.perf_counter() - start
        >>> assert_response_time(duration, 1.0)
    """
    if duration > max_seconds:
        raise AssertionError(
            f"Response took {duration:.3f}s, exceeding limit of {max_seconds}s"
        )
