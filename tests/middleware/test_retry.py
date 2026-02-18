"""Tests for RetryMiddleware constructor (task 2.1.1).

Covers: attribute storage, default values, custom is_retryable predicate,
BaseAgentClient inheritance, and semaphore passthrough via super().__init__().
"""

from __future__ import annotations

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit.middleware.retry import RetryMiddleware


class TestRetryMiddlewareConstructor:
    """RetryMiddleware.__init__ — attribute storage and defaults."""

    def test_stores_client(self) -> None:
        """Asserts that the wrapped client is stored as _client."""
        provider = MockProvider()
        middleware = RetryMiddleware(client=provider)
        assert middleware._client is provider

    def test_default_max_retries(self) -> None:
        """Asserts that max_retries defaults to 3."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._max_retries == 3

    def test_custom_max_retries(self) -> None:
        """Asserts that a custom max_retries value is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), max_retries=5)
        assert middleware._max_retries == 5

    def test_default_backoff_base(self) -> None:
        """Asserts that backoff_base defaults to 1.0."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._backoff_base == 1.0

    def test_custom_backoff_base(self) -> None:
        """Asserts that a custom backoff_base value is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), backoff_base=0.5)
        assert middleware._backoff_base == 0.5

    def test_default_is_retryable_is_none(self) -> None:
        """Asserts that is_retryable defaults to None."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._is_retryable is None

    def test_custom_is_retryable_stored(self) -> None:
        """Asserts that a custom is_retryable callable is stored correctly."""
        predicate = lambda exc: isinstance(exc, ValueError)
        middleware = RetryMiddleware(client=MockProvider(), is_retryable=predicate)
        assert middleware._is_retryable is predicate

    def test_is_base_agent_client(self) -> None:
        """Asserts that RetryMiddleware is a BaseAgentClient subclass."""
        middleware = RetryMiddleware(client=MockProvider())
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = RetryMiddleware(client=MockProvider())
        assert middleware._semaphore is None

    def test_zero_max_retries_stored(self) -> None:
        """Asserts that max_retries=0 (no retries) is stored correctly."""
        middleware = RetryMiddleware(client=MockProvider(), max_retries=0)
        assert middleware._max_retries == 0

    def test_float_backoff_base_stored(self) -> None:
        """Asserts that a float backoff_base (e.g. 2.5) is stored as-is."""
        middleware = RetryMiddleware(client=MockProvider(), backoff_base=2.5)
        assert middleware._backoff_base == 2.5

    def test_client_is_itself_a_middleware(self) -> None:
        """Asserts that a middleware can wrap another middleware (composition)."""
        inner = MockProvider()
        outer_retry = RetryMiddleware(client=inner)
        double_retry = RetryMiddleware(client=outer_retry)
        assert double_retry._client is outer_retry
        assert outer_retry._client is inner


class TestRetryMiddlewareStubDelegation:
    """RetryMiddleware stub send_request — delegates to wrapped client."""

    @pytest.mark.asyncio
    async def test_send_request_delegates_to_client(self) -> None:
        """Asserts that send_request passes through to the wrapped client."""
        from mada_modelkit._types import AgentRequest, AgentResponse

        expected = AgentResponse(
            content="hello", model="mock", input_tokens=5, output_tokens=3
        )
        provider = MockProvider(responses=[expected])
        middleware = RetryMiddleware(client=provider)

        request = AgentRequest(prompt="hi")
        result = await middleware.send_request(request)
        assert result is expected
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_send_request_stream_delegates_to_client(self) -> None:
        """Asserts that send_request_stream passes chunks through from wrapped client."""
        from mada_modelkit._types import AgentRequest

        provider = MockProvider()
        middleware = RetryMiddleware(client=provider)

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="hi")):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "mock"
        assert chunks[0].is_final is True