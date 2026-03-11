"""Tests for RateLimitMiddleware.

Validates token bucket and leaky bucket rate limiting algorithms,
burst handling, and integration with the BaseAgentClient contract.
"""

from __future__ import annotations

import asyncio
import pytest

from mada_modelkit._errors import AgentError
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.middleware.rate_limit import RateLimitMiddleware
from helpers import MockProvider


class TestModuleExports:
    """Verify rate_limit.py module structure."""

    def test_all_exports(self) -> None:
        """__all__ contains RateLimitMiddleware."""
        from mada_modelkit.middleware import rate_limit
        assert hasattr(rate_limit, "__all__")
        assert "RateLimitMiddleware" in rate_limit.__all__
        assert len(rate_limit.__all__) == 1

    def test_middleware_importable(self) -> None:
        """RateLimitMiddleware is importable from rate_limit module."""
        from mada_modelkit.middleware.rate_limit import RateLimitMiddleware as Imported
        assert Imported is RateLimitMiddleware


class TestRateLimitMiddlewareConstructor:
    """Test constructor parameter handling and state initialisation."""

    def test_minimal_constructor(self) -> None:
        """Constructor with only client uses defaults."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock)
        assert middleware._client is mock
        assert middleware._requests_per_second == 10.0
        assert middleware._burst_size is None
        assert middleware._strategy == "token_bucket"

    def test_explicit_requests_per_second(self) -> None:
        """requests_per_second parameter is stored."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=5.0)
        assert middleware._requests_per_second == 5.0

    def test_explicit_burst_size(self) -> None:
        """burst_size parameter is stored."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, burst_size=20)
        assert middleware._burst_size == 20

    def test_explicit_strategy_token_bucket(self) -> None:
        """strategy parameter accepts 'token_bucket'."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="token_bucket")
        assert middleware._strategy == "token_bucket"

    def test_explicit_strategy_leaky_bucket(self) -> None:
        """strategy parameter accepts 'leaky_bucket'."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="leaky_bucket")
        assert middleware._strategy == "leaky_bucket"

    def test_token_bucket_state_initialisation(self) -> None:
        """Token bucket strategy initialises tokens and refill state."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="token_bucket")
        assert hasattr(middleware, "_tokens")
        assert hasattr(middleware, "_max_tokens")
        assert hasattr(middleware, "_last_refill")
        assert hasattr(middleware, "_lock")
        assert middleware._tokens == 10.0  # default burst_size = requests_per_second
        assert middleware._max_tokens == 10.0

    def test_token_bucket_custom_burst_size(self) -> None:
        """Token bucket uses burst_size when provided."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, burst_size=15, strategy="token_bucket"
        )
        assert middleware._tokens == 15.0
        assert middleware._max_tokens == 15.0

    def test_leaky_bucket_state_initialisation(self) -> None:
        """Leaky bucket strategy initialises queue."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="leaky_bucket")
        assert hasattr(middleware, "_max_queue_size")
        assert hasattr(middleware, "_queue")
        assert hasattr(middleware, "_processor_task")
        assert middleware._max_queue_size == 20  # default 2 * requests_per_second
        assert isinstance(middleware._queue, asyncio.Queue)
        assert middleware._processor_task is None

    def test_leaky_bucket_custom_burst_size(self) -> None:
        """Leaky bucket uses burst_size for max_queue_size when provided."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, burst_size=30, strategy="leaky_bucket"
        )
        assert middleware._max_queue_size == 30

    def test_invalid_strategy_raises_value_error(self) -> None:
        """Unknown strategy raises ValueError."""
        mock = MockProvider()
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            RateLimitMiddleware(client=mock, strategy="invalid")

    def test_instance_isolation(self) -> None:
        """Multiple instances have independent state."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware1 = RateLimitMiddleware(client=mock1, requests_per_second=5.0)
        middleware2 = RateLimitMiddleware(client=mock2, requests_per_second=10.0)
        assert middleware1._client is mock1
        assert middleware2._client is mock2
        assert middleware1._requests_per_second == 5.0
        assert middleware2._requests_per_second == 10.0
        assert middleware1._tokens != middleware2._tokens

    def test_wraps_middleware(self) -> None:
        """Can wrap another middleware instance."""
        mock = MockProvider()
        from mada_modelkit.middleware.retry import RetryMiddleware
        retry = RetryMiddleware(client=mock, max_retries=2)
        middleware = RateLimitMiddleware(client=retry, requests_per_second=5.0)
        assert middleware._client is retry
        assert retry._client is mock

    def test_super_init_called(self) -> None:
        """Constructor calls super().__init__()."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock)
        # Inherits from BaseAgentClient, so should have _semaphore attribute
        assert hasattr(middleware, "_semaphore")
