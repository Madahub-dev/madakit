"""Tests for RateLimitMiddleware.

Validates token bucket and leaky bucket rate limiting algorithms,
burst handling, and integration with the BaseAgentClient contract.
"""

from __future__ import annotations

import asyncio
import time

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


class TestTokenBucketAlgorithm:
    """Test token bucket refill and acquisition logic."""

    def test_refill_tokens_adds_tokens_over_time(self) -> None:
        """Tokens increase based on elapsed time and requests_per_second."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Start with full bucket
        initial_tokens = middleware._tokens
        assert initial_tokens == 10.0

        # Consume all tokens
        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 0.5  # Simulate 0.5s elapsed

        # Refill should add 5 tokens (10 tokens/sec * 0.5 sec)
        middleware._refill_tokens()
        assert abs(middleware._tokens - 5.0) < 0.01

    def test_refill_tokens_caps_at_max_tokens(self) -> None:
        """Tokens cannot exceed max_tokens."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Simulate 10 seconds elapsed with no consumption
        middleware._last_refill = middleware._last_refill - 10.0
        middleware._refill_tokens()

        # Should cap at max_tokens (10.0), not accumulate to 100
        assert middleware._tokens == middleware._max_tokens
        assert middleware._tokens == 10.0

    def test_refill_tokens_updates_last_refill(self) -> None:
        """_refill_tokens updates _last_refill timestamp."""
        import time
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        old_refill_time = middleware._last_refill
        time.sleep(0.01)  # Small sleep to ensure time advances
        middleware._refill_tokens()

        assert middleware._last_refill > old_refill_time

    @pytest.mark.asyncio
    async def test_acquire_token_consumes_one_token(self) -> None:
        """_acquire_token consumes exactly one token."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        initial_tokens = middleware._tokens
        await middleware._acquire_token()

        assert abs(middleware._tokens - (initial_tokens - 1.0)) < 0.01

    @pytest.mark.asyncio
    async def test_acquire_token_blocks_when_empty(self) -> None:
        """_acquire_token blocks until tokens are available."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=100.0, strategy="token_bucket")

        # Consume all tokens
        middleware._tokens = 0.0

        # This should block briefly, then succeed as tokens refill
        start = time.monotonic()
        await middleware._acquire_token()
        elapsed = time.monotonic() - start

        # Should have waited at least 10ms (one poll interval)
        assert elapsed >= 0.01
        assert middleware._tokens < middleware._max_tokens  # One token consumed

    @pytest.mark.asyncio
    async def test_acquire_token_multiple_sequential(self) -> None:
        """Multiple sequential acquisitions work correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=5, strategy="token_bucket"
        )

        # Start with 5 tokens
        assert middleware._tokens == 5.0

        # Acquire 3 tokens
        await middleware._acquire_token()
        await middleware._acquire_token()
        await middleware._acquire_token()

        # Should have 2 tokens left (plus any refill from elapsed time)
        assert middleware._tokens >= 2.0 and middleware._tokens < 3.0

    @pytest.mark.asyncio
    async def test_acquire_token_concurrent_safe(self) -> None:
        """Concurrent token acquisitions are thread-safe."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=100.0, burst_size=10, strategy="token_bucket"
        )

        # Start with 10 tokens
        assert middleware._tokens == 10.0

        # Acquire 10 tokens concurrently
        await asyncio.gather(*[middleware._acquire_token() for _ in range(10)])

        # All tokens consumed (plus minimal refill during execution)
        assert middleware._tokens < 1.0

    @pytest.mark.asyncio
    async def test_acquire_token_refills_before_checking(self) -> None:
        """_acquire_token refills before checking availability."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Deplete tokens
        middleware._tokens = 0.0
        middleware._last_refill = time.monotonic() - 0.2  # Simulate 0.2s elapsed

        # Should refill 2 tokens (10/sec * 0.2sec), then consume 1
        await middleware._acquire_token()
        assert abs(middleware._tokens - 1.0) < 0.1  # ~1 token remaining

    def test_refill_tokens_with_custom_rate(self) -> None:
        """Token refill respects custom requests_per_second."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=5.0, strategy="token_bucket")

        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 1.0  # 1 second elapsed

        middleware._refill_tokens()
        assert abs(middleware._tokens - 5.0) < 0.01  # 5 tokens/sec * 1 sec

    def test_refill_tokens_fractional_accumulation(self) -> None:
        """Tokens accumulate as floats for fractional rates."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=2.5, strategy="token_bucket")

        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 0.4  # 0.4 seconds

        middleware._refill_tokens()
        assert abs(middleware._tokens - 1.0) < 0.01  # 2.5 * 0.4 = 1.0
