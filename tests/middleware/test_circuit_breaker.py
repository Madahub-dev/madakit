"""Tests for CircuitBreakerMiddleware constructor (task 2.2.1).

Covers: attribute storage, default values, initial state fields
(_failure_count, _state, _last_failure_time, _lock), BaseAgentClient
inheritance, and semaphore absence.
"""

from __future__ import annotations

import asyncio

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit.middleware.circuit_breaker import CircuitBreakerMiddleware


class TestCircuitBreakerConstructor:
    """CircuitBreakerMiddleware.__init__ — attribute storage and initial state."""

    def test_stores_client(self) -> None:
        """Asserts that the wrapped client is stored as _client."""
        provider = MockProvider()
        middleware = CircuitBreakerMiddleware(client=provider)
        assert middleware._client is provider

    def test_default_failure_threshold(self) -> None:
        """Asserts that failure_threshold defaults to 5."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._failure_threshold == 5

    def test_custom_failure_threshold(self) -> None:
        """Asserts that a custom failure_threshold is stored correctly."""
        middleware = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=3)
        assert middleware._failure_threshold == 3

    def test_default_recovery_timeout(self) -> None:
        """Asserts that recovery_timeout defaults to 60.0."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._recovery_timeout == 60.0

    def test_custom_recovery_timeout(self) -> None:
        """Asserts that a custom recovery_timeout is stored correctly."""
        middleware = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=30.0)
        assert middleware._recovery_timeout == 30.0

    def test_initial_failure_count_is_zero(self) -> None:
        """Asserts that _failure_count is initialised to 0."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._failure_count == 0

    def test_initial_state_is_closed(self) -> None:
        """Asserts that _state is initialised to 'closed'."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._state == "closed"

    def test_initial_last_failure_time_is_none(self) -> None:
        """Asserts that _last_failure_time is initialised to None."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._last_failure_time is None

    def test_lock_is_asyncio_lock(self) -> None:
        """Asserts that _lock is an asyncio.Lock instance."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert isinstance(middleware._lock, asyncio.Lock)

    def test_lock_is_unlocked_initially(self) -> None:
        """Asserts that _lock is in an unlocked state after construction."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert not middleware._lock.locked()

    def test_is_base_agent_client(self) -> None:
        """Asserts that CircuitBreakerMiddleware is a BaseAgentClient subclass."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = CircuitBreakerMiddleware(client=MockProvider())
        assert middleware._semaphore is None

    def test_each_instance_gets_own_lock(self) -> None:
        """Asserts that two instances do not share the same asyncio.Lock."""
        m1 = CircuitBreakerMiddleware(client=MockProvider())
        m2 = CircuitBreakerMiddleware(client=MockProvider())
        assert m1._lock is not m2._lock

    def test_each_instance_has_independent_state(self) -> None:
        """Asserts that mutating one instance's state does not affect another."""
        m1 = CircuitBreakerMiddleware(client=MockProvider())
        m2 = CircuitBreakerMiddleware(client=MockProvider())
        m1._failure_count = 3
        assert m2._failure_count == 0

    def test_client_can_be_another_middleware(self) -> None:
        """Asserts that a middleware can wrap another middleware (composition)."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        inner = RetryMiddleware(client=MockProvider())
        outer = CircuitBreakerMiddleware(client=inner)
        assert outer._client is inner
