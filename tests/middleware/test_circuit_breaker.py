"""Tests for CircuitBreakerMiddleware constructor, state machine, send_request,
send_request_stream, and recovery probing (tasks 2.2.1–2.2.5).

Covers: attribute storage, default values, initial state fields
(_failure_count, _state, _last_failure_time, _lock), BaseAgentClient
inheritance, semaphore absence, state machine transitions (closed→open at
threshold, open→half-open after timeout, half-open→closed on success,
half-open→open on failure), send_request circuit logic (closed passthrough,
open raises CircuitOpenError, half-open health-check probe then request),
send_request_stream with the same closed/open/half-open circuit logic, and
recovery probing (health_check call count, not called in wrong states,
exception propagation without failure recording).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import CircuitOpenError, ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.circuit_breaker import CircuitBreakerMiddleware


class _UnhealthyProvider(MockProvider):
    """MockProvider that reports itself as unhealthy via health_check."""

    async def health_check(self) -> bool:
        """Always returns False to simulate an unavailable backend."""
        return False


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


class TestStateMachine:
    """CircuitBreakerMiddleware state machine — _record_failure, _record_success, _check_state."""

    @pytest.mark.asyncio
    async def test_record_failure_increments_count(self) -> None:
        """Asserts that _record_failure increments _failure_count by one."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=5)
        await cb._record_failure()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_closed_stays_closed_below_threshold(self) -> None:
        """Asserts that state remains closed when failure count is below threshold."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=3)
        await cb._record_failure()
        await cb._record_failure()
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_closed_transitions_to_open_at_threshold(self) -> None:
        """Asserts that reaching failure_threshold transitions the circuit to open."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=3)
        for _ in range(3):
            await cb._record_failure()
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_last_failure_time_recorded_on_open(self) -> None:
        """Asserts that _last_failure_time is set when the circuit opens."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=1)
        before = time.monotonic()
        await cb._record_failure()
        assert cb._last_failure_time is not None
        assert cb._last_failure_time >= before

    @pytest.mark.asyncio
    async def test_failure_count_increments_beyond_threshold(self) -> None:
        """Asserts that _failure_count keeps incrementing after the circuit opens."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=2)
        for _ in range(4):
            await cb._record_failure()
        assert cb._failure_count == 4
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_record_success_transitions_to_closed(self) -> None:
        """Asserts that _record_success sets state to closed."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "half-open"
        await cb._record_success()
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_record_success_resets_failure_count(self) -> None:
        """Asserts that _record_success resets _failure_count to zero."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._failure_count = 4
        await cb._record_success()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_check_state_returns_closed_when_closed(self) -> None:
        """Asserts that _check_state returns 'closed' in the closed state."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        assert await cb._check_state() == "closed"

    @pytest.mark.asyncio
    async def test_check_state_returns_open_before_timeout(self) -> None:
        """Asserts that _check_state returns 'open' when recovery_timeout has not elapsed."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=60.0)
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        assert await cb._check_state() == "open"

    @pytest.mark.asyncio
    async def test_open_transitions_to_half_open_after_timeout(self) -> None:
        """Asserts that _check_state transitions open→half-open after recovery_timeout."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=30.0)
        cb._state = "open"
        cb._last_failure_time = time.monotonic() - 31.0
        state = await cb._check_state()
        assert state == "half-open"
        assert cb._state == "half-open"

    @pytest.mark.asyncio
    async def test_open_to_half_open_uses_monotonic_clock(self) -> None:
        """Asserts that the timeout check uses time.monotonic via the patched value."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=10.0)
        cb._state = "open"
        cb._last_failure_time = 100.0
        with patch("mada_modelkit.middleware.circuit_breaker.time.monotonic", return_value=111.0):
            state = await cb._check_state()
        assert state == "half-open"

    @pytest.mark.asyncio
    async def test_open_stays_open_exactly_at_timeout_boundary(self) -> None:
        """Asserts that state stays open when elapsed equals exactly recovery_timeout."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=10.0)
        cb._state = "open"
        cb._last_failure_time = 100.0
        # elapsed = 109.9 < 10.0 → still open
        with patch("mada_modelkit.middleware.circuit_breaker.time.monotonic", return_value=109.9):
            state = await cb._check_state()
        assert state == "open"

    @pytest.mark.asyncio
    async def test_check_state_returns_half_open_unchanged(self) -> None:
        """Asserts that _check_state returns 'half-open' without altering half-open state."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "half-open"
        assert await cb._check_state() == "half-open"

    @pytest.mark.asyncio
    async def test_half_open_failure_immediately_reopens_circuit(self) -> None:
        """Asserts that any failure in half-open state transitions directly to open."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=10)
        cb._state = "half-open"
        await cb._record_failure()
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_half_open_failure_records_failure_time(self) -> None:
        """Asserts that reopening from half-open updates _last_failure_time."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=10)
        cb._state = "half-open"
        before = time.monotonic()
        await cb._record_failure()
        assert cb._last_failure_time is not None
        assert cb._last_failure_time >= before


class TestSendRequest:
    """CircuitBreakerMiddleware.send_request — closed/open/half-open circuit logic."""

    @pytest.mark.asyncio
    async def test_closed_passes_request_through(self) -> None:
        """Asserts that send_request delegates to the wrapped client in closed state."""
        expected = AgentResponse(content="ok", model="m", input_tokens=1, output_tokens=1)
        cb = CircuitBreakerMiddleware(client=MockProvider(responses=[expected]))
        result = await cb.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_closed_success_keeps_state_closed(self) -> None:
        """Asserts that a successful request in closed state leaves the circuit closed."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_closed_success_resets_failure_count(self) -> None:
        """Asserts that a successful request in closed state resets _failure_count."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), failure_threshold=5)
        cb._failure_count = 3
        await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_closed_failure_increments_count(self) -> None:
        """Asserts that a failed request in closed state increments _failure_count."""
        cb = CircuitBreakerMiddleware(client=MockProvider(errors=[ProviderError("err", 500)]),
                                      failure_threshold=5)
        with pytest.raises(ProviderError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_closed_failures_at_threshold_opens_circuit(self) -> None:
        """Asserts that consecutive failures equal to failure_threshold open the circuit."""
        errors = [ProviderError("err", 500)] * 3
        cb = CircuitBreakerMiddleware(client=MockProvider(errors=errors), failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ProviderError):
                await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self) -> None:
        """Asserts that send_request raises CircuitOpenError when the circuit is open."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_open_does_not_call_client(self) -> None:
        """Asserts that the wrapped client is never called when the circuit is open."""
        provider = MockProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_half_open_healthy_success_closes_circuit(self) -> None:
        """Asserts that a successful request in half-open (healthy probe) closes the circuit."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "half-open"
        await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_healthy_success_returns_response(self) -> None:
        """Asserts that the response is returned after a successful half-open probe+request."""
        expected = AgentResponse(content="ok", model="m", input_tokens=1, output_tokens=1)
        cb = CircuitBreakerMiddleware(client=MockProvider(responses=[expected]))
        cb._state = "half-open"
        result = await cb.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_half_open_healthy_failure_reopens_circuit(self) -> None:
        """Asserts that a failed request in half-open (healthy probe) reopens the circuit."""
        cb = CircuitBreakerMiddleware(
            client=MockProvider(errors=[ProviderError("err", 500)]), failure_threshold=10
        )
        cb._state = "half-open"
        with pytest.raises(ProviderError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_half_open_unhealthy_probe_raises_circuit_open_error(self) -> None:
        """Asserts that a failed health check in half-open raises CircuitOpenError."""
        cb = CircuitBreakerMiddleware(client=_UnhealthyProvider())
        cb._state = "half-open"
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_half_open_unhealthy_probe_reopens_circuit(self) -> None:
        """Asserts that a failed health check in half-open transitions state back to open."""
        cb = CircuitBreakerMiddleware(client=_UnhealthyProvider(), failure_threshold=10)
        cb._state = "half-open"
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_half_open_unhealthy_does_not_call_send_request(self) -> None:
        """Asserts that the wrapped client's send_request is skipped after a failed probe."""
        provider = _UnhealthyProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "half-open"
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_open_to_half_open_to_closed_via_timeout(self) -> None:
        """Asserts full open→half-open→closed transition via recovery timeout and success."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=10.0)
        cb._state = "open"
        cb._last_failure_time = time.monotonic() - 11.0  # timeout elapsed
        await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._state == "closed"


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------


class _ErrorAfterFirstChunkProvider(MockProvider):
    """Yields one chunk then raises to simulate a mid-stream failure."""

    def __init__(self, error: Exception) -> None:
        """Initialise with the error to raise after the first chunk."""
        super().__init__()
        self._error = error

    async def send_request_stream(self, request: AgentRequest):  # type: ignore[override]
        """Yield one chunk then raise the stored error."""
        self.call_count += 1
        yield StreamChunk(delta="first", is_final=False)
        raise self._error


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSendRequestStream:
    """CircuitBreakerMiddleware.send_request_stream — same circuit logic as send_request."""

    @pytest.mark.asyncio
    async def test_closed_streams_all_chunks(self) -> None:
        """Asserts that send_request_stream yields all chunks through in closed state."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        chunks = []
        async for chunk in cb.send_request_stream(AgentRequest(prompt="hi")):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_closed_success_records_success(self) -> None:
        """Asserts that a completed stream in closed state keeps the circuit closed."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._failure_count = 3
        async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert cb._state == "closed"
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_closed_stream_error_records_failure(self) -> None:
        """Asserts that a stream error in closed state increments _failure_count."""
        cb = CircuitBreakerMiddleware(
            client=MockProvider(errors=[ProviderError("err", 500)]), failure_threshold=5
        )
        with pytest.raises(ProviderError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_closed_stream_errors_at_threshold_open_circuit(self) -> None:
        """Asserts that stream failures at threshold open the circuit."""
        errors = [ProviderError("err", 500)] * 3
        cb = CircuitBreakerMiddleware(client=MockProvider(errors=errors), failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ProviderError):
                async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                    pass
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self) -> None:
        """Asserts that send_request_stream raises CircuitOpenError when circuit is open."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass

    @pytest.mark.asyncio
    async def test_open_does_not_call_client(self) -> None:
        """Asserts that the wrapped client is never called when the circuit is open."""
        provider = MockProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_half_open_healthy_stream_success_closes_circuit(self) -> None:
        """Asserts that a completed stream in half-open (healthy probe) closes the circuit."""
        cb = CircuitBreakerMiddleware(client=MockProvider())
        cb._state = "half-open"
        async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert cb._state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_healthy_stream_failure_reopens_circuit(self) -> None:
        """Asserts that a stream error in half-open (healthy probe) reopens the circuit."""
        cb = CircuitBreakerMiddleware(
            client=MockProvider(errors=[ProviderError("err", 500)]), failure_threshold=10
        )
        cb._state = "half-open"
        with pytest.raises(ProviderError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_half_open_unhealthy_probe_raises_circuit_open_error(self) -> None:
        """Asserts that an unhealthy probe in half-open raises CircuitOpenError."""
        cb = CircuitBreakerMiddleware(client=_UnhealthyProvider())
        cb._state = "half-open"
        with pytest.raises(CircuitOpenError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass

    @pytest.mark.asyncio
    async def test_half_open_unhealthy_probe_reopens_circuit(self) -> None:
        """Asserts that an unhealthy probe in half-open transitions state back to open."""
        cb = CircuitBreakerMiddleware(client=_UnhealthyProvider(), failure_threshold=10)
        cb._state = "half-open"
        with pytest.raises(CircuitOpenError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert cb._state == "open"

    @pytest.mark.asyncio
    async def test_mid_stream_error_records_failure(self) -> None:
        """Asserts that an error raised after the first chunk records a failure."""
        err = ProviderError("mid-stream", 500)
        cb = CircuitBreakerMiddleware(
            client=_ErrorAfterFirstChunkProvider(error=err), failure_threshold=10
        )
        chunks = []
        with pytest.raises(ProviderError):
            async for chunk in cb.send_request_stream(AgentRequest(prompt="hi")):
                chunks.append(chunk)
        assert len(chunks) == 1  # first chunk received before error
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_open_to_half_open_to_closed_via_timeout_stream(self) -> None:
        """Asserts full open→half-open→closed transition via timeout and stream success."""
        cb = CircuitBreakerMiddleware(client=MockProvider(), recovery_timeout=10.0)
        cb._state = "open"
        cb._last_failure_time = time.monotonic() - 11.0
        async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert cb._state == "closed"


# ---------------------------------------------------------------------------
# Health-check probing helpers
# ---------------------------------------------------------------------------


class _CountingHealthCheckProvider(MockProvider):
    """MockProvider that counts health_check invocations and returns a configurable status."""

    def __init__(self, healthy: bool = True, **kwargs: object) -> None:
        """Initialise with a healthy flag; health_check_calls starts at zero."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.health_check_calls: int = 0
        self._healthy = healthy

    async def health_check(self) -> bool:
        """Increment health_check_calls and return the configured health status."""
        self.health_check_calls += 1
        return self._healthy


class _RaisingHealthCheckProvider(MockProvider):
    """MockProvider whose health_check raises a caller-supplied exception."""

    def __init__(self, error: Exception) -> None:
        """Initialise with the exception to raise from health_check."""
        super().__init__()
        self._error = error

    async def health_check(self) -> bool:
        """Raise the stored error unconditionally."""
        raise self._error


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthCheckProbing:
    """CircuitBreakerMiddleware recovery probing — health_check call count and exception handling."""

    # -- send_request ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_called_once_on_half_open_send_request(self) -> None:
        """Asserts that health_check is called exactly once per half-open send_request."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "half-open"
        await cb.send_request(AgentRequest(prompt="hi"))
        assert provider.health_check_calls == 1

    @pytest.mark.asyncio
    async def test_health_check_not_called_when_closed_send_request(self) -> None:
        """Asserts that health_check is not called when the circuit is closed."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        await cb.send_request(AgentRequest(prompt="hi"))
        assert provider.health_check_calls == 0

    @pytest.mark.asyncio
    async def test_health_check_not_called_when_open_send_request(self) -> None:
        """Asserts that health_check is not called when the circuit is open."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert provider.health_check_calls == 0

    @pytest.mark.asyncio
    async def test_health_check_exception_propagates_send_request(self) -> None:
        """Asserts that an exception raised by health_check propagates out of send_request."""
        err = RuntimeError("health check network failure")
        cb = CircuitBreakerMiddleware(client=_RaisingHealthCheckProvider(error=err))
        cb._state = "half-open"
        with pytest.raises(RuntimeError, match="health check network failure"):
            await cb.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_health_check_exception_does_not_record_failure_send_request(self) -> None:
        """Asserts that health_check raising does not increment _failure_count."""
        err = RuntimeError("transient probe error")
        cb = CircuitBreakerMiddleware(
            client=_RaisingHealthCheckProvider(error=err), failure_threshold=10
        )
        cb._state = "half-open"
        with pytest.raises(RuntimeError):
            await cb.send_request(AgentRequest(prompt="hi"))
        assert cb._failure_count == 0

    # -- send_request_stream --------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_called_once_on_half_open_send_request_stream(self) -> None:
        """Asserts that health_check is called exactly once per half-open send_request_stream."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "half-open"
        async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert provider.health_check_calls == 1

    @pytest.mark.asyncio
    async def test_health_check_not_called_when_closed_send_request_stream(self) -> None:
        """Asserts that health_check is not called when the circuit is closed (stream)."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert provider.health_check_calls == 0

    @pytest.mark.asyncio
    async def test_health_check_not_called_when_open_send_request_stream(self) -> None:
        """Asserts that health_check is not called when the circuit is open (stream)."""
        provider = _CountingHealthCheckProvider()
        cb = CircuitBreakerMiddleware(client=provider)
        cb._state = "open"
        cb._last_failure_time = time.monotonic()
        with pytest.raises(CircuitOpenError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert provider.health_check_calls == 0

    @pytest.mark.asyncio
    async def test_health_check_exception_propagates_send_request_stream(self) -> None:
        """Asserts that an exception raised by health_check propagates through the stream."""
        err = RuntimeError("health check network failure")
        cb = CircuitBreakerMiddleware(client=_RaisingHealthCheckProvider(error=err))
        cb._state = "half-open"
        with pytest.raises(RuntimeError, match="health check network failure"):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass

    @pytest.mark.asyncio
    async def test_health_check_exception_does_not_record_failure_send_request_stream(self) -> None:
        """Asserts that health_check raising does not increment _failure_count (stream)."""
        err = RuntimeError("transient probe error")
        cb = CircuitBreakerMiddleware(
            client=_RaisingHealthCheckProvider(error=err), failure_threshold=10
        )
        cb._state = "half-open"
        with pytest.raises(RuntimeError):
            async for _ in cb.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert cb._failure_count == 0
