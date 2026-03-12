"""Circuit breaker middleware for mada-modelkit.

Wraps any BaseAgentClient and opens a circuit after a configurable failure
threshold, preventing cascading failures. Supports closed, open, and half-open
states with timeout-based recovery. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import CircuitOpenError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["CircuitBreakerMiddleware"]


class CircuitBreakerMiddleware(BaseAgentClient):
    """Middleware that opens a circuit after repeated failures to prevent cascading errors."""

    def __init__(
        self,
        client: BaseAgentClient,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        """Initialise with a wrapped client and circuit breaker configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            failure_threshold: Number of consecutive failures before the circuit opens.
            recovery_timeout: Seconds to wait in the open state before transitioning
                to half-open and probing the backend.
        """
        super().__init__()
        self._client = client
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._state: str = "closed"
        self._last_failure_time: float | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def _record_failure(self) -> None:
        """Record a failure and open the circuit when the threshold is reached.

        In half-open state any single failure immediately reopens the circuit.
        In closed state the circuit opens once _failure_count reaches _failure_threshold.
        Always records _last_failure_time when transitioning to open.
        """
        async with self._lock:
            self._failure_count += 1
            if self._state == "half-open" or self._failure_count >= self._failure_threshold:
                self._state = "open"
                self._last_failure_time = time.monotonic()

    async def _record_success(self) -> None:
        """Record a success, reset the failure count, and close the circuit."""
        async with self._lock:
            self._failure_count = 0
            self._state = "closed"

    async def _check_state(self) -> str:
        """Return the effective current state, applying the open→half-open timeout transition.

        If the circuit is open and recovery_timeout seconds have elapsed since the
        last failure, the state advances to half-open so a probe can be attempted.
        """
        async with self._lock:
            if self._state == "open" and self._last_failure_time is not None:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half-open"
            return self._state

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with circuit breaker protection.

        Closed: passes through to the wrapped client; records failures and successes.
        Open: raises CircuitOpenError immediately without calling the client.
        Half-open: probes the backend via health_check first; if the probe fails the
        circuit reopens and CircuitOpenError is raised; if the probe succeeds the
        request is attempted — success closes the circuit, failure reopens it.
        """
        state = await self._check_state()

        if state == "open":
            raise CircuitOpenError("Circuit breaker is open")

        if state == "half-open":
            healthy = await self._client.health_check()
            if not healthy:
                await self._record_failure()
                raise CircuitOpenError("Circuit breaker reopened: health check failed")

        try:
            response = await self._client.send_request(request)
            await self._record_success()
            return response
        except Exception:
            await self._record_failure()
            raise

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with circuit breaker protection.

        Applies the same closed/open/half-open logic as send_request. Success is
        recorded when the stream exhausts normally; any exception records a failure
        and re-raises. If the consumer abandons the stream early (aclose), neither
        success nor failure is recorded.
        """
        state = await self._check_state()

        if state == "open":
            raise CircuitOpenError("Circuit breaker is open")

        if state == "half-open":
            healthy = await self._client.health_check()
            if not healthy:
                await self._record_failure()
                raise CircuitOpenError("Circuit breaker reopened: health check failed")

        try:
            async for chunk in self._client.send_request_stream(request):
                yield chunk
            await self._record_success()
        except Exception:
            await self._record_failure()
            raise
