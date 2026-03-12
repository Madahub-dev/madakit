"""Shared test helper classes for the mada-modelkit test suite.

Provides MockProvider — a configurable BaseAgentClient implementation used
across unit tests for middleware and composition. Supports pre-loaded
responses, pre-loaded errors, simulated latency, and call counting.
"""

from __future__ import annotations

import asyncio

from madakit._base import BaseAgentClient
from madakit._types import AgentRequest, AgentResponse


class MockProvider(BaseAgentClient):
    """Configurable mock for testing middleware and composition.

    Pops from responses/errors in order. Falls back to a fixed default
    response when both queues are empty.
    """

    def __init__(
        self,
        responses: list[AgentResponse] | None = None,
        errors: list[Exception] | None = None,
        latency: float = 0.0,
        max_concurrent: int | None = None,
    ) -> None:
        """Initialise with optional pre-loaded responses, errors, and latency."""
        super().__init__(max_concurrent=max_concurrent)
        self._responses: list[AgentResponse] = responses or []
        self._errors: list[Exception] = errors or []
        self._latency = latency
        self.call_count = 0

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Return next pre-loaded response or error; fall back to default."""
        self.call_count += 1
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._errors:
            raise self._errors.pop(0)
        if self._responses:
            return self._responses.pop(0)
        return AgentResponse(content="mock", model="mock", input_tokens=10, output_tokens=5)
