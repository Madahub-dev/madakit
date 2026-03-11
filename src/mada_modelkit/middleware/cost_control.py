"""Cost control middleware for mada-modelkit.

Budget tracking, spending alerts, and cost caps for API usage management.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["CostControlMiddleware"]


class CostControlMiddleware(BaseAgentClient):
    """Middleware that tracks and enforces budget caps on API costs."""

    def __init__(
        self,
        client: BaseAgentClient,
        cost_fn: Callable[[AgentResponse], float],
        budget_cap: float | None = None,
        alert_threshold: float = 0.8,
        on_alert: Callable[[float, float], None] | None = None,
    ) -> None:
        """Initialise with a wrapped client and cost control configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            cost_fn: Function to calculate cost from an AgentResponse.
            budget_cap: Maximum allowed spend. None means no cap.
            alert_threshold: Fraction of budget_cap at which to trigger alert (0.0-1.0).
            on_alert: Optional callback invoked when spending crosses alert_threshold.
                Called with (current_spend, threshold_amount).
        """
        super().__init__()
        self._client = client
        self._budget_cap = budget_cap
        self._alert_threshold = alert_threshold
        self._cost_fn = cost_fn
        self._on_alert = on_alert

        # Tracking state
        self._total_spend = 0.0
        self._alert_fired = False

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request and track cost.

        Delegates to wrapped client, then increments spend.
        """
        # Stub: will implement in task 8.2.5
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks and track cost.

        Delegates to wrapped client, streams chunks, then increments spend.
        """
        # Stub: will implement in task 8.2.5
        async for chunk in self._client.send_request_stream(request):
            yield chunk
