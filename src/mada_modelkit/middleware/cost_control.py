"""Cost control middleware for mada-modelkit.

Budget tracking, spending alerts, and cost caps for API usage management.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import BudgetExceededError
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

    def _track_cost(self, response: AgentResponse) -> None:
        """Calculate and accumulate cost for a response.

        Calls cost_fn to determine cost, then adds to total_spend.
        If budget_cap is set and would be exceeded, raises BudgetExceededError.
        If alert threshold is crossed, calls on_alert callback (once).

        Args:
            response: The response to calculate cost for.

        Raises:
            BudgetExceededError: If adding this cost would exceed budget_cap.
        """
        cost = self._cost_fn(response)

        # Check if adding this cost would exceed the cap
        if self._budget_cap is not None and (self._total_spend + cost) > self._budget_cap:
            raise BudgetExceededError(
                f"Request would exceed budget cap of {self._budget_cap:.2f} "
                f"(current: {self._total_spend:.2f}, cost: {cost:.2f})"
            )

        # Add cost to total
        self._total_spend += cost

        # Check if alert threshold crossed (only fire once)
        if (
            self._budget_cap is not None
            and self._on_alert is not None
            and not self._alert_fired
        ):
            threshold_amount = self._budget_cap * self._alert_threshold
            if self._total_spend >= threshold_amount:
                self._alert_fired = True
                self._on_alert(self._total_spend, threshold_amount)

    @property
    def total_spend(self) -> float:
        """Return the current total spend."""
        return self._total_spend

    def reset_budget(self) -> None:
        """Reset budget tracking to start a new period.

        Zeroes total_spend and clears the alert_fired flag, allowing
        alerts to fire again in the new tracking period.
        """
        self._total_spend = 0.0
        self._alert_fired = False

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request and track cost.

        Delegates to wrapped client, then increments spend.
        """
        response = await self._client.send_request(request)
        self._track_cost(response)
        return response

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks and track cost.

        Delegates to wrapped client, streams chunks, then increments spend.
        Constructs a synthetic AgentResponse from the final chunk's metadata
        to calculate cost.
        """
        async for chunk in self._client.send_request_stream(request):
            if chunk.is_final:
                # Build synthetic response from final chunk metadata for cost tracking
                response = AgentResponse(
                    content="",  # Content not needed for cost_fn
                    model=chunk.metadata.get("model", "unknown"),
                    input_tokens=chunk.metadata.get("input_tokens", 0),
                    output_tokens=chunk.metadata.get("output_tokens", 0),
                    metadata=chunk.metadata,
                )
                self._track_cost(response)
            yield chunk
