"""Metrics middleware for mada-modelkit.

Prometheus-compatible metrics export for request/response monitoring.
Requires prometheus_client (optional dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

__all__ = ["MetricsMiddleware"]


class MetricsMiddleware(BaseAgentClient):
    """Middleware that exports Prometheus-compatible metrics."""

    def __init__(
        self,
        client: BaseAgentClient,
        registry: CollectorRegistry | None = None,
        prefix: str = "madakit",
    ) -> None:
        """Initialise with a wrapped client and metrics configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            registry: Optional Prometheus registry. If None, uses default registry.
            prefix: Metric name prefix (default "madakit").

        Raises:
            ImportError: If prometheus_client is not installed.
        """
        super().__init__()
        self._client = client
        self._prefix = prefix

        # Deferred import of prometheus_client
        try:
            from prometheus_client import CollectorRegistry, Counter

            self._prometheus_available = True
        except ImportError as exc:
            raise ImportError(
                "prometheus_client is required for MetricsMiddleware. "
                "Install with: pip install prometheus-client"
            ) from exc

        # Use provided registry or default
        if registry is None:
            from prometheus_client import REGISTRY

            self._registry = REGISTRY
        else:
            self._registry = registry

        # Will initialize metrics in task 9.2.2
        self._metrics_initialized = False

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with metrics collection.

        Records request count, latency, and token usage.
        """
        # Stub: will implement in task 9.2.2+
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with metrics collection.

        Records request count, latency, TTFT, and token usage.
        """
        # Stub: will implement in task 9.2.2+
        async for chunk in self._client.send_request_stream(request):
            yield chunk
