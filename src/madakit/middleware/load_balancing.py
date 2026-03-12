"""Load balancing middleware for madakit.

Weighted routing, health-based distribution, and latency-based selection.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["LoadBalancingMiddleware"]


class LoadBalancingMiddleware(BaseAgentClient):
    """Middleware for load balancing across multiple providers.

    Supports weighted round-robin, health-based routing, and latency-based selection.
    """

    def __init__(
        self,
        providers: list[tuple[BaseAgentClient, float]],
        strategy: str = "weighted",
    ) -> None:
        """Initialise with providers and load balancing strategy.

        Args:
            providers: List of (client, weight) tuples. Weights are relative.
            strategy: Load balancing strategy. Options:
                     - "weighted": Weighted round-robin based on weights
                     - "health": Route only to healthy providers
                     - "latency": Route to provider with lowest average latency

        Raises:
            ValueError: If providers is empty, weights invalid, or strategy unknown.
        """
        super().__init__()

        if not providers:
            raise ValueError("At least one provider is required")

        if any(weight <= 0 for _, weight in providers):
            raise ValueError("All provider weights must be positive")

        if strategy not in ("weighted", "health", "latency"):
            raise ValueError(f"Unknown strategy: {strategy}")

        self._providers = providers
        self._strategy = strategy

        # Weighted round-robin state - use simple counter approach
        self._request_count = 0

        # Normalize weights for weighted round-robin
        total_weight = sum(weight for _, weight in providers)
        self._normalized_weights = [weight / total_weight for _, weight in providers]

        # Build cumulative weight distribution
        self._cumulative_weights = []
        cumsum = 0.0
        for weight in self._normalized_weights:
            cumsum += weight
            self._cumulative_weights.append(cumsum)

        # Latency tracking: {provider_index: [latencies]}
        self._latencies: dict[int, list[float]] = {
            i: [] for i in range(len(providers))
        }
        self._latency_window = 10  # Keep last 10 latencies

    def _weighted_round_robin_select(self) -> int:
        """Select provider index using weighted round-robin.

        Returns:
            Index of selected provider.
        """
        # Use smooth weighted round-robin algorithm
        # Build a pattern where providers appear proportional to their weights
        # but distributed evenly (not in blocks)

        # For a sequence length of 10, distribute based on weights
        sequence_length = 10
        position = self._request_count % sequence_length
        self._request_count += 1

        # Build distribution: for each slot, find which provider it maps to
        cumulative = 0.0
        for i, weight in enumerate(self._normalized_weights):
            cumulative += weight * sequence_length
            if position < cumulative:
                return i

        # Fallback to last provider
        return len(self._providers) - 1

    async def _health_based_select(self) -> int:
        """Select a healthy provider.

        Returns:
            Index of healthy provider.

        Raises:
            MiddlewareError: If no healthy providers available.
        """
        # Check health of all providers
        health_checks = []
        for client, _ in self._providers:
            try:
                is_healthy = await client.health_check()
                health_checks.append(is_healthy)
            except Exception:
                health_checks.append(False)

        # Find healthy providers
        healthy_indices = [i for i, healthy in enumerate(health_checks) if healthy]

        if not healthy_indices:
            raise MiddlewareError("No healthy providers available")

        # Select first healthy provider (could be random or weighted)
        return healthy_indices[0]

    def _latency_based_select(self) -> int:
        """Select provider with lowest average latency.

        Returns:
            Index of provider with lowest latency.
        """
        # Calculate average latency for each provider
        avg_latencies = []
        for i in range(len(self._providers)):
            latency_list = self._latencies[i]
            if latency_list:
                avg_latencies.append((i, sum(latency_list) / len(latency_list)))
            else:
                # No latency data yet - use round-robin to explore
                # This ensures all providers get tried initially
                avg_latencies.append((i, float('inf')))

        # Count how many providers have no data
        no_data_count = sum(1 for _, lat in avg_latencies if lat == float('inf'))

        if no_data_count == len(avg_latencies):
            # No latency data for any provider - round robin
            result = self._request_count % len(self._providers)
            self._request_count += 1
            return result
        elif no_data_count > 0:
            # Some providers have no data - prefer those to explore
            no_data_providers = [i for i, lat in avg_latencies if lat == float('inf')]
            return no_data_providers[0]

        # All have data - sort by latency and return best
        avg_latencies.sort(key=lambda x: x[1])
        return avg_latencies[0][0]

    async def _select_provider(self) -> tuple[BaseAgentClient, int]:
        """Select provider based on configured strategy.

        Returns:
            Tuple of (selected_client, provider_index).

        Raises:
            MiddlewareError: If provider selection fails.
        """
        if self._strategy == "weighted":
            index = self._weighted_round_robin_select()
        elif self._strategy == "health":
            index = await self._health_based_select()
        elif self._strategy == "latency":
            index = self._latency_based_select()
        else:
            raise MiddlewareError(f"Unknown strategy: {self._strategy}")

        client, _ = self._providers[index]
        return client, index

    def _record_latency(self, provider_index: int, latency: float) -> None:
        """Record latency for a provider.

        Args:
            provider_index: Index of provider.
            latency: Latency in seconds.
        """
        if provider_index not in self._latencies:
            self._latencies[provider_index] = []

        self._latencies[provider_index].append(latency)

        # Keep only recent latencies
        if len(self._latencies[provider_index]) > self._latency_window:
            self._latencies[provider_index].pop(0)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Route request to selected provider.

        Args:
            request: The request to send.

        Returns:
            Response from selected provider.

        Raises:
            MiddlewareError: If provider selection fails.
        """
        # Select provider
        client, provider_index = await self._select_provider()

        # Send request and track latency
        start_time = time.perf_counter()
        response = await client.send_request(request)
        latency = time.perf_counter() - start_time

        # Record latency for latency-based routing
        self._record_latency(provider_index, latency)

        return response

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Route streaming request to selected provider.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from selected provider.

        Raises:
            MiddlewareError: If provider selection fails.
        """
        # Select provider
        client, provider_index = await self._select_provider()

        # Stream from selected provider and track latency
        start_time = time.perf_counter()
        first_chunk = True

        async for chunk in client.send_request_stream(request):
            if first_chunk:
                # Record time to first chunk
                latency = time.perf_counter() - start_time
                self._record_latency(provider_index, latency)
                first_chunk = False

            yield chunk
