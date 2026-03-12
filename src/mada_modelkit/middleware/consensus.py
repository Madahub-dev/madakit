"""Consensus middleware for mada-modelkit.

Send to multiple providers and aggregate results via voting or confidence scoring.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["ConsensusMiddleware"]


class ConsensusMiddleware(BaseAgentClient):
    """Middleware for consensus-based response aggregation.

    Sends requests to multiple providers and aggregates results using
    majority voting or confidence scoring strategies.
    """

    def __init__(
        self,
        providers: list[BaseAgentClient],
        strategy: str = "majority",
    ) -> None:
        """Initialize consensus middleware.

        Args:
            providers: List of provider clients to query.
            strategy: Aggregation strategy ("majority" or "confidence").

        Raises:
            ValueError: If providers list is empty or strategy is invalid.
        """
        super().__init__()

        if not providers:
            raise ValueError("providers list cannot be empty")

        if strategy not in ("majority", "confidence"):
            raise ValueError(f"Invalid strategy: {strategy}")

        self._providers = providers
        self._strategy = strategy

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request to all providers and aggregate responses.

        Args:
            request: The request to send.

        Returns:
            Aggregated response based on consensus strategy.

        Raises:
            MiddlewareError: If all providers fail or consensus cannot be reached.
        """
        # Dispatch to all providers in parallel
        tasks = [
            asyncio.create_task(provider.send_request(request))
            for provider in self._providers
        ]

        # Gather all results (including exceptions)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and collect valid responses
        responses: list[AgentResponse] = []
        errors: list[Exception] = []

        for result in results:
            if isinstance(result, Exception):
                errors.append(result)
            else:
                responses.append(result)

        # Check if we have any valid responses
        if not responses:
            error_msg = f"All {len(self._providers)} providers failed"
            if errors:
                error_msg += f": {errors[0]}"
            raise MiddlewareError(error_msg)

        # Apply consensus strategy
        if self._strategy == "majority":
            return self._majority_vote(responses)
        else:  # confidence
            return self._confidence_aggregate(responses)

    def _majority_vote(self, responses: list[AgentResponse]) -> AgentResponse:
        """Select response with most common content.

        Args:
            responses: List of responses from providers.

        Returns:
            Response with the most common content.
        """
        # Count content occurrences
        content_counter = Counter(r.content for r in responses)

        # Find most common content
        most_common_content, _ = content_counter.most_common(1)[0]

        # Return first response with that content
        for response in responses:
            if response.content == most_common_content:
                # Add metadata about consensus
                metadata = response.metadata.copy() if response.metadata else {}
                metadata["consensus_votes"] = content_counter[most_common_content]
                metadata["consensus_total"] = len(responses)

                return AgentResponse(
                    content=response.content,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata=metadata,
                )

        # Fallback (should never reach here)
        return responses[0]

    def _confidence_aggregate(self, responses: list[AgentResponse]) -> AgentResponse:
        """Aggregate responses using confidence scoring.

        Uses token counts and response length as confidence indicators.
        Returns response with highest confidence score.

        Args:
            responses: List of responses from providers.

        Returns:
            Response with highest confidence score.
        """
        # Calculate confidence scores
        scored_responses: list[tuple[float, AgentResponse]] = []

        for response in responses:
            # Confidence based on output tokens (more tokens = more confident)
            # and content length
            confidence = response.output_tokens + len(response.content) * 0.1
            scored_responses.append((confidence, response))

        # Sort by confidence (highest first)
        scored_responses.sort(key=lambda x: x[0], reverse=True)

        # Return highest confidence response with metadata
        best_confidence, best_response = scored_responses[0]

        metadata = best_response.metadata.copy() if best_response.metadata else {}
        metadata["consensus_confidence"] = best_confidence
        metadata["consensus_total"] = len(responses)

        return AgentResponse(
            content=best_response.content,
            model=best_response.model,
            input_tokens=best_response.input_tokens,
            output_tokens=best_response.output_tokens,
            metadata=metadata,
        )

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request.

        Note: Streaming uses first provider only (consensus requires complete responses).

        Args:
            request: The request to send.

        Yields:
            Stream chunks from the first provider.
        """
        # For streaming, use first provider only
        # Consensus requires complete responses for comparison
        async for chunk in self._providers[0].send_request_stream(request):
            yield chunk

    async def close(self) -> None:
        """Close all provider clients."""
        # Close all providers in parallel
        tasks = [provider.close() for provider in self._providers]
        await asyncio.gather(*tasks, return_exceptions=True)
