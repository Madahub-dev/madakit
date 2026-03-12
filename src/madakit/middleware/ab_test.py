"""A/B testing middleware for madakit.

Split traffic between multiple providers for A/B testing and experimentation.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import hashlib
from typing import Any, AsyncIterator, Callable

from madakit._base import BaseAgentClient
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["ABTestMiddleware"]


class ABTestMiddleware(BaseAgentClient):
    """Middleware for A/B testing between multiple provider variants.

    Deterministically splits traffic based on request key hash and variant weights.
    """

    def __init__(
        self,
        variants: list[tuple[BaseAgentClient, float]],
        key_fn: Callable[[AgentRequest], str] | None = None,
    ) -> None:
        """Initialise with variants and optional key function.

        Args:
            variants: List of (client, weight) tuples. Weights are relative.
            key_fn: Optional function to extract key from request.
                   If None, uses request prompt for deterministic splitting.

        Raises:
            ValueError: If variants is empty or weights are invalid.
        """
        super().__init__()

        if not variants:
            raise ValueError("At least one variant is required")

        if any(weight <= 0 for _, weight in variants):
            raise ValueError("All variant weights must be positive")

        self._variants = variants
        self._key_fn = key_fn

        # Normalize weights to sum to 1.0
        total_weight = sum(weight for _, weight in variants)
        self._normalized_weights = [weight / total_weight for _, weight in variants]

        # Compute cumulative distribution for variant selection
        self._cumulative_weights = []
        cumsum = 0.0
        for weight in self._normalized_weights:
            cumsum += weight
            self._cumulative_weights.append(cumsum)

    def _get_request_key(self, request: AgentRequest) -> str:
        """Extract key from request for deterministic splitting.

        Args:
            request: The request to extract key from.

        Returns:
            String key for hash-based splitting.
        """
        if self._key_fn:
            return self._key_fn(request)

        # Default: use prompt as key
        return request.prompt

    def _hash_to_variant_index(self, key: str) -> int:
        """Map a key to a variant index using consistent hashing.

        Args:
            key: String key to hash.

        Returns:
            Index of the selected variant.
        """
        # Hash the key to a number in [0, 1)
        hash_bytes = hashlib.md5(key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        hash_float = (hash_int / (2 ** 64))

        # Select variant based on cumulative weights
        for i, cumulative_weight in enumerate(self._cumulative_weights):
            if hash_float < cumulative_weight:
                return i

        # Fallback to last variant (should not happen with proper normalization)
        return len(self._variants) - 1

    def _select_variant(self, request: AgentRequest) -> tuple[BaseAgentClient, int]:
        """Select a variant client based on request key.

        Args:
            request: The request to route.

        Returns:
            Tuple of (selected_client, variant_index).
        """
        key = self._get_request_key(request)
        variant_index = self._hash_to_variant_index(key)
        client, _ = self._variants[variant_index]
        return client, variant_index

    def _add_variant_metadata(
        self,
        response: AgentResponse,
        variant_index: int,
    ) -> AgentResponse:
        """Add variant metadata to response.

        Args:
            response: Original response.
            variant_index: Index of variant that generated response.

        Returns:
            Response with variant metadata added.
        """
        new_metadata = {**response.metadata, "variant": variant_index}
        return AgentResponse(
            content=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            metadata=new_metadata,
        )

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Route request to selected variant and tag response.

        Args:
            request: The request to send.

        Returns:
            Response with variant metadata.
        """
        client, variant_index = self._select_variant(request)
        response = await client.send_request(request)
        return self._add_variant_metadata(response, variant_index)

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Route streaming request to selected variant.

        Args:
            request: The request to send.

        Yields:
            Stream chunks with variant metadata in final chunk.
        """
        client, variant_index = self._select_variant(request)

        async for chunk in client.send_request_stream(request):
            # Add variant to final chunk metadata
            if chunk.is_final:
                new_metadata = {**chunk.metadata, "variant": variant_index}
                yield StreamChunk(
                    delta=chunk.delta,
                    is_final=chunk.is_final,
                    metadata=new_metadata,
                )
            else:
                yield chunk
