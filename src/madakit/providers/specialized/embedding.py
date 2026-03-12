"""Embedding provider for mada-modelkit.

Generic embedding provider supporting multiple backends.
"""

from __future__ import annotations

import json
from typing import Any

from madakit._types import AgentRequest, AgentResponse
from madakit.providers._http_base import HttpAgentClient

__all__ = ["EmbeddingProvider"]


class EmbeddingProvider(HttpAgentClient):
    """Client for embedding generation.

    Uses OpenAI-compatible embedding format.
    Response content contains JSON-encoded embedding vector.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        base_url: str = "https://api.openai.com/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize embedding provider.

        Args:
            api_key: API key for the embedding service.
            model: Embedding model name (default: text-embedding-ada-002).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            **kwargs,
        )
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"EmbeddingProvider(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build embedding request payload.

        Args:
            request: The request to convert.

        Returns:
            Embedding API payload dict (OpenAI format).
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "input": request.prompt,
        }

        # Some providers support encoding format
        payload["encoding_format"] = "float"

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse embedding response.

        Args:
            data: Response JSON from embedding API.

        Returns:
            Parsed AgentResponse with embedding vector in content.
        """
        # OpenAI-compatible embedding format
        embeddings_data = data.get("data", [])
        if embeddings_data:
            # Get first embedding
            embedding = embeddings_data[0].get("embedding", [])
            # Encode as JSON string in content
            content = json.dumps(embedding)
            dimensions = len(embedding)
        else:
            content = "[]"
            dimensions = 0

        # Token usage
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens)

        return AgentResponse(
            content=content,
            model=data.get("model", self._model),
            input_tokens=input_tokens,
            output_tokens=total_tokens - input_tokens,
            metadata={
                "dimensions": dimensions,
                "artifact_type": "embedding",
            },
        )

    def _endpoint(self) -> str:
        """Return embedding endpoint.

        Returns:
            API endpoint path.
        """
        return "/embeddings"
