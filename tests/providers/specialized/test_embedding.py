"""Tests for EmbeddingProvider.

Covers embedding generation with OpenAI-compatible format.
"""

from __future__ import annotations

import json

import pytest

from madakit._types import AgentRequest
from madakit.providers.specialized.embedding import EmbeddingProvider


class TestModuleExports:
    """Verify embedding module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.providers.specialized import embedding

        assert hasattr(embedding, "__all__")

    def test_all_contains_embedding_provider(self) -> None:
        """__all__ contains EmbeddingProvider."""
        from madakit.providers.specialized.embedding import __all__

        assert "EmbeddingProvider" in __all__


class TestEmbeddingProviderConstructor:
    """Test EmbeddingProvider constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with default model."""
        client = EmbeddingProvider(api_key="test-key")

        assert client._model == "text-embedding-ada-002"
        assert client._api_key == "test-key"

    def test_custom_model(self) -> None:
        """Constructor accepts custom model."""
        client = EmbeddingProvider(api_key="test-key", model="text-embedding-3-large")

        assert client._model == "text-embedding-3-large"

    def test_custom_base_url(self) -> None:
        """Constructor accepts custom base URL."""
        client = EmbeddingProvider(
            api_key="test-key", base_url="https://custom.openai.com/v1"
        )

        assert client._model == "text-embedding-ada-002"

    def test_repr_redacts_api_key(self) -> None:
        """__repr__ redacts API key."""
        client = EmbeddingProvider(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "text-embedding-ada-002" in repr_str


class TestEmbeddingFormat:
    """Test embedding request/response format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates OpenAI embedding format."""
        client = EmbeddingProvider(api_key="test-key")
        request = AgentRequest(prompt="The quick brown fox")

        payload = client._build_payload(request)

        assert payload["model"] == "text-embedding-ada-002"
        assert payload["input"] == "The quick brown fox"
        assert payload["encoding_format"] == "float"

    def test_build_payload_with_custom_model(self) -> None:
        """_build_payload uses custom model."""
        client = EmbeddingProvider(api_key="test-key", model="text-embedding-3-small")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        assert payload["model"] == "text-embedding-3-small"

    def test_parse_response_with_embedding(self) -> None:
        """_parse_response extracts embedding vector."""
        client = EmbeddingProvider(api_key="test-key")

        data = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        response = client._parse_response(data)

        # Embedding should be JSON-encoded
        embedding = json.loads(response.content)
        assert embedding == [0.1, 0.2, 0.3, 0.4]
        assert response.metadata["dimensions"] == 4
        assert response.metadata["artifact_type"] == "embedding"

    def test_parse_response_with_large_embedding(self) -> None:
        """_parse_response handles large embeddings."""
        client = EmbeddingProvider(api_key="test-key")

        # 1536-dimensional embedding (ada-002 size)
        embedding_vector = [0.1] * 1536
        data = {
            "data": [{"embedding": embedding_vector}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        response = client._parse_response(data)

        parsed = json.loads(response.content)
        assert len(parsed) == 1536
        assert response.metadata["dimensions"] == 1536

    def test_parse_response_empty_data(self) -> None:
        """_parse_response handles empty data."""
        client = EmbeddingProvider(api_key="test-key")

        data = {"data": []}

        response = client._parse_response(data)

        assert response.content == "[]"
        assert response.metadata["dimensions"] == 0

    def test_parse_response_with_token_usage(self) -> None:
        """_parse_response extracts token counts."""
        client = EmbeddingProvider(api_key="test-key")

        data = {
            "data": [{"embedding": [0.1, 0.2]}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        response = client._parse_response(data)

        assert response.input_tokens == 10
        assert response.output_tokens == 0

    def test_parse_response_model_field(self) -> None:
        """_parse_response uses model from response."""
        client = EmbeddingProvider(api_key="test-key")

        data = {
            "data": [{"embedding": [0.1]}],
            "model": "text-embedding-3-large",
        }

        response = client._parse_response(data)

        assert response.model == "text-embedding-3-large"

    def test_endpoint_returns_embeddings(self) -> None:
        """_endpoint returns embeddings path."""
        client = EmbeddingProvider(api_key="test-key")

        endpoint = client._endpoint()

        assert endpoint == "/embeddings"


class TestEmbeddingProviderIntegration:
    """Integration tests for EmbeddingProvider."""

    def test_client_initialization(self) -> None:
        """Client initializes with TLS requirement."""
        client = EmbeddingProvider(api_key="test-key", model="text-embedding-3-small")

        assert client._model == "text-embedding-3-small"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = EmbeddingProvider(api_key="key1", model="model-a")
        client2 = EmbeddingProvider(api_key="key2", model="model-b")

        assert client1._model != client2._model
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with EmbeddingProvider(api_key="test-key") as client:
            assert client._model == "text-embedding-ada-002"
