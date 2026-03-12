"""Tests for Replicate AI client.

Covers Replicate custom prediction-based format.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest
from madakit.providers.cloud.replicate import ReplicateClient


class TestModuleExports:
    """Verify replicate module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.providers.cloud import replicate

        assert hasattr(replicate, "__all__")

    def test_all_contains_replicate_client(self) -> None:
        """__all__ contains ReplicateClient."""
        from madakit.providers.cloud.replicate import __all__

        assert "ReplicateClient" in __all__


class TestReplicateClientConstructor:
    """Test ReplicateClient constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with default model."""
        client = ReplicateClient(api_key="test-key")

        assert client._model == "meta/llama-2-70b-chat"
        assert client._api_key == "test-key"

    def test_custom_model(self) -> None:
        """Constructor accepts custom model."""
        client = ReplicateClient(api_key="test-key", model="meta/llama-3-8b")

        assert client._model == "meta/llama-3-8b"

    def test_custom_base_url(self) -> None:
        """Constructor accepts custom base URL."""
        client = ReplicateClient(
            api_key="test-key", base_url="https://custom.replicate.com/v1"
        )

        assert client._model == "meta/llama-2-70b-chat"

    def test_repr_redacts_api_key(self) -> None:
        """__repr__ redacts API key."""
        client = ReplicateClient(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "meta/llama-2-70b-chat" in repr_str


class TestReplicateFormat:
    """Test Replicate custom format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates Replicate prediction format."""
        client = ReplicateClient(api_key="test-key")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        # Replicate prediction format
        assert payload["version"] == "meta/llama-2-70b-chat"
        assert payload["input"]["prompt"] == "Hello"

    def test_build_payload_with_system_prompt(self) -> None:
        """_build_payload includes system_prompt in input."""
        client = ReplicateClient(api_key="test-key")
        request = AgentRequest(
            prompt="What is AI?", system_prompt="You are helpful"
        )

        payload = client._build_payload(request)

        assert payload["input"]["system_prompt"] == "You are helpful"
        assert payload["input"]["prompt"] == "What is AI?"

    def test_build_payload_with_parameters(self) -> None:
        """_build_payload includes optional parameters."""
        client = ReplicateClient(api_key="test-key")
        request = AgentRequest(
            prompt="Hello",
            max_tokens=512,
            temperature=0.9,
            stop=["END"],
        )

        payload = client._build_payload(request)

        assert payload["input"]["max_tokens"] == 512
        assert payload["input"]["temperature"] == 0.9
        assert payload["input"]["stop_sequences"] == ["END"]

    def test_parse_response_string_output(self) -> None:
        """_parse_response handles string output."""
        client = ReplicateClient(api_key="test-key")

        data = {
            "output": "This is a response",
            "metrics": {"input_tokens": 10, "output_tokens": 20},
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"
        assert response.model == "meta/llama-2-70b-chat"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_parse_response_list_output(self) -> None:
        """_parse_response joins list output."""
        client = ReplicateClient(api_key="test-key")

        data = {
            "output": ["This ", "is ", "a ", "response"],
            "metrics": {"input_tokens": 10, "output_tokens": 20},
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"

    def test_parse_response_no_metrics(self) -> None:
        """_parse_response handles missing metrics."""
        client = ReplicateClient(api_key="test-key")

        data = {
            "output": "Response",
        }

        response = client._parse_response(data)

        assert response.content == "Response"
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_endpoint_returns_predictions(self) -> None:
        """_endpoint returns predictions path."""
        client = ReplicateClient(api_key="test-key")

        endpoint = client._endpoint()

        assert endpoint == "/predictions"


class TestReplicateClientIntegration:
    """Integration tests for ReplicateClient."""

    def test_client_initialization(self) -> None:
        """Client initializes with TLS requirement."""
        client = ReplicateClient(api_key="test-key", model="meta/llama-3-8b")

        assert client._model == "meta/llama-3-8b"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = ReplicateClient(api_key="key1", model="meta/llama-2-70b-chat")
        client2 = ReplicateClient(api_key="key2", model="meta/llama-3-8b")

        assert client1._model != client2._model
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with ReplicateClient(api_key="test-key") as client:
            assert client._model == "meta/llama-2-70b-chat"
