"""Tests for Cohere AI client.

Covers Cohere-specific payload format, response parsing, streaming.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest
from madakit.providers.cloud.cohere import CohereClient


class TestModuleExports:
    """Verify cohere module exports."""

    def test_module_has_all(self) -> None:
        from madakit.providers.cloud import cohere

        assert hasattr(cohere, "__all__")

    def test_all_contains_cohere_client(self) -> None:
        from madakit.providers.cloud.cohere import __all__

        assert "CohereClient" in __all__


class TestCohereClientConstructor:
    """Test CohereClient constructor."""

    def test_valid_constructor(self) -> None:
        client = CohereClient(api_key="test-key")

        assert client._model == "command-r-plus"
        assert client._api_key == "test-key"

    def test_custom_model(self) -> None:
        client = CohereClient(api_key="test-key", model="command-r")

        assert client._model == "command-r"

    def test_custom_base_url(self) -> None:
        # Just verify it doesn't raise (base_url is internal)
        client = CohereClient(
            api_key="test-key", base_url="https://custom.cohere.ai/v1"
        )

        assert client._model == "command-r-plus"

    def test_repr_redacts_api_key(self) -> None:
        client = CohereClient(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "command-r-plus" in repr_str


class TestBuildPayload:
    """Test Cohere payload building."""

    def test_basic_payload(self) -> None:
        client = CohereClient(api_key="test-key")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        assert payload["model"] == "command-r-plus"
        assert payload["message"] == "Hello"
        assert "max_tokens" not in payload  # Default not included
        assert "temperature" not in payload  # Default not included

    def test_payload_with_max_tokens(self) -> None:
        client = CohereClient(api_key="test-key")
        request = AgentRequest(prompt="Test", max_tokens=500)

        payload = client._build_payload(request)

        assert payload["max_tokens"] == 500

    def test_payload_with_temperature(self) -> None:
        client = CohereClient(api_key="test-key")
        request = AgentRequest(prompt="Test", temperature=0.9)

        payload = client._build_payload(request)

        assert payload["temperature"] == 0.9

    def test_payload_with_stop_sequences(self) -> None:
        client = CohereClient(api_key="test-key")
        request = AgentRequest(prompt="Test", stop=["STOP", "END"])

        payload = client._build_payload(request)

        assert payload["stop_sequences"] == ["STOP", "END"]

    def test_payload_with_system_prompt(self) -> None:
        client = CohereClient(api_key="test-key")
        request = AgentRequest(
            prompt="What is AI?", system_prompt="You are a helpful assistant"
        )

        payload = client._build_payload(request)

        assert payload["preamble"] == "You are a helpful assistant"
        assert payload["message"] == "What is AI?"


class TestParseResponse:
    """Test Cohere response parsing."""

    def test_parse_basic_response(self) -> None:
        client = CohereClient(api_key="test-key")

        data = {
            "text": "This is a response",
            "meta": {
                "billed_units": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                }
            },
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"
        assert response.model == "command-r-plus"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_parse_response_without_tokens(self) -> None:
        client = CohereClient(api_key="test-key")

        data = {"text": "Response without token info"}

        response = client._parse_response(data)

        assert response.content == "Response without token info"
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_parse_empty_response(self) -> None:
        client = CohereClient(api_key="test-key")

        data = {}

        response = client._parse_response(data)

        assert response.content == ""


class TestEndpoint:
    """Test Cohere endpoint."""

    def test_endpoint_returns_chat(self) -> None:
        client = CohereClient(api_key="test-key")

        endpoint = client._endpoint()

        assert endpoint == "/chat"


class TestCohereClientIntegration:
    """Integration tests for CohereClient."""

    def test_client_initialization(self) -> None:
        client = CohereClient(api_key="test-key", model="command-r")

        # Verify client is properly initialized
        assert client._model == "command-r"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        client1 = CohereClient(api_key="key1", model="command-r-plus")
        client2 = CohereClient(api_key="key2", model="command-r")

        # Clients are independent
        assert client1._model != client2._model
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with CohereClient(api_key="test-key") as client:
            # Client is usable within context
            assert client._model == "command-r-plus"

        # Context manager exited without error
