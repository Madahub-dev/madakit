"""Tests for Mistral AI client.

Covers Mistral OpenAI-compatible format.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest
from madakit.providers.cloud.mistral import MistralClient


class TestModuleExports:
    """Verify mistral module exports."""

    def test_module_has_all(self) -> None:
        from madakit.providers.cloud import mistral

        assert hasattr(mistral, "__all__")

    def test_all_contains_mistral_client(self) -> None:
        from madakit.providers.cloud.mistral import __all__

        assert "MistralClient" in __all__


class TestMistralClientConstructor:
    """Test MistralClient constructor."""

    def test_valid_constructor(self) -> None:
        client = MistralClient(api_key="test-key")

        assert client._model == "mistral-medium"
        assert client._api_key == "test-key"

    def test_custom_model(self) -> None:
        client = MistralClient(api_key="test-key", model="mistral-small")

        assert client._model == "mistral-small"

    def test_custom_base_url(self) -> None:
        client = MistralClient(
            api_key="test-key", base_url="https://custom.mistral.ai/v1"
        )

        assert client._model == "mistral-medium"

    def test_repr_redacts_api_key(self) -> None:
        client = MistralClient(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "mistral-medium" in repr_str


class TestOpenAICompatibility:
    """Test OpenAI-compatible format."""

    def test_build_payload_format(self) -> None:
        client = MistralClient(api_key="test-key")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        # OpenAI-compatible format
        assert payload["model"] == "mistral-medium"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_build_payload_with_system_prompt(self) -> None:
        client = MistralClient(api_key="test-key")
        request = AgentRequest(
            prompt="What is AI?", system_prompt="You are helpful"
        )

        payload = client._build_payload(request)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {
            "role": "system",
            "content": "You are helpful",
        }
        assert payload["messages"][1] == {"role": "user", "content": "What is AI?"}

    def test_parse_response_format(self) -> None:
        client = MistralClient(api_key="test-key")

        data = {
            "choices": [{"message": {"content": "This is a response"}}],
            "model": "mistral-medium",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"
        assert response.model == "mistral-medium"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_endpoint_returns_chat_completions(self) -> None:
        client = MistralClient(api_key="test-key")

        endpoint = client._endpoint()

        assert endpoint == "/chat/completions"


class TestMistralClientIntegration:
    """Integration tests for MistralClient."""

    def test_client_initialization(self) -> None:
        client = MistralClient(api_key="test-key", model="mistral-small")

        assert client._model == "mistral-small"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        client1 = MistralClient(api_key="key1", model="mistral-medium")
        client2 = MistralClient(api_key="key2", model="mistral-small")

        assert client1._model != client2._model
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with MistralClient(api_key="test-key") as client:
            assert client._model == "mistral-medium"
