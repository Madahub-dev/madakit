"""Tests for Jan client.

Covers Jan OpenAI-compatible local server.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest
from madakit.providers.local_server.jan import JanClient


class TestModuleExports:
    """Verify jan module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.providers.local_server import jan

        assert hasattr(jan, "__all__")

    def test_all_contains_jan_client(self) -> None:
        """__all__ contains JanClient."""
        from madakit.providers.local_server.jan import __all__

        assert "JanClient" in __all__


class TestJanClientConstructor:
    """Test JanClient constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with model."""
        client = JanClient(model="mistral-7b")

        assert client._model == "mistral-7b"

    def test_custom_base_url(self) -> None:
        """Constructor accepts custom base URL."""
        client = JanClient(model="mistral-7b", base_url="http://localhost:5678/v1")

        assert client._model == "mistral-7b"
        assert "5678" in client._base_url

    def test_repr_shows_model_and_url(self) -> None:
        """__repr__ shows model and base URL."""
        client = JanClient(model="mistral-7b")

        repr_str = repr(client)

        assert "mistral-7b" in repr_str
        assert "localhost:1337" in repr_str


class TestOpenAICompatibility:
    """Test OpenAI-compatible format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates OpenAI format."""
        client = JanClient(model="mistral-7b")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        # OpenAI-compatible format
        assert payload["model"] == "mistral-7b"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_build_payload_with_system_prompt(self) -> None:
        """_build_payload includes system message."""
        client = JanClient(model="mistral-7b")
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
        """_parse_response handles OpenAI format."""
        client = JanClient(model="mistral-7b")

        data = {
            "choices": [{"message": {"content": "This is a response"}}],
            "model": "mistral-7b",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"
        assert response.model == "mistral-7b"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_endpoint_returns_chat_completions(self) -> None:
        """_endpoint returns chat completions path."""
        client = JanClient(model="mistral-7b")

        endpoint = client._endpoint()

        assert endpoint == "/chat/completions"


class TestJanClientIntegration:
    """Integration tests for JanClient."""

    def test_client_initialization(self) -> None:
        """Client initializes without TLS requirement."""
        client = JanClient(model="mistral-7b")

        assert client._model == "mistral-7b"
        assert client._require_tls is False

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = JanClient(model="model-a")
        client2 = JanClient(model="model-b")

        assert client1._model != client2._model

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with JanClient(model="mistral-7b") as client:
            assert client._model == "mistral-7b"
