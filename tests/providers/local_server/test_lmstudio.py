"""Tests for LM Studio client.

Covers LM Studio OpenAI-compatible local server.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest
from mada_modelkit.providers.local_server.lmstudio import LMStudioClient


class TestModuleExports:
    """Verify lmstudio module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from mada_modelkit.providers.local_server import lmstudio

        assert hasattr(lmstudio, "__all__")

    def test_all_contains_lmstudio_client(self) -> None:
        """__all__ contains LMStudioClient."""
        from mada_modelkit.providers.local_server.lmstudio import __all__

        assert "LMStudioClient" in __all__


class TestLMStudioClientConstructor:
    """Test LMStudioClient constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with model."""
        client = LMStudioClient(model="llama-3-8b")

        assert client._model == "llama-3-8b"

    def test_custom_base_url(self) -> None:
        """Constructor accepts custom base URL."""
        client = LMStudioClient(
            model="llama-3-8b", base_url="http://localhost:5678/v1"
        )

        assert client._model == "llama-3-8b"
        assert "5678" in client._base_url

    def test_repr_shows_model_and_url(self) -> None:
        """__repr__ shows model and base URL."""
        client = LMStudioClient(model="llama-3-8b")

        repr_str = repr(client)

        assert "llama-3-8b" in repr_str
        assert "localhost:1234" in repr_str


class TestOpenAICompatibility:
    """Test OpenAI-compatible format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates OpenAI format."""
        client = LMStudioClient(model="llama-3-8b")
        request = AgentRequest(prompt="Hello")

        payload = client._build_payload(request)

        # OpenAI-compatible format
        assert payload["model"] == "llama-3-8b"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_build_payload_with_system_prompt(self) -> None:
        """_build_payload includes system message."""
        client = LMStudioClient(model="llama-3-8b")
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
        client = LMStudioClient(model="llama-3-8b")

        data = {
            "choices": [{"message": {"content": "This is a response"}}],
            "model": "llama-3-8b",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        response = client._parse_response(data)

        assert response.content == "This is a response"
        assert response.model == "llama-3-8b"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    def test_endpoint_returns_chat_completions(self) -> None:
        """_endpoint returns chat completions path."""
        client = LMStudioClient(model="llama-3-8b")

        endpoint = client._endpoint()

        assert endpoint == "/chat/completions"


class TestLMStudioClientIntegration:
    """Integration tests for LMStudioClient."""

    def test_client_initialization(self) -> None:
        """Client initializes without TLS requirement."""
        client = LMStudioClient(model="llama-3-8b")

        assert client._model == "llama-3-8b"
        assert client._require_tls is False

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = LMStudioClient(model="model-a")
        client2 = LMStudioClient(model="model-b")

        assert client1._model != client2._model

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with LMStudioClient(model="llama-3-8b") as client:
            assert client._model == "llama-3-8b"
