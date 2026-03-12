"""Tests for Groq client."""

from __future__ import annotations

import pytest
from madakit.providers.cloud.groq import GroqClient


class TestGroqClient:
    """Test Groq client."""

    def test_constructor(self) -> None:
        client = GroqClient(api_key="test-key")
        assert client._model == "llama3-70b-8192"

    def test_custom_model(self) -> None:
        client = GroqClient(api_key="test-key", model="mixtral-8x7b-32768")
        assert client._model == "mixtral-8x7b-32768"

    def test_repr_redacts_key(self) -> None:
        client = GroqClient(api_key="secret-123")
        assert "***" in repr(client)
        assert "secret-123" not in repr(client)

    def test_tls_required(self) -> None:
        client = GroqClient(api_key="test-key")
        assert client._require_tls is True

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with GroqClient(api_key="test-key") as client:
            assert client._model == "llama3-70b-8192"
