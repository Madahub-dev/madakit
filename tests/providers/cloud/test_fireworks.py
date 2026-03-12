"""Tests for Fireworks AI client."""

from __future__ import annotations

import pytest
from mada_modelkit.providers.cloud.fireworks import FireworksClient


class TestFireworksClient:
    """Test Fireworks AI client."""

    def test_constructor(self) -> None:
        client = FireworksClient(api_key="test-key")
        assert client._model == "accounts/fireworks/models/llama-v3p1-70b-instruct"

    def test_custom_model(self) -> None:
        client = FireworksClient(api_key="test-key", model="custom-model")
        assert client._model == "custom-model"

    def test_repr_redacts_key(self) -> None:
        client = FireworksClient(api_key="secret-123")
        assert "***" in repr(client)
        assert "secret-123" not in repr(client)

    def test_tls_required(self) -> None:
        client = FireworksClient(api_key="test-key")
        assert client._require_tls is True

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with FireworksClient(api_key="test-key") as client:
            assert client._model == "accounts/fireworks/models/llama-v3p1-70b-instruct"
