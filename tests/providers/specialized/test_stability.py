"""Tests for Stability AI client.

Covers Stability AI image generation with custom format.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest
from madakit.providers.specialized.stability import StabilityAIClient


class TestModuleExports:
    """Verify stability module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.providers.specialized import stability

        assert hasattr(stability, "__all__")

    def test_all_contains_stability_client(self) -> None:
        """__all__ contains StabilityAIClient."""
        from madakit.providers.specialized.stability import __all__

        assert "StabilityAIClient" in __all__


class TestStabilityAIClientConstructor:
    """Test StabilityAIClient constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with default model."""
        client = StabilityAIClient(api_key="test-key")

        assert client._model == "stable-diffusion-xl-1024-v1-0"
        assert client._api_key == "test-key"

    def test_custom_model(self) -> None:
        """Constructor accepts custom model."""
        client = StabilityAIClient(api_key="test-key", model="stable-diffusion-v1-5")

        assert client._model == "stable-diffusion-v1-5"

    def test_custom_base_url(self) -> None:
        """Constructor accepts custom base URL."""
        client = StabilityAIClient(
            api_key="test-key", base_url="https://custom.stability.ai/v1"
        )

        assert client._model == "stable-diffusion-xl-1024-v1-0"

    def test_repr_redacts_api_key(self) -> None:
        """__repr__ redacts API key."""
        client = StabilityAIClient(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "stable-diffusion-xl-1024-v1-0" in repr_str


class TestStabilityFormat:
    """Test Stability AI custom format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates Stability format."""
        client = StabilityAIClient(api_key="test-key")
        request = AgentRequest(prompt="A beautiful sunset")

        payload = client._build_payload(request)

        # Stability AI format with text_prompts array
        assert "text_prompts" in payload
        assert payload["text_prompts"][0]["text"] == "A beautiful sunset"
        assert payload["text_prompts"][0]["weight"] == 1.0

    def test_build_payload_with_negative_prompt(self) -> None:
        """_build_payload uses system_prompt as negative prompt."""
        client = StabilityAIClient(api_key="test-key")
        request = AgentRequest(
            prompt="A beautiful sunset", system_prompt="blurry, low quality"
        )

        payload = client._build_payload(request)

        assert len(payload["text_prompts"]) == 2
        assert payload["text_prompts"][0]["weight"] == 1.0
        assert payload["text_prompts"][1]["text"] == "blurry, low quality"
        assert payload["text_prompts"][1]["weight"] == -1.0

    def test_build_payload_with_steps(self) -> None:
        """_build_payload maps max_tokens to steps."""
        client = StabilityAIClient(api_key="test-key")
        request = AgentRequest(prompt="A sunset", max_tokens=50)

        payload = client._build_payload(request)

        assert payload["steps"] == 50

    def test_build_payload_caps_steps(self) -> None:
        """_build_payload caps steps at 150."""
        client = StabilityAIClient(api_key="test-key")
        request = AgentRequest(prompt="A sunset", max_tokens=500)

        payload = client._build_payload(request)

        assert payload["steps"] == 150

    def test_build_payload_with_cfg_scale(self) -> None:
        """_build_payload maps temperature to cfg_scale."""
        client = StabilityAIClient(api_key="test-key")
        request = AgentRequest(prompt="A sunset", temperature=1.5)

        payload = client._build_payload(request)

        # 1.5 * 17.5 = 26.25
        assert "cfg_scale" in payload
        assert payload["cfg_scale"] == 26.25

    def test_parse_response_with_base64(self) -> None:
        """_parse_response extracts base64 image."""
        client = StabilityAIClient(api_key="test-key")

        data = {
            "artifacts": [
                {
                    "base64": "iVBORw0KGgoAAAANS...",
                    "finishReason": "SUCCESS",
                }
            ]
        }

        response = client._parse_response(data)

        assert response.content == "iVBORw0KGgoAAAANS..."
        assert response.metadata["finish_reason"] == "SUCCESS"
        assert response.metadata["artifact_type"] == "image"

    def test_parse_response_with_url(self) -> None:
        """_parse_response extracts URL if no base64."""
        client = StabilityAIClient(api_key="test-key")

        data = {
            "artifacts": [
                {
                    "url": "https://cdn.stability.ai/image.png",
                    "finishReason": "SUCCESS",
                }
            ]
        }

        response = client._parse_response(data)

        assert response.content == "https://cdn.stability.ai/image.png"

    def test_parse_response_empty_artifacts(self) -> None:
        """_parse_response handles empty artifacts."""
        client = StabilityAIClient(api_key="test-key")

        data = {"artifacts": []}

        response = client._parse_response(data)

        assert response.content == ""
        assert response.metadata["finish_reason"] == "ERROR"

    def test_endpoint_includes_model(self) -> None:
        """_endpoint includes engine/model ID."""
        client = StabilityAIClient(api_key="test-key", model="stable-diffusion-v1-5")

        endpoint = client._endpoint()

        assert "stable-diffusion-v1-5" in endpoint
        assert "/text-to-image" in endpoint


class TestStabilityAIClientIntegration:
    """Integration tests for StabilityAIClient."""

    def test_client_initialization(self) -> None:
        """Client initializes with TLS requirement."""
        client = StabilityAIClient(
            api_key="test-key", model="stable-diffusion-v1-5"
        )

        assert client._model == "stable-diffusion-v1-5"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = StabilityAIClient(api_key="key1", model="model-a")
        client2 = StabilityAIClient(api_key="key2", model="model-b")

        assert client1._model != client2._model
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with StabilityAIClient(api_key="test-key") as client:
            assert client._model == "stable-diffusion-xl-1024-v1-0"
