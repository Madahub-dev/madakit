"""Tests for ElevenLabs client.

Covers ElevenLabs TTS with custom format.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest
from mada_modelkit.providers.specialized.elevenlabs import ElevenLabsClient


class TestModuleExports:
    """Verify elevenlabs module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from mada_modelkit.providers.specialized import elevenlabs

        assert hasattr(elevenlabs, "__all__")

    def test_all_contains_elevenlabs_client(self) -> None:
        """__all__ contains ElevenLabsClient."""
        from mada_modelkit.providers.specialized.elevenlabs import __all__

        assert "ElevenLabsClient" in __all__


class TestElevenLabsClientConstructor:
    """Test ElevenLabsClient constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with defaults."""
        client = ElevenLabsClient(api_key="test-key")

        assert client._voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert client._model == "eleven_monolingual_v1"
        assert client._api_key == "test-key"

    def test_custom_voice(self) -> None:
        """Constructor accepts custom voice ID."""
        client = ElevenLabsClient(api_key="test-key", voice_id="custom-voice-123")

        assert client._voice_id == "custom-voice-123"

    def test_custom_model(self) -> None:
        """Constructor accepts custom model."""
        client = ElevenLabsClient(
            api_key="test-key", model="eleven_multilingual_v2"
        )

        assert client._model == "eleven_multilingual_v2"

    def test_repr_redacts_api_key(self) -> None:
        """__repr__ redacts API key."""
        client = ElevenLabsClient(api_key="secret-key-12345")

        repr_str = repr(client)

        assert "***" in repr_str
        assert "secret-key-12345" not in repr_str
        assert "21m00Tcm4TlvDq8ikWAM" in repr_str


class TestElevenLabsFormat:
    """Test ElevenLabs custom format."""

    def test_build_payload_format(self) -> None:
        """_build_payload creates ElevenLabs format."""
        client = ElevenLabsClient(api_key="test-key")
        request = AgentRequest(prompt="Hello, world!")

        payload = client._build_payload(request)

        assert payload["text"] == "Hello, world!"
        assert payload["model_id"] == "eleven_monolingual_v1"

    def test_build_payload_with_voice_settings(self) -> None:
        """_build_payload includes voice settings from temperature."""
        client = ElevenLabsClient(api_key="test-key")
        request = AgentRequest(prompt="Hello", temperature=1.5)

        payload = client._build_payload(request)

        assert "voice_settings" in payload
        # stability = 1.0 - 1.5 = -0.5, but max(0.0, -0.5) = 0.0
        assert payload["voice_settings"]["stability"] == 0.0
        assert payload["voice_settings"]["similarity_boost"] == 0.75

    def test_build_payload_stability_mapping(self) -> None:
        """_build_payload maps temperature to stability correctly."""
        client = ElevenLabsClient(api_key="test-key")
        request = AgentRequest(prompt="Hello", temperature=0.3)

        payload = client._build_payload(request)

        # stability = 1.0 - 0.3 = 0.7
        assert payload["voice_settings"]["stability"] == 0.7

    def test_parse_response_with_url(self) -> None:
        """_parse_response extracts audio URL."""
        client = ElevenLabsClient(api_key="test-key")

        data = {"audio_url": "https://api.elevenlabs.io/audio/123.mp3"}

        response = client._parse_response(data)

        assert response.content == "https://api.elevenlabs.io/audio/123.mp3"
        assert response.metadata["artifact_type"] == "audio"
        assert response.metadata["voice_id"] == "21m00Tcm4TlvDq8ikWAM"

    def test_parse_response_with_base64(self) -> None:
        """_parse_response extracts base64 audio."""
        client = ElevenLabsClient(api_key="test-key")

        data = {"audio_base64": "UklGRiQAAABXQVZF..."}

        response = client._parse_response(data)

        assert response.content == "UklGRiQAAABXQVZF..."

    def test_parse_response_prefers_url(self) -> None:
        """_parse_response prefers URL over base64."""
        client = ElevenLabsClient(api_key="test-key")

        data = {
            "audio_url": "https://api.elevenlabs.io/audio/123.mp3",
            "audio_base64": "UklGRiQAAABXQVZF...",
        }

        response = client._parse_response(data)

        assert response.content == "https://api.elevenlabs.io/audio/123.mp3"

    def test_parse_response_empty_data(self) -> None:
        """_parse_response handles empty response."""
        client = ElevenLabsClient(api_key="test-key")

        data = {}

        response = client._parse_response(data)

        assert response.content == ""

    def test_endpoint_includes_voice_id(self) -> None:
        """_endpoint includes voice ID."""
        client = ElevenLabsClient(api_key="test-key", voice_id="custom-voice-123")

        endpoint = client._endpoint()

        assert "custom-voice-123" in endpoint
        assert "/text-to-speech/" in endpoint


class TestElevenLabsClientIntegration:
    """Integration tests for ElevenLabsClient."""

    def test_client_initialization(self) -> None:
        """Client initializes with TLS requirement."""
        client = ElevenLabsClient(api_key="test-key")

        assert client._voice_id == "21m00Tcm4TlvDq8ikWAM"
        assert client._require_tls is True

    def test_multiple_clients_independent(self) -> None:
        """Multiple clients maintain independent state."""
        client1 = ElevenLabsClient(api_key="key1", voice_id="voice1")
        client2 = ElevenLabsClient(api_key="key2", voice_id="voice2")

        assert client1._voice_id != client2._voice_id
        assert client1._api_key != client2._api_key

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as async context manager."""
        async with ElevenLabsClient(api_key="test-key") as client:
            assert client._voice_id == "21m00Tcm4TlvDq8ikWAM"
