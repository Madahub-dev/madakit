"""ElevenLabs client for mada-modelkit.

ElevenLabs provides text-to-speech capabilities.
"""

from __future__ import annotations

from typing import Any

from madakit._types import AgentRequest, AgentResponse
from madakit.providers._http_base import HttpAgentClient

__all__ = ["ElevenLabsClient"]


class ElevenLabsClient(HttpAgentClient):
    """Client for ElevenLabs text-to-speech.

    Uses custom format for TTS requests.
    Response content contains audio data or URL.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_monolingual_v1",
        base_url: str = "https://api.elevenlabs.io/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize ElevenLabs client.

        Args:
            api_key: ElevenLabs API key.
            voice_id: Voice ID to use (default: Rachel).
            model: TTS model ID (default: eleven_monolingual_v1).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={
                "xi-api-key": api_key,
            },
            **kwargs,
        )
        self._voice_id = voice_id
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"ElevenLabsClient(voice_id={self._voice_id!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build ElevenLabs TTS payload.

        Args:
            request: The request to convert.

        Returns:
            ElevenLabs API payload dict.
        """
        payload: dict[str, Any] = {
            "text": request.prompt,
            "model_id": self._model,
        }

        # Voice settings from temperature (map to stability/similarity)
        if request.temperature != 0.7:
            # Higher temperature = more variation
            stability = max(0.0, 1.0 - request.temperature)
            payload["voice_settings"] = {
                "stability": stability,
                "similarity_boost": 0.75,
            }

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse ElevenLabs response.

        Args:
            data: Response JSON from ElevenLabs.

        Returns:
            Parsed AgentResponse with audio content.
        """
        # ElevenLabs may return audio URL or binary data
        # For binary responses, this would need special handling
        audio_url = data.get("audio_url", "")
        audio_base64 = data.get("audio_base64", "")
        content = audio_url or audio_base64

        return AgentResponse(
            content=content,
            model=self._model,
            input_tokens=0,  # TTS doesn't use token counting
            output_tokens=0,
            metadata={
                "voice_id": self._voice_id,
                "artifact_type": "audio",
            },
        )

    def _endpoint(self) -> str:
        """Return ElevenLabs TTS endpoint.

        Returns:
            API endpoint path with voice ID.
        """
        return f"/text-to-speech/{self._voice_id}"
