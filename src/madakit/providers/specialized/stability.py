"""Stability AI client for mada-modelkit.

Stability AI provides image generation capabilities.
"""

from __future__ import annotations

from typing import Any

from madakit._types import AgentRequest, AgentResponse
from madakit.providers._http_base import HttpAgentClient

__all__ = ["StabilityAIClient"]


class StabilityAIClient(HttpAgentClient):
    """Client for Stability AI image generation.

    Uses custom format for image generation requests.
    Response content contains the generated image artifact.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "stable-diffusion-xl-1024-v1-0",
        base_url: str = "https://api.stability.ai/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Stability AI client.

        Args:
            api_key: Stability AI API key.
            model: Model/engine ID (default: stable-diffusion-xl-1024-v1-0).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            **kwargs,
        )
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"StabilityAIClient(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build Stability AI request payload.

        Args:
            request: The request to convert.

        Returns:
            Stability AI API payload dict.
        """
        # Build text prompts array
        text_prompts = [{"text": request.prompt, "weight": 1.0}]

        # Add negative prompt if system_prompt is used
        if request.system_prompt:
            text_prompts.append({"text": request.system_prompt, "weight": -1.0})

        payload: dict[str, Any] = {
            "text_prompts": text_prompts,
        }

        # Add generation parameters
        if request.max_tokens != 1024:
            payload["steps"] = min(request.max_tokens, 150)

        # Map temperature to cfg_scale (guidance strength)
        if request.temperature != 0.7:
            # Scale temperature [0,2] to cfg_scale [0,35]
            payload["cfg_scale"] = min(request.temperature * 17.5, 35)

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse Stability AI response.

        Args:
            data: Response JSON from Stability AI.

        Returns:
            Parsed AgentResponse with image artifact.
        """
        # Stability AI returns artifacts array
        artifacts = data.get("artifacts", [])
        if artifacts:
            # Get first artifact
            artifact = artifacts[0]
            # Base64 image data or URL
            content = artifact.get("base64", artifact.get("url", ""))
            finish_reason = artifact.get("finishReason", "SUCCESS")
        else:
            content = ""
            finish_reason = "ERROR"

        return AgentResponse(
            content=content,
            model=self._model,
            input_tokens=0,  # Image generation doesn't use token counting
            output_tokens=0,
            metadata={
                "finish_reason": finish_reason,
                "artifact_type": "image",
            },
        )

    def _endpoint(self) -> str:
        """Return Stability AI generation endpoint.

        Returns:
            API endpoint path with engine ID.
        """
        return f"/generation/{self._model}/text-to-image"
