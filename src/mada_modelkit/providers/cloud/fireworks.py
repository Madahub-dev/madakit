"""Fireworks AI client for mada-modelkit.

Fireworks API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["FireworksClient"]


class FireworksClient(OpenAICompatMixin, HttpAgentClient):
    """Client for Fireworks AI API.

    Fireworks uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Fireworks client.

        Args:
            api_key: Fireworks API key.
            model: Model name (default: llama-v3p1-70b-instruct).
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
        return f"FireworksClient(model={self._model!r}, api_key='***')"
