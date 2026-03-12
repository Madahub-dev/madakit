"""Mistral AI client for mada-modelkit.

Mistral API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["MistralClient"]


class MistralClient(OpenAICompatMixin, HttpAgentClient):
    """Client for Mistral AI API.

    Mistral uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-medium",
        base_url: str = "https://api.mistral.ai/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Mistral client.

        Args:
            api_key: Mistral API key.
            model: Model name (default: mistral-medium).
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
        return f"MistralClient(model={self._model!r}, api_key='***')"
