"""Groq client for mada-modelkit.

Groq API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["GroqClient"]


class GroqClient(OpenAICompatMixin, HttpAgentClient):
    """Client for Groq API.

    Groq uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "llama3-70b-8192",
        base_url: str = "https://api.groq.com/openai/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Groq client.

        Args:
            api_key: Groq API key.
            model: Model name (default: llama3-70b-8192).
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
        return f"GroqClient(model={self._model!r}, api_key='***')"
