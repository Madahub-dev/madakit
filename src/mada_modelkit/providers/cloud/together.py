"""Together AI client for mada-modelkit.

Together API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["TogetherClient"]


class TogetherClient(OpenAICompatMixin, HttpAgentClient):
    """Client for Together AI API.

    Together uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        base_url: str = "https://api.together.xyz/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Together client.

        Args:
            api_key: Together API key.
            model: Model name (default: Mixtral-8x7B).
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
        return f"TogetherClient(model={self._model!r}, api_key='***')"
