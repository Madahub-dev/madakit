"""LM Studio client for mada-modelkit.

LM Studio API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["LMStudioClient"]


class LMStudioClient(OpenAICompatMixin, HttpAgentClient):
    """Client for LM Studio local server.

    LM Studio uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = False

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize LM Studio client.

        Args:
            model: Model name (required).
            base_url: API base URL (default: http://localhost:1234/v1).
            **kwargs: Additional arguments for HttpAgentClient.
        """
        self._model = model
        self._base_url = base_url
        super().__init__(
            base_url=base_url,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"LMStudioClient(model={self._model!r}, base_url={self._base_url!r})"
