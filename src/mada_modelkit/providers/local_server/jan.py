"""Jan client for mada-modelkit.

Jan API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["JanClient"]


class JanClient(OpenAICompatMixin, HttpAgentClient):
    """Client for Jan local server.

    Jan uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = False

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1337/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Jan client.

        Args:
            model: Model name (required).
            base_url: API base URL (default: http://localhost:1337/v1).
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
        return f"JanClient(model={self._model!r}, base_url={self._base_url!r})"
