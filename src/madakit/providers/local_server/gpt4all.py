"""GPT4All client for mada-modelkit.

GPT4All API is OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin

__all__ = ["GPT4AllClient"]


class GPT4AllClient(OpenAICompatMixin, HttpAgentClient):
    """Client for GPT4All local server.

    GPT4All uses OpenAI-compatible format for requests and responses.
    """

    _require_tls: bool = False

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:4891/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize GPT4All client.

        Args:
            model: Model name (required).
            base_url: API base URL (default: http://localhost:4891/v1).
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
        return f"GPT4AllClient(model={self._model!r}, base_url={self._base_url!r})"
