"""LocalAI local server provider for mada-modelkit.

Provides LocalAIClient — an OpenAICompatMixin + HttpAgentClient subclass that
targets a locally-running LocalAI server. No API key required. TLS is not
enforced (local server). Uses the OpenAI-compatible wire format via
OpenAICompatMixin. All behaviour — payload building, response parsing, endpoint
routing, health check, and streaming — is fully inherited.
"""

from __future__ import annotations

from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin

__all__ = ["LocalAIClient"]


class LocalAIClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for a locally-running LocalAI server.

    Thin subclass of ``OpenAICompatMixin`` + ``HttpAgentClient``. Targets the
    LocalAI OpenAI-compatible endpoint at ``http://localhost:8080/v1`` by
    default. No API key required; TLS is not enforced. All payload building,
    response parsing, endpoint routing, health check, and streaming are
    inherited. ``model`` is required — LocalAI serves whichever model the
    server was started with and the caller must name it explicitly.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8080/v1",
        **kwargs: object,
    ) -> None:
        """Initialise the LocalAI client.

        Args:
            model: Model identifier served by the LocalAI instance (required).
            base_url: Base URL of the LocalAI server. Defaults to
                ``"http://localhost:8080/v1"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._base_url = base_url
        super().__init__(base_url=base_url, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Return a repr showing the model identifier and server base URL."""
        return f"LocalAIClient(model={self._model!r}, base_url={self._base_url!r})"
