"""vLLM local server provider for mada-modelkit.

Provides VllmClient — an OpenAICompatMixin + HttpAgentClient subclass that
targets a locally-running vLLM server. No API key required. TLS is not
enforced (local server). Uses the OpenAI-compatible wire format via
OpenAICompatMixin. Uses the default HttpAgentClient health_check (``GET /``)
and the default send_request_stream from BaseAgentClient.
"""

from __future__ import annotations

from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin

__all__ = ["VllmClient"]


class VllmClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for a locally-running vLLM server.

    Thin subclass of ``OpenAICompatMixin`` + ``HttpAgentClient``. Targets the
    vLLM OpenAI-compatible endpoint at ``http://localhost:8000/v1`` by default.
    No API key required; TLS is not enforced. All payload building, response
    parsing, endpoint routing, health check, and streaming are inherited.
    Unlike Ollama, ``model`` has no default — vLLM loads a single model at
    server startup and the caller must name it explicitly.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        **kwargs: object,
    ) -> None:
        """Initialise the vLLM client.

        Args:
            model: Model identifier served by the vLLM instance (required).
            base_url: Base URL of the vLLM server. Defaults to
                ``"http://localhost:8000/v1"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._base_url = base_url
        super().__init__(base_url=base_url, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        """Return a repr showing the model identifier and server base URL."""
        return f"VllmClient(model={self._model!r}, base_url={self._base_url!r})"
