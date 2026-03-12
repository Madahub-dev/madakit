"""DeepSeek cloud provider for mada-modelkit.

Provides DeepSeekClient — a thin OpenAICompatMixin + HttpAgentClient subclass
that targets the DeepSeek chat-completions API. Requires the ``httpx`` optional
extra. TLS is enforced; the API key is passed as a Bearer token and redacted in
repr. Uses the OpenAI-compatible wire format via OpenAICompatMixin.
"""

from __future__ import annotations

from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin

__all__ = ["DeepSeekClient"]


class DeepSeekClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for the DeepSeek chat-completions API.

    Thin subclass of ``OpenAICompatMixin`` + ``HttpAgentClient``. DeepSeek
    speaks the OpenAI-compatible wire format, so all payload building and
    response parsing is inherited from the mixin. TLS is always enforced; pass
    ``connect_timeout``, ``read_timeout``, or ``max_concurrent`` as keyword
    arguments to tune behaviour.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        **kwargs: object,
    ) -> None:
        """Initialise the DeepSeek client.

        Args:
            api_key: DeepSeek API key; placed in ``Authorization: Bearer`` header.
            model: Chat model identifier. Defaults to ``"deepseek-chat"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._api_key = api_key
        super().__init__(
            base_url="https://api.deepseek.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted."""
        return f"DeepSeekClient(model={self._model!r}, api_key=***)"
