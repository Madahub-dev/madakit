"""OpenAI cloud provider for mada-modelkit.

Provides OpenAIClient — an HttpAgentClient + OpenAICompatMixin subclass that
targets the OpenAI chat-completions API. Requires the ``httpx`` optional extra.
TLS is enforced; the API key is passed as a Bearer token and redacted in repr.
"""

from __future__ import annotations

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin

__all__ = ["OpenAIClient"]


class OpenAIClient(OpenAICompatMixin, HttpAgentClient):
    """HTTP client for the OpenAI chat-completions API.

    Combines ``OpenAICompatMixin`` for OpenAI wire-format handling with
    ``HttpAgentClient`` for the shared async HTTP pipeline. TLS is always
    enforced; pass ``connect_timeout``, ``read_timeout``, or
    ``max_concurrent`` as keyword arguments to tune behaviour.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        **kwargs: object,
    ) -> None:
        """Initialise the OpenAI client.

        Args:
            api_key: OpenAI API key; placed in ``Authorization: Bearer`` header.
            model: Chat model identifier. Defaults to ``"gpt-4o-mini"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._api_key = api_key
        super().__init__(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted."""
        return f"OpenAIClient(model={self._model!r}, api_key=***)"
