"""OpenAI chat-completions format mixin for mada-modelkit providers.

Provides OpenAICompatMixin — concrete implementations of ``_build_payload``,
``_parse_response``, and ``_endpoint`` for any provider that speaks the OpenAI
chat-completions wire format (OpenAI, DeepSeek, Ollama, vLLM, LocalAI).
Requires no additional dependencies beyond ``httpx``.
"""

from __future__ import annotations

from typing import Any

from mada_modelkit._types import AgentRequest, AgentResponse

__all__ = ["OpenAICompatMixin"]


class OpenAICompatMixin:
    """Mixin that implements the OpenAI chat-completions wire format.

    Concrete providers combine this mixin with ``HttpAgentClient`` via multiple
    inheritance, e.g. ``class OpenAIClient(OpenAICompatMixin, HttpAgentClient)``.
    The provider subclass must set ``_model`` before calling any method.
    """

    _model: str

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build an OpenAI-compatible chat-completions request body.

        Constructs a ``messages`` list with an optional system message followed
        by the user message.  Includes ``model``, ``max_tokens``,
        ``temperature``, and ``stop`` (only when provided).
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            payload["stop"] = request.stop
        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse an OpenAI-compatible chat-completions response (stub; task 3.2.2)."""
        raise NotImplementedError

    def _endpoint(self) -> str:
        """Return the chat-completions endpoint path (stub; task 3.2.3)."""
        raise NotImplementedError
