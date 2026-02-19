"""OpenAI chat-completions format mixin for mada-modelkit providers.

Provides OpenAICompatMixin — concrete implementations of ``_build_payload``,
``_parse_response``, and ``_endpoint`` for any provider that speaks the OpenAI
chat-completions wire format (OpenAI, DeepSeek, Ollama, vLLM, LocalAI).
Requires no additional dependencies beyond ``httpx``.
"""

from __future__ import annotations

import base64
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

        When ``request.attachments`` is non-empty, the user message
        ``content`` becomes a list of OpenAI content blocks: one
        ``image_url`` block per attachment (base64 data URI) followed by a
        ``text`` block for the prompt.  Without attachments ``content`` is a
        plain string.
        """
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.attachments:
            user_content: str | list[dict[str, Any]] = [
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{att.media_type};base64,"
                                f"{base64.b64encode(att.content).decode('ascii')}"
                            ),
                        },
                    }
                    for att in request.attachments
                ],
                {"type": "text", "text": request.prompt},
            ]
        else:
            user_content = request.prompt

        messages.append({"role": "user", "content": user_content})
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
        """Parse an OpenAI-compatible chat-completions JSON response.

        Extracts ``choices[0].message.content`` as the response text,
        ``usage.prompt_tokens`` and ``usage.completion_tokens`` as token
        counts (defaulting to 0 when absent), and ``data["model"]`` as the
        model name (falling back to ``self._model`` when absent).
        """
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return AgentResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _endpoint(self) -> str:
        """Return the OpenAI chat-completions endpoint path."""
        return "/chat/completions"
