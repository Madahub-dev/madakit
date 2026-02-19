"""Gemini cloud provider for mada-modelkit.

Provides GeminiClient — an HttpAgentClient subclass that targets the Google
Gemini generateContent API. Requires the ``httpx`` optional extra. TLS is
enforced; the API key is passed as an x-goog-api-key header and redacted in
repr. Does NOT use OpenAICompatMixin — Gemini uses its own wire format.
"""

from __future__ import annotations

import base64
from typing import Any

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient

__all__ = ["GeminiClient"]


class GeminiClient(HttpAgentClient):
    """HTTP client for the Google Gemini generateContent API.

    Subclasses ``HttpAgentClient`` directly (no OpenAICompatMixin) because
    Gemini uses its own request/response format. TLS is always enforced; pass
    ``connect_timeout``, ``read_timeout``, or ``max_concurrent`` as keyword
    arguments to tune behaviour.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        **kwargs: object,
    ) -> None:
        """Initialise the Gemini client.

        Args:
            api_key: Google AI API key; placed in ``x-goog-api-key`` header.
            model: Gemini model identifier. Defaults to ``"gemini-2.0-flash"``.
            **kwargs: Forwarded to ``HttpAgentClient`` (e.g. ``connect_timeout``,
                ``read_timeout``, ``max_concurrent``).
        """
        self._model = model
        self._api_key = api_key
        super().__init__(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            headers={"x-goog-api-key": api_key},
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        """Return a repr with the API key redacted."""
        return f"GeminiClient(model={self._model!r}, api_key=***)"

    def _endpoint(self) -> str:
        """Return the Gemini generateContent endpoint path for the current model."""
        return f"/models/{self._model}:generateContent"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build the Gemini generateContent request payload.

        Formats the request using Gemini's wire format:

        - ``contents`` holds a single user turn whose ``parts`` list contains
          one ``inlineData`` block per attachment (base64-encoded bytes +
          mimeType) followed by a ``text`` part for the prompt.
        - ``systemInstruction`` is a top-level field with a ``parts`` list;
          omitted when ``request.system_prompt`` is absent.
        - ``generationConfig`` carries ``maxOutputTokens``, ``temperature``,
          and ``stopSequences`` (omitted when ``request.stop`` is ``None``).
        - The model is encoded in the endpoint URL, not the payload body.
        """
        parts: list[dict[str, Any]] = [
            {
                "inlineData": {
                    "mimeType": att.media_type,
                    "data": base64.b64encode(att.content).decode("ascii"),
                }
            }
            for att in request.attachments
        ]
        parts.append({"text": request.prompt})

        generation_config: dict[str, Any] = {
            "maxOutputTokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.stop:
            generation_config["stopSequences"] = request.stop

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_prompt}]
            }
        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse the Gemini generateContent response. Implemented in task 4.3.3."""
        raise NotImplementedError
