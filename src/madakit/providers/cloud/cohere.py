"""Cohere AI client for mada-modelkit.

Cohere API has a custom format (not OpenAI-compatible).
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from madakit._errors import ProviderError
from madakit._types import AgentRequest, AgentResponse, StreamChunk
from madakit.providers._http_base import HttpAgentClient

__all__ = ["CohereClient"]


class CohereClient(HttpAgentClient):
    """Client for Cohere AI API.

    Cohere uses a custom chat format with message objects.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "command-r-plus",
        base_url: str = "https://api.cohere.ai/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Cohere client.

        Args:
            api_key: Cohere API key.
            model: Model name (default: command-r-plus).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            **kwargs,
        )
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"CohereClient(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build Cohere-format request payload.

        Args:
            request: The request to convert.

        Returns:
            Cohere API payload dict.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "message": request.prompt,
        }

        # Add optional parameters
        if request.max_tokens != 1024:
            payload["max_tokens"] = request.max_tokens

        if request.temperature != 0.7:
            payload["temperature"] = request.temperature

        if request.stop:
            payload["stop_sequences"] = request.stop

        # Cohere uses preamble for system prompt
        if request.system_prompt:
            payload["preamble"] = request.system_prompt

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse Cohere API response.

        Args:
            data: Response JSON from Cohere API.

        Returns:
            Parsed AgentResponse.
        """
        # Cohere response format
        content = data.get("text", "")

        # Token usage
        meta = data.get("meta", {})
        billed_units = meta.get("billed_units", {})
        input_tokens = billed_units.get("input_tokens", 0)
        output_tokens = billed_units.get("output_tokens", 0)

        return AgentResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _endpoint(self) -> str:
        """Return Cohere chat endpoint.

        Returns:
            API endpoint path.
        """
        return "/chat"

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request to Cohere API.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from Cohere API.

        Raises:
            ProviderError: If the request fails.
        """
        payload = self._build_payload(request)
        payload["stream"] = True

        try:
            async with self._client.stream(  # type: ignore[attr-defined]
                "POST",
                self._build_url(self._endpoint()),
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            ) as response:
                self._check_status(response)

                # Cohere streaming format: JSON lines with event-stream
                async for line in response.aiter_lines():
                    if not line or line.startswith("event:"):
                        continue

                    # Remove "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]

                    # Skip empty lines
                    if not line.strip():
                        continue

                    try:
                        chunk_data = self._json.loads(line)

                        # Check for stream end
                        if chunk_data.get("event_type") == "stream-end":
                            # Final chunk with metadata
                            meta = chunk_data.get("response", {}).get("meta", {})
                            billed_units = meta.get("billed_units", {})

                            yield StreamChunk(
                                delta="",
                                is_final=True,
                                metadata={
                                    "model": self._model,
                                    "input_tokens": billed_units.get("input_tokens", 0),
                                    "output_tokens": billed_units.get(
                                        "output_tokens", 0
                                    ),
                                },
                            )
                            break

                        # Regular text chunk
                        if chunk_data.get("event_type") == "text-generation":
                            text = chunk_data.get("text", "")
                            if text:
                                yield StreamChunk(
                                    delta=text,
                                    is_final=False,
                                )

                    except self._json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(f"Cohere streaming request failed: {e}") from e
