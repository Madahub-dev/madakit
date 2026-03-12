"""Replicate AI client for mada-modelkit.

Replicate API has a custom prediction-based format.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.providers._http_base import HttpAgentClient

__all__ = ["ReplicateClient"]


class ReplicateClient(HttpAgentClient):
    """Client for Replicate AI API.

    Replicate uses a prediction-based format with model versions.
    """

    _require_tls: bool = True

    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-2-70b-chat",
        base_url: str = "https://api.replicate.com/v1",
        **kwargs: Any,
    ) -> None:
        """Initialize Replicate client.

        Args:
            api_key: Replicate API key.
            model: Model identifier (default: meta/llama-2-70b-chat).
            base_url: API base URL.
            **kwargs: Additional arguments for HttpAgentClient.
        """
        super().__init__(
            base_url=base_url,
            headers={
                "Authorization": f"Token {api_key}",
            },
            **kwargs,
        )
        self._model = model
        self._api_key = api_key

    def __repr__(self) -> str:
        """Return string representation with redacted API key."""
        return f"ReplicateClient(model={self._model!r}, api_key='***')"

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Build Replicate-format request payload.

        Args:
            request: The request to convert.

        Returns:
            Replicate API payload dict.
        """
        # Build input parameters
        input_params: dict[str, Any] = {
            "prompt": request.prompt,
        }

        # Add optional parameters
        if request.max_tokens != 1024:
            input_params["max_tokens"] = request.max_tokens

        if request.temperature != 0.7:
            input_params["temperature"] = request.temperature

        if request.stop:
            input_params["stop_sequences"] = request.stop

        # Add system prompt if present
        if request.system_prompt:
            input_params["system_prompt"] = request.system_prompt

        payload: dict[str, Any] = {
            "version": self._model,
            "input": input_params,
        }

        return payload

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Parse Replicate API response.

        Args:
            data: Response JSON from Replicate API.

        Returns:
            Parsed AgentResponse.
        """
        # Replicate returns output as list of strings or single string
        output = data.get("output", "")
        if isinstance(output, list):
            content = "".join(output)
        else:
            content = str(output)

        # Replicate includes metrics in response
        metrics = data.get("metrics", {})

        # Token usage may not always be available
        input_tokens = metrics.get("input_tokens", 0)
        output_tokens = metrics.get("output_tokens", 0)

        return AgentResponse(
            content=content,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _endpoint(self) -> str:
        """Return Replicate predictions endpoint.

        Returns:
            API endpoint path.
        """
        return "/predictions"

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request to Replicate API.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from Replicate API.

        Raises:
            ProviderError: If the request fails.
        """
        payload = self._build_payload(request)
        payload["stream"] = True

        try:
            async with self._client.stream(
                "POST",
                self._build_url(self._endpoint()),
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            ) as response:
                self._check_status(response)

                # Replicate streaming format: server-sent events
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

                        # Check for completion
                        if chunk_data.get("status") == "succeeded":
                            # Final chunk with output
                            output = chunk_data.get("output", "")
                            if isinstance(output, list):
                                text = "".join(output)
                            else:
                                text = str(output)

                            metrics = chunk_data.get("metrics", {})

                            yield StreamChunk(
                                delta=text,
                                is_final=True,
                                metadata={
                                    "model": self._model,
                                    "input_tokens": metrics.get("input_tokens", 0),
                                    "output_tokens": metrics.get("output_tokens", 0),
                                },
                            )
                            break

                        # Regular output chunk
                        if "output" in chunk_data:
                            output = chunk_data["output"]
                            if isinstance(output, list):
                                text = "".join(output)
                            else:
                                text = str(output)

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
            raise ProviderError(f"Replicate streaming request failed: {e}") from e
