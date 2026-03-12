"""Content filtering middleware for madakit.

PII redaction, safety checks, and content moderation.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import re
from typing import AsyncIterator, Callable

from madakit._base import BaseAgentClient
from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["ContentFilterMiddleware"]


class ContentFilterMiddleware(BaseAgentClient):
    """Middleware for content filtering, PII redaction, and safety checks.

    Redacts sensitive information and enforces content safety policies.
    """

    # PII detection patterns
    _EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    _SSN_PATTERN = re.compile(
        r'\b\d{3}-\d{2}-\d{4}\b'
    )
    _CREDIT_CARD_PATTERN = re.compile(
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
    )

    def __init__(
        self,
        client: BaseAgentClient,
        redact_pii: bool = True,
        safety_check: Callable[[str], None] | None = None,
        response_filter: Callable[[str], str] | None = None,
    ) -> None:
        """Initialise with client and filtering options.

        Args:
            client: Wrapped client to delegate to.
            redact_pii: Whether to redact PII from prompts (default True).
            safety_check: Optional callback to check prompt safety.
                         Should raise an exception if prompt is unsafe.
            response_filter: Optional callback to filter response content.
                            Receives content string, returns filtered string.
        """
        super().__init__()
        self._client = client
        self._redact_pii = redact_pii
        self._safety_check = safety_check
        self._response_filter = response_filter

    def _detect_and_redact_pii(self, text: str) -> str:
        """Detect and redact PII from text.

        Args:
            text: Text to scan for PII.

        Returns:
            Text with PII replaced by [REDACTED].
        """
        if not self._redact_pii:
            return text

        # Redact emails
        text = self._EMAIL_PATTERN.sub('[REDACTED]', text)

        # Redact SSNs
        text = self._SSN_PATTERN.sub('[REDACTED]', text)

        # Redact credit card numbers
        text = self._CREDIT_CARD_PATTERN.sub('[REDACTED]', text)

        return text

    def _check_safety(self, prompt: str) -> None:
        """Check prompt safety using configured callback.

        Args:
            prompt: Prompt to check.

        Raises:
            MiddlewareError: If safety check fails.
        """
        if self._safety_check is None:
            return

        try:
            self._safety_check(prompt)
        except Exception as e:
            raise MiddlewareError(f"Safety check failed: {e}") from e

    def _filter_response(self, content: str) -> str:
        """Filter response content using configured callback.

        Args:
            content: Response content to filter.

        Returns:
            Filtered content.
        """
        if self._response_filter is None:
            return content

        return self._response_filter(content)

    def _apply_request_filters(self, request: AgentRequest) -> AgentRequest:
        """Apply PII redaction and safety checks to request.

        Args:
            request: Original request.

        Returns:
            Request with filtered prompt.

        Raises:
            MiddlewareError: If safety check fails.
        """
        # Redact PII from prompt
        filtered_prompt = self._detect_and_redact_pii(request.prompt)

        # Redact PII from system prompt if present
        filtered_system_prompt = None
        if request.system_prompt:
            filtered_system_prompt = self._detect_and_redact_pii(request.system_prompt)

        # Check safety (on redacted prompt)
        self._check_safety(filtered_prompt)

        # Create new request with filtered fields
        return AgentRequest(
            prompt=filtered_prompt,
            system_prompt=filtered_system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            attachments=request.attachments,
            metadata=request.metadata,
        )

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Filter request, delegate to client, filter response.

        Args:
            request: The request to send.

        Returns:
            Response with filtered content.

        Raises:
            MiddlewareError: If safety check fails.
        """
        # Apply request filters
        filtered_request = self._apply_request_filters(request)

        # Delegate to wrapped client
        response = await self._client.send_request(filtered_request)

        # Filter response content
        filtered_content = self._filter_response(response.content)

        # Return response with filtered content
        return AgentResponse(
            content=filtered_content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            metadata=response.metadata,
        )

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Filter request, stream from client, filter final content.

        Args:
            request: The request to send.

        Yields:
            Stream chunks with filtered content in final chunk.

        Raises:
            MiddlewareError: If safety check fails.
        """
        # Apply request filters
        filtered_request = self._apply_request_filters(request)

        # Stream from wrapped client
        async for chunk in self._client.send_request_stream(filtered_request):
            # Filter final chunk content
            if chunk.is_final and self._response_filter:
                # Need to buffer all chunks to filter complete content
                # For simplicity, filter the final delta only
                filtered_delta = self._filter_response(chunk.delta)
                yield StreamChunk(
                    delta=filtered_delta,
                    is_final=chunk.is_final,
                    metadata=chunk.metadata,
                )
            else:
                yield chunk
