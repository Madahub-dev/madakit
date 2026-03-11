"""Logging middleware for mada-modelkit.

Structured logging for requests, responses, and errors with correlation IDs.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["LoggingMiddleware"]


class LoggingMiddleware(BaseAgentClient):
    """Middleware that logs requests, responses, and errors with structured context."""

    def __init__(
        self,
        client: BaseAgentClient,
        logger: logging.Logger | None = None,
        log_level: str = "INFO",
        include_prompts: bool = False,
    ) -> None:
        """Initialise with a wrapped client and logging configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            logger: Optional logger instance. If None, creates a default logger.
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            include_prompts: Whether to include prompt text in logs (may contain PII).
        """
        super().__init__()
        self._client = client
        self._logger = logger or logging.getLogger(__name__)
        self._log_level = getattr(logging, log_level.upper())
        self._include_prompts = include_prompts

        # Set logger level to the configured level
        self._logger.setLevel(self._log_level)

    def _generate_request_id(self) -> str:
        """Generate a unique request ID for correlation."""
        return str(uuid.uuid4())

    def _get_or_generate_request_id(self, request: AgentRequest) -> str:
        """Get existing request ID from metadata or generate a new one.

        Args:
            request: The request that may contain an existing request ID.

        Returns:
            The request ID (existing or newly generated).
        """
        # Check if request already has an ID in metadata
        existing_id = request.metadata.get("request_id")
        if existing_id:
            return str(existing_id)
        return self._generate_request_id()

    def _propagate_request_id(self, request: AgentRequest, request_id: str) -> AgentRequest:
        """Create a new request with request_id in metadata for propagation.

        Args:
            request: The original request.
            request_id: The request ID to propagate.

        Returns:
            A new AgentRequest with request_id in metadata.
        """
        # Add request_id to metadata for downstream propagation
        new_metadata = {**request.metadata, "request_id": request_id}
        return AgentRequest(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            attachments=request.attachments,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
            metadata=new_metadata,
        )

    def _log_request_start(self, request_id: str, request: AgentRequest) -> None:
        """Log the start of a request with ID and metadata.

        Args:
            request_id: Unique correlation ID for this request.
            request: The request being sent.
        """
        log_data = {
            "event": "request_start",
            "request_id": request_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "metadata": request.metadata,
        }

        if self._include_prompts:
            log_data["prompt"] = request.prompt
            log_data["system_prompt"] = request.system_prompt

        self._logger.info("Request started", extra=log_data)

    def _log_response_completion(
        self, request_id: str, response: AgentResponse, duration_ms: float
    ) -> None:
        """Log the completion of a request with response details.

        Args:
            request_id: Correlation ID from the request.
            response: The response received.
            duration_ms: Request duration in milliseconds.
        """
        log_data = {
            "event": "request_complete",
            "request_id": request_id,
            "duration_ms": duration_ms,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }

        self._logger.info("Request completed", extra=log_data)

    def _log_error(self, request_id: str, exception: Exception, duration_ms: float) -> None:
        """Log an error that occurred during request processing.

        Args:
            request_id: Correlation ID from the request.
            exception: The exception that was raised.
            duration_ms: Time elapsed before error occurred.
        """
        log_data = {
            "event": "request_error",
            "request_id": request_id,
            "duration_ms": duration_ms,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        }

        self._logger.error("Request failed", extra=log_data, exc_info=True)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with logging.

        Logs request start, completion, and any errors.
        Propagates request_id in request metadata for downstream tracing.
        """
        request_id = self._get_or_generate_request_id(request)
        self._log_request_start(request_id, request)

        # Propagate request_id in metadata
        request_with_id = self._propagate_request_id(request, request_id)

        start_time = time.perf_counter()
        try:
            response = await self._client.send_request(request_with_id)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._log_response_completion(request_id, response, duration_ms)
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._log_error(request_id, exc, duration_ms)
            raise

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with logging.

        Logs request start, first chunk arrival, completion, and any errors.
        Propagates request_id in request metadata for downstream tracing.
        """
        request_id = self._get_or_generate_request_id(request)
        self._log_request_start(request_id, request)

        # Propagate request_id in metadata
        request_with_id = self._propagate_request_id(request, request_id)

        start_time = time.perf_counter()
        final_chunk = None

        try:
            async for chunk in self._client.send_request_stream(request_with_id):
                if chunk.is_final:
                    final_chunk = chunk
                yield chunk

            # Log completion with metadata from final chunk
            if final_chunk is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000.0

                # Build synthetic response from final chunk metadata
                response = AgentResponse(
                    content="",  # Not included in logs
                    model=final_chunk.metadata.get("model", "unknown"),
                    input_tokens=final_chunk.metadata.get("input_tokens", 0),
                    output_tokens=final_chunk.metadata.get("output_tokens", 0),
                    metadata=final_chunk.metadata,
                )
                self._log_response_completion(request_id, response, duration_ms)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self._log_error(request_id, exc, duration_ms)
            raise
