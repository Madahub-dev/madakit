"""Logging middleware for mada-modelkit.

Structured logging for requests, responses, and errors with correlation IDs.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import logging
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

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with logging.

        Logs request start, completion, and any errors.
        """
        # Stub: will implement in task 9.1.2
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with logging.

        Logs request start, first chunk arrival, completion, and any errors.
        """
        # Stub: will implement in task 9.1.2
        async for chunk in self._client.send_request_stream(request):
            yield chunk
