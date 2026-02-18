"""Abstract base class for all mada-modelkit providers and middleware.

Defines BaseAgentClient — the single interface every provider and middleware
implements. One abstract method (send_request); all other methods are virtual
with sensible defaults. No external dependencies — stdlib only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk


class BaseAgentClient(ABC):
    """Common interface for every AI backend and middleware layer."""

    @abstractmethod
    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute a request and return a complete response."""
        ...

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks. Default: call send_request, yield single final chunk."""
        response = await self.send_request(request)
        yield StreamChunk(delta=response.content, is_final=True)

    async def generate(self, prompt: str, **kwargs: Any) -> AgentResponse:
        """Convenience: build AgentRequest from prompt and kwargs, call send_request."""
        return await self.send_request(AgentRequest(prompt=prompt, **kwargs))

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        """Convenience: build AgentRequest from prompt and kwargs, call send_request_stream."""
        async for chunk in self.send_request_stream(AgentRequest(prompt=prompt, **kwargs)):
            yield chunk

    async def health_check(self) -> bool:
        """Check if the backend is available and responsive. Returns True by default."""
        return True

    async def cancel(self) -> None:
        """Request cancellation of in-progress work. Best-effort, no-op by default."""

    async def close(self) -> None:
        """Release resources (HTTP client, model, executor). No-op by default."""

    async def __aenter__(self) -> BaseAgentClient:
        """Enter the async context manager, returning self."""
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Exit the async context manager, calling close()."""
        await self.close()
