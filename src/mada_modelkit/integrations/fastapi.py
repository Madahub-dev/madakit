"""FastAPI integration for mada-modelkit.

Provides dependency injection and streaming response helpers for FastAPI.
Requires the optional fastapi dependency.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, TYPE_CHECKING

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import StreamingResponse

__all__ = ["get_client", "stream_response"]

# Deferred import check
try:
    from fastapi import Request
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


def get_client(request: Request, client_key: str = "madakit_client") -> BaseAgentClient:
    """FastAPI dependency for injecting mada-modelkit client.

    Use this as a FastAPI dependency to access the client stored in app state.

    Args:
        request: The FastAPI request object.
        client_key: Key in app.state where client is stored (default: "madakit_client").

    Returns:
        The mada-modelkit client from app state.

    Raises:
        ImportError: If fastapi is not installed.
        ValueError: If client not found in app state.

    Example:
        ```python
        from fastapi import FastAPI, Depends
        from mada_modelkit.integrations.fastapi import get_client

        app = FastAPI()
        app.state.madakit_client = MyClient()

        @app.get("/generate")
        async def generate(client: BaseAgentClient = Depends(get_client)):
            response = await client.send_request(AgentRequest(prompt="Hello"))
            return {"response": response.content}
        ```
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI integration requires fastapi. "
            "Install with: pip install mada-modelkit[fastapi]"
        )

    if not hasattr(request.app.state, client_key):
        raise ValueError(
            f"Client not found in app.state.{client_key}. "
            f"Set it in your FastAPI app: app.state.{client_key} = client"
        )

    return getattr(request.app.state, client_key)


async def _stream_generator(
    client: BaseAgentClient, request: AgentRequest
) -> AsyncIterator[str]:
    """Internal generator for SSE streaming.

    Args:
        client: The mada-modelkit client.
        request: The agent request.

    Yields:
        SSE-formatted chunks.
    """
    async for chunk in client.send_request_stream(request):
        if chunk.delta:
            # Format as Server-Sent Events
            yield f"data: {chunk.delta}\n\n"

        if chunk.is_final:
            # Send final event
            yield "data: [DONE]\n\n"


def stream_response(
    client: BaseAgentClient,
    request: AgentRequest,
    media_type: str = "text/event-stream",
) -> StreamingResponse:
    """Create a FastAPI StreamingResponse from a mada-modelkit stream.

    Args:
        client: The mada-modelkit client.
        request: The agent request to stream.
        media_type: Response media type (default: text/event-stream for SSE).

    Returns:
        FastAPI StreamingResponse with SSE stream.

    Raises:
        ImportError: If fastapi is not installed.

    Example:
        ```python
        @app.get("/stream")
        async def stream_generate(client: BaseAgentClient = Depends(get_client)):
            request = AgentRequest(prompt="Tell me a story")
            return stream_response(client, request)
        ```
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI integration requires fastapi. "
            "Install with: pip install mada-modelkit[fastapi]"
        )

    return StreamingResponse(
        _stream_generator(client, request),
        media_type=media_type,
    )
