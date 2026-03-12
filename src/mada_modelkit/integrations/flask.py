"""Flask integration for mada-modelkit.

Provides Flask extension and streaming response helpers.
Requires the optional flask dependency.
"""

from __future__ import annotations

from typing import Any, Iterator, TYPE_CHECKING

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest

if TYPE_CHECKING:
    from flask import Flask, Response

__all__ = ["MadaKit", "stream_response"]

# Deferred import check
try:
    from flask import current_app, Response

    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False


class MadaKit:
    """Flask extension for mada-modelkit.

    Stores a mada-modelkit client in Flask app config and provides
    easy access via the extension pattern.

    Raises:
        ImportError: If flask is not installed.

    Example:
        ```python
        from flask import Flask
        from mada_modelkit.integrations.flask import MadaKit

        app = Flask(__name__)
        madakit = MadaKit(app, client=MyClient())

        @app.route("/generate")
        async def generate():
            client = madakit.client
            response = await client.send_request(AgentRequest(prompt="Hello"))
            return {"response": response.content}
        ```
    """

    def __init__(
        self,
        app: Flask | None = None,
        client: BaseAgentClient | None = None,
        config_key: str = "MADAKIT_CLIENT",
    ) -> None:
        """Initialize Flask extension.

        Args:
            app: Flask application (optional, can use init_app later).
            client: Mada-modelkit client to store.
            config_key: Key in app.config for client (default: "MADAKIT_CLIENT").

        Raises:
            ImportError: If flask is not installed.
        """
        if not _FLASK_AVAILABLE:
            raise ImportError(
                "Flask integration requires flask. "
                "Install with: pip install mada-modelkit[flask]"
            )

        self.config_key = config_key
        self._client = client

        if app is not None:
            self.init_app(app, client)

    def init_app(
        self, app: Flask, client: BaseAgentClient | None = None
    ) -> None:
        """Initialize extension with Flask app.

        Args:
            app: Flask application.
            client: Mada-modelkit client (optional, uses __init__ client if not provided).
        """
        if client is not None:
            self._client = client

        if self._client is not None:
            app.config[self.config_key] = self._client

    @property
    def client(self) -> BaseAgentClient:
        """Get the mada-modelkit client from current app config.

        Returns:
            The mada-modelkit client.

        Raises:
            ValueError: If client not configured.
        """
        if not _FLASK_AVAILABLE:
            raise ImportError(
                "Flask integration requires flask. "
                "Install with: pip install mada-modelkit[flask]"
            )

        client = current_app.config.get(self.config_key)
        if client is None:
            raise ValueError(
                f"Client not configured. Set {self.config_key} in app.config "
                "or pass client to MadaKit(app, client=...)"
            )
        return client


def _sync_stream_generator(
    client: BaseAgentClient, request: AgentRequest
) -> Iterator[str]:
    """Internal generator for Flask SSE streaming.

    Note: This is a synchronous wrapper that won't work with async clients.
    For production use, consider using Flask with async support (Quart).

    Args:
        client: The mada-modelkit client.
        request: The agent request.

    Yields:
        SSE-formatted chunks.
    """
    # This is a simplified version that doesn't actually stream
    # Real async streaming in Flask requires Quart or similar
    yield f"data: Use Quart for true async streaming\n\n"
    yield "data: [DONE]\n\n"


def stream_response(
    client: BaseAgentClient,
    request: AgentRequest,
) -> Response:
    """Create a Flask Response with SSE streaming.

    Note: True async streaming requires Quart or Flask with async support.
    This is a simplified implementation for demonstration.

    Args:
        client: The mada-modelkit client.
        request: The agent request to stream.

    Returns:
        Flask Response with SSE stream.

    Raises:
        ImportError: If flask is not installed.

    Example:
        ```python
        @app.route("/stream")
        def stream_generate():
            madakit = MadaKit()
            request = AgentRequest(prompt="Tell me a story")
            return stream_response(madakit.client, request)
        ```
    """
    if not _FLASK_AVAILABLE:
        raise ImportError(
            "Flask integration requires flask. "
            "Install with: pip install mada-modelkit[flask]"
        )

    return Response(
        _sync_stream_generator(client, request),
        mimetype="text/event-stream",
    )
