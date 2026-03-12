"""Tests for Flask integration.

Covers Flask extension and streaming helpers.
"""

from __future__ import annotations

import pytest

# Check if flask is available
try:
    from flask import Flask

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if FLASK_AVAILABLE:
    from madakit.integrations.flask import MadaKit, stream_response

from helpers import MockProvider

pytestmark = pytest.mark.skipif(
    not FLASK_AVAILABLE, reason="flask not installed"
)


class TestModuleExports:
    """Verify flask module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.integrations import flask

        assert hasattr(flask, "__all__")

    def test_all_contains_madakit(self) -> None:
        """__all__ contains MadaKit."""
        from madakit.integrations.flask import __all__

        assert "MadaKit" in __all__

    def test_all_contains_stream_response(self) -> None:
        """__all__ contains stream_response."""
        from madakit.integrations.flask import __all__

        assert "stream_response" in __all__


class TestMadaKitExtension:
    """Test MadaKit Flask extension."""

    def test_init_with_app(self) -> None:
        """Extension initializes with app."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit(app, client=client)

        assert "MADAKIT_CLIENT" in app.config
        assert app.config["MADAKIT_CLIENT"] is client

    def test_init_app_later(self) -> None:
        """Extension can use init_app pattern."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit()
        madakit.init_app(app, client=client)

        assert "MADAKIT_CLIENT" in app.config

    def test_custom_config_key(self) -> None:
        """Extension supports custom config key."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit(app, client=client, config_key="MY_CLIENT")

        assert "MY_CLIENT" in app.config
        assert madakit.config_key == "MY_CLIENT"

    def test_client_property(self) -> None:
        """client property retrieves from app config."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit(app, client=client)

        with app.app_context():
            retrieved = madakit.client
            assert retrieved is client

    def test_client_property_raises_when_not_configured(self) -> None:
        """client property raises when not configured."""
        app = Flask(__name__)
        madakit = MadaKit()
        madakit.config_key = "MADAKIT_CLIENT"

        with app.app_context():
            with pytest.raises(ValueError, match="not configured"):
                _ = madakit.client


class TestStreamResponse:
    """Test stream_response helper."""

    def test_stream_response_creates_response(self) -> None:
        """stream_response creates Flask Response."""
        client = MockProvider()
        from madakit._types import AgentRequest

        request = AgentRequest(prompt="Hello")
        response = stream_response(client, request)

        assert response.mimetype == "text/event-stream"

    def test_stream_response_is_generator(self) -> None:
        """stream_response returns generator response."""
        client = MockProvider()
        from madakit._types import AgentRequest

        request = AgentRequest(prompt="Test")
        response = stream_response(client, request)

        # Response should have data
        assert response.response is not None


class TestFlaskIntegration:
    """Integration tests with Flask."""

    def test_extension_in_route(self) -> None:
        """Extension works in Flask route."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit(app, client=client)

        @app.route("/test")
        def test_route():
            c = madakit.client
            return {"client_type": type(c).__name__}

        test_client = app.test_client()
        response = test_client.get("/test")

        assert response.status_code == 200
        data = response.get_json()
        assert data["client_type"] == "MockProvider"

    def test_multiple_routes_share_extension(self) -> None:
        """Multiple routes can access same extension."""
        app = Flask(__name__)
        client = MockProvider()
        madakit = MadaKit(app, client=client)

        @app.route("/route1")
        def route1():
            return {"route": 1}

        @app.route("/route2")
        def route2():
            return {"route": 2}

        test_client = app.test_client()
        response1 = test_client.get("/route1")
        response2 = test_client.get("/route2")

        assert response1.status_code == 200
        assert response2.status_code == 200
