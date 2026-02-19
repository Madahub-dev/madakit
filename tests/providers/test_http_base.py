"""Tests for HttpAgentClient constructor (task 3.1.1).

Covers: base_url storage, default timeout values (connect=5.0, read=60.0),
custom timeout configuration, headers merged into client, no headers default,
max_concurrent semaphore, _require_tls class variable default (False),
BaseAgentClient inheritance, httpx.AsyncClient creation, and per-instance
client independence.
"""

from __future__ import annotations

import httpx
import pytest

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing (overrides only the abstract path)
# ---------------------------------------------------------------------------


class _ConcreteHttpClient(HttpAgentClient):
    """Minimal concrete HttpAgentClient for constructor tests."""

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Satisfy the abstract contract; not called during constructor tests."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TLS-enforcing subclass for 3.1.2 forward-compatibility check
# ---------------------------------------------------------------------------


class _TlsHttpClient(_ConcreteHttpClient):
    """HttpAgentClient subclass with _require_tls = True."""

    _require_tls: bool = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHttpAgentClientConstructor:
    """HttpAgentClient.__init__ — attribute storage and httpx.AsyncClient creation."""

    def test_is_base_agent_client(self) -> None:
        """Asserts that HttpAgentClient is a BaseAgentClient subclass."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert isinstance(client, BaseAgentClient)

    def test_creates_httpx_async_client(self) -> None:
        """Asserts that _http_client is an httpx.AsyncClient instance."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_base_url_stored_in_http_client(self) -> None:
        """Asserts that the base_url is passed to the underlying httpx.AsyncClient."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert str(client._http_client.base_url) == "https://api.example.com"

    def test_default_connect_timeout_is_five(self) -> None:
        """Asserts that connect_timeout defaults to 5.0 seconds."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert client._http_client.timeout.connect == 5.0

    def test_default_read_timeout_is_sixty(self) -> None:
        """Asserts that read_timeout defaults to 60.0 seconds."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert client._http_client.timeout.read == 60.0

    def test_write_timeout_matches_connect_timeout(self) -> None:
        """Asserts that write timeout mirrors connect_timeout."""
        client = _ConcreteHttpClient(base_url="https://api.example.com", connect_timeout=3.0)
        assert client._http_client.timeout.write == 3.0

    def test_pool_timeout_matches_connect_timeout(self) -> None:
        """Asserts that pool timeout mirrors connect_timeout."""
        client = _ConcreteHttpClient(base_url="https://api.example.com", connect_timeout=3.0)
        assert client._http_client.timeout.pool == 3.0

    def test_custom_connect_timeout(self) -> None:
        """Asserts that a custom connect_timeout is applied to the httpx client."""
        client = _ConcreteHttpClient(base_url="https://api.example.com", connect_timeout=10.0)
        assert client._http_client.timeout.connect == 10.0

    def test_custom_read_timeout(self) -> None:
        """Asserts that a custom read_timeout is applied to the httpx client."""
        client = _ConcreteHttpClient(base_url="https://api.example.com", read_timeout=120.0)
        assert client._http_client.timeout.read == 120.0

    def test_custom_headers_present_in_http_client(self) -> None:
        """Asserts that custom headers are included in the httpx client's headers."""
        client = _ConcreteHttpClient(
            base_url="https://api.example.com",
            headers={"Authorization": "Bearer token123"},
        )
        assert client._http_client.headers["authorization"] == "Bearer token123"

    def test_no_headers_defaults_to_empty(self) -> None:
        """Asserts that omitting headers does not set any custom header keys."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        # httpx adds default headers; our custom headers should not be present
        assert "authorization" not in client._http_client.headers

    def test_max_concurrent_none_leaves_semaphore_none(self) -> None:
        """Asserts that omitting max_concurrent leaves _semaphore as None."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert client._semaphore is None

    def test_max_concurrent_creates_semaphore(self) -> None:
        """Asserts that max_concurrent=N creates an asyncio.Semaphore."""
        import asyncio
        client = _ConcreteHttpClient(base_url="https://api.example.com", max_concurrent=4)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_require_tls_class_variable_defaults_to_false(self) -> None:
        """Asserts that HttpAgentClient._require_tls is False by default."""
        assert HttpAgentClient._require_tls is False

    def test_http_url_accepted_when_require_tls_false(self) -> None:
        """Asserts that http:// URL is accepted when _require_tls is False."""
        client = _ConcreteHttpClient(base_url="http://localhost:11434")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_two_instances_have_independent_http_clients(self) -> None:
        """Asserts that two instances do not share the same httpx.AsyncClient."""
        c1 = _ConcreteHttpClient(base_url="https://api.example.com")
        c2 = _ConcreteHttpClient(base_url="https://api.example.com")
        assert c1._http_client is not c2._http_client

    def test_tls_subclass_variable_is_true(self) -> None:
        """Asserts that a subclass can set _require_tls = True."""
        assert _TlsHttpClient._require_tls is True

    def test_tls_subclass_accepts_https_url(self) -> None:
        """Asserts that a _require_tls subclass accepts an https:// URL without error."""
        client = _TlsHttpClient(base_url="https://api.example.com")
        assert isinstance(client._http_client, httpx.AsyncClient)


class TestModuleExports:
    """Module-level exports — __all__ and public name availability."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined in _http_base."""
        import mada_modelkit.providers._http_base as mod
        assert hasattr(mod, "__all__")

    def test_http_agent_client_in_all(self) -> None:
        """Asserts that 'HttpAgentClient' is listed in __all__."""
        import mada_modelkit.providers._http_base as mod
        assert "HttpAgentClient" in mod.__all__

    def test_http_agent_client_importable(self) -> None:
        """Asserts that HttpAgentClient can be imported from the module."""
        from mada_modelkit.providers._http_base import HttpAgentClient as HAC
        assert HAC is HttpAgentClient
