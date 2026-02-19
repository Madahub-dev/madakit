
"""Tests for HttpAgentClient: constructor, TLS, abstract methods, send_request, health_check, close (tasks 3.1.1–3.1.6).

Covers: base_url storage, default timeout values (connect=5.0, read=60.0),
custom timeout configuration, headers merged into client, no headers default,
max_concurrent semaphore, _require_tls class variable default (False),
BaseAgentClient inheritance, httpx.AsyncClient creation, per-instance
client independence, ValueError raised for http:// when _require_tls=True,
error message content, https:// acceptance when _require_tls=True,
_build_payload/_parse_response/_endpoint declared abstract, instantiation
blocked without all three methods, overrides callable, send_request pipeline
(success, non-2xx ProviderError with status_code, ConnectError wrapping,
TimeoutException wrapping), health_check (True on any HTTP response, False on
ConnectError/TimeoutException), close (delegates to httpx aclose, context
manager releases connection on exit).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient


# ---------------------------------------------------------------------------
# Minimal concrete subclass — implements all three abstract methods
# ---------------------------------------------------------------------------


class _ConcreteHttpClient(HttpAgentClient):
    """Minimal concrete HttpAgentClient that satisfies all abstract contracts."""

    def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
        """Return a fixed minimal payload for tests."""
        return {"prompt": request.prompt}

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Return a fixed AgentResponse for tests."""
        return AgentResponse(content=str(data), model="test", input_tokens=0, output_tokens=0)

    def _endpoint(self) -> str:
        """Return a fixed endpoint for tests."""
        return "/test"


# ---------------------------------------------------------------------------
# TLS-enforcing subclass
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


class TestAbstractMethods:
    """HttpAgentClient._build_payload / _parse_response / _endpoint — abstract contract."""

    def test_instantiating_without_build_payload_raises(self) -> None:
        """Asserts that omitting _build_payload prevents instantiation."""
        class _Missing(HttpAgentClient):
            def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
                return AgentResponse(content="", model="", input_tokens=0, output_tokens=0)
            def _endpoint(self) -> str:
                return "/test"
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                raise NotImplementedError

        with pytest.raises(TypeError):
            _Missing(base_url="https://api.example.com")  # type: ignore[abstract]

    def test_instantiating_without_parse_response_raises(self) -> None:
        """Asserts that omitting _parse_response prevents instantiation."""
        class _Missing(HttpAgentClient):
            def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
                return {}
            def _endpoint(self) -> str:
                return "/test"
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                raise NotImplementedError

        with pytest.raises(TypeError):
            _Missing(base_url="https://api.example.com")  # type: ignore[abstract]

    def test_instantiating_without_endpoint_raises(self) -> None:
        """Asserts that omitting _endpoint prevents instantiation."""
        class _Missing(HttpAgentClient):
            def _build_payload(self, request: AgentRequest) -> dict[str, Any]:
                return {}
            def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
                return AgentResponse(content="", model="", input_tokens=0, output_tokens=0)
            async def send_request(self, request: AgentRequest) -> AgentResponse:
                raise NotImplementedError

        with pytest.raises(TypeError):
            _Missing(base_url="https://api.example.com")  # type: ignore[abstract]

    def test_concrete_subclass_with_all_three_is_instantiable(self) -> None:
        """Asserts that providing all three abstract methods allows instantiation."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert isinstance(client, HttpAgentClient)

    def test_build_payload_returns_dict(self) -> None:
        """Asserts that _build_payload returns a dict on the concrete subclass."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        result = client._build_payload(AgentRequest(prompt="hi"))
        assert isinstance(result, dict)

    def test_build_payload_includes_prompt(self) -> None:
        """Asserts that the concrete _build_payload includes the request prompt."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        payload = client._build_payload(AgentRequest(prompt="hello"))
        assert payload.get("prompt") == "hello"

    def test_parse_response_returns_agent_response(self) -> None:
        """Asserts that _parse_response returns an AgentResponse on the concrete subclass."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        result = client._parse_response({"content": "ok"})
        assert isinstance(result, AgentResponse)

    def test_endpoint_returns_string(self) -> None:
        """Asserts that _endpoint returns a str on the concrete subclass."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert isinstance(client._endpoint(), str)

    def test_endpoint_starts_with_slash(self) -> None:
        """Asserts that the concrete _endpoint value starts with '/'."""
        client = _ConcreteHttpClient(base_url="https://api.example.com")
        assert client._endpoint().startswith("/")


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


class TestTlsEnforcement:
    """HttpAgentClient._require_tls — http:// rejection and https:// acceptance."""

    def test_http_url_raises_value_error_when_require_tls_true(self) -> None:
        """Asserts that a ValueError is raised when http:// is used with _require_tls=True."""
        with pytest.raises(ValueError):
            _TlsHttpClient(base_url="http://api.example.com")

    def test_error_message_contains_offending_url(self) -> None:
        """Asserts that the ValueError message includes the http:// URL."""
        url = "http://api.example.com"
        with pytest.raises(ValueError, match=url):
            _TlsHttpClient(base_url=url)

    def test_https_url_does_not_raise_when_require_tls_true(self) -> None:
        """Asserts that an https:// URL is accepted without error when _require_tls=True."""
        client = _TlsHttpClient(base_url="https://api.example.com")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_http_url_does_not_raise_when_require_tls_false(self) -> None:
        """Asserts that an http:// URL is silently accepted when _require_tls=False."""
        client = _ConcreteHttpClient(base_url="http://localhost:11434")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_require_tls_check_uses_startswith_not_substring(self) -> None:
        """Asserts that 'https://host/http://path' is not rejected (check is prefix-only)."""
        client = _TlsHttpClient(base_url="https://host/http://path")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_tls_error_raised_before_httpx_client_created(self) -> None:
        """Asserts that the TLS ValueError is raised before any httpx client is constructed."""
        with pytest.raises(ValueError):
            _TlsHttpClient(base_url="http://api.example.com")
        # If we reach here the constructor raised before completing — no client leak

    def test_no_client_attribute_on_failed_construction(self) -> None:
        """Asserts that _http_client is not set when TLS validation fails."""
        instance = object.__new__(_TlsHttpClient)
        try:
            _TlsHttpClient.__init__(instance, base_url="http://api.example.com")  # type: ignore[arg-type]
        except ValueError:
            pass
        assert not hasattr(instance, "_http_client")


# ---------------------------------------------------------------------------
# Helpers for send_request tests — MockTransport-backed clients
# ---------------------------------------------------------------------------


def _make_client(
    transport: httpx.MockTransport,
    base_url: str = "https://api.example.com",
) -> _ConcreteHttpClient:
    """Return a _ConcreteHttpClient whose httpx client uses a MockTransport."""
    client = _ConcreteHttpClient(base_url=base_url)
    client._http_client = httpx.AsyncClient(
        base_url=base_url,
        transport=transport,
    )
    return client


def _ok_transport(body: dict[str, Any]) -> httpx.MockTransport:
    """Return a MockTransport that always responds 200 with *body* as JSON."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body).encode())

    return httpx.MockTransport(handler)


def _error_transport(status_code: int, text: str = "error") -> httpx.MockTransport:
    """Return a MockTransport that always responds with *status_code*."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=text)

    return httpx.MockTransport(handler)


def _connect_error_transport() -> httpx.MockTransport:
    """Return a MockTransport that always raises httpx.ConnectError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    return httpx.MockTransport(handler)


def _timeout_transport() -> httpx.MockTransport:
    """Return a MockTransport that always raises httpx.ReadTimeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return httpx.MockTransport(handler)


class TestSendRequest:
    """HttpAgentClient.send_request — pipeline, error wrapping, and status codes."""

    async def test_success_returns_agent_response(self) -> None:
        """Asserts that a 200 response produces an AgentResponse."""
        client = _make_client(_ok_transport({"key": "value"}))
        result = await client.send_request(AgentRequest(prompt="hi"))
        assert isinstance(result, AgentResponse)

    async def test_success_passes_payload_to_build_payload(self) -> None:
        """Asserts that the prompt from AgentRequest appears in the POST body."""
        captured: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.content)
            return httpx.Response(200, content=json.dumps({}).encode())

        client = _make_client(httpx.MockTransport(handler))
        await client.send_request(AgentRequest(prompt="hello"))
        body = json.loads(captured[0])
        assert body.get("prompt") == "hello"

    async def test_success_uses_endpoint_path(self) -> None:
        """Asserts that the request is sent to the path returned by _endpoint()."""
        captured_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(request.url.path)
            return httpx.Response(200, content=json.dumps({}).encode())

        client = _make_client(httpx.MockTransport(handler))
        await client.send_request(AgentRequest(prompt="hi"))
        assert captured_paths[0].endswith("/test")

    async def test_non_2xx_raises_provider_error(self) -> None:
        """Asserts that a non-2xx HTTP response raises ProviderError."""
        client = _make_client(_error_transport(400, "bad request"))
        with pytest.raises(ProviderError):
            await client.send_request(AgentRequest(prompt="hi"))

    async def test_non_2xx_provider_error_has_status_code(self) -> None:
        """Asserts that the ProviderError carries the HTTP status code."""
        client = _make_client(_error_transport(422, "unprocessable"))
        with pytest.raises(ProviderError) as exc_info:
            await client.send_request(AgentRequest(prompt="hi"))
        assert exc_info.value.status_code == 422

    async def test_500_raises_provider_error_with_status_code(self) -> None:
        """Asserts that a 500 response raises ProviderError with status_code=500."""
        client = _make_client(_error_transport(500, "internal server error"))
        with pytest.raises(ProviderError) as exc_info:
            await client.send_request(AgentRequest(prompt="hi"))
        assert exc_info.value.status_code == 500

    async def test_connect_error_raises_provider_error(self) -> None:
        """Asserts that httpx.ConnectError is wrapped in ProviderError."""
        client = _make_client(_connect_error_transport())
        with pytest.raises(ProviderError):
            await client.send_request(AgentRequest(prompt="hi"))

    async def test_connect_error_chained_as_cause(self) -> None:
        """Asserts that the original ConnectError is chained via __cause__."""
        client = _make_client(_connect_error_transport())
        with pytest.raises(ProviderError) as exc_info:
            await client.send_request(AgentRequest(prompt="hi"))
        assert isinstance(exc_info.value.__cause__, httpx.ConnectError)

    async def test_timeout_raises_provider_error(self) -> None:
        """Asserts that httpx.TimeoutException is wrapped in ProviderError."""
        client = _make_client(_timeout_transport())
        with pytest.raises(ProviderError):
            await client.send_request(AgentRequest(prompt="hi"))

    async def test_timeout_chained_as_cause(self) -> None:
        """Asserts that the original TimeoutException is chained via __cause__."""
        client = _make_client(_timeout_transport())
        with pytest.raises(ProviderError) as exc_info:
            await client.send_request(AgentRequest(prompt="hi"))
        assert isinstance(exc_info.value.__cause__, httpx.TimeoutException)

    async def test_provider_error_not_raised_on_success(self) -> None:
        """Asserts that a 200 response does not raise any exception."""
        client = _make_client(_ok_transport({"result": "ok"}))
        result = await client.send_request(AgentRequest(prompt="hi"))
        assert result is not None

    async def test_error_message_contains_status_code(self) -> None:
        """Asserts that the ProviderError message includes the HTTP status code."""
        client = _make_client(_error_transport(404, "not found"))
        with pytest.raises(ProviderError, match="404"):
            await client.send_request(AgentRequest(prompt="hi"))


class TestHealthCheck:
    """HttpAgentClient.health_check — reachability probe via GET /."""

    async def test_returns_true_on_200(self) -> None:
        """Asserts that a 200 response from GET / returns True."""
        client = _make_client(_ok_transport({}))
        assert await client.health_check() is True

    async def test_returns_true_on_error_status(self) -> None:
        """Asserts that a non-2xx HTTP response still returns True (server is up)."""
        client = _make_client(_error_transport(404))
        assert await client.health_check() is True

    async def test_returns_true_on_500(self) -> None:
        """Asserts that a 500 response still returns True (server is reachable)."""
        client = _make_client(_error_transport(500))
        assert await client.health_check() is True

    async def test_returns_false_on_connect_error(self) -> None:
        """Asserts that a ConnectError causes health_check to return False."""
        client = _make_client(_connect_error_transport())
        assert await client.health_check() is False

    async def test_returns_false_on_timeout(self) -> None:
        """Asserts that a TimeoutException causes health_check to return False."""
        client = _make_client(_timeout_transport())
        assert await client.health_check() is False

    async def test_does_not_raise_on_connect_error(self) -> None:
        """Asserts that health_check never propagates ConnectError."""
        client = _make_client(_connect_error_transport())
        result = await client.health_check()  # must not raise
        assert result is False

    async def test_does_not_raise_on_timeout(self) -> None:
        """Asserts that health_check never propagates TimeoutException."""
        client = _make_client(_timeout_transport())
        result = await client.health_check()  # must not raise
        assert result is False

    async def test_issues_get_request(self) -> None:
        """Asserts that health_check uses the GET method."""
        captured_methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_methods.append(request.method)
            return httpx.Response(200)

        client = _make_client(httpx.MockTransport(handler))
        await client.health_check()
        assert captured_methods == ["GET"]


class TestClose:
    """HttpAgentClient.close — delegates aclose to the underlying httpx client."""

    async def test_close_does_not_raise(self) -> None:
        """Asserts that calling close() completes without raising."""
        client = _make_client(_ok_transport({}))
        await client.close()  # must not raise

    async def test_close_marks_http_client_as_closed(self) -> None:
        """Asserts that the underlying httpx.AsyncClient is closed after close()."""
        client = _make_client(_ok_transport({}))
        await client.close()
        assert client._http_client.is_closed

    async def test_context_manager_calls_close(self) -> None:
        """Asserts that exiting the async context manager closes the httpx client."""
        async with _make_client(_ok_transport({})) as client:
            pass
        assert client._http_client.is_closed

    async def test_context_manager_closes_on_exception(self) -> None:
        """Asserts that the httpx client is closed even when the body raises."""
        client = _make_client(_ok_transport({}))
        try:
            async with client:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert client._http_client.is_closed

    async def test_close_is_idempotent(self) -> None:
        """Asserts that calling close() twice does not raise."""
        client = _make_client(_ok_transport({}))
        await client.close()
        await client.close()  # second call must not raise
