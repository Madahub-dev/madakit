"""Tests for OpenAIClient: constructor, __repr__, send_request_stream, and integration (tasks 4.1.1–4.1.4).

Covers: OpenAICompatMixin and HttpAgentClient inheritance, default model
"gpt-4o-mini", custom model stored as _model, api_key stored as _api_key,
_require_tls class variable is True, base_url fixed to api.openai.com/v1,
Authorization Bearer header set from api_key, http:// URL rejected by TLS
enforcement, kwargs forwarded (connect_timeout, read_timeout, max_concurrent),
httpx.AsyncClient created, per-instance client independence; __repr__:
format "OpenAIClient(model=..., api_key=***)", key not present, model
present, is a str; send_request_stream: SSE delta chunks yielded, final
StreamChunk on [DONE], stream=True in payload, non-2xx raises ProviderError,
ConnectError/TimeoutException wrapped as ProviderError, empty delta chunks
not suppressed, non-data lines skipped; integration: send_request round-trip,
auth header forwarded in both paths, /chat/completions route, health_check,
close/context manager.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from madakit._errors import ProviderError
from madakit._types import AgentRequest, StreamChunk
from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin
from madakit.providers.cloud.openai import OpenAIClient


class TestOpenAIClientConstructor:
    """OpenAIClient.__init__ — attribute storage and inherited configuration."""

    def test_is_openai_compat_mixin(self) -> None:
        """Asserts that OpenAIClient is an OpenAICompatMixin instance."""
        client = OpenAIClient(api_key="sk-test")
        assert isinstance(client, OpenAICompatMixin)

    def test_is_http_agent_client(self) -> None:
        """Asserts that OpenAIClient is an HttpAgentClient instance."""
        client = OpenAIClient(api_key="sk-test")
        assert isinstance(client, HttpAgentClient)

    def test_default_model_is_gpt_4o_mini(self) -> None:
        """Asserts that the default model is 'gpt-4o-mini'."""
        client = OpenAIClient(api_key="sk-test")
        assert client._model == "gpt-4o-mini"

    def test_custom_model_stored(self) -> None:
        """Asserts that a custom model name is stored in _model."""
        client = OpenAIClient(api_key="sk-test", model="gpt-4o")
        assert client._model == "gpt-4o"

    def test_api_key_stored(self) -> None:
        """Asserts that the api_key is stored in _api_key."""
        client = OpenAIClient(api_key="sk-secret")
        assert client._api_key == "sk-secret"

    def test_require_tls_is_true(self) -> None:
        """Asserts that _require_tls is True on OpenAIClient."""
        assert OpenAIClient._require_tls is True

    def test_base_url_is_openai(self) -> None:
        """Asserts that the httpx client base_url points to api.openai.com/v1."""
        client = OpenAIClient(api_key="sk-test")
        assert "api.openai.com" in str(client._http_client.base_url)

    def test_base_url_includes_v1(self) -> None:
        """Asserts that the base_url path includes '/v1'."""
        client = OpenAIClient(api_key="sk-test")
        assert "/v1" in str(client._http_client.base_url)

    def test_authorization_header_set(self) -> None:
        """Asserts that the Authorization header is present in the httpx client."""
        client = OpenAIClient(api_key="sk-test")
        assert "authorization" in client._http_client.headers

    def test_authorization_header_uses_bearer(self) -> None:
        """Asserts that the Authorization header value starts with 'Bearer '."""
        client = OpenAIClient(api_key="sk-test")
        assert client._http_client.headers["authorization"].startswith("Bearer ")

    def test_authorization_header_contains_api_key(self) -> None:
        """Asserts that the api_key appears in the Authorization header value."""
        client = OpenAIClient(api_key="sk-mykey")
        assert "sk-mykey" in client._http_client.headers["authorization"]

    def test_tls_enforcement_rejects_http_subclass(self) -> None:
        """Asserts that _require_tls=True causes ValueError for any http:// base_url."""
        class _HttpOpenAI(OpenAIClient):
            """Subclass that forces an http:// base_url to exercise TLS rejection."""

            def __init__(self) -> None:
                """Bypass OpenAIClient.__init__ to inject an insecure URL."""
                self._model = "gpt-4o-mini"
                self._api_key = "sk-test"
                HttpAgentClient.__init__(  # type: ignore[misc]
                    self,
                    base_url="http://api.openai.com/v1",
                )

        with pytest.raises(ValueError):
            _HttpOpenAI()

    def test_connect_timeout_kwarg_forwarded(self) -> None:
        """Asserts that connect_timeout kwarg is passed through to the httpx client."""
        client = OpenAIClient(api_key="sk-test", connect_timeout=3.0)
        assert client._http_client.timeout.connect == 3.0

    def test_read_timeout_kwarg_forwarded(self) -> None:
        """Asserts that read_timeout kwarg is passed through to the httpx client."""
        client = OpenAIClient(api_key="sk-test", read_timeout=120.0)
        assert client._http_client.timeout.read == 120.0

    def test_max_concurrent_kwarg_creates_semaphore(self) -> None:
        """Asserts that max_concurrent kwarg creates an asyncio.Semaphore."""
        client = OpenAIClient(api_key="sk-test", max_concurrent=5)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_creates_httpx_async_client(self) -> None:
        """Asserts that _http_client is an httpx.AsyncClient instance."""
        client = OpenAIClient(api_key="sk-test")
        assert isinstance(client._http_client, httpx.AsyncClient)

    def test_two_instances_have_independent_http_clients(self) -> None:
        """Asserts that two OpenAIClient instances do not share an httpx client."""
        c1 = OpenAIClient(api_key="sk-a")
        c2 = OpenAIClient(api_key="sk-b")
        assert c1._http_client is not c2._http_client

    def test_different_api_keys_produce_different_headers(self) -> None:
        """Asserts that two instances with different keys have different headers."""
        c1 = OpenAIClient(api_key="sk-alpha")
        c2 = OpenAIClient(api_key="sk-beta")
        assert (
            c1._http_client.headers["authorization"]
            != c2._http_client.headers["authorization"]
        )


class TestRepr:
    """OpenAIClient.__repr__ — key redaction and model visibility."""

    def test_repr_is_str(self) -> None:
        """Asserts that repr() returns a str."""
        client = OpenAIClient(api_key="sk-secret")
        assert isinstance(repr(client), str)

    def test_repr_contains_class_name(self) -> None:
        """Asserts that the repr starts with 'OpenAIClient'."""
        client = OpenAIClient(api_key="sk-secret")
        assert repr(client).startswith("OpenAIClient")

    def test_repr_contains_model(self) -> None:
        """Asserts that the model name appears in the repr."""
        client = OpenAIClient(api_key="sk-secret", model="gpt-4o")
        assert "gpt-4o" in repr(client)

    def test_repr_does_not_contain_api_key(self) -> None:
        """Asserts that the actual api_key value is not present in the repr."""
        client = OpenAIClient(api_key="sk-super-secret")
        assert "sk-super-secret" not in repr(client)

    def test_repr_contains_redacted_placeholder(self) -> None:
        """Asserts that the repr contains '***' in place of the key."""
        client = OpenAIClient(api_key="sk-secret")
        assert "***" in repr(client)

    def test_repr_format(self) -> None:
        """Asserts the exact repr format: OpenAIClient(model=..., api_key=***)."""
        client = OpenAIClient(api_key="sk-secret", model="gpt-4o-mini")
        assert repr(client) == "OpenAIClient(model='gpt-4o-mini', api_key=***)"

    def test_repr_reflects_custom_model(self) -> None:
        """Asserts that the repr reflects a non-default model name."""
        client = OpenAIClient(api_key="sk-x", model="o1-mini")
        assert "o1-mini" in repr(client)


class TestModuleExports:
    """openai module — __all__ and public name availability."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined in providers.cloud.openai."""
        import madakit.providers.cloud.openai as mod
        assert hasattr(mod, "__all__")

    def test_openai_client_in_all(self) -> None:
        """Asserts that 'OpenAIClient' is listed in __all__."""
        import madakit.providers.cloud.openai as mod
        assert "OpenAIClient" in mod.__all__

    def test_openai_client_importable(self) -> None:
        """Asserts that OpenAIClient can be imported from the module."""
        from madakit.providers.cloud.openai import OpenAIClient as OAC
        assert OAC is OpenAIClient


# ---------------------------------------------------------------------------
# SSE helpers for send_request_stream tests
# ---------------------------------------------------------------------------


def _sse_body(*deltas: str, include_done: bool = True) -> bytes:
    """Build a minimal SSE response body with content deltas and optional [DONE]."""
    lines: list[str] = []
    for delta in deltas:
        chunk = {"choices": [{"delta": {"content": delta}}], "model": "gpt-4o-mini"}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    if include_done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _make_streaming_client(handler: object, api_key: str = "sk-test") -> OpenAIClient:
    """Return an OpenAIClient whose httpx client uses a MockTransport."""
    client = OpenAIClient(api_key=api_key)
    client._http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return client


class TestSendRequestStream:
    """OpenAIClient.send_request_stream — SSE parsing and error handling."""

    async def test_yields_stream_chunks(self) -> None:
        """Asserts that send_request_stream yields StreamChunk instances."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("Hello"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert all(isinstance(c, StreamChunk) for c in chunks)

    async def test_delta_content_matches_sse_lines(self) -> None:
        """Asserts that each non-final StreamChunk delta matches the SSE content."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("Hello", " world"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        content_chunks = [c for c in chunks if not c.is_final]
        assert [c.delta for c in content_chunks] == ["Hello", " world"]

    async def test_final_chunk_has_is_final_true(self) -> None:
        """Asserts that the last StreamChunk yielded has is_final=True."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("Hi"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[-1].is_final is True

    async def test_final_chunk_on_done_has_empty_delta(self) -> None:
        """Asserts that the [DONE] sentinel produces a StreamChunk with delta=''."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("Hi"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        final = next(c for c in chunks if c.is_final)
        assert final.delta == ""

    async def test_non_final_chunks_have_is_final_false(self) -> None:
        """Asserts that content StreamChunks have is_final=False."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("A", "B"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        content_chunks = [c for c in chunks if not c.is_final]
        assert all(not c.is_final for c in content_chunks)

    async def test_stream_true_added_to_payload(self) -> None:
        """Asserts that the POST body includes stream=True."""
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(200, content=_sse_body("ok"))

        client = _make_streaming_client(handler)
        async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert captured[0].get("stream") is True

    async def test_non_data_lines_are_skipped(self) -> None:
        """Asserts that comment and empty SSE lines produce no extra chunks."""
        def handler(request: httpx.Request) -> httpx.Response:
            body = (
                b": keep-alive\n\n"
                b"data: " + json.dumps(
                    {"choices": [{"delta": {"content": "hi"}}], "model": "gpt-4o-mini"}
                ).encode() + b"\n\n"
                b"data: [DONE]\n\n"
            )
            return httpx.Response(200, content=body)

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        # Only content chunk + final DONE chunk
        assert len(chunks) == 2

    async def test_non_2xx_raises_provider_error(self) -> None:
        """Asserts that a non-2xx streaming response raises ProviderError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized")

        client = _make_streaming_client(handler)
        with pytest.raises(ProviderError):
            async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
                pass

    async def test_non_2xx_provider_error_has_status_code(self) -> None:
        """Asserts that the ProviderError from a non-2xx response carries the status code."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limited")

        client = _make_streaming_client(handler)
        with pytest.raises(ProviderError) as exc_info:
            async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert exc_info.value.status_code == 429

    async def test_connect_error_raises_provider_error(self) -> None:
        """Asserts that ConnectError during streaming is wrapped as ProviderError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _make_streaming_client(handler)
        with pytest.raises(ProviderError):
            async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
                pass

    async def test_timeout_raises_provider_error(self) -> None:
        """Asserts that TimeoutException during streaming is wrapped as ProviderError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = _make_streaming_client(handler)
        with pytest.raises(ProviderError):
            async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
                pass

    async def test_multiple_deltas_all_yielded(self) -> None:
        """Asserts that all content chunks are yielded before the final chunk."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_sse_body("A", "B", "C"))

        client = _make_streaming_client(handler)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="hi"))]
        content = [c.delta for c in chunks if not c.is_final]
        assert content == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Helper for integration tests — JSON response body in OpenAI non-stream format
# ---------------------------------------------------------------------------


def _json_response(content: str = "Hello!", model: str = "gpt-4o-mini") -> bytes:
    """Return a JSON-encoded non-streaming OpenAI chat-completions response."""
    body = {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return json.dumps(body).encode()


def _make_client(handler: object, api_key: str = "sk-test") -> OpenAIClient:
    """Return an OpenAIClient wired to a MockTransport for non-streaming requests."""
    client = OpenAIClient(api_key=api_key)
    client._http_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        headers={"Authorization": f"Bearer {api_key}"},
    )
    return client


class TestIntegration:
    """OpenAIClient — end-to-end integration across send_request, stream, and lifecycle."""

    async def test_send_request_returns_agent_response(self) -> None:
        """Asserts that send_request returns an AgentResponse with correct content."""
        from madakit._types import AgentResponse

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_json_response("The answer is 42."))

        client = _make_client(handler)
        result = await client.send_request(AgentRequest(prompt="What is 6×7?"))
        assert isinstance(result, AgentResponse)
        assert result.content == "The answer is 42."

    async def test_send_request_posts_to_chat_completions(self) -> None:
        """Asserts that send_request routes to /v1/chat/completions."""
        captured_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(request.url.path)
            return httpx.Response(200, content=_json_response())

        client = _make_client(handler)
        await client.send_request(AgentRequest(prompt="hi"))
        assert captured_paths[0].endswith("/chat/completions")

    async def test_send_request_forwards_authorization_header(self) -> None:
        """Asserts that the Authorization Bearer header is sent with send_request."""
        captured_auth: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_auth.append(request.headers.get("authorization", ""))
            return httpx.Response(200, content=_json_response())

        client = _make_client(handler)
        await client.send_request(AgentRequest(prompt="hi"))
        assert captured_auth[0] == "Bearer sk-test"

    async def test_send_request_stream_forwards_authorization_header(self) -> None:
        """Asserts that the Authorization Bearer header is sent with send_request_stream."""
        captured_auth: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_auth.append(request.headers.get("authorization", ""))
            return httpx.Response(200, content=_sse_body("ok"))

        client = _make_streaming_client(handler)
        async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert captured_auth[0] == "Bearer sk-test"

    async def test_send_request_stream_routes_to_chat_completions(self) -> None:
        """Asserts that send_request_stream also routes to /v1/chat/completions."""
        captured_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(request.url.path)
            return httpx.Response(200, content=_sse_body("ok"))

        client = _make_streaming_client(handler)
        async for _ in client.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert captured_paths[0].endswith("/chat/completions")

    async def test_health_check_returns_true_on_200(self) -> None:
        """Asserts that health_check returns True when the server responds."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        client = _make_client(handler)
        assert await client.health_check() is True

    async def test_health_check_returns_false_on_connect_error(self) -> None:
        """Asserts that health_check returns False on ConnectError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _make_client(handler)
        assert await client.health_check() is False

    async def test_context_manager_closes_http_client(self) -> None:
        """Asserts that the async context manager closes the httpx client on exit."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_json_response())

        async with _make_client(handler) as client:
            await client.send_request(AgentRequest(prompt="hi"))
        assert client._http_client.is_closed

    async def test_token_counts_flow_to_agent_response(self) -> None:
        """Asserts that prompt_tokens and completion_tokens reach the AgentResponse."""
        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"content": "answer"}}],
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 25, "completion_tokens": 12},
            }
            return httpx.Response(200, content=json.dumps(body).encode())

        client = _make_client(handler)
        result = await client.send_request(AgentRequest(prompt="hi"))
        assert result.input_tokens == 25
        assert result.output_tokens == 12
