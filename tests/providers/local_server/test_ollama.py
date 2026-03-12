"""Tests for providers/local_server/ollama.py.

Covers: OllamaClient constructor (task 5.1.1) — default model, custom model,
default base_url, custom base_url, no TLS enforcement, timeout forwarding,
semaphore creation, OpenAICompatMixin inheritance, __repr__ format, and
module exports.
health_check override (task 5.1.2) — queries /api/tags not /.
send_request_stream SSE (task 5.1.3) — OpenAI-compat streaming format.
Comprehensive integration (task 5.1.4) — full round-trip via MockTransport.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from madakit._errors import ProviderError
from madakit._types import AgentRequest, AgentResponse, Attachment, StreamChunk
from madakit.providers._http_base import HttpAgentClient
from madakit.providers._openai_compat import OpenAICompatMixin
from madakit.providers.local_server.ollama import OllamaClient


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for ollama.py."""

    def test_ollama_client_in_all(self) -> None:
        """OllamaClient is listed in __all__."""
        from madakit.providers.local_server import ollama

        assert "OllamaClient" in ollama.__all__

    def test_ollama_client_importable(self) -> None:
        """OllamaClient can be imported directly from its module."""
        from madakit.providers.local_server.ollama import OllamaClient as OC

        assert OC is OllamaClient

    def test_ollama_client_is_subclass_of_http_agent_client(self) -> None:
        """OllamaClient inherits from HttpAgentClient."""
        assert issubclass(OllamaClient, HttpAgentClient)

    def test_ollama_client_uses_openai_compat_mixin(self) -> None:
        """OllamaClient inherits from OpenAICompatMixin."""
        assert issubclass(OllamaClient, OpenAICompatMixin)


# ---------------------------------------------------------------------------
# TestOllamaClientConstructor
# ---------------------------------------------------------------------------


class TestOllamaClientConstructor:
    """OllamaClient constructor (task 5.1.1)."""

    def test_default_model(self) -> None:
        """Default model tag is llama3.2."""
        client = OllamaClient()
        assert client._model == "llama3.2"

    def test_custom_model_stored(self) -> None:
        """Custom model string is stored in _model."""
        client = OllamaClient(model="mistral")
        assert client._model == "mistral"

    def test_default_base_url_is_localhost(self) -> None:
        """Default base_url targets localhost:11434."""
        client = OllamaClient()
        assert "localhost:11434" in str(client._http_client.base_url)

    def test_default_base_url_includes_v1(self) -> None:
        """Default base_url path includes /v1."""
        client = OllamaClient()
        assert "/v1" in str(client._http_client.base_url)

    def test_custom_base_url_accepted(self) -> None:
        """A custom base_url is forwarded to the httpx client."""
        client = OllamaClient(base_url="http://192.168.1.10:11434/v1")
        assert "192.168.1.10" in str(client._http_client.base_url)

    def test_base_url_stored(self) -> None:
        """_base_url attribute stores the constructor argument."""
        client = OllamaClient(base_url="http://myhost:11434/v1")
        assert client._base_url == "http://myhost:11434/v1"

    def test_require_tls_is_false(self) -> None:
        """_require_tls class variable is False (not enforced for local server)."""
        assert OllamaClient._require_tls is False

    def test_http_url_accepted(self) -> None:
        """http:// base_url is accepted without error (no TLS enforcement)."""
        client = OllamaClient(base_url="http://localhost:11434/v1")
        assert str(client._http_client.base_url).startswith("http://")

    def test_no_authorization_header(self) -> None:
        """No Authorization header is set (no API key needed)."""
        client = OllamaClient()
        assert "authorization" not in client._http_client.headers

    def test_connect_timeout_forwarded(self) -> None:
        """connect_timeout kwarg is forwarded to HttpAgentClient."""
        client = OllamaClient(connect_timeout=2.0)
        assert client._http_client.timeout.connect == 2.0

    def test_read_timeout_forwarded(self) -> None:
        """read_timeout kwarg is forwarded to HttpAgentClient."""
        client = OllamaClient(read_timeout=120.0)
        assert client._http_client.timeout.read == 120.0

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates an asyncio.Semaphore."""
        client = OllamaClient(max_concurrent=1)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = OllamaClient()
        assert client._semaphore is None

    def test_endpoint_returns_chat_completions(self) -> None:
        """_endpoint returns /chat/completions (inherited from OpenAICompatMixin)."""
        client = OllamaClient()
        assert client._endpoint() == "/chat/completions"

    def test_different_models_stored_independently(self) -> None:
        """Two clients with different models store them independently."""
        a = OllamaClient(model="llama3.2")
        b = OllamaClient(model="mistral")
        assert a._model == "llama3.2"
        assert b._model == "mistral"

    def test_different_base_urls_stored_independently(self) -> None:
        """Two clients with different base_urls store them independently."""
        a = OllamaClient(base_url="http://host-a:11434/v1")
        b = OllamaClient(base_url="http://host-b:11434/v1")
        assert "host-a" in str(a._http_client.base_url)
        assert "host-b" in str(b._http_client.base_url)


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """OllamaClient.__repr__ (task 5.1.1)."""

    def test_repr_contains_model(self) -> None:
        """repr contains the model tag."""
        client = OllamaClient(model="mistral")
        assert "mistral" in repr(client)

    def test_repr_contains_base_url(self) -> None:
        """repr contains the base URL."""
        client = OllamaClient()
        assert "localhost:11434" in repr(client)

    def test_repr_exact_format_defaults(self) -> None:
        """repr matches the expected format with default arguments."""
        client = OllamaClient()
        assert repr(client) == (
            "OllamaClient(model='llama3.2', base_url='http://localhost:11434/v1')"
        )

    def test_repr_exact_format_custom(self) -> None:
        """repr reflects custom model and base_url."""
        client = OllamaClient(model="mistral", base_url="http://myhost:11434/v1")
        assert repr(client) == (
            "OllamaClient(model='mistral', base_url='http://myhost:11434/v1')"
        )

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = OllamaClient()
        assert isinstance(repr(client), str)


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """OllamaClient.health_check override (task 5.1.2)."""

    @pytest.mark.asyncio
    async def test_health_check_queries_api_tags(self) -> None:
        """health_check sends GET to /api/tags, not /."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request path."""
            captured.append(request)
            return httpx.Response(200, content=b'{"models":[]}')

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(handler),
        )
        await client.health_check()
        assert captured[0].url.path == "/api/tags"

    @pytest.mark.asyncio
    async def test_health_check_does_not_query_root(self) -> None:
        """health_check does NOT send GET to /."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request path."""
            captured.append(request)
            return httpx.Response(200, content=b'{"models":[]}')

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(handler),
        )
        await client.health_check()
        assert captured[0].url.path != "/"

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_200(self) -> None:
        """health_check returns True when /api/tags responds with 200."""
        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=b'{"models":[]}')
            ),
        )
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_error_status(self) -> None:
        """health_check returns True for any HTTP response, including 500."""
        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(500, content=b"error")
            ),
        )
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_connect_error(self) -> None:
        """health_check returns False when the server is unreachable."""

        def raise_connect(request: httpx.Request) -> httpx.Response:
            """Simulate a connection failure."""
            raise httpx.ConnectError("refused")

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(raise_connect),
        )
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_timeout(self) -> None:
        """health_check returns False on TimeoutException."""

        def raise_timeout(request: httpx.Request) -> httpx.Response:
            """Simulate a timeout."""
            raise httpx.TimeoutException("timed out")

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(raise_timeout),
        )
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_does_not_raise(self) -> None:
        """health_check never propagates ConnectError or TimeoutException."""

        def raise_connect(request: httpx.Request) -> httpx.Response:
            """Always fail."""
            raise httpx.ConnectError("refused")

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(raise_connect),
        )
        result = await client.health_check()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_bytes(*deltas: str) -> bytes:
    """Build a minimal OpenAI-compat SSE byte stream from delta strings."""
    lines: list[bytes] = []
    for delta in deltas:
        chunk = json.dumps({"choices": [{"delta": {"content": delta}}]})
        lines.append(f"data: {chunk}\n\n".encode())
    lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


def _streaming_client(body: bytes) -> OllamaClient:
    """Return an OllamaClient whose _http_client streams *body* via MockTransport."""
    client = OllamaClient()
    client._http_client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=body)
        ),
    )
    return client


# ---------------------------------------------------------------------------
# TestSendRequestStream
# ---------------------------------------------------------------------------


class TestSendRequestStream:
    """OllamaClient.send_request_stream SSE (task 5.1.3)."""

    @pytest.mark.asyncio
    async def test_stream_yields_stream_chunks(self) -> None:
        """send_request_stream yields StreamChunk objects."""
        client = _streaming_client(_sse_bytes("Hello"))
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        assert all(isinstance(c, StreamChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_yields_correct_deltas(self) -> None:
        """Each SSE data line produces a StreamChunk with the correct delta."""
        client = _streaming_client(_sse_bytes("Hello", " world"))
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        non_final = [c for c in chunks if not c.is_final]
        assert [c.delta for c in non_final] == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_final_chunk_on_done(self) -> None:
        """[DONE] sentinel produces a final StreamChunk with empty delta."""
        client = _streaming_client(_sse_bytes("Hi"))
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        assert chunks[-1].is_final is True
        assert chunks[-1].delta == ""

    @pytest.mark.asyncio
    async def test_stream_non_final_chunks_have_is_final_false(self) -> None:
        """Content chunks have is_final=False."""
        client = _streaming_client(_sse_bytes("A", "B"))
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        assert all(not c.is_final for c in chunks[:-1])

    @pytest.mark.asyncio
    async def test_stream_payload_includes_stream_true(self) -> None:
        """Payload sent to the server includes stream=True."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_sse_bytes("ok"))

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        async for _ in client.send_request_stream(AgentRequest(prompt="Hi")):
            pass
        assert json.loads(captured[0].content)["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_routes_to_chat_completions(self) -> None:
        """Streaming POST is sent to /chat/completions."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_sse_bytes("ok"))

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        async for _ in client.send_request_stream(AgentRequest(prompt="Hi")):
            pass
        assert captured[0].url.path.endswith("/chat/completions")

    @pytest.mark.asyncio
    async def test_stream_non_2xx_raises_provider_error(self) -> None:
        """A non-2xx response during streaming raises ProviderError."""
        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(503, content=b"unavailable")
            ),
        )
        with pytest.raises(ProviderError):
            async for _ in client.send_request_stream(AgentRequest(prompt="Hi")):
                pass

    @pytest.mark.asyncio
    async def test_stream_connect_error_raises_provider_error(self) -> None:
        """ConnectError during streaming is wrapped as ProviderError."""

        def raise_connect(r: httpx.Request) -> httpx.Response:
            """Simulate connection failure."""
            raise httpx.ConnectError("refused")

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(raise_connect),
        )
        with pytest.raises(ProviderError):
            async for _ in client.send_request_stream(AgentRequest(prompt="Hi")):
                pass

    @pytest.mark.asyncio
    async def test_stream_skips_non_data_lines(self) -> None:
        """Lines that don't start with 'data:' are silently ignored."""
        body = b"event: ping\n\n" + _sse_bytes("hello")
        client = _streaming_client(body)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        non_final = [c for c in chunks if not c.is_final]
        assert len(non_final) == 1
        assert non_final[0].delta == "hello"

    @pytest.mark.asyncio
    async def test_stream_empty_delta_fields_yield_empty_string(self) -> None:
        """Delta chunks with no 'content' key yield delta=''."""
        body = b'data: {"choices":[{"delta":{}}]}\n\ndata: [DONE]\n\n'
        client = _streaming_client(body)
        chunks = [c async for c in client.send_request_stream(AgentRequest(prompt="Hi"))]
        non_final = [c for c in chunks if not c.is_final]
        assert non_final[0].delta == ""


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _openai_json_response(
    content: str = "Hello from Ollama",
    model: str = "llama3.2",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> bytes:
    """Return a minimal OpenAI-compat JSON response as bytes."""
    return json.dumps(
        {
            "choices": [{"message": {"content": content}}],
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }
    ).encode()


def _make_client(*, body: bytes | None = None) -> OllamaClient:
    """Return an OllamaClient whose _http_client uses a MockTransport.

    Always responds with *body* (or a default OpenAI-compat payload).
    """
    if body is None:
        body = _openai_json_response()
    client = OllamaClient()
    client._http_client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=body)
        ),
    )
    return client


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full round-trip tests for OllamaClient via MockTransport (task 5.1.4)."""

    @pytest.mark.asyncio
    async def test_send_request_returns_agent_response(self) -> None:
        """send_request returns an AgentResponse for a basic prompt."""
        client = _make_client()
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert isinstance(response, AgentResponse)

    @pytest.mark.asyncio
    async def test_send_request_content_matches_body(self) -> None:
        """AgentResponse.content equals the text from the response body."""
        body = _openai_json_response(content="Ollama says hi")
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.content == "Ollama says hi"

    @pytest.mark.asyncio
    async def test_send_request_routes_to_chat_completions(self) -> None:
        """send_request POSTs to /chat/completions."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        await client.send_request(AgentRequest(prompt="Hi"))
        assert captured[0].url.path.endswith("/chat/completions")

    @pytest.mark.asyncio
    async def test_no_authorization_header_in_request(self) -> None:
        """No Authorization header is present in the outgoing request."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        await client.send_request(AgentRequest(prompt="Hi"))
        assert "authorization" not in captured[0].headers

    @pytest.mark.asyncio
    async def test_payload_uses_openai_compat_format(self) -> None:
        """Request body follows OpenAI-compat format with 'model' and 'messages'."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        await client.send_request(AgentRequest(prompt="Hello"))
        payload = json.loads(captured[0].content)
        assert "model" in payload
        assert "messages" in payload
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_attachment_produces_image_url_block(self) -> None:
        """An Attachment is serialised as an image_url content block."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(handler),
        )
        att = Attachment(content=b"\xff\xd8\xff", media_type="image/jpeg")
        await client.send_request(AgentRequest(prompt="Describe", attachments=[att]))
        payload = json.loads(captured[0].content)
        content_blocks = payload["messages"][0]["content"]
        assert any(block.get("type") == "image_url" for block in content_blocks)

    @pytest.mark.asyncio
    async def test_non_2xx_raises_provider_error(self) -> None:
        """A non-2xx HTTP response raises ProviderError."""
        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434/v1",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(404, content=b'{"error": "not found"}')
            ),
        )
        with pytest.raises(ProviderError):
            await client.send_request(AgentRequest(prompt="Hi"))

    @pytest.mark.asyncio
    async def test_token_counts_populated(self) -> None:
        """AgentResponse carries prompt and completion token counts."""
        body = _openai_json_response(prompt_tokens=15, completion_tokens=7)
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Count"))
        assert response.input_tokens == 15
        assert response.output_tokens == 7

    @pytest.mark.asyncio
    async def test_model_field_in_response(self) -> None:
        """AgentResponse.model is taken from the JSON 'model' field."""
        body = _openai_json_response(model="mistral")
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.model == "mistral"

    @pytest.mark.asyncio
    async def test_health_check_queries_api_tags_integration(self) -> None:
        """health_check integration: queries /api/tags and returns True on 200."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=b'{"models":[]}')

        client = OllamaClient()
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:11434",
            transport=httpx.MockTransport(handler),
        )
        result = await client.health_check()
        assert result is True
        assert captured[0].url.path == "/api/tags"

    @pytest.mark.asyncio
    async def test_context_manager_closes_cleanly(self) -> None:
        """OllamaClient works as an async context manager without error."""
        async with OllamaClient() as client:
            assert client is not None
