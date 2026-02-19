"""Tests for providers/local_server/vllm.py.

Covers: VllmClient constructor (task 5.2.1) — required model, default base_url,
custom base_url, no TLS enforcement, no authorization header, timeout forwarding,
semaphore creation, OpenAICompatMixin inheritance, __repr__ format, and module
exports.
Comprehensive integration (task 5.2.2) — full round-trip via MockTransport:
/chat/completions routing, OpenAI-compat payload, no auth header, attachment
image_url block, ProviderError on non-2xx, token counts, health_check, context
manager.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse, Attachment
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin
from mada_modelkit.providers.local_server.vllm import VllmClient


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for vllm.py."""

    def test_vllm_client_in_all(self) -> None:
        """VllmClient is listed in __all__."""
        from mada_modelkit.providers.local_server import vllm

        assert "VllmClient" in vllm.__all__

    def test_vllm_client_importable(self) -> None:
        """VllmClient can be imported directly from its module."""
        from mada_modelkit.providers.local_server.vllm import VllmClient as VC

        assert VC is VllmClient

    def test_vllm_client_is_subclass_of_http_agent_client(self) -> None:
        """VllmClient inherits from HttpAgentClient."""
        assert issubclass(VllmClient, HttpAgentClient)

    def test_vllm_client_uses_openai_compat_mixin(self) -> None:
        """VllmClient inherits from OpenAICompatMixin."""
        assert issubclass(VllmClient, OpenAICompatMixin)


# ---------------------------------------------------------------------------
# TestVllmClientConstructor
# ---------------------------------------------------------------------------


class TestVllmClientConstructor:
    """VllmClient constructor (task 5.2.1)."""

    def test_model_required(self) -> None:
        """VllmClient raises TypeError when model is not provided."""
        with pytest.raises(TypeError):
            VllmClient()  # type: ignore[call-arg]

    def test_model_stored(self) -> None:
        """Provided model string is stored in _model."""
        client = VllmClient(model="meta-llama/Llama-3.2-3B-Instruct")
        assert client._model == "meta-llama/Llama-3.2-3B-Instruct"

    def test_default_base_url_is_localhost_8000(self) -> None:
        """Default base_url targets localhost:8000."""
        client = VllmClient(model="llama")
        assert "localhost:8000" in str(client._http_client.base_url)

    def test_default_base_url_includes_v1(self) -> None:
        """Default base_url path includes /v1."""
        client = VllmClient(model="llama")
        assert "/v1" in str(client._http_client.base_url)

    def test_custom_base_url_accepted(self) -> None:
        """A custom base_url is forwarded to the httpx client."""
        client = VllmClient(model="llama", base_url="http://192.168.1.5:8000/v1")
        assert "192.168.1.5" in str(client._http_client.base_url)

    def test_base_url_stored(self) -> None:
        """_base_url attribute stores the constructor argument."""
        client = VllmClient(model="llama", base_url="http://myhost:8000/v1")
        assert client._base_url == "http://myhost:8000/v1"

    def test_require_tls_is_false(self) -> None:
        """_require_tls class variable is False (not enforced for local server)."""
        assert VllmClient._require_tls is False

    def test_http_url_accepted(self) -> None:
        """http:// base_url is accepted without error (no TLS enforcement)."""
        client = VllmClient(model="llama", base_url="http://localhost:8000/v1")
        assert str(client._http_client.base_url).startswith("http://")

    def test_no_authorization_header(self) -> None:
        """No Authorization header is set (no API key needed)."""
        client = VllmClient(model="llama")
        assert "authorization" not in client._http_client.headers

    def test_connect_timeout_forwarded(self) -> None:
        """connect_timeout kwarg is forwarded to HttpAgentClient."""
        client = VllmClient(model="llama", connect_timeout=3.0)
        assert client._http_client.timeout.connect == 3.0

    def test_read_timeout_forwarded(self) -> None:
        """read_timeout kwarg is forwarded to HttpAgentClient."""
        client = VllmClient(model="llama", read_timeout=90.0)
        assert client._http_client.timeout.read == 90.0

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates an asyncio.Semaphore."""
        client = VllmClient(model="llama", max_concurrent=2)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = VllmClient(model="llama")
        assert client._semaphore is None

    def test_endpoint_returns_chat_completions(self) -> None:
        """_endpoint returns /chat/completions (inherited from OpenAICompatMixin)."""
        client = VllmClient(model="llama")
        assert client._endpoint() == "/chat/completions"

    def test_different_models_stored_independently(self) -> None:
        """Two clients with different models store them independently."""
        a = VllmClient(model="llama")
        b = VllmClient(model="mistral")
        assert a._model == "llama"
        assert b._model == "mistral"


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """VllmClient.__repr__ (task 5.2.1)."""

    def test_repr_contains_model(self) -> None:
        """repr contains the model identifier."""
        client = VllmClient(model="meta-llama/Llama-3.2-3B-Instruct")
        assert "meta-llama/Llama-3.2-3B-Instruct" in repr(client)

    def test_repr_contains_base_url(self) -> None:
        """repr contains the base URL."""
        client = VllmClient(model="llama")
        assert "localhost:8000" in repr(client)

    def test_repr_exact_format_defaults(self) -> None:
        """repr matches the expected format with default base_url."""
        client = VllmClient(model="llama")
        assert repr(client) == (
            "VllmClient(model='llama', base_url='http://localhost:8000/v1')"
        )

    def test_repr_exact_format_custom(self) -> None:
        """repr reflects custom model and base_url."""
        client = VllmClient(model="mistral", base_url="http://myhost:8000/v1")
        assert repr(client) == (
            "VllmClient(model='mistral', base_url='http://myhost:8000/v1')"
        )

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = VllmClient(model="llama")
        assert isinstance(repr(client), str)


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def _openai_json_response(
    content: str = "Hello from vLLM",
    model: str = "llama",
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


def _make_client(model: str = "llama", *, body: bytes | None = None) -> VllmClient:
    """Return a VllmClient whose _http_client uses a MockTransport."""
    if body is None:
        body = _openai_json_response(model=model)
    client = VllmClient(model=model)
    client._http_client = httpx.AsyncClient(
        base_url="http://localhost:8000/v1",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=body)
        ),
    )
    return client


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full round-trip tests for VllmClient via MockTransport (task 5.2.2)."""

    @pytest.mark.asyncio
    async def test_send_request_returns_agent_response(self) -> None:
        """send_request returns an AgentResponse for a basic prompt."""
        client = _make_client()
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert isinstance(response, AgentResponse)

    @pytest.mark.asyncio
    async def test_send_request_content_matches_body(self) -> None:
        """AgentResponse.content equals the text from the response body."""
        body = _openai_json_response(content="vLLM says hi")
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.content == "vLLM says hi"

    @pytest.mark.asyncio
    async def test_send_request_routes_to_chat_completions(self) -> None:
        """send_request POSTs to /chat/completions."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
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

        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
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

        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
            transport=httpx.MockTransport(handler),
        )
        await client.send_request(AgentRequest(prompt="Hello"))
        payload = json.loads(captured[0].content)
        assert payload["model"] == "llama"
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_attachment_produces_image_url_block(self) -> None:
        """An Attachment is serialised as an image_url content block."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Capture the request."""
            captured.append(request)
            return httpx.Response(200, content=_openai_json_response())

        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
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
        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(503, content=b'{"error": "unavailable"}')
            ),
        )
        with pytest.raises(ProviderError):
            await client.send_request(AgentRequest(prompt="Hi"))

    @pytest.mark.asyncio
    async def test_token_counts_populated(self) -> None:
        """AgentResponse carries prompt and completion token counts."""
        body = _openai_json_response(prompt_tokens=20, completion_tokens=9)
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Count"))
        assert response.input_tokens == 20
        assert response.output_tokens == 9

    @pytest.mark.asyncio
    async def test_model_field_in_response(self) -> None:
        """AgentResponse.model is taken from the JSON 'model' field."""
        body = _openai_json_response(model="meta-llama/Llama-3.2-3B-Instruct")
        client = _make_client(body=body)
        response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.model == "meta-llama/Llama-3.2-3B-Instruct"

    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_200(self) -> None:
        """health_check (inherited GET /) returns True on 200."""
        client = VllmClient(model="llama")
        client._http_client = httpx.AsyncClient(
            base_url="http://localhost:8000/v1",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=b"ok")
            ),
        )
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_context_manager_closes_cleanly(self) -> None:
        """VllmClient works as an async context manager without error."""
        async with VllmClient(model="llama") as client:
            assert client is not None
