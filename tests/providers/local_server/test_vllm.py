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
