"""Tests for providers/cloud/deepseek.py.

Covers: DeepSeekClient constructor (task 4.4.1) — default model, custom model,
api_key storage, base_url, Authorization Bearer header, TLS enforcement, timeout
forwarding, semaphore creation, OpenAICompatMixin inheritance, and module exports.
__repr__ (task 4.4.2) — API key redacted, model visible, exact format.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin
from mada_modelkit.providers.cloud.deepseek import DeepSeekClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HttpDeepSeek(HttpAgentClient):
    """HttpAgentClient subclass with TLS enabled, bypassing DeepSeekClient init.

    Used to test TLS enforcement in isolation — DeepSeekClient hard-codes an
    https:// URL so the validator can't be exercised through its constructor.
    """

    _require_tls: bool = True

    def _build_payload(self, request: AgentRequest) -> dict:  # type: ignore[override]
        """Stub implementation."""
        raise NotImplementedError

    def _parse_response(self, data: dict) -> AgentResponse:  # type: ignore[override]
        """Stub implementation."""
        raise NotImplementedError

    def _endpoint(self) -> str:
        """Stub implementation."""
        return "/chat/completions"


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for deepseek.py."""

    def test_deepseek_client_in_all(self) -> None:
        """DeepSeekClient is listed in __all__."""
        from mada_modelkit.providers.cloud import deepseek

        assert "DeepSeekClient" in deepseek.__all__

    def test_deepseek_client_importable(self) -> None:
        """DeepSeekClient can be imported directly from its module."""
        from mada_modelkit.providers.cloud.deepseek import DeepSeekClient as DC

        assert DC is DeepSeekClient

    def test_deepseek_client_is_subclass_of_http_agent_client(self) -> None:
        """DeepSeekClient inherits from HttpAgentClient."""
        assert issubclass(DeepSeekClient, HttpAgentClient)

    def test_deepseek_client_uses_openai_compat_mixin(self) -> None:
        """DeepSeekClient inherits from OpenAICompatMixin."""
        assert issubclass(DeepSeekClient, OpenAICompatMixin)


# ---------------------------------------------------------------------------
# TestDeepSeekClientConstructor
# ---------------------------------------------------------------------------


class TestDeepSeekClientConstructor:
    """DeepSeekClient constructor (task 4.4.1)."""

    def test_default_model(self) -> None:
        """Default model is deepseek-chat."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert client._model == "deepseek-chat"

    def test_custom_model_stored(self) -> None:
        """Custom model string is stored in _model."""
        client = DeepSeekClient(api_key="sk-ds-test", model="deepseek-reasoner")
        assert client._model == "deepseek-reasoner"

    def test_api_key_stored(self) -> None:
        """API key is stored in _api_key."""
        client = DeepSeekClient(api_key="sk-ds-abc123")
        assert client._api_key == "sk-ds-abc123"

    def test_base_url_is_deepseek(self) -> None:
        """httpx client base_url targets the DeepSeek API."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert "api.deepseek.com" in str(client._http_client.base_url)

    def test_base_url_uses_https(self) -> None:
        """Base URL scheme is https (TLS enforced)."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert str(client._http_client.base_url).startswith("https://")

    def test_base_url_includes_v1(self) -> None:
        """Base URL path includes /v1."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert "/v1" in str(client._http_client.base_url)

    def test_authorization_bearer_header_set(self) -> None:
        """Authorization header is set to Bearer {api_key}."""
        client = DeepSeekClient(api_key="sk-ds-mykey")
        assert client._http_client.headers["authorization"] == "Bearer sk-ds-mykey"

    def test_require_tls_class_variable(self) -> None:
        """_require_tls class variable is True."""
        assert DeepSeekClient._require_tls is True

    def test_tls_enforcement_rejects_http(self) -> None:
        """HttpAgentClient raises ValueError for http:// when _require_tls=True."""
        with pytest.raises(ValueError, match="TLS"):
            _HttpDeepSeek(base_url="http://api.deepseek.com/v1")

    def test_tls_enforcement_accepts_https(self) -> None:
        """HttpAgentClient accepts https:// when _require_tls=True."""
        client = _HttpDeepSeek(base_url="https://api.deepseek.com/v1")
        assert str(client._http_client.base_url).startswith("https://")

    def test_connect_timeout_forwarded(self) -> None:
        """connect_timeout kwarg is forwarded to HttpAgentClient."""
        client = DeepSeekClient(api_key="sk-ds-test", connect_timeout=2.0)
        assert client._http_client.timeout.connect == 2.0

    def test_read_timeout_forwarded(self) -> None:
        """read_timeout kwarg is forwarded to HttpAgentClient."""
        client = DeepSeekClient(api_key="sk-ds-test", read_timeout=45.0)
        assert client._http_client.timeout.read == 45.0

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates a semaphore."""
        client = DeepSeekClient(api_key="sk-ds-test", max_concurrent=3)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert client._semaphore is None

    def test_endpoint_returns_chat_completions(self) -> None:
        """_endpoint returns /chat/completions (inherited from OpenAICompatMixin)."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert client._endpoint() == "/chat/completions"

    def test_different_api_keys_stored_independently(self) -> None:
        """Two clients with different api_keys store them independently."""
        a = DeepSeekClient(api_key="key-a")
        b = DeepSeekClient(api_key="key-b")
        assert a._api_key == "key-a"
        assert b._api_key == "key-b"


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """DeepSeekClient.__repr__ (task 4.4.2)."""

    def test_repr_does_not_contain_api_key(self) -> None:
        """repr output does not expose the raw API key."""
        client = DeepSeekClient(api_key="sk-ds-secret")
        assert "sk-ds-secret" not in repr(client)

    def test_repr_contains_redacted_placeholder(self) -> None:
        """repr contains '***' in place of the API key."""
        client = DeepSeekClient(api_key="sk-ds-secret")
        assert "***" in repr(client)

    def test_repr_contains_model(self) -> None:
        """repr contains the model identifier."""
        client = DeepSeekClient(api_key="sk-ds-test", model="deepseek-reasoner")
        assert "deepseek-reasoner" in repr(client)

    def test_repr_exact_format_default_model(self) -> None:
        """repr matches the expected format with the default model."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert repr(client) == "DeepSeekClient(model='deepseek-chat', api_key=***)"

    def test_repr_exact_format_custom_model(self) -> None:
        """repr reflects a custom model in the exact format."""
        client = DeepSeekClient(api_key="sk-ds-test", model="deepseek-reasoner")
        assert repr(client) == "DeepSeekClient(model='deepseek-reasoner', api_key=***)"

    def test_repr_different_keys_same_output(self) -> None:
        """Two clients with different keys produce identical repr output."""
        a = DeepSeekClient(api_key="key-one")
        b = DeepSeekClient(api_key="key-two")
        assert repr(a) == repr(b)

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = DeepSeekClient(api_key="sk-ds-test")
        assert isinstance(repr(client), str)
