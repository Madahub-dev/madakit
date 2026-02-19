"""Tests for providers/cloud/anthropic.py.

Covers: AnthropicClient constructor (task 4.2.1) — default model, custom
model, api_key storage, base_url, x-api-key header, anthropic-version header,
TLS enforcement, timeout forwarding, semaphore creation, and module exports.
"""

from __future__ import annotations

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers.cloud.anthropic import AnthropicClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HttpAnthropic(HttpAgentClient):
    """HttpAgentClient subclass with TLS enabled, bypassing AnthropicClient init.

    Used to test TLS enforcement in isolation — AnthropicClient hard-codes
    an https:// URL so we can't exercise the validator through its constructor.
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
        return "/messages"


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for anthropic.py."""

    def test_anthropic_client_in_all(self) -> None:
        """AnthropicClient is listed in __all__."""
        from mada_modelkit.providers.cloud import anthropic

        assert "AnthropicClient" in anthropic.__all__

    def test_anthropic_client_importable(self) -> None:
        """AnthropicClient can be imported directly from its module."""
        from mada_modelkit.providers.cloud.anthropic import AnthropicClient as AC

        assert AC is AnthropicClient

    def test_anthropic_client_is_subclass_of_http_agent_client(self) -> None:
        """AnthropicClient inherits from HttpAgentClient."""
        assert issubclass(AnthropicClient, HttpAgentClient)


# ---------------------------------------------------------------------------
# TestAnthropicClientConstructor
# ---------------------------------------------------------------------------


class TestAnthropicClientConstructor:
    """AnthropicClient constructor (task 4.2.1)."""

    def test_default_model(self) -> None:
        """Default model is claude-sonnet-4-6."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert client._model == "claude-sonnet-4-6"

    def test_custom_model_stored(self) -> None:
        """Custom model string is stored in _model."""
        client = AnthropicClient(api_key="sk-ant-test", model="claude-opus-4-6")
        assert client._model == "claude-opus-4-6"

    def test_api_key_stored(self) -> None:
        """API key is stored in _api_key."""
        client = AnthropicClient(api_key="sk-ant-abc123")
        assert client._api_key == "sk-ant-abc123"

    def test_base_url_is_anthropic(self) -> None:
        """httpx client base_url targets the Anthropic API."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert "api.anthropic.com" in str(client._http_client.base_url)

    def test_base_url_uses_https(self) -> None:
        """Base URL scheme is https (TLS enforced)."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert str(client._http_client.base_url).startswith("https://")

    def test_x_api_key_header_set(self) -> None:
        """x-api-key header is set to the provided api_key."""
        client = AnthropicClient(api_key="sk-ant-mykey")
        assert client._http_client.headers["x-api-key"] == "sk-ant-mykey"

    def test_anthropic_version_header_set(self) -> None:
        """anthropic-version header is set to 2023-06-01."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert client._http_client.headers["anthropic-version"] == "2023-06-01"

    def test_require_tls_class_variable(self) -> None:
        """_require_tls class variable is True."""
        assert AnthropicClient._require_tls is True

    def test_tls_enforcement_rejects_http(self) -> None:
        """HttpAgentClient raises ValueError for http:// when _require_tls=True."""
        with pytest.raises(ValueError, match="TLS"):
            _HttpAnthropic(base_url="http://api.anthropic.com/v1")

    def test_tls_enforcement_accepts_https(self) -> None:
        """HttpAgentClient accepts https:// when _require_tls=True."""
        client = _HttpAnthropic(base_url="https://api.anthropic.com/v1")
        assert str(client._http_client.base_url).startswith("https://")

    def test_connect_timeout_forwarded(self) -> None:
        """connect_timeout kwarg is forwarded to HttpAgentClient."""
        client = AnthropicClient(api_key="sk-ant-test", connect_timeout=2.5)
        assert client._http_client.timeout.connect == 2.5

    def test_read_timeout_forwarded(self) -> None:
        """read_timeout kwarg is forwarded to HttpAgentClient."""
        client = AnthropicClient(api_key="sk-ant-test", read_timeout=90.0)
        assert client._http_client.timeout.read == 90.0

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates a semaphore."""
        import asyncio

        client = AnthropicClient(api_key="sk-ant-test", max_concurrent=5)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert client._semaphore is None

    def test_different_api_keys_stored_independently(self) -> None:
        """Two clients with different api_keys store them independently."""
        a = AnthropicClient(api_key="key-a")
        b = AnthropicClient(api_key="key-b")
        assert a._api_key == "key-a"
        assert b._api_key == "key-b"

    def test_endpoint_returns_messages(self) -> None:
        """_endpoint returns /messages."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert client._endpoint() == "/messages"
