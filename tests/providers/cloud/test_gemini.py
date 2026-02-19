"""Tests for providers/cloud/gemini.py.

Covers: GeminiClient constructor (task 4.3.1) — default model, custom model,
api_key storage, base_url, x-goog-api-key header, TLS enforcement, timeout
forwarding, semaphore creation, dynamic _endpoint per model, and module exports.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers.cloud.gemini import GeminiClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _HttpGemini(HttpAgentClient):
    """HttpAgentClient subclass with TLS enabled, bypassing GeminiClient init.

    Used to test TLS enforcement in isolation — GeminiClient hard-codes an
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
        return "/test"


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for gemini.py."""

    def test_gemini_client_in_all(self) -> None:
        """GeminiClient is listed in __all__."""
        from mada_modelkit.providers.cloud import gemini

        assert "GeminiClient" in gemini.__all__

    def test_gemini_client_importable(self) -> None:
        """GeminiClient can be imported directly from its module."""
        from mada_modelkit.providers.cloud.gemini import GeminiClient as GC

        assert GC is GeminiClient

    def test_gemini_client_is_subclass_of_http_agent_client(self) -> None:
        """GeminiClient inherits from HttpAgentClient."""
        assert issubclass(GeminiClient, HttpAgentClient)


# ---------------------------------------------------------------------------
# TestGeminiClientConstructor
# ---------------------------------------------------------------------------


class TestGeminiClientConstructor:
    """GeminiClient constructor (task 4.3.1)."""

    def test_default_model(self) -> None:
        """Default model is gemini-2.0-flash."""
        client = GeminiClient(api_key="AIza-test")
        assert client._model == "gemini-2.0-flash"

    def test_custom_model_stored(self) -> None:
        """Custom model string is stored in _model."""
        client = GeminiClient(api_key="AIza-test", model="gemini-1.5-pro")
        assert client._model == "gemini-1.5-pro"

    def test_api_key_stored(self) -> None:
        """API key is stored in _api_key."""
        client = GeminiClient(api_key="AIza-abc123")
        assert client._api_key == "AIza-abc123"

    def test_base_url_is_google_generativelanguage(self) -> None:
        """httpx client base_url targets the Gemini API."""
        client = GeminiClient(api_key="AIza-test")
        assert "generativelanguage.googleapis.com" in str(client._http_client.base_url)

    def test_base_url_uses_https(self) -> None:
        """Base URL scheme is https (TLS enforced)."""
        client = GeminiClient(api_key="AIza-test")
        assert str(client._http_client.base_url).startswith("https://")

    def test_x_goog_api_key_header_set(self) -> None:
        """x-goog-api-key header is set to the provided api_key."""
        client = GeminiClient(api_key="AIza-mykey")
        assert client._http_client.headers["x-goog-api-key"] == "AIza-mykey"

    def test_require_tls_class_variable(self) -> None:
        """_require_tls class variable is True."""
        assert GeminiClient._require_tls is True

    def test_tls_enforcement_rejects_http(self) -> None:
        """HttpAgentClient raises ValueError for http:// when _require_tls=True."""
        with pytest.raises(ValueError, match="TLS"):
            _HttpGemini(base_url="http://generativelanguage.googleapis.com/v1beta")

    def test_tls_enforcement_accepts_https(self) -> None:
        """HttpAgentClient accepts https:// when _require_tls=True."""
        client = _HttpGemini(base_url="https://generativelanguage.googleapis.com/v1beta")
        assert str(client._http_client.base_url).startswith("https://")

    def test_connect_timeout_forwarded(self) -> None:
        """connect_timeout kwarg is forwarded to HttpAgentClient."""
        client = GeminiClient(api_key="AIza-test", connect_timeout=3.0)
        assert client._http_client.timeout.connect == 3.0

    def test_read_timeout_forwarded(self) -> None:
        """read_timeout kwarg is forwarded to HttpAgentClient."""
        client = GeminiClient(api_key="AIza-test", read_timeout=120.0)
        assert client._http_client.timeout.read == 120.0

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates a semaphore."""
        client = GeminiClient(api_key="AIza-test", max_concurrent=4)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = GeminiClient(api_key="AIza-test")
        assert client._semaphore is None

    def test_endpoint_includes_model(self) -> None:
        """_endpoint returns a path that contains the model name."""
        client = GeminiClient(api_key="AIza-test", model="gemini-1.5-flash")
        assert "gemini-1.5-flash" in client._endpoint()

    def test_endpoint_includes_generate_content(self) -> None:
        """_endpoint path ends with :generateContent."""
        client = GeminiClient(api_key="AIza-test")
        assert client._endpoint().endswith(":generateContent")

    def test_endpoint_changes_with_model(self) -> None:
        """_endpoint returns a different path when model differs."""
        a = GeminiClient(api_key="AIza-test", model="gemini-2.0-flash")
        b = GeminiClient(api_key="AIza-test", model="gemini-1.5-pro")
        assert a._endpoint() != b._endpoint()

    def test_different_api_keys_stored_independently(self) -> None:
        """Two clients with different api_keys store them independently."""
        a = GeminiClient(api_key="key-a")
        b = GeminiClient(api_key="key-b")
        assert a._api_key == "key-a"
        assert b._api_key == "key-b"
