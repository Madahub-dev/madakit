"""Tests for OpenAIClient constructor and __repr__ (tasks 4.1.1–4.1.2).

Covers: OpenAICompatMixin and HttpAgentClient inheritance, default model
"gpt-4o-mini", custom model stored as _model, api_key stored as _api_key,
_require_tls class variable is True, base_url fixed to api.openai.com/v1,
Authorization Bearer header set from api_key, http:// URL rejected by TLS
enforcement, kwargs forwarded (connect_timeout, read_timeout, max_concurrent),
httpx.AsyncClient created, per-instance client independence; __repr__:
format "OpenAIClient(model=..., api_key=***)", key not present, model
present, is a str.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin
from mada_modelkit.providers.cloud.openai import OpenAIClient


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
        import mada_modelkit.providers.cloud.openai as mod
        assert hasattr(mod, "__all__")

    def test_openai_client_in_all(self) -> None:
        """Asserts that 'OpenAIClient' is listed in __all__."""
        import mada_modelkit.providers.cloud.openai as mod
        assert "OpenAIClient" in mod.__all__

    def test_openai_client_importable(self) -> None:
        """Asserts that OpenAIClient can be imported from the module."""
        from mada_modelkit.providers.cloud.openai import OpenAIClient as OAC
        assert OAC is OpenAIClient
