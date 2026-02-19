"""Tests for providers/cloud/anthropic.py.

Covers: AnthropicClient constructor (task 4.2.1) — default model, custom
model, api_key storage, base_url, x-api-key header, anthropic-version header,
TLS enforcement, timeout forwarding, semaphore creation, and module exports.
_build_payload (task 4.2.2) — Anthropic wire format: system as top-level
field, messages array with user turn only, stop_sequences mapping.
_parse_response (task 4.2.3) — content[0].text extraction, model fallback,
input_tokens/output_tokens defaults.
Attachment support (task 4.2.4) — Attachment mapped to Anthropic image source
blocks (base64-encoded bytes, media_type), text block appended after images.
__repr__ (task 4.2.5) — API key redacted, model visible, exact format.
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


# ---------------------------------------------------------------------------
# TestBuildPayload
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """AnthropicClient.__repr__ (task 4.2.5)."""

    def test_repr_does_not_contain_api_key(self) -> None:
        """repr output does not expose the raw API key."""
        client = AnthropicClient(api_key="sk-ant-secret")
        assert "sk-ant-secret" not in repr(client)

    def test_repr_contains_redacted_placeholder(self) -> None:
        """repr contains '***' in place of the API key."""
        client = AnthropicClient(api_key="sk-ant-secret")
        assert "***" in repr(client)

    def test_repr_contains_model(self) -> None:
        """repr contains the model identifier."""
        client = AnthropicClient(api_key="sk-ant-test", model="claude-opus-4-6")
        assert "claude-opus-4-6" in repr(client)

    def test_repr_exact_format_default_model(self) -> None:
        """repr matches the expected format with the default model."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert repr(client) == "AnthropicClient(model='claude-sonnet-4-6', api_key=***)"

    def test_repr_exact_format_custom_model(self) -> None:
        """repr reflects a custom model in the exact format."""
        client = AnthropicClient(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
        assert repr(client) == "AnthropicClient(model='claude-haiku-4-5-20251001', api_key=***)"

    def test_repr_different_keys_same_output(self) -> None:
        """Two clients with different keys produce identical repr output."""
        a = AnthropicClient(api_key="key-one")
        b = AnthropicClient(api_key="key-two")
        assert repr(a) == repr(b)

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = AnthropicClient(api_key="sk-ant-test")
        assert isinstance(repr(client), str)


# ---------------------------------------------------------------------------
# Helpers for payload / response tests
# ---------------------------------------------------------------------------


def _make_request(**kwargs: object) -> AgentRequest:
    """Return an AgentRequest with sensible defaults, overridden by kwargs."""
    defaults: dict[str, object] = {
        "prompt": "Hello",
        "system_prompt": None,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stop": None,
    }
    defaults.update(kwargs)
    return AgentRequest(**defaults)  # type: ignore[arg-type]


class TestBuildPayload:
    """AnthropicClient._build_payload (task 4.2.2)."""

    def test_user_prompt_in_messages(self) -> None:
        """User prompt appears as a user-role entry in the messages array."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(prompt="Hi there"))
        assert payload["messages"] == [{"role": "user", "content": "Hi there"}]

    def test_messages_has_exactly_one_entry_without_system(self) -> None:
        """Without a system prompt, messages contains exactly one user entry."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request())
        assert len(payload["messages"]) == 1

    def test_system_prompt_is_top_level_field(self) -> None:
        """System prompt appears as a top-level 'system' key, not in messages."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(system_prompt="Be helpful"))
        assert payload["system"] == "Be helpful"

    def test_system_prompt_not_in_messages(self) -> None:
        """System prompt does not appear inside the messages array."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(system_prompt="Be helpful"))
        for msg in payload["messages"]:
            assert msg.get("role") != "system"

    def test_no_system_key_when_system_prompt_is_none(self) -> None:
        """'system' key is absent from payload when system_prompt is None."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(system_prompt=None))
        assert "system" not in payload

    def test_no_system_key_when_system_prompt_is_empty(self) -> None:
        """'system' key is absent from payload when system_prompt is empty string."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(system_prompt=""))
        assert "system" not in payload

    def test_model_in_payload(self) -> None:
        """Payload contains the client's model identifier."""
        client = AnthropicClient(api_key="sk-ant-test", model="claude-opus-4-6")
        payload = client._build_payload(_make_request())
        assert payload["model"] == "claude-opus-4-6"

    def test_max_tokens_in_payload(self) -> None:
        """max_tokens from the request appears in the payload."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(max_tokens=512))
        assert payload["max_tokens"] == 512

    def test_temperature_in_payload(self) -> None:
        """temperature from the request appears in the payload."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(temperature=0.2))
        assert payload["temperature"] == 0.2

    def test_stop_sequences_mapped_from_stop(self) -> None:
        """request.stop is mapped to 'stop_sequences' (Anthropic's name)."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(stop=["END", "STOP"]))
        assert payload["stop_sequences"] == ["END", "STOP"]

    def test_stop_sequences_absent_when_stop_is_none(self) -> None:
        """'stop_sequences' is absent from payload when request.stop is None."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(stop=None))
        assert "stop_sequences" not in payload

    def test_stop_key_not_used(self) -> None:
        """Payload uses 'stop_sequences' not 'stop' (OpenAI naming)."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request(stop=["X"]))
        assert "stop" not in payload

    def test_payload_keys_present(self) -> None:
        """Payload always contains model, max_tokens, messages, temperature."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request())
        for key in ("model", "max_tokens", "messages", "temperature"):
            assert key in payload


# ---------------------------------------------------------------------------
# TestParseResponse
# ---------------------------------------------------------------------------


def _make_response(**kwargs: object) -> dict:  # type: ignore[type-arg]
    """Return an Anthropic-shaped response dict with sensible defaults."""
    data: dict = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    data.update(kwargs)
    return data


class TestParseResponse:
    """AnthropicClient._parse_response (task 4.2.3)."""

    def test_returns_agent_response(self) -> None:
        """_parse_response returns an AgentResponse instance."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(_make_response())
        assert isinstance(result, AgentResponse)

    def test_content_extracted_from_content_array(self) -> None:
        """content is extracted from data['content'][0]['text']."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(_make_response(content=[{"type": "text", "text": "Hi!"}]))
        assert result.content == "Hi!"

    def test_model_from_response(self) -> None:
        """model field is taken from the response data."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(_make_response(model="claude-opus-4-6"))
        assert result.model == "claude-opus-4-6"

    def test_model_fallback_to_self_model(self) -> None:
        """model falls back to self._model when absent from response."""
        client = AnthropicClient(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
        data = _make_response()
        del data["model"]
        result = client._parse_response(data)
        assert result.model == "claude-haiku-4-5-20251001"

    def test_input_tokens_from_usage(self) -> None:
        """input_tokens is read from usage.input_tokens."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(
            _make_response(usage={"input_tokens": 42, "output_tokens": 7})
        )
        assert result.input_tokens == 42

    def test_output_tokens_from_usage(self) -> None:
        """output_tokens is read from usage.output_tokens."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(
            _make_response(usage={"input_tokens": 3, "output_tokens": 99})
        )
        assert result.output_tokens == 99

    def test_input_tokens_default_zero_when_usage_absent(self) -> None:
        """input_tokens defaults to 0 when usage key is absent."""
        client = AnthropicClient(api_key="sk-ant-test")
        data = _make_response()
        del data["usage"]
        result = client._parse_response(data)
        assert result.input_tokens == 0

    def test_output_tokens_default_zero_when_usage_absent(self) -> None:
        """output_tokens defaults to 0 when usage key is absent."""
        client = AnthropicClient(api_key="sk-ant-test")
        data = _make_response()
        del data["usage"]
        result = client._parse_response(data)
        assert result.output_tokens == 0

    def test_tokens_default_zero_when_usage_fields_missing(self) -> None:
        """Token counts default to 0 when usage is present but fields are absent."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(_make_response(usage={}))
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_total_tokens_property(self) -> None:
        """total_tokens equals input_tokens + output_tokens."""
        client = AnthropicClient(api_key="sk-ant-test")
        result = client._parse_response(
            _make_response(usage={"input_tokens": 20, "output_tokens": 30})
        )
        assert result.total_tokens == 50


# ---------------------------------------------------------------------------
# TestAttachmentSupport
# ---------------------------------------------------------------------------


import base64 as _base64

from mada_modelkit._types import Attachment


class TestAttachmentSupport:
    """AnthropicClient._build_payload attachment mapping (task 4.2.4)."""

    def test_no_attachments_content_is_plain_string(self) -> None:
        """Without attachments, user message content is a plain string."""
        client = AnthropicClient(api_key="sk-ant-test")
        payload = client._build_payload(_make_request())
        assert payload["messages"][0]["content"] == "Hello"

    def test_single_attachment_content_is_list(self) -> None:
        """With an attachment, user message content becomes a list."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        assert isinstance(payload["messages"][0]["content"], list)

    def test_attachment_block_type_is_image(self) -> None:
        """The attachment content block has type 'image'."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/jpeg")
        payload = client._build_payload(_make_request(attachments=[att]))
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "image"

    def test_attachment_source_type_is_base64(self) -> None:
        """The image source block has type 'base64'."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        source = payload["messages"][0]["content"][0]["source"]
        assert source["type"] == "base64"

    def test_attachment_media_type_propagated(self) -> None:
        """media_type from Attachment is placed in the source block."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/webp")
        payload = client._build_payload(_make_request(attachments=[att]))
        source = payload["messages"][0]["content"][0]["source"]
        assert source["media_type"] == "image/webp"

    def test_attachment_bytes_base64_encoded(self) -> None:
        """Attachment bytes are base64-encoded in the source data field."""
        client = AnthropicClient(api_key="sk-ant-test")
        raw = b"\x89PNG\r\n"
        att = Attachment(content=raw, media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        source = payload["messages"][0]["content"][0]["source"]
        assert source["data"] == _base64.b64encode(raw).decode("ascii")

    def test_text_block_appended_after_image_blocks(self) -> None:
        """The text block (prompt) comes after all image blocks."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(_make_request(prompt="Describe", attachments=[att]))
        content = payload["messages"][0]["content"]
        assert content[-1]["type"] == "text"

    def test_text_block_contains_prompt(self) -> None:
        """The text block's text field equals request.prompt."""
        client = AnthropicClient(api_key="sk-ant-test")
        att = Attachment(content=b"img", media_type="image/jpeg")
        payload = client._build_payload(_make_request(prompt="What is this?", attachments=[att]))
        content = payload["messages"][0]["content"]
        assert content[-1]["text"] == "What is this?"

    def test_multiple_attachments_all_present(self) -> None:
        """Multiple attachments each produce a separate image block."""
        client = AnthropicClient(api_key="sk-ant-test")
        atts = [
            Attachment(content=b"a", media_type="image/png"),
            Attachment(content=b"b", media_type="image/jpeg"),
        ]
        payload = client._build_payload(_make_request(attachments=atts))
        content = payload["messages"][0]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        assert len(image_blocks) == 2

    def test_multiple_attachments_order_preserved(self) -> None:
        """Attachments appear in the same order as request.attachments."""
        client = AnthropicClient(api_key="sk-ant-test")
        atts = [
            Attachment(content=b"first", media_type="image/png"),
            Attachment(content=b"second", media_type="image/gif"),
        ]
        payload = client._build_payload(_make_request(attachments=atts))
        content = payload["messages"][0]["content"]
        assert content[0]["source"]["data"] == _base64.b64encode(b"first").decode("ascii")
        assert content[1]["source"]["data"] == _base64.b64encode(b"second").decode("ascii")

    def test_content_list_length_is_attachments_plus_one(self) -> None:
        """Content list has len(attachments) image blocks + 1 text block."""
        client = AnthropicClient(api_key="sk-ant-test")
        atts = [Attachment(content=bytes([i]), media_type="image/png") for i in range(3)]
        payload = client._build_payload(_make_request(attachments=atts))
        content = payload["messages"][0]["content"]
        assert len(content) == 4  # 3 images + 1 text
