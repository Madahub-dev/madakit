"""Tests for OpenAICompatMixin: _build_payload, _parse_response, _endpoint, and integration (tasks 3.2.1–3.2.4).

Covers: messages list structure with and without system_prompt, user message
always present, model field from _model attribute, max_tokens and temperature
fields, stop key absent when stop=None, stop key present and correct when
stop is a list; _parse_response: content from choices[0].message.content,
model from data["model"] with fallback to _model, input_tokens from
usage.prompt_tokens (default 0), output_tokens from usage.completion_tokens
(default 0), AgentResponse type returned; _endpoint: returns the string
"/chat/completions", is a str, starts with "/"; module exports (__all__,
importable); integration: mixin + HttpAgentClient full round-trip via
MockTransport, system prompt + stop together, model override, POST to
/chat/completions endpoint.
Attachment support (gap-fill): Attachment mapped to OpenAI image_url content
blocks (data URI with base64 bytes), text block appended after images.
"""

from __future__ import annotations

import base64 as _base64
import json
from typing import Any

import httpx
import pytest

from mada_modelkit._types import AgentRequest, AgentResponse, Attachment
from mada_modelkit.providers._http_base import HttpAgentClient
from mada_modelkit.providers._openai_compat import OpenAICompatMixin


# ---------------------------------------------------------------------------
# Minimal concrete subclass for testing _build_payload in isolation
# ---------------------------------------------------------------------------


class _ConcreteCompat(OpenAICompatMixin):
    """Minimal OpenAICompatMixin subclass with _model set for tests."""

    _model: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildPayload:
    """OpenAICompatMixin._build_payload — messages format and field inclusion."""

    def test_messages_key_is_present(self) -> None:
        """Asserts that the payload contains a 'messages' key."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert "messages" in payload

    def test_user_message_always_present(self) -> None:
        """Asserts that a user-role message is always included in messages."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hello"))
        roles = [m["role"] for m in payload["messages"]]
        assert "user" in roles

    def test_user_message_content_matches_prompt(self) -> None:
        """Asserts that the user message content equals the request prompt."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="what is 2+2?"))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert user_msg["content"] == "what is 2+2?"

    def test_no_system_message_when_system_prompt_is_none(self) -> None:
        """Asserts that no system-role message is added when system_prompt is None."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        roles = [m["role"] for m in payload["messages"]]
        assert "system" not in roles

    def test_system_message_added_when_system_prompt_set(self) -> None:
        """Asserts that a system-role message is added when system_prompt is provided."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", system_prompt="You are helpful.")
        )
        roles = [m["role"] for m in payload["messages"]]
        assert "system" in roles

    def test_system_message_content_matches_system_prompt(self) -> None:
        """Asserts that the system message content equals the request system_prompt."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", system_prompt="Be concise.")
        )
        sys_msg = next(m for m in payload["messages"] if m["role"] == "system")
        assert sys_msg["content"] == "Be concise."

    def test_system_message_comes_before_user_message(self) -> None:
        """Asserts that system message is the first element in the messages list."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", system_prompt="sys")
        )
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_two_messages_with_system_prompt(self) -> None:
        """Asserts that exactly two messages are present when system_prompt is set."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", system_prompt="sys")
        )
        assert len(payload["messages"]) == 2

    def test_one_message_without_system_prompt(self) -> None:
        """Asserts that exactly one message is present when system_prompt is None."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert len(payload["messages"]) == 1

    def test_model_field_uses_class_attribute(self) -> None:
        """Asserts that the payload 'model' field equals the _model attribute."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert payload["model"] == "gpt-4o-mini"

    def test_model_field_reflects_overridden_model(self) -> None:
        """Asserts that a subclass with a different _model is used in the payload."""
        class _CustomModel(_ConcreteCompat):
            """Subclass with overridden model name."""
            _model = "gpt-4o"

        client = _CustomModel()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert payload["model"] == "gpt-4o"

    def test_max_tokens_field_present(self) -> None:
        """Asserts that the payload includes a 'max_tokens' field."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert "max_tokens" in payload

    def test_max_tokens_matches_request(self) -> None:
        """Asserts that 'max_tokens' in the payload equals the request value."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi", max_tokens=512))
        assert payload["max_tokens"] == 512

    def test_temperature_field_present(self) -> None:
        """Asserts that the payload includes a 'temperature' field."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert "temperature" in payload

    def test_temperature_matches_request(self) -> None:
        """Asserts that 'temperature' in the payload equals the request value."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi", temperature=0.0))
        assert payload["temperature"] == 0.0

    def test_stop_key_absent_when_stop_is_none(self) -> None:
        """Asserts that 'stop' is not included in the payload when stop=None."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert "stop" not in payload

    def test_stop_key_present_when_stop_provided(self) -> None:
        """Asserts that 'stop' is included in the payload when stop is a list."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", stop=["END", "\n"])
        )
        assert "stop" in payload

    def test_stop_value_matches_request(self) -> None:
        """Asserts that the payload 'stop' value equals the request stop list."""
        client = _ConcreteCompat()
        payload = client._build_payload(
            AgentRequest(prompt="hi", stop=["END"])
        )
        assert payload["stop"] == ["END"]

    def test_payload_is_dict(self) -> None:
        """Asserts that _build_payload returns a dict."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        assert isinstance(payload, dict)


class TestBuildPayloadAttachments:
    """OpenAICompatMixin._build_payload — attachment mapping to image_url blocks."""

    def test_no_attachments_user_content_is_string(self) -> None:
        """Without attachments, user message content is a plain string."""
        client = _ConcreteCompat()
        payload = client._build_payload(AgentRequest(prompt="hi"))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert user_msg["content"] == "hi"

    def test_single_attachment_content_is_list(self) -> None:
        """With an attachment, user message content becomes a list."""
        client = _ConcreteCompat()
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=[att]))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert isinstance(user_msg["content"], list)

    def test_attachment_block_type_is_image_url(self) -> None:
        """The attachment content block has type 'image_url'."""
        client = _ConcreteCompat()
        att = Attachment(content=b"img", media_type="image/jpeg")
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=[att]))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert user_msg["content"][0]["type"] == "image_url"

    def test_attachment_url_is_data_uri(self) -> None:
        """image_url block URL is a base64 data URI with correct media_type prefix."""
        client = _ConcreteCompat()
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=[att]))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        url = user_msg["content"][0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_attachment_data_is_base64_encoded(self) -> None:
        """Attachment bytes are base64-encoded inside the data URI."""
        client = _ConcreteCompat()
        raw = b"\x89PNG\r\n"
        att = Attachment(content=raw, media_type="image/png")
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=[att]))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        url = user_msg["content"][0]["image_url"]["url"]
        encoded = _base64.b64encode(raw).decode("ascii")
        assert url.endswith(encoded)

    def test_text_block_appended_after_image_blocks(self) -> None:
        """The text block (prompt) is the last element in the content list."""
        client = _ConcreteCompat()
        att = Attachment(content=b"img", media_type="image/jpeg")
        payload = client._build_payload(AgentRequest(prompt="Describe", attachments=[att]))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert user_msg["content"][-1]["type"] == "text"
        assert user_msg["content"][-1]["text"] == "Describe"

    def test_multiple_attachments_all_present(self) -> None:
        """Multiple attachments each produce a separate image_url block."""
        client = _ConcreteCompat()
        atts = [
            Attachment(content=b"a", media_type="image/png"),
            Attachment(content=b"b", media_type="image/jpeg"),
        ]
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=atts))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        image_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
        assert len(image_blocks) == 2

    def test_multiple_attachments_order_preserved(self) -> None:
        """Attachments appear in the same order as request.attachments."""
        client = _ConcreteCompat()
        atts = [
            Attachment(content=b"first", media_type="image/png"),
            Attachment(content=b"second", media_type="image/gif"),
        ]
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=atts))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert _base64.b64encode(b"first").decode("ascii") in user_msg["content"][0]["image_url"]["url"]
        assert _base64.b64encode(b"second").decode("ascii") in user_msg["content"][1]["image_url"]["url"]

    def test_content_list_length_is_attachments_plus_one(self) -> None:
        """Content list has len(attachments) image_url blocks + 1 text block."""
        client = _ConcreteCompat()
        atts = [Attachment(content=bytes([i]), media_type="image/png") for i in range(3)]
        payload = client._build_payload(AgentRequest(prompt="hi", attachments=atts))
        user_msg = next(m for m in payload["messages"] if m["role"] == "user")
        assert len(user_msg["content"]) == 4  # 3 images + 1 text

    def test_system_prompt_still_added_with_attachments(self) -> None:
        """System prompt message is still prepended when attachments are present."""
        client = _ConcreteCompat()
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(
            AgentRequest(prompt="hi", system_prompt="Be helpful", attachments=[att])
        )
        roles = [m["role"] for m in payload["messages"]]
        assert roles[0] == "system"
        assert roles[1] == "user"


# ---------------------------------------------------------------------------
# Helpers — minimal OpenAI-format response dicts
# ---------------------------------------------------------------------------


def _make_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict:
    """Return a minimal OpenAI-compatible chat-completions response dict."""
    return {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class TestParseResponse:
    """OpenAICompatMixin._parse_response — extraction from OpenAI JSON format."""

    def test_returns_agent_response(self) -> None:
        """Asserts that _parse_response returns an AgentResponse instance."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response())
        assert isinstance(result, AgentResponse)

    def test_content_extracted_from_choices(self) -> None:
        """Asserts that content comes from choices[0].message.content."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response(content="The answer is 42."))
        assert result.content == "The answer is 42."

    def test_model_extracted_from_data(self) -> None:
        """Asserts that model is taken from the top-level 'model' key."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response(model="gpt-4o"))
        assert result.model == "gpt-4o"

    def test_model_falls_back_to_class_attribute_when_absent(self) -> None:
        """Asserts that model falls back to self._model when 'model' key is missing."""
        client = _ConcreteCompat()
        data = _make_response()
        del data["model"]
        result = client._parse_response(data)
        assert result.model == "gpt-4o-mini"

    def test_input_tokens_from_prompt_tokens(self) -> None:
        """Asserts that input_tokens is read from usage.prompt_tokens."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response(prompt_tokens=42))
        assert result.input_tokens == 42

    def test_output_tokens_from_completion_tokens(self) -> None:
        """Asserts that output_tokens is read from usage.completion_tokens."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response(completion_tokens=17))
        assert result.output_tokens == 17

    def test_input_tokens_defaults_to_zero_when_usage_absent(self) -> None:
        """Asserts that input_tokens is 0 when the usage key is missing entirely."""
        client = _ConcreteCompat()
        data = _make_response()
        del data["usage"]
        result = client._parse_response(data)
        assert result.input_tokens == 0

    def test_output_tokens_defaults_to_zero_when_usage_absent(self) -> None:
        """Asserts that output_tokens is 0 when the usage key is missing entirely."""
        client = _ConcreteCompat()
        data = _make_response()
        del data["usage"]
        result = client._parse_response(data)
        assert result.output_tokens == 0

    def test_input_tokens_defaults_to_zero_when_prompt_tokens_absent(self) -> None:
        """Asserts that input_tokens is 0 when prompt_tokens is absent from usage."""
        client = _ConcreteCompat()
        data = _make_response()
        del data["usage"]["prompt_tokens"]
        result = client._parse_response(data)
        assert result.input_tokens == 0

    def test_output_tokens_defaults_to_zero_when_completion_tokens_absent(self) -> None:
        """Asserts that output_tokens is 0 when completion_tokens is absent from usage."""
        client = _ConcreteCompat()
        data = _make_response()
        del data["usage"]["completion_tokens"]
        result = client._parse_response(data)
        assert result.output_tokens == 0

    def test_total_tokens_is_sum_of_input_and_output(self) -> None:
        """Asserts that total_tokens equals input_tokens + output_tokens."""
        client = _ConcreteCompat()
        result = client._parse_response(_make_response(prompt_tokens=10, completion_tokens=5))
        assert result.total_tokens == 15


class TestEndpoint:
    """OpenAICompatMixin._endpoint — chat-completions path value."""

    def test_endpoint_returns_chat_completions(self) -> None:
        """Asserts that _endpoint returns '/chat/completions'."""
        client = _ConcreteCompat()
        assert client._endpoint() == "/chat/completions"

    def test_endpoint_returns_str(self) -> None:
        """Asserts that _endpoint returns a str."""
        client = _ConcreteCompat()
        assert isinstance(client._endpoint(), str)

    def test_endpoint_starts_with_slash(self) -> None:
        """Asserts that the endpoint path starts with '/'."""
        client = _ConcreteCompat()
        assert client._endpoint().startswith("/")


class TestModuleExports:
    """_openai_compat module — __all__ and public name availability."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined in _openai_compat."""
        import mada_modelkit.providers._openai_compat as mod
        assert hasattr(mod, "__all__")

    def test_openai_compat_mixin_in_all(self) -> None:
        """Asserts that 'OpenAICompatMixin' is listed in __all__."""
        import mada_modelkit.providers._openai_compat as mod
        assert "OpenAICompatMixin" in mod.__all__

    def test_openai_compat_mixin_importable(self) -> None:
        """Asserts that OpenAICompatMixin can be imported from the module."""
        from mada_modelkit.providers._openai_compat import OpenAICompatMixin as OAC
        assert OAC is OpenAICompatMixin


# ---------------------------------------------------------------------------
# Combined subclass for integration tests
# ---------------------------------------------------------------------------


class _OpenAICompatClient(OpenAICompatMixin, HttpAgentClient):
    """Concrete provider combining OpenAICompatMixin with HttpAgentClient."""

    _model: str = "gpt-4o-mini"


def _compat_response(content: str = "ok", model: str = "gpt-4o-mini") -> bytes:
    """Return a JSON-encoded OpenAI-compatible response body."""
    body = {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
    }
    return json.dumps(body).encode()


class TestIntegration:
    """OpenAICompatMixin + HttpAgentClient — full end-to-end round-trip scenarios."""

    def _make_combined(
        self,
        handler: Any,
        model: str = "gpt-4o-mini",
    ) -> _OpenAICompatClient:
        """Return a combined client wired to a MockTransport handler."""
        client = _OpenAICompatClient(base_url="https://api.example.com")
        client._model = model
        client._http_client = httpx.AsyncClient(
            base_url="https://api.example.com",
            transport=httpx.MockTransport(handler),
        )
        return client

    async def test_full_round_trip_returns_agent_response(self) -> None:
        """Asserts that a full send_request call through the mixin returns AgentResponse."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_compat_response("Hello, world!"))

        client = self._make_combined(handler)
        result = await client.send_request(AgentRequest(prompt="hi"))
        assert isinstance(result, AgentResponse)
        assert result.content == "Hello, world!"

    async def test_send_request_posts_to_chat_completions(self) -> None:
        """Asserts that the POST is sent to /chat/completions."""
        captured_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_paths.append(request.url.path)
            return httpx.Response(200, content=_compat_response())

        client = self._make_combined(handler)
        await client.send_request(AgentRequest(prompt="hi"))
        assert captured_paths[0].endswith("/chat/completions")

    async def test_payload_includes_model_from_class(self) -> None:
        """Asserts that the POST body contains the _model attribute value."""
        captured_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_bodies.append(json.loads(request.content))
            return httpx.Response(200, content=_compat_response(model="gpt-4o"))

        client = self._make_combined(handler, model="gpt-4o")
        await client.send_request(AgentRequest(prompt="hi"))
        assert captured_bodies[0]["model"] == "gpt-4o"

    async def test_system_prompt_and_stop_together(self) -> None:
        """Asserts that system_prompt and stop are both reflected in the payload."""
        captured_bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_bodies.append(json.loads(request.content))
            return httpx.Response(200, content=_compat_response())

        client = self._make_combined(handler)
        await client.send_request(
            AgentRequest(prompt="hi", system_prompt="Be brief.", stop=["END"])
        )
        body = captured_bodies[0]
        roles = [m["role"] for m in body["messages"]]
        assert "system" in roles
        assert body["stop"] == ["END"]

    async def test_token_counts_propagated_to_response(self) -> None:
        """Asserts that token counts from the response body reach AgentResponse."""
        def handler(request: httpx.Request) -> httpx.Response:
            body = {
                "choices": [{"message": {"content": "answer"}}],
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }
            return httpx.Response(200, content=json.dumps(body).encode())

        client = self._make_combined(handler)
        result = await client.send_request(AgentRequest(prompt="hi"))
        assert result.input_tokens == 20
        assert result.output_tokens == 10

    async def test_context_manager_closes_client(self) -> None:
        """Asserts that the async context manager closes the underlying httpx client."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_compat_response())

        client = self._make_combined(handler)
        async with client:
            await client.send_request(AgentRequest(prompt="hi"))
        assert client._http_client.is_closed
