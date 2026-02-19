"""Tests for OpenAICompatMixin._build_payload and _parse_response (tasks 3.2.1–3.2.2).

Covers: messages list structure with and without system_prompt, user message
always present, model field from _model attribute, max_tokens and temperature
fields, stop key absent when stop=None, stop key present and correct when
stop is a list; _parse_response: content from choices[0].message.content,
model from data["model"] with fallback to _model, input_tokens from
usage.prompt_tokens (default 0), output_tokens from usage.completion_tokens
(default 0), AgentResponse type returned.
"""

from __future__ import annotations

from typing import Any

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse
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
