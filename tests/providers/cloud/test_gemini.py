"""Tests for providers/cloud/gemini.py.

Covers: GeminiClient constructor (task 4.3.1) — default model, custom model,
api_key storage, base_url, x-goog-api-key header, TLS enforcement, timeout
forwarding, semaphore creation, dynamic _endpoint per model, and module exports.
_build_payload (task 4.3.2) — Gemini wire format: contents/parts structure,
systemInstruction top-level field, generationConfig with maxOutputTokens and
stopSequences, inlineData attachment blocks.
_parse_response (task 4.3.3) — candidates[0].content.parts[0].text extraction,
modelVersion with _model fallback, promptTokenCount/candidatesTokenCount defaults.
"""

from __future__ import annotations

import asyncio
import base64 as _base64

import pytest

from mada_modelkit._types import AgentRequest, AgentResponse, Attachment
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


# ---------------------------------------------------------------------------
# TestBuildPayload
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
    """GeminiClient._build_payload (task 4.3.2) — Gemini wire format."""

    def test_contents_key_present(self) -> None:
        """Payload contains a 'contents' key."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request())
        assert "contents" in payload

    def test_contents_has_one_user_turn(self) -> None:
        """contents list has a single user-role entry."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request())
        assert len(payload["contents"]) == 1
        assert payload["contents"][0]["role"] == "user"

    def test_prompt_in_parts_as_text(self) -> None:
        """The prompt appears as a text part in the user turn's parts."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(prompt="Hi there"))
        parts = payload["contents"][0]["parts"]
        text_parts = [p for p in parts if "text" in p]
        assert any(p["text"] == "Hi there" for p in text_parts)

    def test_text_part_is_last(self) -> None:
        """The text part is always the last element in parts."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request())
        parts = payload["contents"][0]["parts"]
        assert "text" in parts[-1]

    def test_no_system_instruction_when_none(self) -> None:
        """'systemInstruction' is absent when system_prompt is None."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(system_prompt=None))
        assert "systemInstruction" not in payload

    def test_no_system_instruction_when_empty(self) -> None:
        """'systemInstruction' is absent when system_prompt is empty string."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(system_prompt=""))
        assert "systemInstruction" not in payload

    def test_system_instruction_present_when_set(self) -> None:
        """'systemInstruction' is included when system_prompt is provided."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(system_prompt="Be helpful"))
        assert "systemInstruction" in payload

    def test_system_instruction_text_content(self) -> None:
        """systemInstruction contains the system_prompt text in parts."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(system_prompt="Be concise"))
        assert payload["systemInstruction"]["parts"][0]["text"] == "Be concise"

    def test_generation_config_present(self) -> None:
        """Payload contains a 'generationConfig' key."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request())
        assert "generationConfig" in payload

    def test_max_output_tokens_in_generation_config(self) -> None:
        """generationConfig.maxOutputTokens equals request.max_tokens."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(max_tokens=512))
        assert payload["generationConfig"]["maxOutputTokens"] == 512

    def test_temperature_in_generation_config(self) -> None:
        """generationConfig.temperature equals request.temperature."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(temperature=0.3))
        assert payload["generationConfig"]["temperature"] == 0.3

    def test_stop_sequences_absent_when_none(self) -> None:
        """'stopSequences' is absent from generationConfig when stop is None."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(stop=None))
        assert "stopSequences" not in payload["generationConfig"]

    def test_stop_sequences_mapped_from_stop(self) -> None:
        """request.stop is mapped to generationConfig.stopSequences."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request(stop=["END", "STOP"]))
        assert payload["generationConfig"]["stopSequences"] == ["END", "STOP"]

    def test_model_not_in_payload(self) -> None:
        """Model is encoded in the endpoint URL, not the payload body."""
        client = GeminiClient(api_key="AIza-test", model="gemini-2.0-flash")
        payload = client._build_payload(_make_request())
        assert "model" not in payload

    # --- Attachment tests ---

    def test_no_attachments_single_text_part(self) -> None:
        """Without attachments, parts contains exactly one text part."""
        client = GeminiClient(api_key="AIza-test")
        payload = client._build_payload(_make_request())
        parts = payload["contents"][0]["parts"]
        assert parts == [{"text": "Hello"}]

    def test_attachment_produces_inline_data_block(self) -> None:
        """An attachment produces an inlineData block in parts."""
        client = GeminiClient(api_key="AIza-test")
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        parts = payload["contents"][0]["parts"]
        assert parts[0].get("inlineData") is not None

    def test_attachment_mime_type_propagated(self) -> None:
        """inlineData mimeType matches the Attachment's media_type."""
        client = GeminiClient(api_key="AIza-test")
        att = Attachment(content=b"img", media_type="image/jpeg")
        payload = client._build_payload(_make_request(attachments=[att]))
        mime = payload["contents"][0]["parts"][0]["inlineData"]["mimeType"]
        assert mime == "image/jpeg"

    def test_attachment_data_base64_encoded(self) -> None:
        """inlineData.data contains the base64-encoded attachment bytes."""
        client = GeminiClient(api_key="AIza-test")
        raw = b"\x89PNG\r\n"
        att = Attachment(content=raw, media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        data = payload["contents"][0]["parts"][0]["inlineData"]["data"]
        assert data == _base64.b64encode(raw).decode("ascii")

    def test_attachment_before_text_part(self) -> None:
        """inlineData blocks appear before the text part in parts."""
        client = GeminiClient(api_key="AIza-test")
        att = Attachment(content=b"img", media_type="image/png")
        payload = client._build_payload(_make_request(attachments=[att]))
        parts = payload["contents"][0]["parts"]
        assert "inlineData" in parts[0]
        assert "text" in parts[-1]

    def test_multiple_attachments_all_present(self) -> None:
        """Multiple attachments each produce a separate inlineData block."""
        client = GeminiClient(api_key="AIza-test")
        atts = [
            Attachment(content=b"a", media_type="image/png"),
            Attachment(content=b"b", media_type="image/jpeg"),
        ]
        payload = client._build_payload(_make_request(attachments=atts))
        parts = payload["contents"][0]["parts"]
        inline_parts = [p for p in parts if "inlineData" in p]
        assert len(inline_parts) == 2

    def test_parts_length_with_attachments(self) -> None:
        """parts has len(attachments) + 1 entries (images + text)."""
        client = GeminiClient(api_key="AIza-test")
        atts = [Attachment(content=bytes([i]), media_type="image/png") for i in range(3)]
        payload = client._build_payload(_make_request(attachments=atts))
        parts = payload["contents"][0]["parts"]
        assert len(parts) == 4  # 3 images + 1 text


# ---------------------------------------------------------------------------
# TestParseResponse
# ---------------------------------------------------------------------------


def _make_gemini_response(
    text: str = "Hello",
    model_version: str = "gemini-2.0-flash",
    prompt_tokens: int = 10,
    candidates_tokens: int = 5,
) -> dict:  # type: ignore[type-arg]
    """Return a minimal Gemini generateContent response dict."""
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": prompt_tokens + candidates_tokens,
        },
        "modelVersion": model_version,
    }


class TestParseResponse:
    """GeminiClient._parse_response (task 4.3.3)."""

    def test_returns_agent_response(self) -> None:
        """_parse_response returns an AgentResponse instance."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(_make_gemini_response())
        assert isinstance(result, AgentResponse)

    def test_content_from_candidates_parts_text(self) -> None:
        """content is extracted from candidates[0].content.parts[0].text."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(_make_gemini_response(text="Hi there!"))
        assert result.content == "Hi there!"

    def test_model_from_model_version(self) -> None:
        """model is taken from the 'modelVersion' field."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(_make_gemini_response(model_version="gemini-1.5-pro"))
        assert result.model == "gemini-1.5-pro"

    def test_model_fallback_to_self_model(self) -> None:
        """model falls back to self._model when 'modelVersion' is absent."""
        client = GeminiClient(api_key="AIza-test", model="gemini-1.5-flash")
        data = _make_gemini_response()
        del data["modelVersion"]
        result = client._parse_response(data)
        assert result.model == "gemini-1.5-flash"

    def test_input_tokens_from_prompt_token_count(self) -> None:
        """input_tokens is read from usageMetadata.promptTokenCount."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(_make_gemini_response(prompt_tokens=42))
        assert result.input_tokens == 42

    def test_output_tokens_from_candidates_token_count(self) -> None:
        """output_tokens is read from usageMetadata.candidatesTokenCount."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(_make_gemini_response(candidates_tokens=17))
        assert result.output_tokens == 17

    def test_input_tokens_default_zero_when_usage_absent(self) -> None:
        """input_tokens defaults to 0 when usageMetadata is absent."""
        client = GeminiClient(api_key="AIza-test")
        data = _make_gemini_response()
        del data["usageMetadata"]
        result = client._parse_response(data)
        assert result.input_tokens == 0

    def test_output_tokens_default_zero_when_usage_absent(self) -> None:
        """output_tokens defaults to 0 when usageMetadata is absent."""
        client = GeminiClient(api_key="AIza-test")
        data = _make_gemini_response()
        del data["usageMetadata"]
        result = client._parse_response(data)
        assert result.output_tokens == 0

    def test_tokens_default_zero_when_fields_missing(self) -> None:
        """Token counts default to 0 when usageMetadata fields are absent."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(
            {**_make_gemini_response(), "usageMetadata": {}}
        )
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_total_tokens_property(self) -> None:
        """total_tokens equals input_tokens + output_tokens."""
        client = GeminiClient(api_key="AIza-test")
        result = client._parse_response(
            _make_gemini_response(prompt_tokens=20, candidates_tokens=30)
        )
        assert result.total_tokens == 50
