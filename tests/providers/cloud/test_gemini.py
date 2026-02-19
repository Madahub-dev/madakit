"""Tests for providers/cloud/gemini.py.

Covers: GeminiClient constructor (task 4.3.1) — default model, custom model,
api_key storage, base_url, x-goog-api-key header, TLS enforcement, timeout
forwarding, semaphore creation, dynamic _endpoint per model, and module exports.
_build_payload (task 4.3.2) — Gemini wire format: contents/parts structure,
systemInstruction top-level field, generationConfig with maxOutputTokens and
stopSequences, inlineData attachment blocks.
_parse_response (task 4.3.3) — candidates[0].content.parts[0].text extraction,
modelVersion with _model fallback, promptTokenCount/candidatesTokenCount defaults.
__repr__ (task 4.3.4) — API key redacted, model visible, exact format.
Comprehensive integration (task 4.3.5) — full round-trip via MockTransport:
dynamic endpoint routing, x-goog-api-key header, Gemini payload format,
inlineData attachment, health_check, context manager, token counts. Mock HTTP.
"""

from __future__ import annotations

import asyncio
import base64 as _base64
import json

import httpx
import pytest

from mada_modelkit._errors import ProviderError
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
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """GeminiClient.__repr__ (task 4.3.4)."""

    def test_repr_does_not_contain_api_key(self) -> None:
        """repr output does not expose the raw API key."""
        client = GeminiClient(api_key="AIza-secret")
        assert "AIza-secret" not in repr(client)

    def test_repr_contains_redacted_placeholder(self) -> None:
        """repr contains '***' in place of the API key."""
        client = GeminiClient(api_key="AIza-secret")
        assert "***" in repr(client)

    def test_repr_contains_model(self) -> None:
        """repr contains the model identifier."""
        client = GeminiClient(api_key="AIza-test", model="gemini-1.5-pro")
        assert "gemini-1.5-pro" in repr(client)

    def test_repr_exact_format_default_model(self) -> None:
        """repr matches the expected format with the default model."""
        client = GeminiClient(api_key="AIza-test")
        assert repr(client) == "GeminiClient(model='gemini-2.0-flash', api_key=***)"

    def test_repr_exact_format_custom_model(self) -> None:
        """repr reflects a custom model in the exact format."""
        client = GeminiClient(api_key="AIza-test", model="gemini-1.5-pro")
        assert repr(client) == "GeminiClient(model='gemini-1.5-pro', api_key=***)"

    def test_repr_different_keys_same_output(self) -> None:
        """Two clients with different keys produce identical repr output."""
        a = GeminiClient(api_key="key-one")
        b = GeminiClient(api_key="key-two")
        assert repr(a) == repr(b)

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = GeminiClient(api_key="AIza-test")
        assert isinstance(repr(client), str)


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


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


def _gemini_json_response(
    text: str = "Hello",
    model_version: str = "gemini-2.0-flash",
    prompt_tokens: int = 10,
    candidates_tokens: int = 5,
) -> bytes:
    """Return a minimal Gemini generateContent JSON response as bytes."""
    return json.dumps(
        {
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
    ).encode()


def _make_client(
    handler: httpx.MockTransport | None = None,
    api_key: str = "AIza-test",
    model: str = "gemini-2.0-flash",
) -> GeminiClient:
    """Return a GeminiClient with an injected MockTransport."""
    if handler is None:
        handler = httpx.MockTransport(
            lambda r: httpx.Response(200, content=_gemini_json_response())
        )
    client = GeminiClient(api_key=api_key, model=model)
    client._http_client = httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        transport=handler,
        headers={"x-goog-api-key": api_key},
    )
    return client


class TestIntegration:
    """End-to-end integration tests for GeminiClient using MockTransport."""

    async def test_send_request_returns_agent_response(self) -> None:
        """send_request returns an AgentResponse with the parsed content."""
        client = _make_client()
        result = await client.send_request(AgentRequest(prompt="Hi"))
        assert isinstance(result, AgentResponse)
        assert result.content == "Hello"

    async def test_send_request_posts_to_generate_content_endpoint(self) -> None:
        """send_request issues a POST to /models/{model}:generateContent."""
        captured: list[str] = []

        def handler(r: httpx.Request) -> httpx.Response:
            captured.append(r.url.path)
            return httpx.Response(200, content=_gemini_json_response())

        client = _make_client(httpx.MockTransport(handler), model="gemini-1.5-pro")
        await client.send_request(AgentRequest(prompt="Hi"))
        assert "gemini-1.5-pro" in captured[0]
        assert captured[0].endswith(":generateContent")

    async def test_send_request_forwards_x_goog_api_key_header(self) -> None:
        """send_request includes the x-goog-api-key header in every request."""
        captured: list[str] = []

        def handler(r: httpx.Request) -> httpx.Response:
            captured.append(r.headers.get("x-goog-api-key", ""))
            return httpx.Response(200, content=_gemini_json_response())

        client = _make_client(httpx.MockTransport(handler), api_key="AIza-mykey")
        await client.send_request(AgentRequest(prompt="Hi"))
        assert captured[0] == "AIza-mykey"

    async def test_send_request_payload_uses_gemini_format(self) -> None:
        """Payload contains 'contents' array (not 'messages') per Gemini format."""
        captured: list[dict] = []

        def handler(r: httpx.Request) -> httpx.Response:
            captured.append(json.loads(r.content))
            return httpx.Response(200, content=_gemini_json_response())

        client = _make_client(httpx.MockTransport(handler))
        await client.send_request(
            AgentRequest(prompt="Hello", system_prompt="Be helpful")
        )
        payload = captured[0]
        assert "contents" in payload
        assert "systemInstruction" in payload
        assert "messages" not in payload

    async def test_send_request_with_attachment_sends_inline_data(self) -> None:
        """When request has an attachment, payload parts contain an inlineData block."""
        captured: list[dict] = []

        def handler(r: httpx.Request) -> httpx.Response:
            captured.append(json.loads(r.content))
            return httpx.Response(200, content=_gemini_json_response())

        client = _make_client(httpx.MockTransport(handler))
        att = Attachment(content=b"imgdata", media_type="image/png")
        await client.send_request(AgentRequest(prompt="Describe", attachments=[att]))
        parts = captured[0]["contents"][0]["parts"]
        assert any("inlineData" in p for p in parts)

    async def test_send_request_raises_provider_error_on_non_2xx(self) -> None:
        """send_request wraps a non-2xx response as ProviderError with status_code."""
        client = _make_client(
            httpx.MockTransport(lambda r: httpx.Response(403, content=b"Forbidden"))
        )
        with pytest.raises(ProviderError) as exc_info:
            await client.send_request(AgentRequest(prompt="Hi"))
        assert exc_info.value.status_code == 403

    async def test_health_check_returns_true_on_200(self) -> None:
        """health_check returns True when the server responds with 200."""
        client = _make_client(
            httpx.MockTransport(lambda r: httpx.Response(200, content=b"{}"))
        )
        assert await client.health_check() is True

    async def test_health_check_returns_false_on_connect_error(self) -> None:
        """health_check returns False on ConnectError."""
        def handler(r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _make_client(httpx.MockTransport(handler))
        assert await client.health_check() is False

    async def test_context_manager_closes_http_client(self) -> None:
        """Using GeminiClient as a context manager closes the httpx client."""
        async with GeminiClient(api_key="AIza-test") as client:
            is_closed_inside = client._http_client.is_closed
        assert not is_closed_inside
        assert client._http_client.is_closed

    async def test_token_counts_flow_to_agent_response(self) -> None:
        """Token counts from the Gemini response reach AgentResponse."""
        client = _make_client(
            httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    content=_gemini_json_response(
                        prompt_tokens=30, candidates_tokens=45
                    ),
                )
            )
        )
        result = await client.send_request(AgentRequest(prompt="Hi"))
        assert result.input_tokens == 30
        assert result.output_tokens == 45
        assert result.total_tokens == 75
