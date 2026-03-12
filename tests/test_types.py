"""Tests for madakit._types.

Covers all five public dataclasses (Attachment, AgentRequest, AgentResponse,
StreamChunk, TrackingStats): construction, defaults, the total_tokens property,
reset() snapshot-then-zero semantics, field-name annotations, and cross-type
integration patterns.
"""

from __future__ import annotations

import dataclasses

from madakit._types import (
    AgentRequest,
    AgentResponse,
    Attachment,
    StreamChunk,
    TrackingStats,
)


class TestAttachment:
    """Tests for the Attachment dataclass."""

    def test_construction(self) -> None:
        """Required fields are stored and filename defaults to None."""
        att = Attachment(content=b"hello", media_type="text/plain")
        assert att.content == b"hello"
        assert att.media_type == "text/plain"
        assert att.filename is None

    def test_with_filename(self) -> None:
        """Optional filename is stored when provided."""
        att = Attachment(content=b"\x89PNG", media_type="image/png", filename="logo.png")
        assert att.content == b"\x89PNG"
        assert att.media_type == "image/png"
        assert att.filename == "logo.png"

    def test_content_is_bytes(self) -> None:
        """content field holds bytes, not str."""
        att = Attachment(content=b"", media_type="application/octet-stream")
        assert isinstance(att.content, bytes)

    def test_equality(self) -> None:
        """Two Attachments with the same fields are equal."""
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"x", media_type="text/plain")
        assert a == b

    def test_inequality_content(self) -> None:
        """Different content yields inequality."""
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"y", media_type="text/plain")
        assert a != b

    def test_inequality_media_type(self) -> None:
        """Different media_type yields inequality."""
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"x", media_type="image/png")
        assert a != b

    def test_default_filename_none(self) -> None:
        """filename is None when not supplied."""
        att = Attachment(content=b"data", media_type="application/pdf")
        assert att.filename is None


class TestAgentRequest:
    """Tests for the AgentRequest dataclass."""

    def test_construction_minimal(self) -> None:
        """Only prompt is required; all other fields use defaults."""
        req = AgentRequest(prompt="Hello")
        assert req.prompt == "Hello"

    def test_default_system_prompt_none(self) -> None:
        """system_prompt defaults to None."""
        req = AgentRequest(prompt="p")
        assert req.system_prompt is None

    def test_default_attachments_empty_list(self) -> None:
        """attachments defaults to an empty list."""
        req = AgentRequest(prompt="p")
        assert req.attachments == []

    def test_default_max_tokens(self) -> None:
        """max_tokens defaults to 1024."""
        req = AgentRequest(prompt="p")
        assert req.max_tokens == 1024

    def test_default_temperature(self) -> None:
        """temperature defaults to 0.7."""
        req = AgentRequest(prompt="p")
        assert req.temperature == 0.7

    def test_default_stop_none(self) -> None:
        """stop defaults to None."""
        req = AgentRequest(prompt="p")
        assert req.stop is None

    def test_default_metadata_empty_dict(self) -> None:
        """metadata defaults to an empty dict."""
        req = AgentRequest(prompt="p")
        assert req.metadata == {}

    def test_all_fields_set(self) -> None:
        """All 7 fields are stored correctly when explicitly provided."""
        att = Attachment(content=b"\x89PNG", media_type="image/png")
        req = AgentRequest(
            prompt="describe this image",
            system_prompt="You are helpful.",
            attachments=[att],
            max_tokens=512,
            temperature=0.3,
            stop=["<end>"],
            metadata={"top_p": 0.9},
        )
        assert req.prompt == "describe this image"
        assert req.system_prompt == "You are helpful."
        assert req.attachments == [att]
        assert req.max_tokens == 512
        assert req.temperature == 0.3
        assert req.stop == ["<end>"]
        assert req.metadata == {"top_p": 0.9}

    def test_attachments_default_factory_independent(self) -> None:
        """Mutating one instance's attachments does not affect another."""
        r1 = AgentRequest(prompt="a")
        r2 = AgentRequest(prompt="b")
        r1.attachments.append(Attachment(content=b"x", media_type="text/plain"))
        assert r2.attachments == []

    def test_metadata_default_factory_independent(self) -> None:
        """Mutating one instance's metadata does not affect another."""
        r1 = AgentRequest(prompt="a")
        r2 = AgentRequest(prompt="b")
        r1.metadata["key"] = "value"
        assert r2.metadata == {}

    def test_equality(self) -> None:
        """Two AgentRequests with the same fields are equal."""
        r1 = AgentRequest(prompt="hello", max_tokens=256)
        r2 = AgentRequest(prompt="hello", max_tokens=256)
        assert r1 == r2

    def test_inequality(self) -> None:
        """Different prompts yield inequality."""
        r1 = AgentRequest(prompt="hello")
        r2 = AgentRequest(prompt="world")
        assert r1 != r2


class TestAgentResponse:
    """Tests for the AgentResponse dataclass."""

    def test_construction(self) -> None:
        """All required fields are stored correctly."""
        resp = AgentResponse(content="Hi", model="gpt-4o", input_tokens=10, output_tokens=5)
        assert resp.content == "Hi"
        assert resp.model == "gpt-4o"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    def test_default_metadata_empty_dict(self) -> None:
        """metadata defaults to an empty dict."""
        resp = AgentResponse(content="Hi", model="m", input_tokens=1, output_tokens=1)
        assert resp.metadata == {}

    def test_total_tokens_property(self) -> None:
        """total_tokens returns input_tokens + output_tokens."""
        resp = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        assert resp.total_tokens == 15

    def test_total_tokens_zero(self) -> None:
        """total_tokens returns 0 when both token counts are 0."""
        resp = AgentResponse(content="", model="m", input_tokens=0, output_tokens=0)
        assert resp.total_tokens == 0

    def test_metadata_set(self) -> None:
        """Explicitly supplied metadata is stored."""
        resp = AgentResponse(
            content="Hi",
            model="m",
            input_tokens=1,
            output_tokens=1,
            metadata={"finish_reason": "stop"},
        )
        assert resp.metadata["finish_reason"] == "stop"

    def test_metadata_default_factory_independent(self) -> None:
        """Mutating one instance's metadata does not affect another."""
        r1 = AgentResponse(content="a", model="m", input_tokens=1, output_tokens=1)
        r2 = AgentResponse(content="b", model="m", input_tokens=1, output_tokens=1)
        r1.metadata["key"] = "value"
        assert r2.metadata == {}

    def test_equality(self) -> None:
        """Two AgentResponses with the same fields are equal."""
        r1 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        r2 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        assert r1 == r2

    def test_inequality(self) -> None:
        """Different content yields inequality."""
        r1 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        r2 = AgentResponse(content="Bye", model="m", input_tokens=10, output_tokens=5)
        assert r1 != r2


class TestStreamChunk:
    """Tests for the StreamChunk dataclass."""

    def test_construction(self) -> None:
        """delta is stored correctly."""
        chunk = StreamChunk(delta="hello")
        assert chunk.delta == "hello"

    def test_default_is_final_false(self) -> None:
        """is_final defaults to False."""
        chunk = StreamChunk(delta="x")
        assert chunk.is_final is False

    def test_default_metadata_empty_dict(self) -> None:
        """metadata defaults to an empty dict."""
        chunk = StreamChunk(delta="x")
        assert chunk.metadata == {}

    def test_is_final_true(self) -> None:
        """is_final=True marks the terminal chunk."""
        chunk = StreamChunk(delta="", is_final=True)
        assert chunk.is_final is True

    def test_delta_empty_string(self) -> None:
        """delta may be an empty string (e.g. on the final chunk)."""
        chunk = StreamChunk(delta="")
        assert chunk.delta == ""

    def test_metadata_set(self) -> None:
        """Explicitly supplied metadata is stored."""
        chunk = StreamChunk(delta="tok", metadata={"ttft_ms": 42.0})
        assert chunk.metadata["ttft_ms"] == 42.0

    def test_metadata_default_factory_independent(self) -> None:
        """Mutating one instance's metadata does not affect another."""
        c1 = StreamChunk(delta="a")
        c2 = StreamChunk(delta="b")
        c1.metadata["key"] = "val"
        assert c2.metadata == {}

    def test_equality(self) -> None:
        """Two StreamChunks with the same fields are equal."""
        c1 = StreamChunk(delta="hi", is_final=True)
        c2 = StreamChunk(delta="hi", is_final=True)
        assert c1 == c2

    def test_inequality(self) -> None:
        """Different deltas yield inequality."""
        c1 = StreamChunk(delta="hi")
        c2 = StreamChunk(delta="bye")
        assert c1 != c2


class TestTrackingStats:
    """Tests for the TrackingStats dataclass and its reset() method."""

    def test_construction_defaults(self) -> None:
        """All counters start at zero and total_cost_usd is None."""
        stats = TrackingStats()
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_inference_ms == 0.0
        assert stats.total_ttft_ms == 0.0
        assert stats.total_cost_usd is None

    def test_construction_with_values(self) -> None:
        """All 6 fields are stored when explicitly provided."""
        stats = TrackingStats(
            total_requests=5,
            total_input_tokens=100,
            total_output_tokens=200,
            total_inference_ms=1500.0,
            total_ttft_ms=80.0,
            total_cost_usd=0.003,
        )
        assert stats.total_requests == 5
        assert stats.total_input_tokens == 100
        assert stats.total_output_tokens == 200
        assert stats.total_inference_ms == 1500.0
        assert stats.total_ttft_ms == 80.0
        assert stats.total_cost_usd == 0.003

    def test_reset_returns_snapshot(self) -> None:
        """reset() returns a TrackingStats with the pre-reset values."""
        stats = TrackingStats(
            total_requests=3,
            total_input_tokens=50,
            total_output_tokens=100,
            total_inference_ms=900.0,
            total_ttft_ms=45.0,
            total_cost_usd=0.001,
        )
        snapshot = stats.reset()
        assert snapshot.total_requests == 3
        assert snapshot.total_input_tokens == 50
        assert snapshot.total_output_tokens == 100
        assert snapshot.total_inference_ms == 900.0
        assert snapshot.total_ttft_ms == 45.0
        assert snapshot.total_cost_usd == 0.001

    def test_reset_zeros_original(self) -> None:
        """reset() zeros all counters on the original instance."""
        stats = TrackingStats(
            total_requests=3,
            total_input_tokens=50,
            total_output_tokens=100,
            total_inference_ms=900.0,
            total_ttft_ms=45.0,
            total_cost_usd=0.001,
        )
        stats.reset()
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_inference_ms == 0.0
        assert stats.total_ttft_ms == 0.0
        assert stats.total_cost_usd is None

    def test_reset_snapshot_is_independent(self) -> None:
        """Mutating the original after reset() does not alter the snapshot."""
        stats = TrackingStats(total_requests=7)
        snapshot = stats.reset()
        stats.total_requests = 99
        assert snapshot.total_requests == 7

    def test_reset_on_defaults_returns_zero_snapshot(self) -> None:
        """reset() on a freshly constructed instance returns an all-zero snapshot."""
        stats = TrackingStats()
        snapshot = stats.reset()
        assert snapshot.total_requests == 0
        assert snapshot.total_cost_usd is None


class TestTypeAnnotations:
    """Verify field names and that total_tokens is a property, not a field."""

    def test_attachment_field_names(self) -> None:
        """Attachment exposes exactly the three expected fields."""
        names = {f.name for f in dataclasses.fields(Attachment)}
        assert names == {"content", "media_type", "filename"}

    def test_agent_request_field_names(self) -> None:
        """AgentRequest exposes exactly the seven expected fields."""
        names = {f.name for f in dataclasses.fields(AgentRequest)}
        assert names == {
            "prompt",
            "system_prompt",
            "attachments",
            "max_tokens",
            "temperature",
            "stop",
            "metadata",
        }

    def test_agent_response_field_names(self) -> None:
        """AgentResponse exposes exactly the five expected fields (total_tokens excluded)."""
        names = {f.name for f in dataclasses.fields(AgentResponse)}
        assert names == {"content", "model", "input_tokens", "output_tokens", "metadata"}

    def test_stream_chunk_field_names(self) -> None:
        """StreamChunk exposes exactly the three expected fields."""
        names = {f.name for f in dataclasses.fields(StreamChunk)}
        assert names == {"delta", "is_final", "metadata"}

    def test_tracking_stats_field_names(self) -> None:
        """TrackingStats exposes exactly the six expected counter fields."""
        names = {f.name for f in dataclasses.fields(TrackingStats)}
        assert names == {
            "total_requests",
            "total_input_tokens",
            "total_output_tokens",
            "total_inference_ms",
            "total_ttft_ms",
            "total_cost_usd",
        }

    def test_total_tokens_is_property(self) -> None:
        """total_tokens is a @property descriptor on the class."""
        assert isinstance(AgentResponse.total_tokens, property)

    def test_total_tokens_not_a_dataclass_field(self) -> None:
        """total_tokens is not registered as a dataclass field."""
        names = {f.name for f in dataclasses.fields(AgentResponse)}
        assert "total_tokens" not in names

    def test_all_types_are_dataclasses(self) -> None:
        """All five public types are proper dataclasses."""
        for cls in (Attachment, AgentRequest, AgentResponse, StreamChunk, TrackingStats):
            assert dataclasses.is_dataclass(cls)


class TestIntegration:
    """Cross-type usage patterns."""

    def test_agent_request_with_multiple_attachments(self) -> None:
        """AgentRequest stores a heterogeneous attachment list correctly."""
        atts = [
            Attachment(content=b"\x89PNG", media_type="image/png", filename="a.png"),
            Attachment(content=b"%PDF", media_type="application/pdf"),
        ]
        req = AgentRequest(prompt="Analyse these", attachments=atts)
        assert len(req.attachments) == 2
        assert req.attachments[0].filename == "a.png"
        assert req.attachments[1].filename is None

    def test_agent_response_total_tokens_large_values(self) -> None:
        """total_tokens handles large token counts without overflow."""
        resp = AgentResponse(content="...", model="m", input_tokens=8192, output_tokens=4096)
        assert resp.total_tokens == 12288

    def test_tracking_stats_multiple_resets(self) -> None:
        """Successive reset() calls each return accurate snapshots and zero the live counters."""
        stats = TrackingStats(total_requests=10, total_input_tokens=500)
        snap1 = stats.reset()
        stats.total_requests = 3
        stats.total_input_tokens = 60
        snap2 = stats.reset()
        assert snap1.total_requests == 10
        assert snap1.total_input_tokens == 500
        assert snap2.total_requests == 3
        assert snap2.total_input_tokens == 60
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0

    def test_stream_chunk_final_with_metadata(self) -> None:
        """A terminal StreamChunk carries both is_final=True and metadata."""
        chunk = StreamChunk(delta="", is_final=True, metadata={"finish_reason": "stop"})
        assert chunk.is_final is True
        assert chunk.metadata["finish_reason"] == "stop"
