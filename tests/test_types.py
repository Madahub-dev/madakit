from __future__ import annotations

import dataclasses

from mada_modelkit._types import AgentRequest, AgentResponse, Attachment, StreamChunk, TrackingStats


class TestAttachment:
    def test_construction(self) -> None:
        att = Attachment(content=b"hello", media_type="text/plain")
        assert att.content == b"hello"
        assert att.media_type == "text/plain"
        assert att.filename is None

    def test_with_filename(self) -> None:
        att = Attachment(content=b"\x89PNG", media_type="image/png", filename="logo.png")
        assert att.content == b"\x89PNG"
        assert att.media_type == "image/png"
        assert att.filename == "logo.png"

    def test_content_is_bytes(self) -> None:
        att = Attachment(content=b"", media_type="application/octet-stream")
        assert isinstance(att.content, bytes)

    def test_equality(self) -> None:
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"x", media_type="text/plain")
        assert a == b

    def test_inequality_content(self) -> None:
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"y", media_type="text/plain")
        assert a != b

    def test_inequality_media_type(self) -> None:
        a = Attachment(content=b"x", media_type="text/plain")
        b = Attachment(content=b"x", media_type="image/png")
        assert a != b

    def test_default_filename_none(self) -> None:
        att = Attachment(content=b"data", media_type="application/pdf")
        assert att.filename is None


class TestAgentRequest:
    def test_construction_minimal(self) -> None:
        req = AgentRequest(prompt="Hello")
        assert req.prompt == "Hello"

    def test_default_system_prompt_none(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.system_prompt is None

    def test_default_attachments_empty_list(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.attachments == []

    def test_default_max_tokens(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.max_tokens == 1024

    def test_default_temperature(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.temperature == 0.7

    def test_default_stop_none(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.stop is None

    def test_default_metadata_empty_dict(self) -> None:
        req = AgentRequest(prompt="p")
        assert req.metadata == {}

    def test_all_fields_set(self) -> None:
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
        r1 = AgentRequest(prompt="a")
        r2 = AgentRequest(prompt="b")
        r1.attachments.append(Attachment(content=b"x", media_type="text/plain"))
        assert r2.attachments == []

    def test_metadata_default_factory_independent(self) -> None:
        r1 = AgentRequest(prompt="a")
        r2 = AgentRequest(prompt="b")
        r1.metadata["key"] = "value"
        assert r2.metadata == {}

    def test_equality(self) -> None:
        r1 = AgentRequest(prompt="hello", max_tokens=256)
        r2 = AgentRequest(prompt="hello", max_tokens=256)
        assert r1 == r2

    def test_inequality(self) -> None:
        r1 = AgentRequest(prompt="hello")
        r2 = AgentRequest(prompt="world")
        assert r1 != r2


class TestAgentResponse:
    def test_construction(self) -> None:
        resp = AgentResponse(content="Hi", model="gpt-4o", input_tokens=10, output_tokens=5)
        assert resp.content == "Hi"
        assert resp.model == "gpt-4o"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5

    def test_default_metadata_empty_dict(self) -> None:
        resp = AgentResponse(content="Hi", model="m", input_tokens=1, output_tokens=1)
        assert resp.metadata == {}

    def test_total_tokens_property(self) -> None:
        resp = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        assert resp.total_tokens == 15

    def test_total_tokens_zero(self) -> None:
        resp = AgentResponse(content="", model="m", input_tokens=0, output_tokens=0)
        assert resp.total_tokens == 0

    def test_metadata_set(self) -> None:
        resp = AgentResponse(
            content="Hi", model="m", input_tokens=1, output_tokens=1, metadata={"finish_reason": "stop"}
        )
        assert resp.metadata["finish_reason"] == "stop"

    def test_metadata_default_factory_independent(self) -> None:
        r1 = AgentResponse(content="a", model="m", input_tokens=1, output_tokens=1)
        r2 = AgentResponse(content="b", model="m", input_tokens=1, output_tokens=1)
        r1.metadata["key"] = "value"
        assert r2.metadata == {}

    def test_equality(self) -> None:
        r1 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        r2 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        assert r1 == r2

    def test_inequality(self) -> None:
        r1 = AgentResponse(content="Hi", model="m", input_tokens=10, output_tokens=5)
        r2 = AgentResponse(content="Bye", model="m", input_tokens=10, output_tokens=5)
        assert r1 != r2


class TestStreamChunk:
    def test_construction(self) -> None:
        chunk = StreamChunk(delta="hello")
        assert chunk.delta == "hello"

    def test_default_is_final_false(self) -> None:
        chunk = StreamChunk(delta="x")
        assert chunk.is_final is False

    def test_default_metadata_empty_dict(self) -> None:
        chunk = StreamChunk(delta="x")
        assert chunk.metadata == {}

    def test_is_final_true(self) -> None:
        chunk = StreamChunk(delta="", is_final=True)
        assert chunk.is_final is True

    def test_delta_empty_string(self) -> None:
        chunk = StreamChunk(delta="")
        assert chunk.delta == ""

    def test_metadata_set(self) -> None:
        chunk = StreamChunk(delta="tok", metadata={"ttft_ms": 42.0})
        assert chunk.metadata["ttft_ms"] == 42.0

    def test_metadata_default_factory_independent(self) -> None:
        c1 = StreamChunk(delta="a")
        c2 = StreamChunk(delta="b")
        c1.metadata["key"] = "val"
        assert c2.metadata == {}

    def test_equality(self) -> None:
        c1 = StreamChunk(delta="hi", is_final=True)
        c2 = StreamChunk(delta="hi", is_final=True)
        assert c1 == c2

    def test_inequality(self) -> None:
        c1 = StreamChunk(delta="hi")
        c2 = StreamChunk(delta="bye")
        assert c1 != c2


class TestTrackingStats:
    def test_construction_defaults(self) -> None:
        stats = TrackingStats()
        assert stats.total_requests == 0
        assert stats.total_input_tokens == 0
        assert stats.total_output_tokens == 0
        assert stats.total_inference_ms == 0.0
        assert stats.total_ttft_ms == 0.0
        assert stats.total_cost_usd is None

    def test_construction_with_values(self) -> None:
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
        stats = TrackingStats(total_requests=7)
        snapshot = stats.reset()
        stats.total_requests = 99
        assert snapshot.total_requests == 7

    def test_reset_on_defaults_returns_zero_snapshot(self) -> None:
        stats = TrackingStats()
        snapshot = stats.reset()
        assert snapshot.total_requests == 0
        assert snapshot.total_cost_usd is None


class TestTypeAnnotations:
    """Verify field names and that total_tokens is a property, not a field."""

    def test_attachment_field_names(self) -> None:
        names = {f.name for f in dataclasses.fields(Attachment)}
        assert names == {"content", "media_type", "filename"}

    def test_agent_request_field_names(self) -> None:
        names = {f.name for f in dataclasses.fields(AgentRequest)}
        assert names == {"prompt", "system_prompt", "attachments", "max_tokens", "temperature", "stop", "metadata"}

    def test_agent_response_field_names(self) -> None:
        names = {f.name for f in dataclasses.fields(AgentResponse)}
        assert names == {"content", "model", "input_tokens", "output_tokens", "metadata"}

    def test_stream_chunk_field_names(self) -> None:
        names = {f.name for f in dataclasses.fields(StreamChunk)}
        assert names == {"delta", "is_final", "metadata"}

    def test_tracking_stats_field_names(self) -> None:
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
        assert isinstance(AgentResponse.total_tokens, property)

    def test_total_tokens_not_a_dataclass_field(self) -> None:
        names = {f.name for f in dataclasses.fields(AgentResponse)}
        assert "total_tokens" not in names

    def test_all_types_are_dataclasses(self) -> None:
        for cls in (Attachment, AgentRequest, AgentResponse, StreamChunk, TrackingStats):
            assert dataclasses.is_dataclass(cls)


class TestIntegration:
    """Cross-type usage patterns."""

    def test_agent_request_with_multiple_attachments(self) -> None:
        atts = [
            Attachment(content=b"\x89PNG", media_type="image/png", filename="a.png"),
            Attachment(content=b"%PDF", media_type="application/pdf"),
        ]
        req = AgentRequest(prompt="Analyse these", attachments=atts)
        assert len(req.attachments) == 2
        assert req.attachments[0].filename == "a.png"
        assert req.attachments[1].filename is None

    def test_agent_response_total_tokens_large_values(self) -> None:
        resp = AgentResponse(content="...", model="m", input_tokens=8192, output_tokens=4096)
        assert resp.total_tokens == 12288

    def test_tracking_stats_multiple_resets(self) -> None:
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
        chunk = StreamChunk(delta="", is_final=True, metadata={"finish_reason": "stop"})
        assert chunk.is_final is True
        assert chunk.metadata["finish_reason"] == "stop"
