from __future__ import annotations

from mada_modelkit._types import AgentRequest, AgentResponse, Attachment


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
