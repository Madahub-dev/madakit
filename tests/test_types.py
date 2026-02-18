from __future__ import annotations

from mada_modelkit._types import Attachment


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
