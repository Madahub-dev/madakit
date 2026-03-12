"""Tests for LlamaIndex integration.

Covers MadaKitLLM and MadaKitEmbedding wrappers.
"""

from __future__ import annotations

import pytest

# Check if llama-index is available
try:
    from llama_index.core.llms import ChatMessage

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

if LLAMAINDEX_AVAILABLE:
    from madakit.integrations.llamaindex import MadaKitLLM, MadaKitEmbedding

from helpers import MockProvider

pytestmark = pytest.mark.skipif(
    not LLAMAINDEX_AVAILABLE, reason="llama-index not installed"
)


class TestModuleExports:
    """Verify llamaindex module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from madakit.integrations import llamaindex

        assert hasattr(llamaindex, "__all__")

    def test_all_contains_madakit_llm(self) -> None:
        """__all__ contains MadaKitLLM."""
        from madakit.integrations.llamaindex import __all__

        assert "MadaKitLLM" in __all__

    def test_all_contains_madakit_embedding(self) -> None:
        """__all__ contains MadaKitEmbedding."""
        from madakit.integrations.llamaindex import __all__

        assert "MadaKitEmbedding" in __all__


class TestMadaKitLLMConstructor:
    """Test MadaKitLLM constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with client."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        assert llm.client is client

    def test_metadata_property(self) -> None:
        """metadata property returns LLM info."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        metadata = llm.metadata

        assert metadata["model"] == "madakit"
        assert metadata["max_tokens"] == 1024
        assert metadata["temperature"] == 0.7


class TestMadaKitLLMCompletion:
    """Test MadaKitLLM completion methods."""

    def test_sync_complete_raises(self) -> None:
        """Sync complete raises NotImplementedError."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        with pytest.raises(NotImplementedError, match="async methods"):
            llm.complete("Hello")

    @pytest.mark.asyncio
    async def test_async_complete(self) -> None:
        """Async acomplete delegates to client."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        response = await llm.acomplete("Hello")

        assert response.text == "Mock response to: Hello"

    @pytest.mark.asyncio
    async def test_async_complete_with_system_prompt(self) -> None:
        """Async acomplete includes system prompt."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)
        llm.system_prompt = "Be concise"

        response = await llm.acomplete("What is AI?")

        assert "What is AI?" in response.text


class TestMadaKitLLMChat:
    """Test MadaKitLLM chat methods."""

    def test_sync_chat_raises(self) -> None:
        """Sync chat raises NotImplementedError."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        messages = [ChatMessage(role="user", content="Hello")]

        with pytest.raises(NotImplementedError, match="async methods"):
            llm.chat(messages)

    @pytest.mark.asyncio
    async def test_async_chat(self) -> None:
        """Async achat delegates to client."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        messages = [ChatMessage(role="user", content="Hello")]
        response = await llm.achat(messages)

        assert response.message.role == "assistant"
        assert "Hello" in response.message.content

    @pytest.mark.asyncio
    async def test_async_chat_with_system_message(self) -> None:
        """Async achat handles system messages."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        messages = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="What is AI?"),
        ]
        response = await llm.achat(messages)

        assert response.message.role == "assistant"

    @pytest.mark.asyncio
    async def test_async_chat_with_conversation(self) -> None:
        """Async achat handles multi-turn conversations."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        messages = [
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
        ]
        response = await llm.achat(messages)

        assert response.message.role == "assistant"


class TestMadaKitEmbeddingConstructor:
    """Test MadaKitEmbedding constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with client."""
        client = MockProvider()
        embedding = MadaKitEmbedding(client=client)

        assert embedding.client is client


class TestMadaKitEmbeddingMethods:
    """Test MadaKitEmbedding embedding methods."""

    def test_sync_query_embedding_raises(self) -> None:
        """Sync query embedding raises NotImplementedError."""
        client = MockProvider()
        embedding = MadaKitEmbedding(client=client)

        with pytest.raises(NotImplementedError, match="async methods"):
            embedding._get_query_embedding("test")

    def test_sync_text_embedding_raises(self) -> None:
        """Sync text embedding raises NotImplementedError."""
        client = MockProvider()
        embedding = MadaKitEmbedding(client=client)

        with pytest.raises(NotImplementedError, match="async methods"):
            embedding._get_text_embedding("test")

    @pytest.mark.asyncio
    async def test_async_query_embedding(self) -> None:
        """Async query embedding returns vector."""
        # Create a mock provider that returns JSON embedding
        client = MockProvider()
        # Override the response to return JSON array
        client._response_content = "[0.1, 0.2, 0.3]"

        embedding = MadaKitEmbedding(client=client)
        vector = await embedding._aget_query_embedding("test query")

        assert isinstance(vector, list)
        assert len(vector) == 3
        assert vector == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_async_text_embedding(self) -> None:
        """Async text embedding returns vector."""
        client = MockProvider()
        client._response_content = "[0.5, 0.6]"

        embedding = MadaKitEmbedding(client=client)
        vector = await embedding._aget_text_embedding("test text")

        assert isinstance(vector, list)
        assert len(vector) == 2
        assert vector == [0.5, 0.6]


class TestMadaKitLLMIntegration:
    """Integration tests for LlamaIndex."""

    @pytest.mark.asyncio
    async def test_multiple_completions(self) -> None:
        """Multiple completions work correctly."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        response1 = await llm.acomplete("First")
        response2 = await llm.acomplete("Second")

        assert "First" in response1.text
        assert "Second" in response2.text

    @pytest.mark.asyncio
    async def test_chat_and_complete_together(self) -> None:
        """Chat and complete can be used together."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        completion = await llm.acomplete("Complete this")
        messages = [ChatMessage(role="user", content="Chat this")]
        chat = await llm.achat(messages)

        assert completion.text
        assert chat.message.content

    @pytest.mark.asyncio
    async def test_embedding_multiple_texts(self) -> None:
        """Embedding multiple texts works."""
        client = MockProvider()
        client._response_content = "[0.1, 0.2]"

        embedding = MadaKitEmbedding(client=client)

        vec1 = await embedding._aget_text_embedding("text 1")
        vec2 = await embedding._aget_text_embedding("text 2")

        assert vec1 == [0.1, 0.2]
        assert vec2 == [0.1, 0.2]
