"""LlamaIndex integration for mada-modelkit.

Provides LlamaIndex LLM and Embedding wrappers for mada-modelkit clients.
Requires the optional llama-index dependency.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from madakit._base import BaseAgentClient
from madakit._types import AgentRequest

if TYPE_CHECKING:
    from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse
    from llama_index.core.embeddings import BaseEmbedding

__all__ = ["MadaKitLLM", "MadaKitEmbedding"]

# Deferred import check
try:
    from llama_index.core.llms import LLM as _LLM_BASE
    from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
    from llama_index.core.embeddings import BaseEmbedding as _EMBEDDING_BASE

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    _LLM_BASE = object
    _EMBEDDING_BASE = object


class MadaKitLLM(_LLM_BASE):  # type: ignore[misc]
    """LlamaIndex LLM wrapper for mada-modelkit clients.

    Wraps any BaseAgentClient as a LlamaIndex LLM for use in
    LlamaIndex query engines, agents, and workflows.

    Raises:
        ImportError: If llama-index is not installed.
    """

    client: BaseAgentClient
    """The mada-modelkit client to wrap."""

    system_prompt: str | None = None
    """Optional system prompt."""

    max_tokens: int = 1024
    """Maximum tokens for generation."""

    temperature: float = 0.7
    """Temperature for generation."""

    def __init__(self, client: BaseAgentClient, **kwargs: Any) -> None:
        """Initialize MadaKitLLM.

        Args:
            client: The mada-modelkit client to wrap.
            **kwargs: Additional keyword arguments.

        Raises:
            ImportError: If llama-index is not installed.
        """
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex integration requires llama-index. "
                "Install with: pip install mada-modelkit[llamaindex]"
            )
        self.client = client
        super().__init__(**kwargs)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return LLM metadata."""
        return {
            "model": "madakit",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion.

        Args:
            prompt: The prompt text.
            **kwargs: Additional arguments.

        Returns:
            CompletionResponse with generated text.
        """
        request = AgentRequest(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = await self.client.send_request(request)

        return CompletionResponse(text=response.content)

    @llm_chat_callback()
    async def achat(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat.

        Args:
            messages: List of chat messages.
            **kwargs: Additional arguments.

        Returns:
            ChatResponse with generated message.
        """
        # Convert messages to prompt
        prompt_parts = []
        system_prompt = None

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                prompt_parts.append(msg.content)
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n".join(prompt_parts)

        request = AgentRequest(
            prompt=prompt,
            system_prompt=system_prompt or self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = await self.client.send_request(request)

        # Create chat message from response
        message = ChatMessage(role="assistant", content=response.content)

        return ChatResponse(message=message)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Sync completion not supported."""
        raise NotImplementedError("Use async methods: await llm.acomplete()")

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Sync chat not supported."""
        raise NotImplementedError("Use async methods: await llm.achat()")


class MadaKitEmbedding(_EMBEDDING_BASE):  # type: ignore[misc]
    """LlamaIndex Embedding wrapper for mada-modelkit embedding providers.

    Wraps an embedding provider (e.g., EmbeddingProvider) as a
    LlamaIndex embedding model.

    Raises:
        ImportError: If llama-index is not installed.
    """

    client: BaseAgentClient
    """The mada-modelkit embedding client to wrap."""

    def __init__(self, client: BaseAgentClient, **kwargs: Any) -> None:
        """Initialize MadaKitEmbedding.

        Args:
            client: The mada-modelkit embedding client.
            **kwargs: Additional keyword arguments.

        Raises:
            ImportError: If llama-index is not installed.
        """
        if not _LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex integration requires llama-index. "
                "Install with: pip install mada-modelkit[llamaindex]"
            )
        self.client = client
        super().__init__(**kwargs)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for query text.

        Args:
            query: The query text.

        Returns:
            Embedding vector as list of floats.
        """
        request = AgentRequest(prompt=query)
        response = await self.client.send_request(request)

        # Parse JSON-encoded embedding from content
        embedding = json.loads(response.content)
        return embedding

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Get embedding for text.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return await self._aget_query_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Sync query embedding not supported."""
        raise NotImplementedError("Use async methods: await embed_model.aget_query_embedding()")

    def _get_text_embedding(self, text: str) -> list[float]:
        """Sync text embedding not supported."""
        raise NotImplementedError("Use async methods: await embed_model.aget_text_embedding()")
