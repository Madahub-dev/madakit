"""LangChain integration for mada-modelkit.

Provides LangChain LLM wrapper for mada-modelkit clients.
Requires the optional langchain dependency.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, TYPE_CHECKING

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest

if TYPE_CHECKING:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

__all__ = ["MadaKitLLM"]

# Deferred import check
try:
    from langchain.llms.base import LLM as _LLM_BASE
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    _LLM_BASE = object  # Fallback base class


class MadaKitLLM(_LLM_BASE):  # type: ignore[misc]
    """LangChain LLM wrapper for mada-modelkit clients.

    Wraps any BaseAgentClient as a LangChain LLM, enabling use in
    LangChain chains, agents, and workflows.

    Raises:
        ImportError: If langchain is not installed.
    """

    client: BaseAgentClient
    """The mada-modelkit client to wrap."""

    system_prompt: str | None = None
    """Optional system prompt to include in all requests."""

    max_tokens: int = 1024
    """Maximum tokens for generation."""

    temperature: float = 0.7
    """Temperature for generation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MadaKitLLM.

        Args:
            **kwargs: Keyword arguments including client.

        Raises:
            ImportError: If langchain is not installed.
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain integration requires langchain. "
                "Install with: pip install mada-modelkit[langchain]"
            )
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "madakit"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous call (not supported).

        Args:
            prompt: The prompt text.
            stop: Stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: Synchronous calls not supported.
        """
        raise NotImplementedError(
            "MadaKitLLM requires async usage. Use await llm.agenerate() or chains."
        )

    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous call to the wrapped client.

        Args:
            prompt: The prompt text.
            stop: Stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Additional arguments.

        Returns:
            Generated text response.
        """
        # Build request
        request = AgentRequest(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop or [],
        )

        # Fire on_llm_start callback if manager present
        if run_manager:
            await run_manager.on_llm_start(
                serialized={"name": self._llm_type},
                prompts=[prompt],
            )

        try:
            # Call wrapped client
            response = await self.client.send_request(request)

            # Fire on_llm_end callback
            if run_manager:
                await run_manager.on_llm_end(response=response.content)

            return response.content

        except Exception as e:
            # Fire on_llm_error callback
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response from the wrapped client.

        Args:
            prompt: The prompt text.
            stop: Stop sequences.
            run_manager: LangChain callback manager.
            **kwargs: Additional arguments.

        Yields:
            Text chunks from the response stream.
        """
        # Build request
        request = AgentRequest(
            prompt=prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop or [],
        )

        # Fire on_llm_start callback
        if run_manager:
            await run_manager.on_llm_start(
                serialized={"name": self._llm_type},
                prompts=[prompt],
            )

        try:
            # Stream from wrapped client
            full_response = ""
            async for chunk in self.client.send_request_stream(request):
                if chunk.delta:
                    full_response += chunk.delta
                    # Fire on_llm_new_token callback
                    if run_manager:
                        await run_manager.on_llm_new_token(token=chunk.delta)
                    yield chunk.delta

            # Fire on_llm_end callback
            if run_manager:
                await run_manager.on_llm_end(response=full_response)

        except Exception as e:
            # Fire on_llm_error callback
            if run_manager:
                await run_manager.on_llm_error(e)
            raise
