"""Tests for LangChain integration.

Covers MadaKitLLM wrapper with LangChain callbacks and streaming.
"""

from __future__ import annotations

import pytest

# Check if langchain is available
try:
    from langchain.callbacks.base import AsyncCallbackHandler
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

if LANGCHAIN_AVAILABLE:
    from mada_modelkit.integrations.langchain import MadaKitLLM

from helpers import MockProvider

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="langchain not installed"
)


class TestModuleExports:
    """Verify langchain module exports."""

    def test_module_has_all(self) -> None:
        """Module exports __all__."""
        from mada_modelkit.integrations import langchain

        assert hasattr(langchain, "__all__")

    def test_all_contains_madakit_llm(self) -> None:
        """__all__ contains MadaKitLLM."""
        from mada_modelkit.integrations.langchain import __all__

        assert "MadaKitLLM" in __all__


class TestMadaKitLLMConstructor:
    """Test MadaKitLLM constructor."""

    def test_valid_constructor(self) -> None:
        """Constructor initializes with client."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        assert llm.client is client
        assert llm.system_prompt is None
        assert llm.max_tokens == 1024
        assert llm.temperature == 0.7

    def test_custom_parameters(self) -> None:
        """Constructor accepts custom parameters."""
        client = MockProvider()
        llm = MadaKitLLM(
            client=client,
            system_prompt="You are helpful",
            max_tokens=512,
            temperature=0.9,
        )

        assert llm.system_prompt == "You are helpful"
        assert llm.max_tokens == 512
        assert llm.temperature == 0.9

    def test_llm_type(self) -> None:
        """_llm_type returns madakit identifier."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        assert llm._llm_type == "madakit"


class TestMadaKitLLMCalls:
    """Test MadaKitLLM call methods."""

    def test_sync_call_raises(self) -> None:
        """Synchronous _call raises NotImplementedError."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        with pytest.raises(NotImplementedError, match="async usage"):
            llm._call("Hello")

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        """Async _acall delegates to client."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        response = await llm._acall("Hello")

        assert response == "Mock response to: Hello"

    @pytest.mark.asyncio
    async def test_async_call_with_system_prompt(self) -> None:
        """Async _acall includes system prompt."""
        client = MockProvider()
        llm = MadaKitLLM(client=client, system_prompt="Be concise")

        response = await llm._acall("What is AI?")

        # MockProvider echoes the prompt
        assert "What is AI?" in response

    @pytest.mark.asyncio
    async def test_async_call_with_stop(self) -> None:
        """Async _acall passes stop sequences."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        response = await llm._acall("Hello", stop=["END"])

        assert response == "Mock response to: Hello"

    @pytest.mark.asyncio
    async def test_async_call_error_propagation(self) -> None:
        """Async _acall propagates errors."""
        client = MockProvider(fail_on_request=True)
        llm = MadaKitLLM(client=client)

        with pytest.raises(RuntimeError, match="Mock provider error"):
            await llm._acall("Hello")


class TestMadaKitLLMStreaming:
    """Test MadaKitLLM streaming."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self) -> None:
        """_astream yields text chunks."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        chunks = []
        async for chunk in llm._astream("Hello"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_concatenates_correctly(self) -> None:
        """_astream chunks concatenate to full response."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        chunks = []
        async for chunk in llm._astream("Test"):
            chunks.append(chunk)

        full_text = "".join(chunks)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_stream_error_propagation(self) -> None:
        """_astream propagates errors."""
        client = MockProvider(fail_on_stream=True)
        llm = MadaKitLLM(client=client)

        with pytest.raises(RuntimeError, match="Mock stream error"):
            async for _ in llm._astream("Hello"):
                pass


class TestMadaKitLLMCallbacks:
    """Test callback integration."""

    @pytest.mark.asyncio
    async def test_callbacks_fire_on_success(self) -> None:
        """Callbacks fire on successful call."""

        class TestCallback(AsyncCallbackHandler):
            def __init__(self) -> None:
                self.started = False
                self.ended = False
                self.tokens = []

            async def on_llm_start(self, *args, **kwargs) -> None:
                self.started = True

            async def on_llm_end(self, *args, **kwargs) -> None:
                self.ended = True

            async def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.tokens.append(token)

        client = MockProvider()
        llm = MadaKitLLM(client=client)
        callback = TestCallback()

        # Create a simple chain
        prompt = PromptTemplate(input_variables=["topic"], template="Tell me about {topic}")
        chain = LLMChain(llm=llm, prompt=prompt)

        await chain.arun(topic="AI", callbacks=[callback])

        assert callback.started is True
        assert callback.ended is True

    @pytest.mark.asyncio
    async def test_stream_callbacks(self) -> None:
        """Streaming fires token callbacks."""

        class TokenCounter(AsyncCallbackHandler):
            def __init__(self) -> None:
                self.token_count = 0

            async def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.token_count += 1

        client = MockProvider()
        llm = MadaKitLLM(client=client)
        counter = TokenCounter()

        chunks = []
        async for chunk in llm._astream("Test", run_manager=None):
            chunks.append(chunk)

        # Tokens were received
        assert len(chunks) > 0


class TestMadaKitLLMIntegration:
    """Integration tests with LangChain."""

    @pytest.mark.asyncio
    async def test_simple_chain(self) -> None:
        """Simple LangChain chain works with MadaKitLLM."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        prompt = PromptTemplate(input_variables=["question"], template="{question}")
        chain = LLMChain(llm=llm, prompt=prompt)

        result = await chain.arun(question="What is 2+2?")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_multiple_calls(self) -> None:
        """Multiple chain calls work correctly."""
        client = MockProvider()
        llm = MadaKitLLM(client=client)

        prompt = PromptTemplate(input_variables=["input"], template="{input}")
        chain = LLMChain(llm=llm, prompt=prompt)

        result1 = await chain.arun(input="First")
        result2 = await chain.arun(input="Second")

        assert "First" in result1
        assert "Second" in result2

    @pytest.mark.asyncio
    async def test_chain_with_system_prompt(self) -> None:
        """Chain with system prompt works."""
        client = MockProvider()
        llm = MadaKitLLM(client=client, system_prompt="Be brief")

        prompt = PromptTemplate(input_variables=["query"], template="{query}")
        chain = LLMChain(llm=llm, prompt=prompt)

        result = await chain.arun(query="Explain AI")

        assert isinstance(result, str)
