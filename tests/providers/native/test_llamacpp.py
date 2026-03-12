"""Tests for providers/native/llamacpp.py.

Covers: LlamaCppClient constructor (task 6.1.1) — required model_path,
default n_ctx, custom n_ctx, _llm initialised to None, ThreadPoolExecutor
created with max_workers=1, semaphore creation, BaseAgentClient inheritance,
__repr__ format, and module exports.
__aenter__ model loading (task 6.1.2) — loads model via executor; deferred import.
send_request + _sync_generate (tasks 6.1.3, 6.1.4) — lazy load fallback;
executor dispatch; AgentResponse construction from llama-cpp output.
cancel (task 6.1.5) — abort flag propagation.
close (task 6.1.6) — executor shutdown; _llm cleared.
Comprehensive integration (task 6.1.7) — Mock llama_cpp.Llama end-to-end:
lazy loading, response construction, cancel, close, context manager.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from madakit._base import BaseAgentClient
from madakit._errors import ProviderError
from madakit._types import AgentRequest, AgentResponse
from madakit.providers.native.llamacpp import LlamaCppClient


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for llamacpp.py."""

    def test_llamacpp_client_in_all(self) -> None:
        """LlamaCppClient is listed in __all__."""
        from madakit.providers.native import llamacpp

        assert "LlamaCppClient" in llamacpp.__all__

    def test_llamacpp_client_importable(self) -> None:
        """LlamaCppClient can be imported directly from its module."""
        from madakit.providers.native.llamacpp import LlamaCppClient as LC

        assert LC is LlamaCppClient

    def test_llamacpp_client_is_subclass_of_base_agent_client(self) -> None:
        """LlamaCppClient inherits from BaseAgentClient."""
        assert issubclass(LlamaCppClient, BaseAgentClient)

    def test_llamacpp_client_is_not_subclass_of_http_client(self) -> None:
        """LlamaCppClient does NOT inherit from HttpAgentClient (no HTTP)."""
        from madakit.providers._http_base import HttpAgentClient

        assert not issubclass(LlamaCppClient, HttpAgentClient)


# ---------------------------------------------------------------------------
# TestLlamaCppClientConstructor
# ---------------------------------------------------------------------------


class TestLlamaCppClientConstructor:
    """LlamaCppClient constructor (task 6.1.1)."""

    def test_model_path_required(self) -> None:
        """LlamaCppClient raises TypeError when model_path is not provided."""
        with pytest.raises(TypeError):
            LlamaCppClient()  # type: ignore[call-arg]

    def test_model_path_stored(self) -> None:
        """Provided model_path is stored in _model_path."""
        client = LlamaCppClient(model_path="/models/llama.gguf")
        assert client._model_path == "/models/llama.gguf"

    def test_default_n_ctx(self) -> None:
        """Default context window size is 2048."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._n_ctx == 2048

    def test_custom_n_ctx_stored(self) -> None:
        """Custom n_ctx value is stored."""
        client = LlamaCppClient(model_path="model.gguf", n_ctx=4096)
        assert client._n_ctx == 4096

    def test_llm_is_none_after_construction(self) -> None:
        """_llm is None after construction — model loading is deferred."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None

    def test_executor_is_thread_pool_executor(self) -> None:
        """_executor is a ThreadPoolExecutor."""
        client = LlamaCppClient(model_path="model.gguf")
        assert isinstance(client._executor, ThreadPoolExecutor)

    def test_executor_has_single_worker(self) -> None:
        """The ThreadPoolExecutor is created with max_workers=1."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._executor._max_workers == 1

    def test_two_clients_have_independent_executors(self) -> None:
        """Two clients each get their own executor instance."""
        a = LlamaCppClient(model_path="model.gguf")
        b = LlamaCppClient(model_path="model.gguf")
        assert a._executor is not b._executor

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates an asyncio.Semaphore."""
        client = LlamaCppClient(model_path="model.gguf", max_concurrent=1)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._semaphore is None

    def test_different_model_paths_stored_independently(self) -> None:
        """Two clients with different model_paths store them independently."""
        a = LlamaCppClient(model_path="/models/a.gguf")
        b = LlamaCppClient(model_path="/models/b.gguf")
        assert a._model_path == "/models/a.gguf"
        assert b._model_path == "/models/b.gguf"


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """LlamaCppClient.__repr__ (task 6.1.1)."""

    def test_repr_contains_model_path(self) -> None:
        """repr contains the model_path."""
        client = LlamaCppClient(model_path="/models/llama.gguf")
        assert "/models/llama.gguf" in repr(client)

    def test_repr_contains_n_ctx(self) -> None:
        """repr contains the n_ctx value."""
        client = LlamaCppClient(model_path="model.gguf", n_ctx=4096)
        assert "4096" in repr(client)

    def test_repr_exact_format_defaults(self) -> None:
        """repr matches the expected format with default n_ctx."""
        client = LlamaCppClient(model_path="model.gguf")
        assert repr(client) == "LlamaCppClient(model_path='model.gguf', n_ctx=2048)"

    def test_repr_exact_format_custom(self) -> None:
        """repr reflects custom model_path and n_ctx."""
        client = LlamaCppClient(model_path="/opt/models/llama.gguf", n_ctx=4096)
        assert repr(client) == (
            "LlamaCppClient(model_path='/opt/models/llama.gguf', n_ctx=4096)"
        )

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = LlamaCppClient(model_path="model.gguf")
        assert isinstance(repr(client), str)


# ---------------------------------------------------------------------------
# TestAenter
# ---------------------------------------------------------------------------


class TestAenter:
    """LlamaCppClient.__aenter__ model loading (task 6.1.2)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the client instance."""
        client = LlamaCppClient(model_path="model.gguf")
        with patch.object(client, "_load_model", return_value=MagicMock()):
            result = await client.__aenter__()
        assert result is client

    @pytest.mark.asyncio
    async def test_aenter_sets_llm(self) -> None:
        """_llm is set to a non-None value after __aenter__."""
        client = LlamaCppClient(model_path="model.gguf")
        mock_llm = MagicMock()
        with patch.object(client, "_load_model", return_value=mock_llm):
            await client.__aenter__()
        assert client._llm is mock_llm

    @pytest.mark.asyncio
    async def test_aenter_calls_load_model_once(self) -> None:
        """__aenter__ calls _load_model exactly once."""
        client = LlamaCppClient(model_path="model.gguf")
        mock_load = MagicMock(return_value=MagicMock())
        with patch.object(client, "_load_model", mock_load):
            await client.__aenter__()
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_aenter_llm_was_none_before(self) -> None:
        """_llm is None before __aenter__ is called."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None

    @pytest.mark.asyncio
    async def test_aenter_via_context_manager(self) -> None:
        """async with sets _llm via __aenter__."""
        with patch.object(
            LlamaCppClient, "_load_model", return_value=MagicMock()
        ):
            async with LlamaCppClient(model_path="model.gguf") as client:
                assert client._llm is not None

    @pytest.mark.asyncio
    async def test_aenter_dispatches_via_executor(self) -> None:
        """__aenter__ dispatches _load_model through run_in_executor."""
        client = LlamaCppClient(model_path="model.gguf")
        call_thread_ids: list[int] = []
        import threading

        def load_and_record() -> MagicMock:
            """Record the thread id of the loader call."""
            call_thread_ids.append(threading.current_thread().ident or 0)
            return MagicMock()

        with patch.object(client, "_load_model", side_effect=load_and_record):
            await client.__aenter__()
        # The load ran in a worker thread, not the main asyncio thread.
        assert call_thread_ids[0] != threading.main_thread().ident

    def test_deferred_import_not_at_module_level(self) -> None:
        """llama_cpp is not imported when llamacpp.py is imported."""
        import sys

        # llamacpp module is importable without llama_cpp installed.
        import madakit.providers.native.llamacpp  # noqa: F401

        assert "llama_cpp" not in sys.modules


# ---------------------------------------------------------------------------
# TestSendRequest
# ---------------------------------------------------------------------------


def _loaded_client() -> LlamaCppClient:
    """Return a LlamaCppClient with _llm pre-set to a MagicMock (already loaded)."""
    client = LlamaCppClient(model_path="model.gguf")
    client._llm = MagicMock()
    return client


def _fake_response() -> AgentResponse:
    """Return a minimal AgentResponse for use as a mock return value."""
    return AgentResponse(
        content="hello", model="model.gguf", input_tokens=5, output_tokens=3
    )


# ---------------------------------------------------------------------------
# TestSyncGenerate
# ---------------------------------------------------------------------------


def _llm_output(
    text: str = "response text",
    prompt_tokens: int = 8,
    completion_tokens: int = 4,
) -> dict:
    """Return a minimal llama-cpp-python output dict."""
    return {
        "choices": [{"text": text}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class TestSyncGenerate:
    """LlamaCppClient._sync_generate (task 6.1.4)."""

    def test_sync_generate_returns_agent_response(self) -> None:
        """_sync_generate returns an AgentResponse."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert isinstance(result, AgentResponse)

    def test_sync_generate_content_from_choices(self) -> None:
        """AgentResponse.content is taken from choices[0].text."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output(text="The answer is 42"))
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert result.content == "The answer is 42"

    def test_sync_generate_model_is_model_path(self) -> None:
        """AgentResponse.model is set to the _model_path string."""
        client = LlamaCppClient(model_path="/opt/llama.gguf")
        client._llm = MagicMock(return_value=_llm_output())
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert result.model == "/opt/llama.gguf"

    def test_sync_generate_input_tokens_from_usage(self) -> None:
        """input_tokens is read from usage.prompt_tokens."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output(prompt_tokens=12))
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert result.input_tokens == 12

    def test_sync_generate_output_tokens_from_usage(self) -> None:
        """output_tokens is read from usage.completion_tokens."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output(completion_tokens=7))
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert result.output_tokens == 7

    def test_sync_generate_tokens_default_to_zero_when_absent(self) -> None:
        """Token counts default to 0 when usage is absent from the output."""
        client = _loaded_client()
        client._llm = MagicMock(
            return_value={"choices": [{"text": "hi"}]}
        )
        result = client._sync_generate(AgentRequest(prompt="Hi"))
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_sync_generate_passes_prompt(self) -> None:
        """The prompt string is passed as the first positional argument to _llm."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        client._sync_generate(AgentRequest(prompt="Tell me a story"))
        assert client._llm.call_args[0][0] == "Tell me a story"

    def test_sync_generate_passes_max_tokens(self) -> None:
        """max_tokens from the request is forwarded to _llm."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        client._sync_generate(AgentRequest(prompt="Hi", max_tokens=512))
        assert client._llm.call_args[1]["max_tokens"] == 512

    def test_sync_generate_passes_temperature(self) -> None:
        """temperature from the request is forwarded to _llm."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        client._sync_generate(AgentRequest(prompt="Hi", temperature=0.5))
        assert client._llm.call_args[1]["temperature"] == 0.5

    def test_sync_generate_passes_stop_list(self) -> None:
        """stop sequences from the request are forwarded to _llm."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        client._sync_generate(AgentRequest(prompt="Hi", stop=["END", "\n"]))
        assert client._llm.call_args[1]["stop"] == ["END", "\n"]

    def test_sync_generate_converts_none_stop_to_empty_list(self) -> None:
        """When request.stop is None, _llm receives stop=[] not stop=None."""
        client = _loaded_client()
        client._llm = MagicMock(return_value=_llm_output())
        client._sync_generate(AgentRequest(prompt="Hi"))
        assert client._llm.call_args[1]["stop"] == []


# ---------------------------------------------------------------------------
# TestCancel
# ---------------------------------------------------------------------------


class TestCancel:
    """LlamaCppClient.cancel (task 6.1.5)."""

    @pytest.mark.asyncio
    async def test_cancel_is_noop_when_llm_is_none(self) -> None:
        """cancel() does not raise when _llm is None (model not loaded)."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None
        await client.cancel()  # must not raise

    @pytest.mark.asyncio
    async def test_cancel_is_noop_when_llm_has_no_abort(self) -> None:
        """cancel() does not raise when _llm lacks an abort attribute."""
        client = _loaded_client()
        client._llm = MagicMock(spec=[])  # no attributes at all
        await client.cancel()  # must not raise

    @pytest.mark.asyncio
    async def test_cancel_calls_abort_when_available(self) -> None:
        """cancel() calls _llm.abort() when _llm has an abort method."""
        client = _loaded_client()
        client._llm = MagicMock()  # MagicMock has all attributes by default
        await client.cancel()
        client._llm.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_does_not_call_abort_when_llm_none(self) -> None:
        """abort() is never called when _llm is None."""
        client = LlamaCppClient(model_path="model.gguf")
        # No way to verify a call on None; just assert no AttributeError raised.
        await client.cancel()

    @pytest.mark.asyncio
    async def test_cancel_callable_multiple_times(self) -> None:
        """cancel() can be called multiple times without error."""
        client = _loaded_client()
        await client.cancel()
        await client.cancel()
        await client.cancel()


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    """LlamaCppClient.close (task 6.1.6)."""

    @pytest.mark.asyncio
    async def test_close_sets_llm_to_none(self) -> None:
        """close() sets _llm to None, releasing the model reference."""
        client = _loaded_client()
        assert client._llm is not None
        await client.close()
        assert client._llm is None

    @pytest.mark.asyncio
    async def test_close_shuts_down_executor(self) -> None:
        """close() calls executor.shutdown(wait=False)."""
        client = _loaded_client()
        with patch.object(client._executor, "shutdown") as mock_shutdown:
            await client.close()
        mock_shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test_close_is_noop_when_llm_already_none(self) -> None:
        """close() does not raise when _llm is already None."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None
        await client.close()  # must not raise

    @pytest.mark.asyncio
    async def test_close_callable_multiple_times(self) -> None:
        """close() can be called multiple times without error."""
        client = _loaded_client()
        await client.close()
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager_calls_close(self) -> None:
        """__aexit__ invokes close(), setting _llm to None."""
        with patch.object(LlamaCppClient, "_load_model", return_value=MagicMock()):
            async with LlamaCppClient(model_path="model.gguf") as client:
                assert client._llm is not None
            assert client._llm is None


class TestSendRequest:
    """LlamaCppClient.send_request (task 6.1.3)."""

    @pytest.mark.asyncio
    async def test_send_request_returns_agent_response(self) -> None:
        """send_request returns an AgentResponse."""
        client = _loaded_client()
        with patch.object(client, "_sync_generate", return_value=_fake_response()):
            result = await client.send_request(AgentRequest(prompt="Hi"))
        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_send_request_content_from_sync_generate(self) -> None:
        """AgentResponse.content matches what _sync_generate returned."""
        client = _loaded_client()
        with patch.object(client, "_sync_generate", return_value=_fake_response()):
            result = await client.send_request(AgentRequest(prompt="Hi"))
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_send_request_calls_sync_generate(self) -> None:
        """send_request calls _sync_generate exactly once with the request."""
        client = _loaded_client()
        mock_gen = MagicMock(return_value=_fake_response())
        with patch.object(client, "_sync_generate", mock_gen):
            request = AgentRequest(prompt="Test")
            await client.send_request(request)
        mock_gen.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_send_request_lazy_loads_when_llm_is_none(self) -> None:
        """send_request calls __aenter__ to load the model when _llm is None."""
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None
        with patch.object(client, "_load_model", return_value=MagicMock()):
            with patch.object(client, "_sync_generate", return_value=_fake_response()):
                await client.send_request(AgentRequest(prompt="Hi"))
        assert client._llm is not None

    @pytest.mark.asyncio
    async def test_send_request_does_not_reload_when_llm_set(self) -> None:
        """send_request does not call _load_model when _llm is already loaded."""
        client = _loaded_client()
        mock_load = MagicMock(return_value=MagicMock())
        with patch.object(client, "_load_model", mock_load):
            with patch.object(client, "_sync_generate", return_value=_fake_response()):
                await client.send_request(AgentRequest(prompt="Hi"))
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_request_wraps_exception_as_provider_error(self) -> None:
        """Exceptions from _sync_generate are wrapped as ProviderError."""
        client = _loaded_client()
        with patch.object(
            client, "_sync_generate", side_effect=RuntimeError("inference failed")
        ):
            with pytest.raises(ProviderError, match="inference failed"):
                await client.send_request(AgentRequest(prompt="Hi"))

    @pytest.mark.asyncio
    async def test_send_request_dispatches_to_executor(self) -> None:
        """_sync_generate is run in a worker thread via the executor."""
        import threading

        client = _loaded_client()
        call_thread_ids: list[int] = []

        def recording_gen(request: AgentRequest) -> AgentResponse:
            """Record thread id and return a fake response."""
            call_thread_ids.append(threading.current_thread().ident or 0)
            return _fake_response()

        with patch.object(client, "_sync_generate", side_effect=recording_gen):
            await client.send_request(AgentRequest(prompt="Hi"))
        assert call_thread_ids[0] != threading.main_thread().ident


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests for LlamaCppClient with mocked llama_cpp.Llama."""

    @pytest.mark.asyncio
    async def test_full_pipeline_via_context_manager(self) -> None:
        """Context manager loads the model and send_request returns a correct response."""
        mock_llm = MagicMock(return_value=_llm_output(text="Paris"))
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="/models/geo.gguf") as client:
                response = await client.send_request(
                    AgentRequest(prompt="Capital of France?")
                )
        assert response.content == "Paris"
        assert response.model == "/models/geo.gguf"

    @pytest.mark.asyncio
    async def test_lazy_load_on_first_send_request(self) -> None:
        """send_request triggers lazy model loading when called without __aenter__."""
        mock_llm = MagicMock(return_value=_llm_output(text="lazy"))
        client = LlamaCppClient(model_path="model.gguf")
        assert client._llm is None
        with patch.object(client, "_load_model", return_value=mock_llm):
            response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.content == "lazy"
        assert client._llm is not None

    @pytest.mark.asyncio
    async def test_model_loaded_only_once_across_requests(self) -> None:
        """Multiple send_request calls reuse the same loaded model without re-loading."""
        mock_llm = MagicMock(return_value=_llm_output())
        mock_load = MagicMock(return_value=mock_llm)
        client = LlamaCppClient(model_path="model.gguf")
        with patch.object(client, "_load_model", mock_load):
            await client.send_request(AgentRequest(prompt="First"))
            await client.send_request(AgentRequest(prompt="Second"))
            await client.send_request(AgentRequest(prompt="Third"))
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_counts_propagated(self) -> None:
        """input_tokens, output_tokens, and total_tokens are correctly propagated."""
        mock_llm = MagicMock(
            return_value=_llm_output(prompt_tokens=20, completion_tokens=10)
        )
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="model.gguf") as client:
                response = await client.send_request(AgentRequest(prompt="Count me"))
        assert response.input_tokens == 20
        assert response.output_tokens == 10
        assert response.total_tokens == 30

    @pytest.mark.asyncio
    async def test_cancel_calls_abort_on_loaded_model(self) -> None:
        """cancel() calls abort() on the loaded model in the end-to-end flow."""
        mock_llm = MagicMock(return_value=_llm_output())
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="model.gguf") as client:
                await client.cancel()
        mock_llm.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_releases_model_reference(self) -> None:
        """After close(), _llm is None (model reference released)."""
        mock_llm = MagicMock(return_value=_llm_output())
        client = LlamaCppClient(model_path="model.gguf")
        with patch.object(client, "_load_model", return_value=mock_llm):
            await client.__aenter__()
            assert client._llm is not None
            await client.close()
        assert client._llm is None

    @pytest.mark.asyncio
    async def test_context_manager_closes_on_exception(self) -> None:
        """__aexit__ calls close() and sets _llm to None even when an exception is raised."""
        mock_llm = MagicMock(return_value=_llm_output())
        client_ref: list[LlamaCppClient] = []
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            with pytest.raises(ValueError, match="inner error"):
                async with LlamaCppClient(model_path="model.gguf") as client:
                    client_ref.append(client)
                    raise ValueError("inner error")
        assert client_ref[0]._llm is None

    @pytest.mark.asyncio
    async def test_send_request_stream_yields_full_content(self) -> None:
        """send_request_stream (inherited default) yields one final chunk with full content."""
        mock_llm = MagicMock(return_value=_llm_output(text="streamed content"))
        chunks: list = []
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="model.gguf") as client:
                async for chunk in client.send_request_stream(
                    AgentRequest(prompt="Stream me")
                ):
                    chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].delta == "streamed content"
        assert chunks[0].is_final is True

    @pytest.mark.asyncio
    async def test_provider_error_on_inference_failure(self) -> None:
        """ProviderError is raised when llm inference raises an exception."""
        mock_llm = MagicMock(side_effect=RuntimeError("GPU OOM"))
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="model.gguf") as client:
                with pytest.raises(ProviderError, match="GPU OOM"):
                    await client.send_request(AgentRequest(prompt="Crash"))

    @pytest.mark.asyncio
    async def test_stacked_with_retry_middleware(self) -> None:
        """LlamaCppClient works correctly when wrapped with RetryMiddleware."""
        from madakit.middleware.retry import RetryMiddleware

        mock_llm = MagicMock(return_value=_llm_output(text="retried response"))
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="model.gguf") as llama_client:
                retry_client = RetryMiddleware(llama_client, max_retries=1)
                response = await retry_client.send_request(AgentRequest(prompt="Hi"))
        assert response.content == "retried response"

    @pytest.mark.asyncio
    async def test_model_path_in_response(self) -> None:
        """AgentResponse.model reflects the _model_path exactly."""
        mock_llm = MagicMock(return_value=_llm_output())
        with patch.object(LlamaCppClient, "_load_model", return_value=mock_llm):
            async with LlamaCppClient(model_path="/very/specific/path.gguf") as client:
                response = await client.send_request(AgentRequest(prompt="Hi"))
        assert response.model == "/very/specific/path.gguf"