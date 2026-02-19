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

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.providers.native.llamacpp import LlamaCppClient


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for llamacpp.py."""

    def test_llamacpp_client_in_all(self) -> None:
        """LlamaCppClient is listed in __all__."""
        from mada_modelkit.providers.native import llamacpp

        assert "LlamaCppClient" in llamacpp.__all__

    def test_llamacpp_client_importable(self) -> None:
        """LlamaCppClient can be imported directly from its module."""
        from mada_modelkit.providers.native.llamacpp import LlamaCppClient as LC

        assert LC is LlamaCppClient

    def test_llamacpp_client_is_subclass_of_base_agent_client(self) -> None:
        """LlamaCppClient inherits from BaseAgentClient."""
        assert issubclass(LlamaCppClient, BaseAgentClient)

    def test_llamacpp_client_is_not_subclass_of_http_client(self) -> None:
        """LlamaCppClient does NOT inherit from HttpAgentClient (no HTTP)."""
        from mada_modelkit.providers._http_base import HttpAgentClient

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
        import mada_modelkit.providers.native.llamacpp  # noqa: F401

        assert "llama_cpp" not in sys.modules