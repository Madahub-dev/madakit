"""Tests for providers/native/transformers.py.

Covers: TransformersClient constructor (task 6.2.1) — required model_name,
default device, custom device, _model and _tokenizer initialised to None,
ThreadPoolExecutor with max_workers=1, _stop_flag initialised to False,
semaphore support, BaseAgentClient inheritance, __repr__ format, and module
exports.
__aenter__ model + tokenizer loading (task 6.2.2) — loads both via executor;
returns self; deferred import; both None before aenter.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from mada_modelkit._base import BaseAgentClient
from mada_modelkit.providers.native.transformers import TransformersClient


# ---------------------------------------------------------------------------
# TestModuleExports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Module-level export contract for transformers.py."""

    def test_transformers_client_in_all(self) -> None:
        """TransformersClient is listed in __all__."""
        from mada_modelkit.providers.native import transformers

        assert "TransformersClient" in transformers.__all__

    def test_transformers_client_importable(self) -> None:
        """TransformersClient can be imported directly from its module."""
        from mada_modelkit.providers.native.transformers import (
            TransformersClient as TC,
        )

        assert TC is TransformersClient

    def test_transformers_client_is_subclass_of_base_agent_client(self) -> None:
        """TransformersClient inherits from BaseAgentClient."""
        assert issubclass(TransformersClient, BaseAgentClient)

    def test_transformers_client_is_not_subclass_of_http_client(self) -> None:
        """TransformersClient does NOT inherit from HttpAgentClient (no HTTP)."""
        from mada_modelkit.providers._http_base import HttpAgentClient

        assert not issubclass(TransformersClient, HttpAgentClient)


# ---------------------------------------------------------------------------
# TestTransformersClientConstructor
# ---------------------------------------------------------------------------


class TestTransformersClientConstructor:
    """TransformersClient constructor (task 6.2.1)."""

    def test_model_name_required(self) -> None:
        """TransformersClient raises TypeError when model_name is not provided."""
        with pytest.raises(TypeError):
            TransformersClient()  # type: ignore[call-arg]

    def test_model_name_stored(self) -> None:
        """Provided model_name is stored in _model_name."""
        client = TransformersClient(model_name="gpt2")
        assert client._model_name == "gpt2"

    def test_default_device_is_auto(self) -> None:
        """Default device is 'auto'."""
        client = TransformersClient(model_name="gpt2")
        assert client._device == "auto"

    def test_custom_device_stored(self) -> None:
        """Custom device value is stored in _device."""
        client = TransformersClient(model_name="gpt2", device="cuda:0")
        assert client._device == "cuda:0"

    def test_model_is_none_after_construction(self) -> None:
        """_model is None after construction — model loading is deferred."""
        client = TransformersClient(model_name="gpt2")
        assert client._model is None

    def test_tokenizer_is_none_after_construction(self) -> None:
        """_tokenizer is None after construction — tokenizer loading is deferred."""
        client = TransformersClient(model_name="gpt2")
        assert client._tokenizer is None

    def test_stop_flag_is_false_after_construction(self) -> None:
        """_stop_flag is False after construction."""
        client = TransformersClient(model_name="gpt2")
        assert client._stop_flag is False

    def test_executor_is_thread_pool_executor(self) -> None:
        """_executor is a ThreadPoolExecutor."""
        client = TransformersClient(model_name="gpt2")
        assert isinstance(client._executor, ThreadPoolExecutor)

    def test_executor_has_single_worker(self) -> None:
        """The ThreadPoolExecutor is created with max_workers=1."""
        client = TransformersClient(model_name="gpt2")
        assert client._executor._max_workers == 1

    def test_two_clients_have_independent_executors(self) -> None:
        """Two clients each get their own executor instance."""
        a = TransformersClient(model_name="gpt2")
        b = TransformersClient(model_name="gpt2")
        assert a._executor is not b._executor

    def test_max_concurrent_creates_semaphore(self) -> None:
        """max_concurrent kwarg creates an asyncio.Semaphore."""
        client = TransformersClient(model_name="gpt2", max_concurrent=2)
        assert isinstance(client._semaphore, asyncio.Semaphore)

    def test_no_semaphore_by_default(self) -> None:
        """_semaphore is None when max_concurrent is not set."""
        client = TransformersClient(model_name="gpt2")
        assert client._semaphore is None

    def test_different_model_names_stored_independently(self) -> None:
        """Two clients with different model_names store them independently."""
        a = TransformersClient(model_name="gpt2")
        b = TransformersClient(model_name="meta-llama/Llama-2-7b-hf")
        assert a._model_name == "gpt2"
        assert b._model_name == "meta-llama/Llama-2-7b-hf"


# ---------------------------------------------------------------------------
# TestRepr
# ---------------------------------------------------------------------------


class TestRepr:
    """TransformersClient.__repr__ (task 6.2.1)."""

    def test_repr_contains_model_name(self) -> None:
        """repr contains the model_name."""
        client = TransformersClient(model_name="gpt2")
        assert "gpt2" in repr(client)

    def test_repr_contains_device(self) -> None:
        """repr contains the device value."""
        client = TransformersClient(model_name="gpt2", device="cuda:0")
        assert "cuda:0" in repr(client)

    def test_repr_exact_format_defaults(self) -> None:
        """repr matches the expected format with default device."""
        client = TransformersClient(model_name="gpt2")
        assert repr(client) == "TransformersClient(model_name='gpt2', device='auto')"

    def test_repr_exact_format_custom(self) -> None:
        """repr reflects custom model_name and device."""
        client = TransformersClient(
            model_name="meta-llama/Llama-2-7b-hf", device="cuda:0"
        )
        assert repr(client) == (
            "TransformersClient(model_name='meta-llama/Llama-2-7b-hf', device='cuda:0')"
        )

    def test_repr_is_string(self) -> None:
        """repr returns a str."""
        client = TransformersClient(model_name="gpt2")
        assert isinstance(repr(client), str)

    def test_deferred_import_not_at_module_level(self) -> None:
        """transformers is not imported when transformers.py is imported."""
        import sys

        import mada_modelkit.providers.native.transformers  # noqa: F401

        assert "transformers" not in sys.modules


# ---------------------------------------------------------------------------
# TestAenter
# ---------------------------------------------------------------------------


class TestAenter:
    """TransformersClient.__aenter__ model + tokenizer loading (task 6.2.2)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self) -> None:
        """__aenter__ returns the client instance."""
        client = TransformersClient(model_name="gpt2")
        mock_model, mock_tok = MagicMock(), MagicMock()
        with patch.object(client, "_load_model", return_value=(mock_model, mock_tok)):
            result = await client.__aenter__()
        assert result is client

    @pytest.mark.asyncio
    async def test_aenter_sets_model(self) -> None:
        """_model is set to the value returned by _load_model after __aenter__."""
        client = TransformersClient(model_name="gpt2")
        mock_model, mock_tok = MagicMock(), MagicMock()
        with patch.object(client, "_load_model", return_value=(mock_model, mock_tok)):
            await client.__aenter__()
        assert client._model is mock_model

    @pytest.mark.asyncio
    async def test_aenter_sets_tokenizer(self) -> None:
        """_tokenizer is set to the value returned by _load_model after __aenter__."""
        client = TransformersClient(model_name="gpt2")
        mock_model, mock_tok = MagicMock(), MagicMock()
        with patch.object(client, "_load_model", return_value=(mock_model, mock_tok)):
            await client.__aenter__()
        assert client._tokenizer is mock_tok

    @pytest.mark.asyncio
    async def test_aenter_calls_load_model_once(self) -> None:
        """__aenter__ calls _load_model exactly once."""
        client = TransformersClient(model_name="gpt2")
        mock_load = MagicMock(return_value=(MagicMock(), MagicMock()))
        with patch.object(client, "_load_model", mock_load):
            await client.__aenter__()
        mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_aenter_model_was_none_before(self) -> None:
        """_model is None before __aenter__ is called."""
        client = TransformersClient(model_name="gpt2")
        assert client._model is None

    @pytest.mark.asyncio
    async def test_aenter_tokenizer_was_none_before(self) -> None:
        """_tokenizer is None before __aenter__ is called."""
        client = TransformersClient(model_name="gpt2")
        assert client._tokenizer is None

    @pytest.mark.asyncio
    async def test_aenter_via_context_manager(self) -> None:
        """async with sets both _model and _tokenizer via __aenter__."""
        mock_model, mock_tok = MagicMock(), MagicMock()
        with patch.object(
            TransformersClient, "_load_model", return_value=(mock_model, mock_tok)
        ):
            async with TransformersClient(model_name="gpt2") as client:
                assert client._model is mock_model
                assert client._tokenizer is mock_tok

    @pytest.mark.asyncio
    async def test_aenter_dispatches_via_executor(self) -> None:
        """__aenter__ dispatches _load_model through run_in_executor (worker thread)."""
        client = TransformersClient(model_name="gpt2")
        call_thread_ids: list[int] = []

        def load_and_record() -> tuple[MagicMock, MagicMock]:
            """Record the thread id and return mock model + tokenizer."""
            call_thread_ids.append(threading.current_thread().ident or 0)
            return MagicMock(), MagicMock()

        with patch.object(client, "_load_model", side_effect=load_and_record):
            await client.__aenter__()
        assert call_thread_ids[0] != threading.main_thread().ident
