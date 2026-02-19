"""llama-cpp-python native provider for mada-modelkit.

Provides LlamaCppClient — a BaseAgentClient subclass that runs llama-cpp-python
in-process. No HTTP server required. Blocking inference is dispatched via a
single-thread ThreadPoolExecutor so it never blocks the asyncio event loop.
Requires the ``llamacpp`` optional extra (``llama-cpp-python>=0.2``).
The ``llama_cpp`` module is imported lazily inside ``__aenter__`` / ``_load_model``
to keep the zero-dep core intact.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse

__all__ = ["LlamaCppClient"]


class LlamaCppClient(BaseAgentClient):
    """In-process client that runs inference via llama-cpp-python.

    Loads a GGUF model file into memory on ``__aenter__`` (or lazily on first
    ``send_request``). All blocking inference runs in a single-thread
    ``ThreadPoolExecutor`` so the asyncio event loop stays responsive.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        **kwargs: object,
    ) -> None:
        """Initialise the LlamaCpp client.

        Stores configuration only — no model is loaded until ``__aenter__``
        (or the first ``send_request`` call). Creates a single-thread
        ``ThreadPoolExecutor`` for blocking inference dispatch.

        Args:
            model_path: Path to the GGUF model file on disk.
            n_ctx: Context window size in tokens. Defaults to ``2048``.
            **kwargs: Forwarded to ``BaseAgentClient`` (e.g. ``max_concurrent``).
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._init_kwargs = kwargs
        self._llm: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __repr__(self) -> str:
        """Return a repr showing the model path and context size."""
        return f"LlamaCppClient(model_path={self._model_path!r}, n_ctx={self._n_ctx})"

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute inference and return a complete AgentResponse.

        Loads the model lazily if ``__aenter__`` was not called. Dispatches
        ``_sync_generate`` to the single-thread executor so the event loop is
        never blocked.
        """
        if self._llm is None:
            await self.__aenter__()
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, partial(self._sync_generate, request)
            )
        except Exception as exc:
            raise ProviderError(f"LlamaCpp inference failed: {exc}") from exc

    async def __aenter__(self) -> LlamaCppClient:
        """Load the GGUF model via the executor (non-blocking to the event loop).

        Returns:
            self, to support ``async with LlamaCppClient(...) as client:``
        """
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(self._executor, self._load_model)
        return self

    def _load_model(self) -> Any:
        """Load and return a ``llama_cpp.Llama`` instance (runs in executor).

        Performs the deferred import of ``llama_cpp`` so the zero-dep core
        remains importable without the optional extra installed.
        """
        from llama_cpp import Llama  # type: ignore[import-untyped]

        return Llama(model_path=self._model_path, n_ctx=self._n_ctx)

    def _sync_generate(self, request: AgentRequest) -> AgentResponse:
        """Run synchronous inference and return an AgentResponse (runs in executor).

        Args:
            request: The inference request.

        Returns:
            An ``AgentResponse`` built from the llama-cpp-python output dict.
        """
        output = self._llm(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop or [],
        )
        usage = output.get("usage", {})
        return AgentResponse(
            content=output["choices"][0]["text"],
            model=self._model_path,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def cancel(self) -> None:
        """Signal the running inference to abort (no-op if not running)."""
        if self._llm is not None and hasattr(self._llm, "abort"):
            self._llm.abort()

    async def close(self) -> None:
        """Shut down the executor and release the model reference."""
        self._executor.shutdown(wait=False)
        self._llm = None
