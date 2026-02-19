"""HuggingFace Transformers native provider for mada-modelkit.

Provides TransformersClient — a BaseAgentClient subclass that runs
AutoModelForCausalLM inference in-process. No HTTP server required. Blocking
inference is dispatched via a single-thread ThreadPoolExecutor so it never
blocks the asyncio event loop. Cooperative cancellation is implemented via a
custom StoppingCriteria flag.

Requires the ``transformers`` optional extra
(``transformers>=4.40``, ``torch>=2.0``). The ``transformers`` and ``torch``
modules are imported lazily inside ``__aenter__`` / ``_load_model`` to keep
the zero-dep core intact.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
from mada_modelkit._types import AgentRequest, AgentResponse

__all__ = ["TransformersClient"]


class TransformersClient(BaseAgentClient):
    """In-process client that runs inference via HuggingFace Transformers.

    Loads an ``AutoModelForCausalLM`` and matching ``AutoTokenizer`` on
    ``__aenter__`` (or lazily on first ``send_request``). All blocking
    inference runs in a single-thread ``ThreadPoolExecutor`` so the asyncio
    event loop stays responsive. Cooperative cancellation is achieved by
    raising a ``StoppingCriteria`` flag checked between generation steps.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        **kwargs: object,
    ) -> None:
        """Initialise the Transformers client.

        Stores configuration only — no model or tokenizer is loaded until
        ``__aenter__`` (or the first ``send_request`` call). Creates a
        single-thread ``ThreadPoolExecutor`` for blocking inference dispatch
        and a boolean stop flag for cooperative cancellation.

        Args:
            model_name: HuggingFace model identifier (e.g. ``"gpt2"``) or
                local path to a model directory.
            device: Device map string forwarded to ``from_pretrained``.
                Defaults to ``"auto"`` (let Accelerate choose).
            **kwargs: Forwarded to ``BaseAgentClient`` (e.g. ``max_concurrent``).
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._stop_flag: bool = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __repr__(self) -> str:
        """Return a repr showing the model name and device."""
        return (
            f"TransformersClient(model_name={self._model_name!r}, device={self._device!r})"
        )

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute inference and return a complete AgentResponse.

        Loads the model lazily if ``__aenter__`` was not called. Dispatches
        ``_sync_generate`` to the single-thread executor so the event loop is
        never blocked.
        """
        if self._model is None:
            await self.__aenter__()
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self._executor, partial(self._sync_generate, request)
            )
        except Exception as exc:
            raise ProviderError(f"Transformers inference failed: {exc}") from exc

    async def __aenter__(self) -> TransformersClient:
        """Load the model and tokenizer via the executor (non-blocking).

        Returns:
            self, to support ``async with TransformersClient(...) as client:``
        """
        loop = asyncio.get_event_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            self._executor, self._load_model
        )
        return self

    def _load_model(self) -> tuple[Any, Any]:
        """Load and return ``(model, tokenizer)`` (runs in executor).

        Performs the deferred import of ``transformers`` so the zero-dep core
        remains importable without the optional extra installed.

        Returns:
            A ``(AutoModelForCausalLM, AutoTokenizer)`` pair.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name, device_map=self._device
        )
        return model, tokenizer

    def _sync_generate(self, request: AgentRequest) -> AgentResponse:
        """Run synchronous inference and return an AgentResponse (runs in executor).

        Resets the stop flag before each call. Uses the tokenizer to encode
        the prompt, runs ``model.generate`` with a ``StoppingCriteriaList``
        wrapping ``_CancelCriteria``, then decodes the output tokens.

        Args:
            request: The inference request.

        Returns:
            An ``AgentResponse`` built from the decoded generation output.
        """
        from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore[import-untyped]

        self._stop_flag = False

        class _CancelCriteria(StoppingCriteria):
            """Stopping criteria that checks the client's stop flag."""

            def __init__(self_inner) -> None:
                """Initialise _CancelCriteria."""
                super().__init__()

            def __call__(self_inner, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                """Return True when the client's stop flag is set."""
                return self._stop_flag

        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[-1]

        generate_kwargs: dict[str, Any] = {
            "stopping_criteria": StoppingCriteriaList([_CancelCriteria()]),
        }
        if request.max_tokens is not None:
            generate_kwargs["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            generate_kwargs["temperature"] = request.temperature
            generate_kwargs["do_sample"] = True

        output_ids = self._model.generate(input_ids, **generate_kwargs)
        new_tokens = output_ids[0][input_length:]
        content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        input_tokens = input_length
        output_tokens = len(new_tokens)
        return AgentResponse(
            content=content,
            model=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def cancel(self) -> None:
        """Signal the running inference to stop via the StoppingCriteria flag."""
        self._stop_flag = True

    async def close(self) -> None:
        """Shut down the executor and release the model and tokenizer references."""
        self._executor.shutdown(wait=False)
        self._model = None
        self._tokenizer = None
