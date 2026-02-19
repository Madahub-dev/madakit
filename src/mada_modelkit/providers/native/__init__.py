"""Native (in-process) provider implementations for mada-modelkit.

Contains clients that load models in-process rather than over HTTP (LlamaCpp,
Transformers). No server required. Blocking inference is dispatched via a
single-thread ThreadPoolExecutor to avoid blocking the asyncio event loop.
Provider dependencies (llama-cpp-python, transformers) are optional extras
imported inside each sub-module.
"""
