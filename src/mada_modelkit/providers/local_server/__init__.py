"""Local server provider implementations for mada-modelkit.

Contains HTTP clients for locally-running inference servers (Ollama, vLLM,
LocalAI). No API keys required. TLS is not enforced. All providers use the
OpenAI-compatible wire format via OpenAICompatMixin.
"""
