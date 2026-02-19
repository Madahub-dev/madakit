"""Cloud provider implementations for mada-modelkit.

Contains HTTP clients for external AI APIs (OpenAI, Anthropic, Gemini,
DeepSeek). All cloud providers enforce TLS and require API keys. Provider
dependencies (httpx) are optional extras imported inside each sub-module.
"""
