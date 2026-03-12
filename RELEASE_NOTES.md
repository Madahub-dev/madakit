# madakit 1.0.0 Released

**March 12, 2026** — We're excited to announce the first stable release of madakit, a composable AI client library for Python.

## What is madakit?

madakit provides a unified interface for working with AI providers — whether you're using OpenAI, Anthropic, local models via Ollama, or running inference with Transformers. It's designed to be:

- **Composable** — stack middleware like LEGO bricks
- **Provider-agnostic** — one interface for 21 providers
- **Zero-dependency** — core library uses only Python stdlib
- **Type-safe** — full mypy strict mode support
- **Production-ready** — retry, circuit breakers, caching, metrics, and more

## Highlights

### 21 Providers

Work with any AI provider through a single `BaseAgentClient` interface:

- **Cloud:** OpenAI, Anthropic, Gemini, DeepSeek, Cohere, Mistral, Together, Groq, Fireworks, Replicate
- **Local servers:** Ollama, vLLM, LocalAI, llama.cpp server, LM Studio, Jan, GPT4All
- **Native:** Transformers, llama.cpp Python bindings
- **Specialized:** Stability AI (images), ElevenLabs (TTS), Embeddings

### 16 Middleware

Enhance any provider with production middleware:

- **Reliability:** Retry, Circuit Breaker, Fallback
- **Performance:** Caching, Load Balancing, Batching
- **Observability:** Logging, Metrics, Tracking
- **Control:** Timeout, Rate Limiting, Cost Control
- **Intelligence:** A/B Testing, Consensus, Content Filtering, Prompt Templates, Function Calling, Stream Aggregation

### Framework Integrations

Drop-in integrations for:

- **LangChain** — LLM wrapper with callback support
- **LlamaIndex** — LLM and embedding wrappers
- **FastAPI** — dependency injection and streaming
- **Flask** — extension class and helpers

### Developer Experience

- **Configuration files** — YAML/JSON with environment variable substitution
- **Scaffolding CLI** — generate provider, middleware, and test boilerplate
- **Testing utilities** — enhanced mocks, assertion helpers, pytest fixtures
- **Migration tools** — migrate from LangChain with automated code conversion

## Quick Example

```python
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient
from madakit.middleware import RetryMiddleware, CachingMiddleware

# Create provider
provider = OpenAIClient(api_key="sk-...")

# Wrap with middleware
client = RetryMiddleware(
    client=CachingMiddleware(
        client=provider,
        ttl_seconds=3600
    ),
    max_attempts=3
)

# Send request
request = AgentRequest(prompt="Explain asyncio in one sentence")
response = await client.send_request(request)
print(response.content)
```

## Installation

```bash
# Core library (zero dependencies)
pip install madakit

# With cloud providers
pip install madakit[cloud]

# With local server providers
pip install madakit[local]

# With native providers (Transformers, llama.cpp)
pip install madakit[native]

# Everything
pip install madakit[all]
```

## Documentation

Comprehensive documentation is available:

- **[Architecture Guide](https://madakit.readthedocs.io/en/latest/architecture/)** — layer design, ABC contract, design patterns
- **[User Guide](https://madakit.readthedocs.io/en/latest/user-guide/)** — installation, provider selection, middleware configuration
- **[API Reference](https://madakit.readthedocs.io/en/latest/api-reference/)** — complete API documentation
- **[Tutorial](https://madakit.readthedocs.io/en/latest/tutorial/)** — quickstart, recipes, advanced patterns
- **[Extension Guide](https://madakit.readthedocs.io/en/latest/extension-guide/)** — building custom providers and middleware

## Architecture

madakit is built on three core principles:

1. **Zero core dependencies** — types, errors, base client, and all middleware use only Python's standard library
2. **Provider agnostic** — middleware operates through the ABC contract only, never coupled to specific providers
3. **Composable by design** — every middleware implements `BaseAgentClient`, making them infinitely stackable

```
┌─────────────────────────────────────────┐
│         Your Application                │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  LoggingMiddleware                      │
│    ↓                                    │
│  TimeoutMiddleware                      │
│    ↓                                    │
│  TrackingMiddleware                     │
│    ↓                                    │
│  RetryMiddleware                        │
│    ↓                                    │
│  CircuitBreakerMiddleware               │
│    ↓                                    │
│  CachingMiddleware                      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Provider (OpenAI, Anthropic, etc.)     │
└─────────────────────────────────────────┘
```

## Testing

madakit has been thoroughly tested with:

- **2,100+ tests** — comprehensive coverage across all components
- **Type safety** — mypy strict mode for all library code
- **Cross-platform** — CI tests on Linux, macOS, and Windows
- **Multiple Python versions** — tested on Python 3.11 and 3.12

## What's Next?

This v1.0.0 release represents a stable foundation. Future plans include:

- Additional cloud providers (Azure OpenAI, AWS Bedrock)
- Additional middleware (semantic caching, multi-step reasoning)
- Enhanced observability (OpenTelemetry integration)
- Performance optimizations (connection pooling, request batching improvements)

## Get Involved

madakit is open source and we welcome contributions!

- **GitHub:** [Madahub-dev/madakit](https://github.com/Madahub-dev/madakit)
- **Issues:** [Report bugs or request features](https://github.com/Madahub-dev/madakit/issues)
- **Contributing:** [Contribution guidelines](https://github.com/Madahub-dev/madakit/blob/main/CONTRIBUTING.md)

## Acknowledgments

Special thanks to the Python community and the open source projects that inspire madakit's design: LangChain, LlamaIndex, httpx, and the entire asyncio ecosystem.

## License

madakit is licensed under the [MIT License](https://github.com/Madahub-dev/madakit/blob/main/LICENSE).

---

**Install now:** `pip install madakit`

**Documentation:** [madakit.readthedocs.io](https://madakit.readthedocs.io)

**GitHub:** [Madahub-dev/madakit](https://github.com/Madahub-dev/madakit)
