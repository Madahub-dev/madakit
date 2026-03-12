# madakit

**Composable AI client library — cloud, local, and native providers behind one async interface, layered with middleware.**

---

## What is madakit?

madakit is a Python library that provides a unified, composable interface for working with AI providers. Whether you're using OpenAI, Anthropic, local models via Ollama, or running inference with Transformers, madakit gives you:

- **One interface** — `BaseAgentClient` ABC that all providers implement
- **Zero core dependencies** — types, errors, and middleware use only stdlib
- **Composable middleware** — stack retry, caching, circuit breakers, and more like LEGO bricks
- **Type-safe** — full type annotations with mypy strict mode
- **Async-first** — built on asyncio with streaming support

## Key Features

### 21 Providers

Work with any AI provider through a single interface:

- **Cloud:** OpenAI, Anthropic, Gemini, DeepSeek, Cohere, Mistral, Together, Groq, Fireworks, Replicate
- **Local servers:** Ollama, vLLM, LocalAI, llama.cpp server, LM Studio, Jan, GPT4All
- **Native:** Transformers, llama.cpp Python bindings
- **Specialized:** Stability AI (images), ElevenLabs (TTS), Embeddings

### 16 Middleware

Enhance any provider with production-ready middleware:

- **Reliability:** Retry, Circuit Breaker, Fallback
- **Performance:** Caching, Load Balancing, Batching
- **Observability:** Logging, Metrics, Tracking
- **Control:** Timeout, Rate Limiting, Cost Control
- **Intelligence:** A/B Testing, Consensus, Content Filtering, Prompt Templates, Function Calling, Stream Aggregation

### Framework Integrations

- **LangChain** — drop-in LLM wrapper
- **LlamaIndex** — LLM and embedding wrappers
- **FastAPI** — dependency injection and streaming
- **Flask** — extension class and helpers

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

## Why madakit?

### Provider Agnostic

Switch between OpenAI, Anthropic, local models, and more without changing your code:

```python
# Same interface for all providers
from madakit.providers.cloud.openai import OpenAIClient
from madakit.providers.cloud.anthropic import AnthropicClient
from madakit.providers.local_server.ollama import OllamaClient

# All implement BaseAgentClient
clients = [
    OpenAIClient(api_key="..."),
    AnthropicClient(api_key="..."),
    OllamaClient(model="llama3.1")
]

for client in clients:
    response = await client.send_request(request)
```

### Composable by Design

Stack middleware to build exactly the behavior you need:

```python
from madakit.middleware import (
    LoggingMiddleware,
    TimeoutMiddleware,
    TrackingMiddleware,
    RetryMiddleware,
    CircuitBreakerMiddleware,
    CachingMiddleware,
)

# Production stack
client = LoggingMiddleware(
    client=TimeoutMiddleware(
        client=TrackingMiddleware(
            client=RetryMiddleware(
                client=CircuitBreakerMiddleware(
                    client=CachingMiddleware(
                        client=provider,
                        ttl_seconds=3600
                    ),
                    failure_threshold=5
                ),
                max_attempts=3
            )
        ),
        timeout_seconds=30.0
    )
)
```

### Zero Lock-In

- **No proprietary abstractions** — standard async/await Python
- **No hidden magic** — explicit middleware composition
- **No mandatory dependencies** — use only what you need
- **No vendor coupling** — swap providers anytime

## Next Steps

<div class="grid cards" markdown>

- :material-clock-fast: **[Quickstart](getting-started/quickstart.md)**

    Get up and running in 5 minutes

- :material-book-open-variant: **[Tutorial](tutorial.md)**

    Learn with hands-on examples

- :material-cog: **[User Guide](user-guide.md)**

    Deep dive into providers and middleware

- :material-code-braces: **[API Reference](api-reference.md)**

    Complete API documentation

</div>

## Community

- **GitHub:** [madahub/madakit](https://github.com/madahub/madakit)
- **Issues:** [Report bugs or request features](https://github.com/madahub/madakit/issues)
- **Contributing:** [Contribution guidelines](contributing.md)

## License

madakit is licensed under the [MIT License](https://github.com/madahub/madakit/blob/main/LICENSE).
