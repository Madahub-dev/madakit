# madakit

**Composable AI client library — cloud, local, and native providers behind one async interface, layered with middleware.**

[![PyPI version](https://badge.fury.io/py/madakit.svg)](https://pypi.org/project/madakit/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Features

- **Zero core dependencies** — types, errors, middleware use only stdlib
- **21 providers** — OpenAI, Anthropic, Gemini, DeepSeek, Ollama, vLLM, Transformers, and more
- **16 middleware** — retry, circuit breaker, caching, tracking, fallback, rate limiting, cost control, timeout, logging, metrics, A/B testing, content filtering, prompt templates, load balancing, batching, consensus
- **Async-first** — built on asyncio with streaming support
- **Type-safe** — full type annotations, mypy strict mode
- **Composable** — stack middleware like LEGO bricks
- **Framework integrations** — LangChain, LlamaIndex, FastAPI, Flask
- **Developer tools** — scaffolding CLI, testing utilities, migration helpers

## Installation

```bash
# Core library (zero dependencies)
pip install madakit

# With cloud providers (httpx)
pip install madakit[cloud]

# With local server providers (httpx)
pip install madakit[local]

# With native providers (transformers, llama-cpp-python)
pip install madakit[native]

# With metrics support (prometheus-client)
pip install madakit[metrics]

# Everything
pip install madakit[all]
```

## Quickstart

```python
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient
from madakit.middleware import RetryMiddleware, CachingMiddleware

# Create a provider
provider = OpenAIClient(api_key="sk-...")

# Wrap with middleware
client = RetryMiddleware(
    client=CachingMiddleware(
        client=provider,
        ttl_seconds=3600
    ),
    max_attempts=3
)

# Send a request
request = AgentRequest(prompt="Explain Python asyncio in one sentence")
response = await client.send_request(request)
print(response.content)
# "Asyncio is Python's built-in library for writing concurrent code using async/await syntax."
```

## Streaming

```python
async for chunk in client.send_request_stream(request):
    print(chunk.delta_content, end="", flush=True)
```

## Provider Examples

### Cloud Providers

```python
# OpenAI
from madakit.providers.cloud.openai import OpenAIClient
client = OpenAIClient(api_key="sk-...", model="gpt-4")

# Anthropic
from madakit.providers.cloud.anthropic import AnthropicClient
client = AnthropicClient(api_key="sk-ant-...", model="claude-3-5-sonnet-20241022")

# Google Gemini
from madakit.providers.cloud.gemini import GeminiClient
client = GeminiClient(api_key="...", model="gemini-1.5-pro")

# DeepSeek
from madakit.providers.cloud.deepseek import DeepSeekClient
client = DeepSeekClient(api_key="...", model="deepseek-chat")
```

### Local Server Providers

```python
# Ollama
from madakit.providers.local_server.ollama import OllamaClient
client = OllamaClient(model="llama3.1")

# vLLM
from madakit.providers.local_server.vllm import VLLMClient
client = VLLMClient(model="meta-llama/Llama-3.1-70B", base_url="http://localhost:8000")

# LM Studio
from madakit.providers.local_server.lmstudio import LMStudioClient
client = LMStudioClient(model="llama-3.1-8b")
```

### Native Providers

```python
# Transformers (Hugging Face)
from madakit.providers.native.transformers import TransformersClient
client = TransformersClient(model_name="meta-llama/Llama-3.2-1B-Instruct")

# llama.cpp
from madakit.providers.native.llamacpp import LlamaCppClient
client = LlamaCppClient(model_path="./models/llama-3.1-8b.gguf")
```

## Middleware Recipes

### Production Stack

```python
from madakit.middleware import (
    RetryMiddleware,
    CircuitBreakerMiddleware,
    CachingMiddleware,
    TrackingMiddleware,
    TimeoutMiddleware,
    LoggingMiddleware,
)

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

### Fallback Chain

```python
from madakit.middleware import FallbackMiddleware

client = FallbackMiddleware(
    primary=openai_client,
    fallbacks=[anthropic_client, gemini_client]
)
```

### Multi-Provider Consensus

```python
from madakit.middleware import ConsensusMiddleware

client = ConsensusMiddleware(
    providers=[openai_client, anthropic_client, gemini_client],
    strategy="majority"
)
```

## Configuration Files

```yaml
# config.yaml
provider:
  type: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

middleware:
  - type: cache
    params:
      ttl_seconds: 3600
  - type: retry
    params:
      max_attempts: 3
  - type: timeout
    params:
      timeout_seconds: 30.0
```

```python
from madakit.config import ConfigLoader

loader = ConfigLoader()
client = loader.from_yaml("config.yaml")
```

## Framework Integrations

### LangChain

```python
from madakit.integrations.langchain import MadaKitLLM

llm = MadaKitLLM(client=client)
result = await llm.apredict("What is the capital of France?")
```

### FastAPI

```python
from fastapi import FastAPI, Depends
from madakit.integrations.fastapi import get_client, stream_response

app = FastAPI()
app.state.madakit_client = client

@app.post("/chat")
async def chat(request: dict, client=Depends(get_client)):
    response = await client.send_request(AgentRequest(**request))
    return response

@app.post("/chat/stream")
async def chat_stream(request: dict, client=Depends(get_client)):
    return stream_response(client.send_request_stream(AgentRequest(**request)))
```

## Documentation

- [Architecture Guide](docs/architecture.md) — layer design, ABC contract, design patterns
- [User Guide](docs/user-guide.md) — installation, provider selection, middleware configuration
- [API Reference](docs/api-reference.md) — complete API documentation
- [Tutorial](docs/tutorial.md) — quickstart, recipes, advanced patterns
- [Extension Guide](docs/extension-guide.md) — building custom providers and middleware

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy src/
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **Documentation:** [readthedocs.io](https://madakit.readthedocs.io)
- **PyPI:** [pypi.org/project/madakit](https://pypi.org/project/madakit/)
- **GitHub:** [github.com/Madahub-dev/madakit](https://github.com/Madahub-dev/madakit)
- **Issues:** [github.com/Madahub-dev/madakit/issues](https://github.com/Madahub-dev/madakit/issues)
