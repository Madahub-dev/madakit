# User Guide

This guide walks you through practical usage of **madakit**, from installation to advanced patterns.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Choosing a Provider](#choosing-a-provider)
4. [Configuring Middleware](#configuring-middleware)
5. [Common Patterns](#common-patterns)
6. [Configuration Files](#configuration-files)
7. [Framework Integration](#framework-integration)
8. [Best Practices](#best-practices)

---

## Installation

### Core Package (Zero Dependencies)

```bash
pip install madakit
```

Installs types, errors, ABC, and all middleware with **zero external dependencies**.

### Provider Groups

Install only what you need:

```bash
# Cloud providers (requires httpx)
pip install madakit[cloud]

# Local server providers (requires httpx)
pip install madakit[local]

# Native providers
pip install madakit[llamacpp]      # llama-cpp-python
pip install madakit[transformers]  # transformers + torch

# Metrics middleware
pip install madakit[metrics]       # prometheus-client

# Framework integrations
pip install madakit[langchain]
pip install madakit[llamaindex]
pip install madakit[fastapi]
pip install madakit[flask]

# Everything
pip install madakit[all]

# Development tools
pip install madakit[dev]
```

---

## Quick Start

### Basic Request

```python
import asyncio
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient

async def main():
    client = OpenAIClient(api_key="sk-...")

    request = AgentRequest(
        prompt="Explain Python asyncio in one sentence",
        max_tokens=100,
        temperature=0.7
    )

    response = await client.send_request(request)
    print(response.content)
    print(f"Tokens: {response.total_tokens}")

asyncio.run(main())
```

### Streaming Request

```python
async def stream_example():
    client = OpenAIClient(api_key="sk-...")

    request = AgentRequest(prompt="Write a haiku about coding")

    async for chunk in client.send_request_stream(request):
        print(chunk.delta, end="", flush=True)
        if chunk.is_final:
            print()  # Newline

asyncio.run(stream_example())
```

### With Middleware

```python
from madakit import (
    RetryMiddleware,
    CachingMiddleware,
    TrackingMiddleware,
)

async def with_middleware():
    # Build stack: outermost → innermost
    client = RetryMiddleware(
        CachingMiddleware(
            TrackingMiddleware(
                OpenAIClient(api_key="sk-...")
            ),
            ttl=3600.0  # Cache for 1 hour
        ),
        max_retries=3
    )

    request = AgentRequest(prompt="What is machine learning?")

    # First request: cache miss, tracked, retried on failure
    response = await client.send_request(request)

    # Second request: cache hit, no API call
    response2 = await client.send_request(request)

    # Check tracking stats
    tracker = client._client._client  # Unwrap to TrackingMiddleware
    stats = tracker.stats
    print(f"Total requests: {stats.total_requests}")  # 1 (cache hit doesn't count)

asyncio.run(with_middleware())
```

---

## Choosing a Provider

### Decision Matrix

| Use Case | Provider Type | Examples | When to Use |
|----------|---------------|----------|-------------|
| **Production API** | Cloud | OpenAI, Anthropic, Gemini | Best quality, scalable, pay-per-use |
| **Local Development** | Local Server | Ollama, vLLM, LM Studio | Free, private, no internet required |
| **Edge Deployment** | Native | llama.cpp, Transformers | Offline, embedded, full control |
| **Image Generation** | Specialized | StabilityAI | Non-chat use cases |
| **Text-to-Speech** | Specialized | ElevenLabs | Audio output |
| **Embeddings** | Specialized | EmbeddingProvider | Vector search, RAG |

### Cloud Providers

#### OpenAI
```python
from madakit.providers.cloud.openai import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    model="gpt-4o-mini"  # or "gpt-4o", "o1", etc.
)
```

#### Anthropic
```python
from madakit.providers.cloud.anthropic import AnthropicClient

client = AnthropicClient(
    api_key="sk-ant-...",
    model="claude-sonnet-4-6"  # or "claude-opus-4-6"
)
```

#### Google Gemini
```python
from madakit.providers.cloud.gemini import GeminiClient

client = GeminiClient(
    api_key="AIza...",
    model="gemini-2.0-flash"
)
```

#### DeepSeek
```python
from madakit.providers.cloud.deepseek import DeepSeekClient

client = DeepSeekClient(
    api_key="sk-...",
    model="deepseek-chat"
)
```

#### Other Cloud Providers
```python
from madakit.providers.cloud.cohere import CohereClient
from madakit.providers.cloud.mistral import MistralClient
from madakit.providers.cloud.together import TogetherClient
from madakit.providers.cloud.groq import GroqClient
from madakit.providers.cloud.fireworks import FireworksClient
from madakit.providers.cloud.replicate import ReplicateClient
```

### Local Server Providers

#### Ollama
```python
from madakit.providers.local_server.ollama import OllamaClient

client = OllamaClient(
    model="llama3.2",
    base_url="http://localhost:11434/v1"  # Default
)
```

#### vLLM
```python
from madakit.providers.local_server.vllm import VllmClient

client = VllmClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1"
)
```

#### LM Studio, Jan, GPT4All
```python
from madakit.providers.local_server.lmstudio import LMStudioClient
from madakit.providers.local_server.jan import JanClient
from madakit.providers.local_server.gpt4all import GPT4AllClient

# Default ports: 1234, 1337, 4891
```

### Native Providers

#### llama.cpp
```python
from madakit.providers.native.llamacpp import LlamaCppClient

async with LlamaCppClient(
    model_path="/path/to/model.gguf",
    n_ctx=2048
) as client:
    response = await client.send_request(request)
```

#### Transformers
```python
from madakit.providers.native.transformers import TransformersClient

async with TransformersClient(
    model_name="gpt2",
    device="auto"  # or "cpu", "cuda", "cuda:0"
) as client:
    response = await client.send_request(request)
```

---

## Configuring Middleware

### Resilience Stack

```python
from madakit import (
    RetryMiddleware,
    CircuitBreakerMiddleware,
    FallbackMiddleware,
    TimeoutMiddleware,
)

client = TimeoutMiddleware(
    RetryMiddleware(
        CircuitBreakerMiddleware(
            FallbackMiddleware(
                primary=OpenAIClient(api_key="..."),
                fallbacks=[AnthropicClient(api_key="...")]
            ),
            failure_threshold=5,
            recovery_timeout=60.0
        ),
        max_retries=3,
        backoff_base=1.0
    ),
    timeout_seconds=30.0
)
```

**What this does:**
1. **Timeout** (30s) — Fail requests taking too long
2. **Retry** (3 attempts, exponential backoff) — Retry transient errors
3. **Circuit Breaker** (5 failures → open for 60s) — Fail-fast when provider is down
4. **Fallback** (OpenAI → Anthropic) — Switch providers on failure

### Performance Stack

```python
from madakit import (
    CachingMiddleware,
    RateLimitMiddleware,
    LoadBalancingMiddleware,
)

# Load balance across multiple providers
providers = [
    (OpenAIClient(api_key="..."), 0.7),  # 70% traffic
    (AnthropicClient(api_key="..."), 0.3),  # 30% traffic
]

client = RateLimitMiddleware(
    CachingMiddleware(
        LoadBalancingMiddleware(
            providers=providers,
            strategy="weighted"
        ),
        ttl=3600.0,
        max_entries=1000
    ),
    requests_per_second=10.0,
    strategy="token_bucket"
)
```

### Observability Stack

```python
from madakit import (
    LoggingMiddleware,
    MetricsMiddleware,
    TrackingMiddleware,
    CostControlMiddleware,
)

def cost_fn(response):
    """Calculate cost per response."""
    input_cost = response.input_tokens * 0.0001  # $0.10 per 1M tokens
    output_cost = response.output_tokens * 0.0003  # $0.30 per 1M tokens
    return input_cost + output_cost

client = LoggingMiddleware(
    MetricsMiddleware(
        CostControlMiddleware(
            TrackingMiddleware(
                OpenAIClient(api_key="...")
            ),
            cost_fn=cost_fn,
            budget_cap=10.0,  # $10 budget
            alert_threshold=0.8,  # Alert at $8
            on_alert=lambda current, threshold: print(f"Budget alert: ${current:.2f}")
        ),
        track_labels=True  # Add model/status labels
    ),
    log_level="INFO",
    include_prompts=False  # Don't log prompts (PII)
)
```

### Advanced Stack

```python
from madakit import (
    ABTestMiddleware,
    ContentFilterMiddleware,
    PromptTemplateMiddleware,
    FunctionCallingMiddleware,
)
from madakit.tools import ToolRegistry, Tool

# A/B test two models
variants = [
    (OpenAIClient(api_key="...", model="gpt-4o-mini"), 0.5),
    (OpenAIClient(api_key="...", model="gpt-4o"), 0.5),
]

# Tool registry
registry = ToolRegistry()
registry.register(Tool(
    name="get_weather",
    description="Get current weather for a location",
    function=lambda location: f"Sunny in {location}",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
))

# Stack
client = PromptTemplateMiddleware(
    ContentFilterMiddleware(
        FunctionCallingMiddleware(
            ABTestMiddleware(
                variants=variants,
                key_fn=lambda req: req.metadata.get("user_id", "default")
            ),
            registry=registry,
            max_iterations=3
        ),
        redact_pii=True
    ),
    templates={
        "weather": "What's the weather like in {{ location }}?"
    }
)
```

---

## Common Patterns

### Pattern 1: Hedged Requests (Fast Fallback)

```python
from madakit import FallbackMiddleware

client = FallbackMiddleware(
    primary=OpenAIClient(api_key="..."),
    fallbacks=[AnthropicClient(api_key="...")],
    fast_fail_ms=500  # Start fallback after 500ms
)

# If primary doesn't respond in 500ms, start fallback in parallel
# First response wins, loser is cancelled
response = await client.send_request(request)
```

### Pattern 2: Consensus Voting

```python
from madakit.middleware.consensus import ConsensusMiddleware

client = ConsensusMiddleware(
    providers=[
        OpenAIClient(api_key="..."),
        AnthropicClient(api_key="..."),
        GeminiClient(api_key="..."),
    ],
    strategy="majority"  # or "confidence"
)

# Sends to all 3 providers, returns most common response
response = await client.send_request(request)
print(response.metadata["votes"])  # e.g., 2 out of 3
```

### Pattern 3: Streaming with Caching

```python
from madakit import CachingMiddleware

client = CachingMiddleware(OpenAIClient(api_key="..."))

# First stream: cache miss, stream-through + buffer
async for chunk in client.send_request_stream(request):
    print(chunk.delta, end="", flush=True)
# Full response cached on is_final=True

# Second stream: cache hit, single chunk
async for chunk in client.send_request_stream(request):
    print(chunk.delta, end="", flush=True)  # Prints entire response at once
```

### Pattern 4: Per-User Rate Limiting

```python
from madakit import RateLimitMiddleware

client = RateLimitMiddleware(
    OpenAIClient(api_key="..."),
    requests_per_second=10.0,
    key_fn=lambda req: req.metadata.get("user_id", "anonymous")
)

# Each user gets independent 10 req/s bucket
request = AgentRequest(
    prompt="Hello",
    metadata={"user_id": "user123"}
)
response = await client.send_request(request)
```

### Pattern 5: Multimodal Requests

```python
from madakit import AgentRequest, Attachment

# Image input
with open("chart.png", "rb") as f:
    image_data = f.read()

request = AgentRequest(
    prompt="Describe this chart",
    attachments=[
        Attachment(
            content=image_data,
            media_type="image/png",
            filename="chart.png"
        )
    ]
)

# Works with Anthropic, Gemini, OpenAI GPT-4o
client = AnthropicClient(api_key="...")
response = await client.send_request(request)
```

---

## Configuration Files

### YAML Configuration

```yaml
# config.yaml
provider:
  type: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}  # Environment variable substitution

middleware:
  - type: timeout
    params:
      timeout_seconds: 30.0

  - type: retry
    params:
      max_retries: 3
      backoff_base: 1.0

  - type: circuit_breaker
    params:
      failure_threshold: 5
      recovery_timeout: 60.0

  - type: cache
    params:
      ttl: 3600.0
      max_entries: 1000

  - type: tracking
    params:
      cost_fn: null  # Set in code
```

### Load Configuration

```python
from madakit.config import ConfigLoader

loader = ConfigLoader.from_yaml("config.yaml")
client = loader.build_stack()

# Use configured stack
response = await client.send_request(request)
```

### JSON Configuration

```json
{
  "provider": {
    "type": "anthropic",
    "model": "claude-sonnet-4-6",
    "api_key": "${ANTHROPIC_API_KEY}"
  },
  "middleware": [
    {"type": "retry", "params": {"max_retries": 3}},
    {"type": "cache", "params": {"ttl": 3600.0}}
  ]
}
```

```python
loader = ConfigLoader.from_json("config.json")
client = loader.build_stack()
```

---

## Framework Integration

### FastAPI

```python
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from madakit.integrations.fastapi import get_client, stream_response
from madakit.providers.cloud.openai import OpenAIClient

app = FastAPI()
app.state.client = OpenAIClient(api_key="...")

@app.post("/generate")
async def generate(
    prompt: str,
    client = Depends(get_client)
):
    request = AgentRequest(prompt=prompt)
    response = await client.send_request(request)
    return {"content": response.content}

@app.post("/stream")
async def stream(
    prompt: str,
    client = Depends(get_client)
):
    request = AgentRequest(prompt=prompt)
    return stream_response(client, request)
```

### Flask

```python
from flask import Flask, request, jsonify
from madakit.integrations.flask import MadaKit
from madakit.providers.cloud.openai import OpenAIClient

app = Flask(__name__)
madakit = MadaKit()

app.config["MADAKIT_CLIENT"] = OpenAIClient(api_key="...")
madakit.init_app(app)

@app.route("/generate", methods=["POST"])
async def generate():
    prompt = request.json["prompt"]
    req = AgentRequest(prompt=prompt)
    response = await madakit.client.send_request(req)
    return jsonify({"content": response.content})
```

### LangChain

```python
from madakit.integrations.langchain import MadaKitLLM
from madakit.providers.cloud.openai import OpenAIClient

llm = MadaKitLLM(
    client=OpenAIClient(api_key="..."),
    system_prompt="You are a helpful assistant",
    max_tokens=100
)

# Use in LangChain chains
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("Tell me about {topic}")
)

result = await chain.arun(topic="asyncio")
```

### LlamaIndex

```python
from madakit.integrations.llamaindex import MadaKitLLM, MadaKitEmbedding
from madakit.providers.cloud.openai import OpenAIClient

llm = MadaKitLLM(client=OpenAIClient(api_key="..."))
embed_model = MadaKitEmbedding(client=EmbeddingProvider(api_key="..."))

# Use in LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("docs").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model
)

query_engine = index.as_query_engine()
response = await query_engine.aquery("What is madakit?")
```

---

## Best Practices

### 1. **Use Context Managers**
Always use `async with` to ensure proper cleanup:

```python
async with OpenAIClient(api_key="...") as client:
    response = await client.send_request(request)
# close() called automatically
```

### 2. **Handle Errors Gracefully**
Catch specific exceptions:

```python
from madakit import RetryExhaustedError, CircuitOpenError

try:
    response = await client.send_request(request)
except RetryExhaustedError as e:
    print(f"All retries failed: {e.last_error}")
except CircuitOpenError:
    print("Circuit is open, try again later")
```

### 3. **Cache Aggressively**
Cache identical requests to save cost:

```python
from madakit import CachingMiddleware

client = CachingMiddleware(
    OpenAIClient(api_key="..."),
    ttl=3600.0,  # 1 hour
    max_entries=10000  # Large cache for high traffic
)
```

### 4. **Monitor Everything**
Use tracking and metrics in production:

```python
from madakit import TrackingMiddleware, MetricsMiddleware

client = MetricsMiddleware(
    TrackingMiddleware(OpenAIClient(api_key="..."))
)

# Expose Prometheus metrics at /metrics
```

### 5. **Set Budgets**
Prevent runaway costs:

```python
from madakit import CostControlMiddleware

client = CostControlMiddleware(
    OpenAIClient(api_key="..."),
    cost_fn=calculate_cost,
    budget_cap=100.0,  # $100/day
    alert_threshold=0.8
)
```

### 6. **Test with Local Providers**
Use Ollama or llama.cpp in development:

```python
import os

if os.getenv("ENV") == "production":
    client = OpenAIClient(api_key="...")
else:
    client = OllamaClient(model="llama3.2")  # Free, local
```

### 7. **Use Fallbacks**
Always have a backup provider:

```python
from madakit import FallbackMiddleware

client = FallbackMiddleware(
    primary=OpenAIClient(api_key="..."),
    fallbacks=[
        AnthropicClient(api_key="..."),  # Fallback 1
        DeepSeekClient(api_key="..."),   # Fallback 2
    ]
)
```

### 8. **Redact PII**
Filter sensitive data:

```python
from madakit import ContentFilterMiddleware

client = ContentFilterMiddleware(
    OpenAIClient(api_key="..."),
    redact_pii=True  # Removes emails, SSNs, credit cards
)
```

### 9. **Template Prompts**
Reuse prompt templates:

```python
from madakit import PromptTemplateMiddleware

client = PromptTemplateMiddleware(
    OpenAIClient(api_key="..."),
    templates={
        "summarize": "Summarize this text:\n\n{{ text }}",
        "translate": "Translate to {{ language }}:\n\n{{ text }}"
    }
)

request = AgentRequest(
    prompt="",  # Ignored
    metadata={
        "template_name": "summarize",
        "variables": {"text": "Long document..."}
    }
)
```

### 10. **Use Configuration Files**
Externalize configuration:

```python
# config.yaml defines provider + middleware stack
loader = ConfigLoader.from_yaml("config.yaml")
client = loader.build_stack()

# Swap providers by changing config, no code changes
```

---

## Next Steps

- [Architecture Guide](architecture.md) — Understand the layer design
- [API Reference](api-reference.md) — Complete API documentation
- [Tutorial](tutorial.md) — Hands-on examples
- [Extension Guide](extension-guide.md) — Build custom providers/middleware
