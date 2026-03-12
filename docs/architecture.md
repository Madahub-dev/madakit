# Architecture Guide

**madakit** is a composable AI client library built on three foundational layers: a universal abstract interface, reusable middleware, and pluggable providers. This architecture enables you to switch between AI providers, stack resilience patterns, and compose sophisticated request pipelines—all with zero core dependencies.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Layer Architecture](#layer-architecture)
3. [The ABC Contract](#the-abc-contract)
4. [Middleware Layer](#middleware-layer)
5. [Provider Layer](#provider-layer)
6. [Request Flow](#request-flow)
7. [Type System](#type-system)
8. [Error Handling](#error-handling)
9. [Design Patterns](#design-patterns)

---

## Core Principles

### 1. **Zero Core Dependencies**
The foundation (types, errors, ABC) and all middleware use only Python's standard library. This ensures:
- Fast installation
- Minimal dependency conflicts
- Long-term stability
- Easy auditing

### 2. **Provider Agnostic**
All providers (cloud, local server, native) implement the same `BaseAgentClient` interface. Switch from OpenAI to Anthropic to llama.cpp without changing your application code.

### 3. **Composable Middleware**
Middleware wraps clients using the decorator pattern. Stack as many layers as needed:
```python
client = RetryMiddleware(
    CircuitBreakerMiddleware(
        CachingMiddleware(
            TrackingMiddleware(
                OpenAIClient(api_key="...")
            )
        )
    )
)
```

### 4. **Type Safe**
Strict type annotations with `mypy` in strict mode. All public APIs are typed using standard library `dataclasses` (no Pydantic dependency).

### 5. **Async First**
All I/O operations are `async`. Blocking providers (llama.cpp, Transformers) dispatch to thread pools via `asyncio.run_in_executor`.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────┐
│              Application Code                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         BaseAgentClient (ABC)                   │
│  • send_request(AgentRequest) → AgentResponse   │
│  • send_request_stream() → AsyncIterator        │
│  • generate(), health_check(), cancel(), close()│
└─────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌───────────────┐          ┌────────────────┐
│  Middleware   │          │   Providers    │
│   (Stacked)   │          │   (Terminal)   │
└───────────────┘          └────────────────┘
        ↓                           ↓
┌───────────────┐          ┌────────────────┐
│ • Retry       │          │ • Cloud        │
│ • Circuit     │          │   - OpenAI     │
│ • Cache       │          │   - Anthropic  │
│ • Tracking    │          │   - Gemini     │
│ • Fallback    │          │   - DeepSeek   │
│ • RateLimit   │          │   - Cohere     │
│ • CostControl │          │   - Mistral    │
│ • Timeout     │          │   - Together   │
│ • Logging     │          │   - Groq       │
│ • Metrics     │          │   - Fireworks  │
│ • ABTest      │          │   - Replicate  │
│ • Content     │          │ • Local Server │
│ • Template    │          │   - Ollama     │
│ • LoadBalance │          │   - vLLM       │
│ • Batching    │          │   - LocalAI    │
│ • Consensus   │          │   - LMStudio   │
│ • StreamAgg   │          │   - Jan        │
│ • Function    │          │   - GPT4All    │
└───────────────┘          │ • Native       │
                           │   - llama.cpp  │
                           │   - Transformers│
                           │ • Specialized  │
                           │   - StabilityAI│
                           │   - ElevenLabs │
                           │   - Embedding  │
                           └────────────────┘
```

---

## The ABC Contract

**`BaseAgentClient`** is the single interface that everything implements:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator

class BaseAgentClient(ABC):
    @abstractmethod
    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send a request and return a complete response."""
        ...

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Send a request and stream response chunks."""
        # Default implementation: call send_request, yield single chunk
        response = await self.send_request(request)
        yield StreamChunk(delta=response.content, is_final=True)

    async def health_check(self) -> bool:
        """Check if the client is healthy."""
        return True

    async def cancel(self) -> None:
        """Cancel the currently running request."""
        pass  # No-op by default

    async def close(self) -> None:
        """Close the client and release resources."""
        pass  # No-op by default
```

### Key Methods

1. **`send_request`** (abstract)
   - Required implementation
   - Takes `AgentRequest`, returns `AgentResponse`
   - For complete, buffered responses

2. **`send_request_stream`** (virtual)
   - Optional override
   - Returns `AsyncIterator[StreamChunk]`
   - For streaming/real-time responses
   - Default: calls `send_request`, yields single chunk

3. **`generate` / `generate_stream`** (convenience)
   - Build `AgentRequest` from kwargs
   - Delegate to `send_request` / `send_request_stream`

4. **`health_check`** (virtual)
   - Returns `True` by default
   - Providers can override (e.g., HTTP `GET /`)

5. **`cancel`** (virtual)
   - No-op by default
   - Native providers implement (abort flag, StoppingCriteria)

6. **`close`** (virtual)
   - No-op by default
   - Providers implement cleanup (HTTP client, thread pool, model release)

### Context Manager

All clients support `async with`:

```python
async with OpenAIClient(api_key="...") as client:
    response = await client.send_request(request)
# close() called automatically on exit
```

---

## Middleware Layer

Middleware wraps a `BaseAgentClient` and is itself a `BaseAgentClient`. This enables **decorator pattern composition**.

### Middleware Categories

#### **Resilience**
- **RetryMiddleware**: Exponential backoff retry on transient errors (429, 5xx)
- **CircuitBreakerMiddleware**: Fail-fast when provider is down, auto-recovery
- **FallbackMiddleware**: Sequential or hedged fallback to backup providers
- **TimeoutMiddleware**: Request-level timeout enforcement

#### **Performance**
- **CachingMiddleware**: TTL + LRU cache with request coalescing (singleflight)
- **BatchingMiddleware**: Collect requests, dispatch as batch
- **LoadBalancingMiddleware**: Weighted, health-based, or latency-based routing

#### **Observability**
- **TrackingMiddleware**: Token counts, latency, TTFT, cost tracking
- **LoggingMiddleware**: Structured logging with correlation IDs
- **MetricsMiddleware**: Prometheus metrics (counters, histograms, gauges)

#### **Cost & Safety**
- **CostControlMiddleware**: Budget caps, spending alerts
- **RateLimitMiddleware**: Token bucket or leaky bucket rate limiting
- **ContentFilterMiddleware**: PII redaction, safety checks

#### **Advanced**
- **ABTestMiddleware**: A/B traffic splitting with deterministic hashing
- **PromptTemplateMiddleware**: Jinja2-style template rendering
- **ConsensusMiddleware**: Multi-provider voting (majority or confidence)
- **StreamAggregationMiddleware**: Merge or race multiple streams
- **FunctionCallingMiddleware**: Automatic tool execution

### Middleware Contract

Every middleware must implement **both** `send_request` and `send_request_stream`:

```python
class ExampleMiddleware(BaseAgentClient):
    def __init__(self, client: BaseAgentClient):
        super().__init__()
        self._client = client

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        # Pre-processing
        modified_request = self._transform(request)

        # Delegate to wrapped client
        response = await self._client.send_request(modified_request)

        # Post-processing
        return self._transform_response(response)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        modified_request = self._transform(request)
        async for chunk in self._client.send_request_stream(modified_request):
            yield self._transform_chunk(chunk)
```

### Recommended Middleware Order

Stack middleware from **outermost to innermost**:

```python
client = (
    LoggingMiddleware(           # 1. Log everything
        MetricsMiddleware(        # 2. Record metrics
            TimeoutMiddleware(    # 3. Enforce timeout
                RateLimitMiddleware(  # 4. Rate limit
                    CostControlMiddleware(  # 5. Budget enforcement
                        RetryMiddleware(     # 6. Retry transient errors
                            CircuitBreakerMiddleware(  # 7. Fail-fast
                                CachingMiddleware(     # 8. Cache responses
                                    TrackingMiddleware(  # 9. Track tokens/cost
                                        FallbackMiddleware(  # 10. Fallback providers
                                            primary=OpenAIClient(...),
                                            fallbacks=[AnthropicClient(...)]
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
```

**Rationale:**
- Logging/metrics outermost → capture all activity
- Timeout/rate limit early → protect resources
- Retry/circuit breaker before cache → don't cache errors
- Tracking inside cache → don't double-count cache hits
- Provider/fallback innermost → terminal execution

---

## Provider Layer

Providers are **terminal nodes** that execute actual API calls or inference.

### Provider Categories

#### **1. Cloud Providers** (HTTP-based, TLS required)
Inherit from `HttpAgentClient` base class:

```python
class HttpAgentClient(BaseAgentClient):
    _require_tls: bool = False  # Subclasses set True for cloud

    @abstractmethod
    def _build_payload(self, request: AgentRequest) -> dict:
        """Convert AgentRequest to provider-specific JSON."""
        ...

    @abstractmethod
    def _parse_response(self, data: dict) -> AgentResponse:
        """Convert provider JSON to AgentResponse."""
        ...

    @abstractmethod
    def _endpoint(self) -> str:
        """Return API endpoint path."""
        ...
```

**OpenAI-Compatible Providers:**
- Use `OpenAICompatMixin` for standard format
- Examples: OpenAI, DeepSeek, Mistral, Together, Groq, Fireworks

**Custom Format Providers:**
- Implement `_build_payload` / `_parse_response` directly
- Examples: Anthropic, Gemini, Cohere, Replicate

#### **2. Local Server Providers** (HTTP-based, no TLS)
Same as cloud but `_require_tls = False`, no API key:
- Ollama (port 11434)
- vLLM (port 8000)
- LocalAI (port 8080)
- LM Studio (port 1234)
- Jan (port 1337)
- GPT4All (port 4891)

#### **3. Native Providers** (In-process, no HTTP)
Load models directly into Python process:

```python
class LlamaCppClient(BaseAgentClient):
    def __init__(self, model_path: str, n_ctx: int = 2048):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._llm = None  # Lazy load
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def __aenter__(self):
        # Load model in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        self._llm = await loop.run_in_executor(self._executor, self._load_model)
        return self

    def _load_model(self):
        from llama_cpp import Llama  # Deferred import
        return Llama(model_path=self._model_path, n_ctx=self._n_ctx)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        if self._llm is None:
            await self.__aenter__()  # Lazy load
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._sync_generate, request)
        )
```

**Key Pattern:**
- Deferred imports (optional dependencies)
- `ThreadPoolExecutor(max_workers=1)` for blocking I/O
- `run_in_executor` to dispatch sync code
- Lazy loading on first request or `__aenter__`

#### **4. Specialized Providers**
- **StabilityAI**: Image generation
- **ElevenLabs**: Text-to-speech
- **EmbeddingProvider**: Embedding vectors

---

## Request Flow

### Example: Cached, Tracked OpenAI Request

```python
from madakit import (
    AgentRequest,
    CachingMiddleware,
    TrackingMiddleware,
)
from madakit.providers.cloud.openai import OpenAIClient

# Build stack
provider = OpenAIClient(api_key="sk-...")
tracked = TrackingMiddleware(provider)
cached = CachingMiddleware(tracked, ttl=3600.0)

# First request (cache miss)
request = AgentRequest(
    prompt="Explain asyncio",
    max_tokens=100,
    temperature=0.7
)

response = await cached.send_request(request)
# Flow: CachingMiddleware → cache miss → TrackingMiddleware → OpenAIClient → HTTP POST
# Response cached with key hash(prompt, system_prompt, max_tokens, temperature, stop)

# Second identical request (cache hit)
response2 = await cached.send_request(request)
# Flow: CachingMiddleware → cache hit → return immediately
# TrackingMiddleware never called, token count not double-counted

# Check stats
stats = tracked.stats
print(f"Total requests: {stats.total_requests}")  # 1 (cache hit doesn't count)
print(f"Total tokens: {stats.total_input_tokens + stats.total_output_tokens}")
```

### Streaming Flow

```python
async for chunk in cached.send_request_stream(request):
    print(chunk.delta, end="", flush=True)
    if chunk.is_final:
        print()  # Newline after final chunk
```

**Cache stream-through behavior:**
- Cache miss: yield chunks immediately + buffer on side
- On `is_final=True`: write buffer to cache
- On exception: discard buffer, don't cache
- Cache hit: yield single `StreamChunk(delta=content, is_final=True)`

---

## Type System

All types are `@dataclass` classes (zero dependencies):

### **AgentRequest**
```python
@dataclass
class AgentRequest:
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    stop: list[str] | None = None
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

### **AgentResponse**
```python
@dataclass
class AgentResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```

### **StreamChunk**
```python
@dataclass
class StreamChunk:
    delta: str  # Incremental text
    is_final: bool = False  # True on last chunk
    metadata: dict[str, Any] = field(default_factory=dict)
```

### **Attachment** (multimodal)
```python
@dataclass
class Attachment:
    content: bytes  # Image/audio data
    media_type: str  # "image/png", "audio/mpeg"
    filename: str | None = None
```

### **TrackingStats**
```python
@dataclass
class TrackingStats:
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_inference_ms: float = 0.0
    total_ttft_ms: float = 0.0
    total_cost: float = 0.0

    def reset(self) -> TrackingStats:
        """Return snapshot, then zero all counters."""
        snapshot = TrackingStats(**asdict(self))
        # Zero all fields...
        return snapshot
```

---

## Error Handling

### Error Hierarchy

```
Exception
 └── AgentError (base for all madakit errors)
      ├── ProviderError (API failures, network errors)
      │    └── status_code: int | None
      └── MiddlewareError (middleware-specific errors)
           ├── CircuitOpenError
           ├── RetryExhaustedError
           │    └── last_error: Exception
           ├── BudgetExceededError
           ├── MiddlewareError (base)
           └── WorkflowError
```

### Error Handling Patterns

#### **Retry Transient Errors**
```python
from madakit import RetryMiddleware, ProviderError

client = RetryMiddleware(
    OpenAIClient(api_key="..."),
    max_retries=3,
    backoff_base=1.0  # 1s, 2s, 4s
)

try:
    response = await client.send_request(request)
except RetryExhaustedError as e:
    print(f"All retries failed. Last error: {e.last_error}")
```

#### **Circuit Breaker Fail-Fast**
```python
from madakit import CircuitBreakerMiddleware, CircuitOpenError

client = CircuitBreakerMiddleware(
    OpenAIClient(api_key="..."),
    failure_threshold=5,
    recovery_timeout=60.0
)

try:
    response = await client.send_request(request)
except CircuitOpenError:
    print("Circuit is open, failing fast")
```

#### **Fallback to Backup**
```python
from madakit import FallbackMiddleware

client = FallbackMiddleware(
    primary=OpenAIClient(api_key="..."),
    fallbacks=[
        AnthropicClient(api_key="..."),
        DeepSeekClient(api_key="..."),
    ]
)

# Tries primary, then fallbacks in order until one succeeds
response = await client.send_request(request)
```

---

## Design Patterns

### 1. **Decorator Pattern** (Middleware)
Each middleware wraps a client and exposes the same interface:
```python
client = Middleware1(Middleware2(Middleware3(Provider())))
```

### 2. **Strategy Pattern** (Pluggable Providers)
All providers implement `BaseAgentClient`, swap implementations:
```python
def create_client(provider_type: str) -> BaseAgentClient:
    if provider_type == "openai":
        return OpenAIClient(api_key="...")
    elif provider_type == "anthropic":
        return AnthropicClient(api_key="...")
    # Application code unchanged
```

### 3. **Template Method** (HttpAgentClient)
Base class defines skeleton, subclasses fill in specifics:
```python
class HttpAgentClient:
    async def send_request(self, request):
        payload = self._build_payload(request)  # Abstract
        response = await self._client.post(self._endpoint(), json=payload)
        return self._parse_response(response.json())  # Abstract
```

### 4. **Lazy Loading** (Native Providers)
Defer expensive operations until first use:
```python
async def send_request(self, request):
    if self._llm is None:
        await self.__aenter__()  # Load model on first call
    return await self._execute(request)
```

### 5. **Request Coalescing / Singleflight** (CachingMiddleware)
Multiple concurrent identical requests share a single computation:
```python
# Request A and B arrive simultaneously with same cache key
# Only one calls the wrapped client, both wait on same asyncio.Lock
# Both receive the same cached result
```

### 6. **Observer Pattern** (Callbacks)
Middleware can notify observers:
```python
CostControlMiddleware(
    client=...,
    on_alert=lambda current, threshold: print(f"Budget alert: {current}/{threshold}")
)
```

---

## Summary

**madakit's architecture enables:**

✅ **Provider Flexibility** — Switch between OpenAI, Anthropic, llama.cpp without code changes
✅ **Resilience** — Retry, circuit breaker, fallback, timeout out of the box
✅ **Performance** — Caching, batching, request coalescing, load balancing
✅ **Observability** — Logging, metrics, tracking, distributed tracing
✅ **Cost Control** — Budget caps, rate limiting, spending alerts
✅ **Composability** — Stack middleware in any order
✅ **Type Safety** — Full mypy strict mode compliance
✅ **Zero Lock-In** — Core has zero dependencies, providers are optional extras

**Next Steps:**
- [User Guide](user-guide.md) — Practical usage patterns
- [API Reference](api-reference.md) — Complete API documentation
- [Tutorial](tutorial.md) — Hands-on examples
- [Extension Guide](extension-guide.md) — Build custom providers/middleware
