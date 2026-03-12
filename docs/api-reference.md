# API Reference

Complete API documentation for **madakit**.

---

## Core Types

### `AgentRequest`

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

**Fields:**
- `prompt` — Main user prompt (required)
- `system_prompt` — System/instruction prompt (optional)
- `max_tokens` — Maximum tokens to generate (default: 1024)
- `temperature` — Sampling temperature 0.0-2.0 (default: 0.7)
- `stop` — Stop sequences (optional)
- `attachments` — List of multimodal attachments (images, audio)
- `metadata` — Arbitrary metadata for middleware (escape hatch)

---

### `AgentResponse`

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

**Fields:**
- `content` — Generated text response
- `model` — Model identifier used
- `input_tokens` — Prompt token count
- `output_tokens` — Generated token count
- `metadata` — Provider-specific metadata

**Properties:**
- `total_tokens` — Sum of input + output tokens

---

### `StreamChunk`

```python
@dataclass
class StreamChunk:
    delta: str
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Fields:**
- `delta` — Incremental text chunk
- `is_final` — True on last chunk
- `metadata` — Chunk-specific metadata (e.g., TTFT)

---

### `Attachment`

```python
@dataclass
class Attachment:
    content: bytes
    media_type: str
    filename: str | None = None
```

**Fields:**
- `content` — Binary data (image, audio, etc.)
- `media_type` — MIME type ("image/png", "audio/mpeg")
- `filename` — Optional filename

---

### `TrackingStats`

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
```

---

## Core Errors

### `AgentError`

Base exception for all madakit errors.

```python
class AgentError(Exception):
    pass
```

---

### `ProviderError`

Raised by providers on API failures, network errors, HTTP errors.

```python
class ProviderError(AgentError):
    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
```

**Attributes:**
- `status_code` — HTTP status code if applicable (429, 500, etc.)

---

### `MiddlewareError`

Base exception for middleware-raised errors.

```python
class MiddlewareError(AgentError):
    pass
```

---

### `CircuitOpenError`

Raised when circuit breaker is open.

```python
class CircuitOpenError(MiddlewareError):
    pass
```

---

### `RetryExhaustedError`

Raised when all retry attempts are exhausted.

```python
class RetryExhaustedError(MiddlewareError):
    def __init__(self, message: str, last_error: Exception):
        self.last_error = last_error
```

**Attributes:**
- `last_error` — The final exception that caused retry exhaustion

---

### `BudgetExceededError`

Raised when cost budget cap is exceeded.

```python
class BudgetExceededError(MiddlewareError):
    pass
```

---

## BaseAgentClient (ABC)

```python
class BaseAgentClient(ABC):
    @abstractmethod
    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request, return complete response."""

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send request, stream response chunks."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AgentResponse:
        """Convenience: build AgentRequest and call send_request."""

    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Convenience: build AgentRequest and call send_request_stream."""

    async def health_check(self) -> bool:
        """Check if client is healthy. Default: True."""

    async def cancel(self) -> None:
        """Cancel running request. Default: no-op."""

    async def close(self) -> None:
        """Close client, release resources. Default: no-op."""

    async def __aenter__(self) -> BaseAgentClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()
```

---

## Middleware

All middleware follow this pattern:

```python
class SomeMiddleware(BaseAgentClient):
    def __init__(self, client: BaseAgentClient, **params):
        super().__init__()
        self._client = client
```

### RetryMiddleware

```python
RetryMiddleware(
    client: BaseAgentClient,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    is_retryable: Callable[[Exception], bool] | None = None
)
```

**Parameters:**
- `max_retries` — Number of retry attempts
- `backoff_base` — Base delay in seconds (exponential: base * 2^attempt)
- `is_retryable` — Custom function to determine if error is retryable

**Default retryable errors:**
- `ProviderError` with `status_code` 429 or ≥500
- `ProviderError` with `status_code=None` (network errors)

---

### CircuitBreakerMiddleware

```python
CircuitBreakerMiddleware(
    client: BaseAgentClient,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
)
```

**Parameters:**
- `failure_threshold` — Failures before circuit opens
- `recovery_timeout` — Seconds before attempting recovery

**States:**
- `closed` — Normal operation
- `open` — Fail-fast, raise `CircuitOpenError`
- `half-open` — Probe via `health_check()`, attempt recovery

---

### CachingMiddleware

```python
CachingMiddleware(
    client: BaseAgentClient,
    ttl: float = 3600.0,
    max_entries: int = 1000,
    key_fn: Callable[[AgentRequest], str] | None = None
)
```

**Parameters:**
- `ttl` — Time-to-live in seconds
- `max_entries` — Max cache size (LRU eviction)
- `key_fn` — Custom key function (default: hash of prompt/system/max_tokens/temp/stop)

**Features:**
- Request coalescing (singleflight)
- Stream-through caching (buffer on side, write on `is_final`)

---

### TrackingMiddleware

```python
TrackingMiddleware(
    client: BaseAgentClient,
    cost_fn: Callable[[AgentResponse], float] | None = None
)

# Access stats
middleware.stats  # Returns TrackingStats
```

**Parameters:**
- `cost_fn` — Function to calculate cost from response

**Tracked metrics:**
- Total requests
- Input/output/total tokens
- Inference time (ms)
- TTFT (time to first token, ms)
- Total cost (if `cost_fn` provided)

---

### FallbackMiddleware

```python
FallbackMiddleware(
    primary: BaseAgentClient,
    fallbacks: list[BaseAgentClient],
    fast_fail_ms: float | None = None
)
```

**Parameters:**
- `primary` — Primary provider
- `fallbacks` — List of fallback providers (tried in order)
- `fast_fail_ms` — If set, start first fallback after delay (hedged requests)

**Modes:**
- Sequential: try primary, then fallbacks in order
- Hedged: start fallback after delay, first response wins

---

### RateLimitMiddleware

```python
RateLimitMiddleware(
    client: BaseAgentClient,
    requests_per_second: float = 10.0,
    burst_size: int | None = None,
    strategy: str = "token_bucket",  # or "leaky_bucket"
    key_fn: Callable[[AgentRequest], str] | None = None
)
```

**Parameters:**
- `requests_per_second` — Rate limit
- `burst_size` — Max burst tokens (default: 2x rate)
- `strategy` — "token_bucket" or "leaky_bucket"
- `key_fn` — For per-user/per-endpoint limits

---

### CostControlMiddleware

```python
CostControlMiddleware(
    client: BaseAgentClient,
    cost_fn: Callable[[AgentResponse], float],
    budget_cap: float | None = None,
    alert_threshold: float = 0.8,
    on_alert: Callable[[float, float], None] | None = None
)

# Reset budget
middleware.reset_budget()

# Check spending
middleware.total_spend
```

**Parameters:**
- `cost_fn` — Calculate cost from response (required)
- `budget_cap` — Max spending (raises `BudgetExceededError`)
- `alert_threshold` — Fire alert at this fraction of cap
- `on_alert` — Callback `(current_spend, threshold_amount)`

---

### TimeoutMiddleware

```python
TimeoutMiddleware(
    client: BaseAgentClient,
    timeout_seconds: float = 30.0
)
```

**Parameters:**
- `timeout_seconds` — Request timeout (raises `asyncio.TimeoutError`)

**Streaming behavior:** Timeout applies only to first chunk arrival.

---

### LoggingMiddleware

```python
LoggingMiddleware(
    client: BaseAgentClient,
    logger: logging.Logger | None = None,
    log_level: str = "INFO",
    include_prompts: bool = False
)
```

**Parameters:**
- `logger` — Custom logger (default: creates new)
- `log_level` — "DEBUG", "INFO", "WARNING", "ERROR"
- `include_prompts` — Log prompts/system prompts (PII risk)

**Logged:**
- Request start (ID, max_tokens, temperature)
- Response completion (ID, duration, tokens, model)
- Errors (ID, exception, stack trace)

---

### MetricsMiddleware

```python
MetricsMiddleware(
    client: BaseAgentClient,
    registry: CollectorRegistry | None = None,
    prefix: str = "madakit",
    track_labels: bool = False
)
```

**Parameters:**
- `registry` — Prometheus registry
- `prefix` — Metric name prefix
- `track_labels` — Add model/status labels (higher cardinality)

**Metrics:**
- `requests_total` — Counter
- `errors_total` — Counter (label: error_type)
- `request_duration_seconds` — Histogram
- `input_tokens` — Histogram
- `output_tokens` — Histogram
- `ttft_seconds` — Histogram (streaming only)
- `active_requests` — Gauge

---

### ABTestMiddleware

```python
ABTestMiddleware(
    variants: list[tuple[BaseAgentClient, float]],
    key_fn: Callable[[AgentRequest], str] | None = None
)
```

**Parameters:**
- `variants` — List of `(client, weight)` tuples
- `key_fn` — Extract key for deterministic routing (default: hash all fields)

**Metadata added:** `variant` (int, 0-indexed)

---

### ContentFilterMiddleware

```python
ContentFilterMiddleware(
    client: BaseAgentClient,
    redact_pii: bool = True,
    safety_check: Callable[[str], None] | None = None,
    response_filter: Callable[[str], str] | None = None
)
```

**Parameters:**
- `redact_pii` — Redact emails, SSNs, credit cards with `[REDACTED]`
- `safety_check` — Validate prompt before API call (raises on unsafe)
- `response_filter` — Filter response content

---

### PromptTemplateMiddleware

```python
PromptTemplateMiddleware(
    client: BaseAgentClient,
    templates: dict[str, str] = {}
)

# Register template
middleware.register_template(name, template)

# Use template
request = AgentRequest(
    prompt="",  # Ignored
    metadata={
        "template_name": "weather",
        "variables": {"location": "Seattle"}
    }
)
```

**Template syntax:** Jinja2-style `{{ variable }}`

---

### LoadBalancingMiddleware

```python
LoadBalancingMiddleware(
    providers: list[tuple[BaseAgentClient, float]],
    strategy: str = "weighted"  # or "health", "latency"
)
```

**Strategies:**
- `weighted` — Smooth weighted round-robin
- `health` — Route to healthy providers only (health_check)
- `latency` — Route to fastest provider (sliding window)

---

### BatchingMiddleware

```python
BatchingMiddleware(
    client: BaseAgentClient,
    batch_size: int = 10,
    max_wait_ms: float = 100.0
)
```

**Parameters:**
- `batch_size` — Max requests per batch
- `max_wait_ms` — Max wait before dispatching partial batch

**Note:** Streaming bypasses batching.

---

### ConsensusMiddleware

```python
ConsensusMiddleware(
    providers: list[BaseAgentClient],
    strategy: str = "majority"  # or "confidence"
)
```

**Strategies:**
- `majority` — Most common response (voting)
- `confidence` — Highest confidence score (tokens + length)

**Metadata added:** `votes`, `total_providers`

---

### StreamAggregationMiddleware

```python
StreamAggregationMiddleware(
    clients: list[BaseAgentClient],
    strategy: str = "merge"  # or "race"
)
```

**Strategies:**
- `merge` — Interleave chunks from all streams
- `race` — First chunk wins, cancel others

---

### FunctionCallingMiddleware

```python
from madakit.tools import ToolRegistry, Tool

registry = ToolRegistry()
registry.register(Tool(
    name="tool_name",
    description="Tool description",
    function=callable,
    parameters={"type": "object", ...}
))

FunctionCallingMiddleware(
    client: BaseAgentClient,
    registry: ToolRegistry,
    max_iterations: int = 3
)
```

**Tool call format:** `<tool_call name="...">{"arg": "value"}</tool_call>`

---

## Providers

All providers implement `BaseAgentClient`.

### Cloud Providers

#### OpenAIClient

```python
OpenAIClient(
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    **kwargs
)
```

#### AnthropicClient

```python
AnthropicClient(
    api_key: str,
    model: str = "claude-sonnet-4-6",
    base_url: str = "https://api.anthropic.com",
    **kwargs
)
```

#### GeminiClient

```python
GeminiClient(
    api_key: str,
    model: str = "gemini-2.0-flash",
    base_url: str = "https://generativelanguage.googleapis.com",
    **kwargs
)
```

#### DeepSeekClient

```python
DeepSeekClient(
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com/v1",
    **kwargs
)
```

#### CohereClient, MistralClient, TogetherClient, GroqClient, FireworksClient, ReplicateClient

Similar constructors with provider-specific defaults.

---

### Local Server Providers

#### OllamaClient

```python
OllamaClient(
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434/v1",
    **kwargs
)
```

#### VllmClient

```python
VllmClient(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    **kwargs
)
```

#### LocalAIClient, LMStudioClient, JanClient, GPT4AllClient

Similar constructors with different default ports.

---

### Native Providers

#### LlamaCppClient

```python
LlamaCppClient(
    model_path: str,
    n_ctx: int = 2048,
    **kwargs
)

async with LlamaCppClient(...) as client:
    response = await client.send_request(request)
```

#### TransformersClient

```python
TransformersClient(
    model_name: str,
    device: str = "auto",  # or "cpu", "cuda", "cuda:0"
    **kwargs
)

async with TransformersClient(...) as client:
    response = await client.send_request(request)
```

---

### Specialized Providers

#### StabilityAIClient

```python
StabilityAIClient(
    api_key: str,
    model: str = "stable-diffusion-xl-1024-v1-0",
    base_url: str = "https://api.stability.ai",
    **kwargs
)
```

**Returns:** Base64-encoded image or URL in `response.content`

#### ElevenLabsClient

```python
ElevenLabsClient(
    api_key: str,
    voice_id: str,
    model: str = "eleven_monolingual_v1",
    base_url: str = "https://api.elevenlabs.io",
    **kwargs
)
```

**Returns:** Audio URL or base64 in `response.content`

#### EmbeddingProvider

```python
EmbeddingProvider(
    api_key: str,
    model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai.com/v1",
    **kwargs
)
```

**Returns:** JSON-encoded vector in `response.content`, dimensions in `metadata`

---

## Configuration

### ConfigLoader

```python
from madakit.config import ConfigLoader

# From YAML
loader = ConfigLoader.from_yaml("config.yaml")
client = loader.build_stack()

# From JSON
loader = ConfigLoader.from_json("config.json")
client = loader.build_stack()

# From dict
config = {...}
loader = ConfigLoader.from_dict(config)
client = loader.build_stack()
```

**Environment variable substitution:** `${VAR}` or `${VAR:default}`

---

## Tools & Workflows

### ToolRegistry

```python
from madakit.tools import ToolRegistry, Tool

registry = ToolRegistry()

# Register tool
registry.register(Tool(
    name="get_weather",
    description="Get current weather",
    function=lambda location: f"Sunny in {location}",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
))

# Retrieve tool
tool = registry.get("get_weather")

# List all tools
tools = registry.list_tools()

# OpenAPI schemas
schemas = registry.to_openapi_schemas()
```

---

### WorkflowEngine

```python
from madakit.tools import Workflow, Step, WorkflowState

workflow = Workflow()

workflow.add_step(Step(
    name="step1",
    client=client1,
    condition=lambda state: state.variables.get("should_run", True),
    prompt_fn=lambda state: f"Process {state.last_response}"
))

workflow.add_step(Step(
    name="step2",
    client=client2
))

# Execute
initial_state = WorkflowState(variables={"key": "value"})
final_state = await workflow.execute(initial_state)
```

---

## Framework Integrations

### FastAPI

```python
from fastapi import FastAPI, Depends
from madakit.integrations.fastapi import get_client, stream_response

app = FastAPI()
app.state.client = OpenAIClient(api_key="...")

@app.post("/chat")
async def chat(prompt: str, client = Depends(get_client)):
    request = AgentRequest(prompt=prompt)
    response = await client.send_request(request)
    return {"content": response.content}

@app.post("/stream")
async def stream(prompt: str, client = Depends(get_client)):
    request = AgentRequest(prompt=prompt)
    return stream_response(client, request)
```

---

### Flask

```python
from flask import Flask
from madakit.integrations.flask import MadaKit

app = Flask(__name__)
madakit = MadaKit()

app.config["MADAKIT_CLIENT"] = OpenAIClient(api_key="...")
madakit.init_app(app)

@app.route("/chat")
async def chat():
    response = await madakit.client.send_request(request)
    return {"content": response.content}
```

---

### LangChain

```python
from madakit.integrations.langchain import MadaKitLLM

llm = MadaKitLLM(
    client=OpenAIClient(api_key="..."),
    system_prompt="You are helpful",
    max_tokens=100
)

# Use in LangChain chains
result = await llm.agenerate(["What is Python?"])
```

---

### LlamaIndex

```python
from madakit.integrations.llamaindex import MadaKitLLM, MadaKitEmbedding

llm = MadaKitLLM(client=OpenAIClient(api_key="..."))
embed_model = MadaKitEmbedding(client=EmbeddingProvider(api_key="..."))

# Use in LlamaIndex
response = await llm.acomplete("What is AI?")
embedding = await embed_model.aget_query_embedding("query text")
```

---

## CLI Tools

### Scaffolding

```bash
# Generate provider boilerplate
python -m madakit.cli.scaffold provider MyProvider --output my_provider.py

# Generate middleware boilerplate
python -m madakit.cli.scaffold middleware MyMiddleware --output my_middleware.py

# Generate test template
python -m madakit.cli.scaffold test MyProvider --output test_my_provider.py
```

---

### Migration

```bash
# Migrate LangChain code
python -m madakit.cli.migrate langchain input.py --output output.py

# Check compatibility
python -m madakit.cli.migrate check input.py
```

---

## Testing Utilities

```python
from madakit.testing.utils import (
    MockProvider,
    assert_cache_hit,
    assert_cache_miss,
    assert_retry_count,
    assert_response_time
)
from madakit.testing.fixtures import (
    mock_provider,
    sample_request,
    sample_response
)

# Enhanced mock provider
mock = MockProvider(
    responses=[AgentResponse(content="mock", ...)],
    errors=[ProviderError("fail")],
    latency=0.1,
    stream_chunks=[StreamChunk(delta="a"), StreamChunk(delta="b", is_final=True)]
)

# Assertions
assert_cache_hit(response)
assert_retry_count(response, expected=2)
assert_response_time(response, max_ms=100.0)
```

---

## Next Steps

- [Architecture Guide](architecture.md)
- [User Guide](user-guide.md)
- [Tutorial](tutorial.md)
- [Extension Guide](extension-guide.md)
