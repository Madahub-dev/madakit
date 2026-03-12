# Middleware

Middleware wraps providers to add functionality without changing the provider implementation. All middleware implements the same `BaseAgentClient` interface, making them composable.

## Available Middleware

### Reliability

| Middleware | Purpose | Key Parameters |
|------------|---------|----------------|
| `RetryMiddleware` | Retry failed requests with exponential backoff | `max_attempts`, `base_delay_ms`, `max_delay_ms` |
| `CircuitBreakerMiddleware` | Stop requests when failure rate is high | `failure_threshold`, `recovery_timeout_s`, `half_open_max_requests` |
| `FallbackMiddleware` | Fall back to alternate providers on failure | `primary`, `fallbacks` |

### Performance

| Middleware | Purpose | Key Parameters |
|------------|---------|----------------|
| `CachingMiddleware` | Cache responses to avoid duplicate requests | `ttl_seconds`, `max_size` |
| `LoadBalancingMiddleware` | Distribute requests across multiple providers | `providers`, `strategy` (`"weighted"`, `"health"`, `"latency"`) |
| `BatchingMiddleware` | Batch multiple requests for efficiency | `batch_size`, `max_wait_ms` |

### Observability

| Middleware | Purpose | Key Parameters |
|------------|---------|----------------|
| `LoggingMiddleware` | Log requests, responses, and errors | `logger`, `log_level`, `include_prompts` |
| `MetricsMiddleware` | Prometheus metrics (counters, histograms, gauges) | `registry`, `prefix`, `track_labels` |
| `TrackingMiddleware` | Track tokens, latency, and costs | `cost_per_input_token`, `cost_per_output_token` |

### Control

| Middleware | Purpose | Key Parameters |
|------------|---------|----------------|
| `TimeoutMiddleware` | Enforce request timeouts | `timeout_seconds` |
| `RateLimitMiddleware` | Limit request rate (token bucket) | `requests_per_second`, `burst_size` |
| `CostControlMiddleware` | Track and limit API costs | `max_cost`, `alert_threshold`, `cost_fn` |

### Intelligence

| Middleware | Purpose | Key Parameters |
|------------|---------|----------------|
| `ABTestMiddleware` | A/B testing with traffic splitting | `variants` (with weights) |
| `ConsensusMiddleware` | Get consensus from multiple providers | `providers`, `strategy` (`"majority"`, `"confidence"`) |
| `ContentFilterMiddleware` | PII redaction and safety checks | `redact_pii`, `safety_check`, `response_filter` |
| `PromptTemplateMiddleware` | Template management (Jinja2-style) | `templates` |
| `FunctionCallingMiddleware` | Automatic tool execution | `registry`, `max_iterations` |
| `StreamAggregationMiddleware` | Merge or race multiple streams | `clients`, `strategy` (`"merge"`, `"race"`) |

## Middleware Composition

Stack middleware from innermost (closest to provider) to outermost:

```python
from madakit.middleware import (
    LoggingMiddleware,
    TimeoutMiddleware,
    TrackingMiddleware,
    RetryMiddleware,
    CircuitBreakerMiddleware,
    CachingMiddleware,
)

client = LoggingMiddleware(              # 6. Log everything
    client=TimeoutMiddleware(            # 5. Enforce timeout
        client=TrackingMiddleware(       # 4. Track metrics
            client=RetryMiddleware(      # 3. Retry on failure
                client=CircuitBreakerMiddleware(  # 2. Circuit break on high failure
                    client=CachingMiddleware(     # 1. Check cache first
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

**Recommended order:**
1. Caching (innermost — avoid API calls entirely)
2. Circuit breaker (stop requests when provider is down)
3. Retry (retry transient failures)
4. Tracking (measure everything)
5. Timeout (enforce time limits)
6. Logging (outermost — capture all requests)

## Quick Examples

### Basic Retry

```python
from madakit.middleware import RetryMiddleware

client = RetryMiddleware(
    client=provider,
    max_attempts=3
)
```

### Caching with TTL

```python
from madakit.middleware import CachingMiddleware

client = CachingMiddleware(
    client=provider,
    ttl_seconds=3600  # 1 hour
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

### Load Balancing

```python
from madakit.middleware import LoadBalancingMiddleware

client = LoadBalancingMiddleware(
    providers=[
        {"client": openai_client, "weight": 2},
        {"client": anthropic_client, "weight": 1}
    ],
    strategy="weighted"
)
```

### Cost Control

```python
from madakit.middleware import CostControlMiddleware

client = CostControlMiddleware(
    client=provider,
    max_cost=100.0,
    alert_threshold=80.0
)
```

## Custom Middleware

Create custom middleware by subclassing `BaseAgentClient`:

```python
from madakit import BaseAgentClient, AgentRequest, AgentResponse

class MyMiddleware(BaseAgentClient):
    def __init__(self, client: BaseAgentClient):
        self._client = client

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        # Pre-process
        modified_request = transform(request)

        # Call wrapped client
        response = await self._client.send_request(modified_request)

        # Post-process
        modified_response = transform(response)
        return modified_response
```

For detailed middleware documentation, see the [User Guide](../user-guide.md#middleware) and [Extension Guide](../extension-guide.md).
