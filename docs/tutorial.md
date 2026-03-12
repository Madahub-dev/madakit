# Tutorial & Cookbook

Hands-on examples to get you started with **madakit**.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Basic Patterns](#basic-patterns)
3. [Middleware Recipes](#middleware-recipes)
4. [Production Patterns](#production-patterns)
5. [Advanced Use Cases](#advanced-use-cases)

---

## Quickstart

### Installation

```bash
pip install madakit[all]
```

### Your First Request

```python
import asyncio
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient

async def main():
    # Create client
    client = OpenAIClient(api_key="sk-...")

    # Build request
    request = AgentRequest(
        prompt="Explain Python's asyncio in one sentence",
        max_tokens=50
    )

    # Send request
    response = await client.send_request(request)

    # Print response
    print(response.content)
    print(f"Tokens used: {response.total_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Output:**
```
Python's asyncio is a library for writing concurrent code using async/await syntax.
Tokens used: 27
```

---

## Basic Patterns

### Pattern 1: Simple Streaming

```python
async def stream_example():
    client = OpenAIClient(api_key="sk-...")

    request = AgentRequest(prompt="Write a haiku about code")

    print("Response: ", end="", flush=True)
    async for chunk in client.send_request_stream(request):
        print(chunk.delta, end="", flush=True)
        if chunk.is_final:
            print()  # Newline

asyncio.run(stream_example())
```

**Output:**
```
Response: Functions dance in loops,
Variables hold their secrets,
Compilers translate.
```

### Pattern 2: Multimodal Input

```python
async def multimodal_example():
    from madakit import Attachment

    # Read image
    with open("chart.png", "rb") as f:
        image_data = f.read()

    # Create request with image
    request = AgentRequest(
        prompt="Describe this chart in detail",
        attachments=[
            Attachment(
                content=image_data,
                media_type="image/png",
                filename="chart.png"
            )
        ]
    )

    # Use Claude (good for vision)
    from madakit.providers.cloud.anthropic import AnthropicClient
    client = AnthropicClient(api_key="sk-ant-...")

    response = await client.send_request(request)
    print(response.content)

asyncio.run(multimodal_example())
```

### Pattern 3: System Prompts

```python
async def system_prompt_example():
    client = OpenAIClient(api_key="sk-...")

    request = AgentRequest(
        system_prompt="You are a helpful Python tutor. Explain concepts simply with code examples.",
        prompt="What is a decorator?",
        temperature=0.5
    )

    response = await client.send_request(request)
    print(response.content)

asyncio.run(system_prompt_example())
```

---

## Middleware Recipes

### Recipe 1: Resilient Client (Retry + Circuit Breaker)

```python
from madakit import RetryMiddleware, CircuitBreakerMiddleware

async def resilient_client():
    # Build resilient stack
    client = RetryMiddleware(
        CircuitBreakerMiddleware(
            OpenAIClient(api_key="sk-..."),
            failure_threshold=5,
            recovery_timeout=60.0
        ),
        max_retries=3,
        backoff_base=1.0  # 1s, 2s, 4s
    )

    # This will automatically retry on transient errors
    # and fail-fast if provider is down
    try:
        response = await client.send_request(request)
        print(response.content)
    except RetryExhaustedError as e:
        print(f"All retries failed: {e.last_error}")
    except CircuitOpenError:
        print("Circuit is open, provider is down")

asyncio.run(resilient_client())
```

### Recipe 2: Cached Client (Save Cost)

```python
from madakit import CachingMiddleware

async def cached_client():
    client = CachingMiddleware(
        OpenAIClient(api_key="sk-..."),
        ttl=3600.0,  # 1 hour cache
        max_entries=10000
    )

    request = AgentRequest(prompt="What is Python?")

    # First request: cache miss, calls API
    response1 = await client.send_request(request)
    print(f"First: {response1.content[:50]}...")

    # Second request: cache hit, instant
    response2 = await client.send_request(request)
    print(f"Second: {response2.content[:50]}...")

    # Same content, no API call
    assert response1.content == response2.content

asyncio.run(cached_client())
```

### Recipe 3: Tracked Client (Monitor Usage)

```python
from madakit import TrackingMiddleware

async def tracked_client():
    def cost_fn(response):
        """Calculate cost: $0.10/1M input, $0.30/1M output"""
        input_cost = response.input_tokens * 0.0001
        output_cost = response.output_tokens * 0.0003
        return input_cost + output_cost

    client = TrackingMiddleware(
        OpenAIClient(api_key="sk-..."),
        cost_fn=cost_fn
    )

    # Make multiple requests
    for prompt in ["What is AI?", "What is ML?", "What is DL?"]:
        request = AgentRequest(prompt=prompt)
        await client.send_request(request)

    # Check stats
    stats = client.stats
    print(f"Total requests: {stats.total_requests}")
    print(f"Total tokens: {stats.total_input_tokens + stats.total_output_tokens}")
    print(f"Total cost: ${stats.total_cost:.4f}")
    print(f"Avg latency: {stats.total_inference_ms / stats.total_requests:.0f}ms")

asyncio.run(tracked_client())
```

**Output:**
```
Total requests: 3
Total tokens: 456
Total cost: $0.0137
Avg latency: 234ms
```

### Recipe 4: Fallback Client (High Availability)

```python
from madakit import FallbackMiddleware
from madakit.providers.cloud.openai import OpenAIClient
from madakit.providers.cloud.anthropic import AnthropicClient
from madakit.providers.cloud.deepseek import DeepSeekClient

async def fallback_client():
    # Try OpenAI → Anthropic → DeepSeek
    client = FallbackMiddleware(
        primary=OpenAIClient(api_key="sk-..."),
        fallbacks=[
            AnthropicClient(api_key="sk-ant-..."),
            DeepSeekClient(api_key="sk-...")
        ]
    )

    request = AgentRequest(prompt="What is machine learning?")

    # If OpenAI fails, automatically tries Anthropic
    # If Anthropic fails, tries DeepSeek
    response = await client.send_request(request)
    print(f"Response from: {response.model}")
    print(response.content)

asyncio.run(fallback_client())
```

### Recipe 5: Complete Production Stack

```python
from madakit import (
    LoggingMiddleware,
    MetricsMiddleware,
    TimeoutMiddleware,
    RateLimitMiddleware,
    CostControlMiddleware,
    RetryMiddleware,
    CircuitBreakerMiddleware,
    CachingMiddleware,
    TrackingMiddleware,
    FallbackMiddleware,
)

async def production_stack():
    def cost_fn(response):
        return response.input_tokens * 0.0001 + response.output_tokens * 0.0003

    client = (
        LoggingMiddleware(           # 1. Log everything
            MetricsMiddleware(        # 2. Prometheus metrics
                TimeoutMiddleware(    # 3. 30s timeout
                    RateLimitMiddleware(  # 4. 10 req/s
                        CostControlMiddleware(  # 5. $100 budget
                            RetryMiddleware(     # 6. 3 retries
                                CircuitBreakerMiddleware(  # 7. Fail-fast
                                    CachingMiddleware(     # 8. 1h cache
                                        TrackingMiddleware(  # 9. Track stats
                                            FallbackMiddleware(  # 10. OpenAI → Anthropic
                                                primary=OpenAIClient(api_key="..."),
                                                fallbacks=[AnthropicClient(api_key="...")]
                                            )
                                        ),
                                        ttl=3600.0
                                    ),
                                    failure_threshold=5
                                ),
                                max_retries=3
                            ),
                            cost_fn=cost_fn,
                            budget_cap=100.0
                        ),
                        requests_per_second=10.0
                    ),
                    timeout_seconds=30.0
                )
            ),
            log_level="INFO"
        )
    )

    # This client is production-ready
    response = await client.send_request(request)
    print(response.content)

asyncio.run(production_stack())
```

---

## Production Patterns

### Pattern 1: Rate Limiting per User

```python
from madakit import RateLimitMiddleware

async def per_user_rate_limit():
    client = RateLimitMiddleware(
        OpenAIClient(api_key="sk-..."),
        requests_per_second=5.0,
        key_fn=lambda req: req.metadata.get("user_id", "anonymous")
    )

    # Each user gets independent 5 req/s limit
    request = AgentRequest(
        prompt="Hello",
        metadata={"user_id": "user_123"}
    )

    response = await client.send_request(request)
    print(response.content)

asyncio.run(per_user_rate_limit())
```

### Pattern 2: Budget Alerts

```python
from madakit import CostControlMiddleware

async def budget_alerts():
    def cost_fn(response):
        return response.total_tokens * 0.0001

    def alert_callback(current, threshold):
        print(f"⚠️  Budget alert: ${current:.2f} / ${threshold:.2f}")
        # Send email, Slack notification, etc.

    client = CostControlMiddleware(
        OpenAIClient(api_key="sk-..."),
        cost_fn=cost_fn,
        budget_cap=10.0,
        alert_threshold=0.8,  # Alert at $8
        on_alert=alert_callback
    )

    # Make requests until budget exceeded
    for i in range(100):
        try:
            request = AgentRequest(prompt=f"Request {i}")
            await client.send_request(request)
        except BudgetExceededError:
            print(f"Budget exceeded at request {i}")
            break

asyncio.run(budget_alerts())
```

### Pattern 3: A/B Testing

```python
from madakit import ABTestMiddleware

async def ab_testing():
    # 50/50 split between gpt-4o-mini and gpt-4o
    variants = [
        (OpenAIClient(api_key="...", model="gpt-4o-mini"), 0.5),
        (OpenAIClient(api_key="...", model="gpt-4o"), 0.5)
    ]

    client = ABTestMiddleware(
        variants=variants,
        key_fn=lambda req: req.metadata.get("user_id", "default")
    )

    # Same user always gets same variant (deterministic)
    request = AgentRequest(
        prompt="What is Python?",
        metadata={"user_id": "user_123"}
    )

    response = await client.send_request(request)
    print(f"Variant: {response.metadata['variant']}")  # 0 or 1
    print(f"Model: {response.model}")

asyncio.run(ab_testing())
```

### Pattern 4: Content Filtering

```python
from madakit import ContentFilterMiddleware

async def content_filtering():
    def safety_check(prompt):
        """Block unsafe prompts."""
        if any(word in prompt.lower() for word in ["hack", "exploit"]):
            raise ValueError("Unsafe prompt detected")

    client = ContentFilterMiddleware(
        OpenAIClient(api_key="sk-..."),
        redact_pii=True,  # Redact emails, SSNs, credit cards
        safety_check=safety_check
    )

    # This works
    request1 = AgentRequest(prompt="What is Python?")
    await client.send_request(request1)

    # This is blocked
    try:
        request2 = AgentRequest(prompt="How to hack a system?")
        await client.send_request(request2)
    except ValueError as e:
        print(f"Blocked: {e}")

asyncio.run(content_filtering())
```

### Pattern 5: Prompt Templates

```python
from madakit import PromptTemplateMiddleware

async def prompt_templates():
    client = PromptTemplateMiddleware(
        OpenAIClient(api_key="sk-..."),
        templates={
            "summarize": "Summarize the following text:\n\n{{ text }}",
            "translate": "Translate to {{ language }}:\n\n{{ text }}",
            "code_review": "Review this code:\n```{{ language }}\n{{ code }}\n```"
        }
    )

    # Use summarize template
    request = AgentRequest(
        prompt="",  # Ignored
        metadata={
            "template_name": "summarize",
            "variables": {"text": "Long document about AI..."}
        }
    )

    response = await client.send_request(request)
    print(response.content)

asyncio.run(prompt_templates())
```

---

## Advanced Use Cases

### Use Case 1: Multi-Provider Consensus

```python
from madakit.middleware.consensus import ConsensusMiddleware

async def consensus_voting():
    # Send to 3 providers, vote on result
    client = ConsensusMiddleware(
        providers=[
            OpenAIClient(api_key="..."),
            AnthropicClient(api_key="..."),
            GeminiClient(api_key="...")
        ],
        strategy="majority"
    )

    request = AgentRequest(prompt="Is Python better than JavaScript?")

    response = await client.send_request(request)
    print(f"Consensus response: {response.content}")
    print(f"Votes: {response.metadata['votes']} / {response.metadata['total_providers']}")

asyncio.run(consensus_voting())
```

### Use Case 2: Tool Calling (Function Calling)

```python
from madakit import FunctionCallingMiddleware
from madakit.tools import ToolRegistry, Tool

async def function_calling():
    # Define tools
    registry = ToolRegistry()

    registry.register(Tool(
        name="get_weather",
        description="Get current weather for a location",
        function=lambda location: f"The weather in {location} is sunny, 72°F",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    ))

    registry.register(Tool(
        name="calculate",
        description="Perform basic arithmetic",
        function=lambda expression: str(eval(expression)),
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    ))

    # Wrap client with function calling
    client = FunctionCallingMiddleware(
        OpenAIClient(api_key="sk-..."),
        registry=registry,
        max_iterations=3
    )

    # Model can now call tools automatically
    request = AgentRequest(
        prompt="What's the weather in Seattle? Also calculate 15 * 23."
    )

    response = await client.send_request(request)
    print(response.content)

asyncio.run(function_calling())
```

**Output:**
```
The weather in Seattle is sunny, 72°F. 15 * 23 equals 345.
```

### Use Case 3: Multi-Step Workflow

```python
from madakit.tools import Workflow, Step, WorkflowState

async def multi_step_workflow():
    # Step 1: Research
    research_client = OpenAIClient(api_key="...")

    # Step 2: Write (only if research succeeds)
    write_client = AnthropicClient(api_key="...")

    # Step 3: Edit
    edit_client = OpenAIClient(api_key="...", model="gpt-4o")

    workflow = Workflow()

    workflow.add_step(Step(
        name="research",
        client=research_client,
        prompt_fn=lambda state: f"Research: {state.variables['topic']}"
    ))

    workflow.add_step(Step(
        name="write",
        client=write_client,
        condition=lambda state: len(state.last_response) > 100,  # Only if research is detailed
        prompt_fn=lambda state: f"Write article based on: {state.last_response}"
    ))

    workflow.add_step(Step(
        name="edit",
        client=edit_client,
        prompt_fn=lambda state: f"Edit and improve: {state.last_response}"
    ))

    # Execute workflow
    initial_state = WorkflowState(variables={"topic": "Python asyncio"})
    final_state = await workflow.execute(initial_state)

    print("Final article:")
    print(final_state.last_response)

asyncio.run(multi_step_workflow())
```

### Use Case 4: Stream Aggregation (Racing)

```python
from madakit.middleware.stream_aggregation import StreamAggregationMiddleware

async def stream_race():
    # Race 3 providers, first response wins
    client = StreamAggregationMiddleware(
        clients=[
            OpenAIClient(api_key="..."),
            AnthropicClient(api_key="..."),
            DeepSeekClient(api_key="...")
        ],
        strategy="race"
    )

    request = AgentRequest(prompt="Write a haiku")

    # First provider to respond wins, others cancelled
    async for chunk in client.send_request_stream(request):
        print(chunk.delta, end="", flush=True)
        if chunk.is_final:
            print()

asyncio.run(stream_race())
```

### Use Case 5: Local Development → Production

```python
import os

async def environment_aware_client():
    # Use Ollama in dev, OpenAI in prod
    if os.getenv("ENV") == "production":
        from madakit.providers.cloud.openai import OpenAIClient
        client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        from madakit.providers.local_server.ollama import OllamaClient
        client = OllamaClient(model="llama3.2")

    # Same code works in both environments
    request = AgentRequest(prompt="What is Python?")
    response = await client.send_request(request)
    print(response.content)

asyncio.run(environment_aware_client())
```

### Use Case 6: Configuration-Driven Setup

```python
from madakit.config import ConfigLoader

async def config_driven():
    # config.yaml defines entire stack
    loader = ConfigLoader.from_yaml("config.yaml")
    client = loader.build_stack()

    # No code changes to swap providers/middleware
    request = AgentRequest(prompt="What is AI?")
    response = await client.send_request(request)
    print(response.content)

asyncio.run(config_driven())
```

**config.yaml:**
```yaml
provider:
  type: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

middleware:
  - type: retry
    params: {max_retries: 3}

  - type: cache
    params: {ttl: 3600.0}

  - type: tracking
```

---

## Next Steps

- [Architecture Guide](architecture.md) — Understand the design
- [User Guide](user-guide.md) — Detailed usage patterns
- [API Reference](api-reference.md) — Complete API docs
- [Extension Guide](extension-guide.md) — Build custom providers/middleware
