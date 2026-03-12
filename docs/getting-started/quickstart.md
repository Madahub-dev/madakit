# Quickstart

Get started with madakit in 5 minutes.

## Installation

```bash
pip install madakit[cloud]
```

## Your First Request

```python
import asyncio
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient

async def main():
    # Create client
    client = OpenAIClient(api_key="sk-...")

    # Create request
    request = AgentRequest(prompt="What is Python?")

    # Send request
    response = await client.send_request(request)

    # Print response
    print(response.content)
    # "Python is a high-level, interpreted programming language..."

asyncio.run(main())
```

## Add Middleware

Wrap your provider with middleware for production features:

```python
from madakit.middleware import RetryMiddleware, CachingMiddleware

# Wrap provider with caching and retry
client = RetryMiddleware(
    client=CachingMiddleware(
        client=OpenAIClient(api_key="sk-..."),
        ttl_seconds=3600  # Cache for 1 hour
    ),
    max_attempts=3  # Retry up to 3 times
)

response = await client.send_request(request)
```

## Streaming

Stream responses token by token:

```python
async for chunk in client.send_request_stream(request):
    print(chunk.delta_content, end="", flush=True)
```

## Switch Providers

The same code works with any provider:

```python
# OpenAI
from madakit.providers.cloud.openai import OpenAIClient
client = OpenAIClient(api_key="sk-...")

# Anthropic
from madakit.providers.cloud.anthropic import AnthropicClient
client = AnthropicClient(api_key="sk-ant-...")

# Ollama (local)
from madakit.providers.local_server.ollama import OllamaClient
client = OllamaClient(model="llama3.1")

# All use the same interface
response = await client.send_request(request)
```

## Configuration Files

Use YAML for configuration:

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
```

```python
from madakit.config import ConfigLoader

loader = ConfigLoader()
client = loader.from_yaml("config.yaml")
response = await client.send_request(request)
```

## Complete Example

```python
import asyncio
from madakit import AgentRequest
from madakit.providers.cloud.openai import OpenAIClient
from madakit.middleware import (
    RetryMiddleware,
    CachingMiddleware,
    TrackingMiddleware,
    TimeoutMiddleware,
)

async def main():
    # Build client stack
    provider = OpenAIClient(api_key="sk-...")

    client = TimeoutMiddleware(
        client=TrackingMiddleware(
            client=RetryMiddleware(
                client=CachingMiddleware(
                    client=provider,
                    ttl_seconds=3600
                ),
                max_attempts=3
            )
        ),
        timeout_seconds=30.0
    )

    # Send request
    request = AgentRequest(
        prompt="Explain asyncio in one sentence",
        max_tokens=50
    )

    response = await client.send_request(request)

    # Access response
    print(f"Content: {response.content}")
    print(f"Tokens: {response.metadata.get('input_tokens')} in, "
          f"{response.metadata.get('output_tokens')} out")
    print(f"Cached: {response.metadata.get('cache_hit', False)}")

asyncio.run(main())
```

## Next Steps

- [Tutorial](../tutorial.md) — Learn with detailed examples
- [User Guide](../user-guide.md) — Deep dive into features
- [Providers](providers.md) — Explore all 21 providers
- [Middleware](middleware.md) — Stack production middleware
