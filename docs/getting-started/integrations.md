# Framework Integrations

madakit integrates with popular Python frameworks for AI applications.

## LangChain

Drop-in LLM wrapper for LangChain applications.

### Installation

```bash
pip install madakit[cloud] langchain
```

### Usage

```python
from madakit.providers.cloud.openai import OpenAIClient
from madakit.integrations.langchain import MadaKitLLM

# Create madakit client
provider = OpenAIClient(api_key="sk-...")

# Wrap in LangChain LLM
llm = MadaKitLLM(client=provider)

# Use with LangChain
result = await llm.apredict("What is the capital of France?")
print(result)  # "Paris"

# Streaming
async for token in llm.astream("Write a poem"):
    print(token, end="", flush=True)
```

### With Middleware

```python
from madakit.middleware import RetryMiddleware, CachingMiddleware

client = RetryMiddleware(
    client=CachingMiddleware(
        client=OpenAIClient(api_key="sk-..."),
        ttl_seconds=3600
    ),
    max_attempts=3
)

llm = MadaKitLLM(client=client)
```

## LlamaIndex

LLM and embedding wrappers for LlamaIndex.

### Installation

```bash
pip install madakit[cloud] llama-index
```

### LLM Usage

```python
from madakit.providers.cloud.openai import OpenAIClient
from madakit.integrations.llamaindex import MadaKitLLM

llm = MadaKitLLM(client=OpenAIClient(api_key="sk-..."))

# Completion
response = await llm.acomplete("What is Python?")
print(response.text)

# Chat
from llama_index.core.base.llms.types import ChatMessage

messages = [ChatMessage(role="user", content="Hello!")]
response = await llm.achat(messages)
print(response.message.content)
```

### Embedding Usage

```python
from madakit.integrations.llamaindex import MadaKitEmbedding
from madakit.providers.specialized.embedding import EmbeddingProvider

embedding = MadaKitEmbedding(
    client=EmbeddingProvider(api_key="sk-...")
)

# Single query
vector = await embedding.aget_query_embedding("search query")

# Batch
vectors = await embedding.aget_text_embedding_batch(["text1", "text2"])
```

## FastAPI

Dependency injection and streaming helpers for FastAPI applications.

### Installation

```bash
pip install madakit[cloud] fastapi uvicorn
```

### Setup

```python
from fastapi import FastAPI, Depends
from madakit.providers.cloud.openai import OpenAIClient
from madakit.integrations.fastapi import get_client, stream_response
from madakit import AgentRequest

app = FastAPI()

# Store client in app state
app.state.madakit_client = OpenAIClient(api_key="sk-...")
```

### Non-Streaming Endpoint

```python
@app.post("/chat")
async def chat(request: dict, client=Depends(get_client)):
    agent_request = AgentRequest(**request)
    response = await client.send_request(agent_request)
    return {
        "content": response.content,
        "metadata": response.metadata
    }
```

### Streaming Endpoint

```python
@app.post("/chat/stream")
async def chat_stream(request: dict, client=Depends(get_client)):
    agent_request = AgentRequest(**request)
    return stream_response(client.send_request_stream(agent_request))
```

### Client Usage

```python
import httpx
import asyncio

async def test_chat():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/chat",
            json={"prompt": "Hello!"}
        )
        print(response.json())

asyncio.run(test_chat())
```

## Flask

Extension class for Flask applications.

### Installation

```bash
pip install madakit[cloud] flask
```

### Setup

```python
from flask import Flask, request, jsonify
from madakit.providers.cloud.openai import OpenAIClient
from madakit.integrations.flask import MadaKit
from madakit import AgentRequest

app = Flask(__name__)
app.config["MADAKIT_CLIENT"] = OpenAIClient(api_key="sk-...")

madakit = MadaKit(app)
```

### Endpoint

```python
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    agent_request = AgentRequest(**data)

    # Async wrapper for sync Flask
    import asyncio
    response = asyncio.run(madakit.client.send_request(agent_request))

    return jsonify({
        "content": response.content,
        "metadata": response.metadata
    })
```

### Streaming (simplified)

```python
from madakit.integrations.flask import stream_response

@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    agent_request = AgentRequest(**data)

    return stream_response(madakit.client.send_request_stream(agent_request))
```

## Comparison

| Feature | LangChain | LlamaIndex | FastAPI | Flask |
|---------|-----------|------------|---------|-------|
| Use Case | LLM orchestration | RAG applications | Async API server | Sync API server |
| Async Support | Yes | Yes | Yes | Limited |
| Streaming | Yes | Yes | Yes | Simplified |
| Complexity | High | High | Low | Low |

## Best Practices

1. **Reuse clients** — Create one client and reuse it across requests
2. **Use middleware** — Add retry, caching, and timeouts in production
3. **Handle errors** — Catch `AgentError` and return appropriate HTTP status codes
4. **Monitor metrics** — Use `MetricsMiddleware` for observability
5. **Rate limiting** — Use `RateLimitMiddleware` or framework-level rate limiting

For detailed integration examples, see the [Tutorial](../tutorial.md) and [User Guide](../user-guide.md).
