# Providers

madakit supports 21 AI providers across three categories: cloud, local server, and native.

## Cloud Providers

Cloud providers connect to external APIs over HTTPS.

| Provider | Class | Default Model | API Key Required |
|----------|-------|---------------|------------------|
| OpenAI | `OpenAIClient` | `gpt-4` | Yes |
| Anthropic | `AnthropicClient` | `claude-3-5-sonnet-20241022` | Yes |
| Google Gemini | `GeminiClient` | `gemini-1.5-pro` | Yes |
| DeepSeek | `DeepSeekClient` | `deepseek-chat` | Yes |
| Cohere | `CohereClient` | `command-r-plus` | Yes |
| Mistral | `MistralClient` | `mistral-medium` | Yes |
| Together | `TogetherClient` | `Mixtral-8x7B-Instruct` | Yes |
| Groq | `GroqClient` | `llama3-70b-8192` | Yes |
| Fireworks | `FireworksClient` | `llama-v3p1-70b-instruct` | Yes |
| Replicate | `ReplicateClient` | `meta/llama-2-70b-chat` | Yes |

### Example

```python
from madakit.providers.cloud.openai import OpenAIClient
from madakit.providers.cloud.anthropic import AnthropicClient

openai = OpenAIClient(api_key="sk-...")
anthropic = AnthropicClient(api_key="sk-ant-...")
```

## Local Server Providers

Local server providers connect to inference servers running on localhost or your network.

| Provider | Class | Default Port | API Key Required |
|----------|-------|--------------|------------------|
| Ollama | `OllamaClient` | 11434 | No |
| vLLM | `VLLMClient` | 8000 | No |
| LocalAI | `LocalAIClient` | 8080 | No |
| llama.cpp server | `LlamaCppServerClient` | 8080 | No |
| LM Studio | `LMStudioClient` | 1234 | No |
| Jan | `JanClient` | 1337 | No |
| GPT4All | `GPT4AllClient` | 4891 | No |

### Example

```python
from madakit.providers.local_server.ollama import OllamaClient
from madakit.providers.local_server.vllm import VLLMClient

ollama = OllamaClient(model="llama3.1")
vllm = VLLMClient(model="meta-llama/Llama-3.1-70B", base_url="http://localhost:8000")
```

## Native Providers

Native providers run inference in-process using Python libraries.

| Provider | Class | Dependencies | Use Case |
|----------|-------|--------------|----------|
| Transformers | `TransformersClient` | `transformers`, `torch` | Hugging Face models |
| llama.cpp | `LlamaCppClient` | `llama-cpp-python` | GGUF models |

### Example

```python
from madakit.providers.native.transformers import TransformersClient
from madakit.providers.native.llamacpp import LlamaCppClient

# Transformers
transformers = TransformersClient(
    model_name="meta-llama/Llama-3.2-1B-Instruct"
)

# llama.cpp
llamacpp = LlamaCppClient(
    model_path="./models/llama-3.1-8b.gguf"
)
```

## Specialized Providers

Specialized providers for non-chat use cases.

| Provider | Class | Use Case |
|----------|-------|----------|
| Stability AI | `StabilityAIClient` | Image generation |
| ElevenLabs | `ElevenLabsClient` | Text-to-speech |
| Embedding | `EmbeddingProvider` | Text embeddings |

## Provider Selection Guide

Choose based on your requirements:

| Requirement | Recommended Provider |
|-------------|---------------------|
| Production chat app | OpenAI, Anthropic, Gemini |
| Cost-sensitive | DeepSeek, Groq |
| Privacy/on-premise | Ollama, vLLM, LocalAI |
| Low latency | Groq, Fireworks |
| Open source models | Ollama, vLLM, Transformers |
| No internet required | Transformers, llama.cpp |
| GPU acceleration | vLLM, Transformers |
| CPU-only | llama.cpp, Ollama |

For detailed provider documentation, see the [User Guide](../user-guide.md#providers).
