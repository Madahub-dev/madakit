# Changelog

All notable changes to madakit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-03-12

### Added

#### Core Architecture (Phases 1-3)
- **Type system** (`_types.py`) — `AgentRequest`, `AgentResponse`, `StreamChunk`, `Attachment`, `TrackingStats`
- **Error hierarchy** (`_errors.py`) — `AgentError`, `ProviderError`, `MiddlewareError`, `CircuitOpenError`, `RetryExhaustedError`
- **Abstract base client** (`_base.py`) — `BaseAgentClient` ABC with `send_request`, `send_request_stream`, `close`, `cancel` methods
- **HTTP base client** (`providers/_http_base.py`) — TLS enforcement, timeout configuration, connection pooling via httpx
- **OpenAI compatibility mixin** (`providers/_openai_compat.py`) — shared payload/response handling for OpenAI-compatible providers

#### Middleware (Phases 4-6, 8-9, 11, 13)
- **RetryMiddleware** — exponential backoff with jitter (Phase 4)
- **CircuitBreakerMiddleware** — failure detection with half-open state (Phase 5)
- **CachingMiddleware** — in-memory TTL-based caching with LRU eviction (Phase 5)
- **TrackingMiddleware** — token counting, latency, cost tracking (Phase 6)
- **FallbackMiddleware** — primary + fallback chain (Phase 6)
- **RateLimitMiddleware** — token bucket algorithm (Phase 8)
- **CostControlMiddleware** — budget tracking with alerts (Phase 8)
- **TimeoutMiddleware** — request and first-chunk timeouts (Phase 8)
- **LoggingMiddleware** — structured logging with correlation IDs (Phase 9)
- **MetricsMiddleware** — Prometheus metrics (counters, histograms, gauges) (Phase 9)
- **ABTestMiddleware** — deterministic A/B testing with traffic splitting (Phase 11)
- **ContentFilterMiddleware** — PII redaction and safety checks (Phase 11)
- **PromptTemplateMiddleware** — Jinja2-style template management (Phase 11)
- **LoadBalancingMiddleware** — weighted, health-based, latency-based routing (Phase 11)
- **BatchingMiddleware** — request batching with timeout-based dispatch (Phase 13)
- **ConsensusMiddleware** — multi-provider consensus with majority voting (Phase 13)
- **StreamAggregationMiddleware** — merge and race strategies (Phase 13)

#### Cloud Providers (Phases 3, 14)
- **OpenAIClient** — OpenAI GPT-4, GPT-3.5 (Phase 3)
- **AnthropicClient** — Claude 3.5 Sonnet, Opus, Haiku (Phase 3)
- **GeminiClient** — Google Gemini 1.5 Pro, Flash (Phase 3)
- **DeepSeekClient** — DeepSeek Chat, Coder (Phase 3)
- **CohereClient** — Cohere Command R/R+ (Phase 14)
- **MistralClient** — Mistral Medium, Large (Phase 14)
- **TogetherClient** — Together AI Mixtral (Phase 14)
- **GroqClient** — Groq LLaMA (Phase 14)
- **FireworksClient** — Fireworks LLaMA (Phase 14)
- **ReplicateClient** — Replicate predictions API (Phase 14)

#### Local Server Providers (Phases 3, 14)
- **OllamaClient** — Ollama local models (Phase 3)
- **VLLMClient** — vLLM inference server (Phase 3)
- **LocalAIClient** — LocalAI OpenAI-compatible server (Phase 3)
- **LlamaCppServerClient** — llama.cpp server (Phase 3)
- **LMStudioClient** — LM Studio local server (Phase 14)
- **JanClient** — Jan local server (Phase 14)
- **GPT4AllClient** — GPT4All local server (Phase 14)

#### Native Providers (Phases 3, 6, 14)
- **TransformersClient** — Hugging Face Transformers (Phase 6)
- **LlamaCppClient** — llama.cpp Python bindings (Phase 3)

#### Specialized Providers (Phase 14)
- **StabilityAIClient** — Stability AI image generation
- **ElevenLabsClient** — ElevenLabs text-to-speech
- **EmbeddingProvider** — OpenAI-compatible embeddings

#### Configuration (Phase 10)
- **ConfigLoader** — YAML/JSON configuration with environment variable substitution
- **ProviderConfig**, **MiddlewareConfig**, **StackConfig** — dataclass-based configuration schema
- **Dynamic imports** — lazy loading for optional provider dependencies
- **Middleware order validation** — advisory ordering checks

#### Tools & Workflows (Phase 12)
- **Tool registry** — function registration with OpenAPI schema generation
- **FunctionCallingMiddleware** — automatic tool detection and execution
- **WorkflowEngine** — multi-step workflows with conditional branching and state management
- **WorkflowState**, **Step**, **Workflow** — workflow primitives

#### Framework Integrations (Phase 15)
- **LangChain** — `MadaKitLLM` wrapper with callback support
- **LlamaIndex** — `MadaKitLLM` and `MadaKitEmbedding` wrappers
- **FastAPI** — dependency injection and streaming response helpers
- **Flask** — extension class with streaming support

#### Developer Tools (Phase 16)
- **Scaffolding CLI** — generate provider, middleware, and test boilerplate
- **Testing utilities** — enhanced `MockProvider`, assertion helpers, pytest fixtures
- **Migration tools** — LangChain to madakit migration, config conversion, compatibility checking

#### Documentation (Phase 17)
- **Architecture guide** — layer design, ABC contract, design patterns
- **User guide** — installation, provider selection, middleware configuration
- **API reference** — complete API documentation for all components
- **Tutorial** — quickstart, recipes, advanced patterns
- **Extension guide** — building custom providers and middleware

#### Packaging (Phase 7, 17)
- **Zero core dependencies** — stdlib-only for types, errors, base client, middleware
- **Optional dependencies** — cloud, local, native, metrics, framework integrations
- **Type annotations** — full mypy strict mode support
- **Ruff linting** — line-length=100, modern Python formatting
- **pytest suite** — 2,100+ tests with pytest-asyncio
- **py.typed** — PEP 561 type marker for downstream type checking

### Changed

#### Breaking Changes
- **Package renamed** — `mada-modelkit` → `madakit` (shorter, cleaner)
  - **Install:** `pip install madakit` (was `pip install mada-modelkit`)
  - **Import:** `from madakit import *` (was `from mada_modelkit import *`)
  - **Repository:** `github.com/Madahub-dev/madakit` (was `github.com/madahub/mada-modelkit`)

### Fixed
- **Import tests** — updated for 24 exported names (added BatchingMiddleware, ConsensusMiddleware, StreamAggregationMiddleware)
- **TLS enforcement** — all cloud providers enforce HTTPS
- **API key redaction** — `__repr__` methods mask secrets
- **Thread pool cleanup** — TransformersClient properly shuts down executor

### Security
- **No eval/exec** — zero dynamic code execution
- **TLS by default** — HTTPS enforced for all cloud providers
- **API key masking** — credentials redacted in logs and repr
- **Deferred imports** — optional dependencies lazy-loaded to avoid supply chain attacks

## [0.1.0] - 2026-03-01

### Added
- Initial development release
- Core ABC contract and type system
- Basic middleware (Retry, CircuitBreaker, Caching)
- Initial cloud providers (OpenAI, Anthropic, Gemini)

---

[Unreleased]: https://github.com/Madahub-dev/madakit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Madahub-dev/madakit/releases/tag/v1.0.0
[0.1.0]: https://github.com/Madahub-dev/madakit/releases/tag/v0.1.0
