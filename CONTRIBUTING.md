# Contributing to madakit

Thank you for your interest in contributing to madakit! We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations of the Code of Conduct may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly. The project team is obligated to maintain confidentiality with regard to the reporter of an incident.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/madakit.git
   cd madakit
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/madahub/madakit.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip and virtualenv

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dev dependencies
pip install -e ".[dev]"
```

The `dev` extra includes:
- `pytest`, `pytest-asyncio` — testing framework
- `pytest-cov` — coverage reporting
- `ruff` — linting and formatting
- `mypy` — static type checking
- All optional dependencies (cloud, local, native, metrics, integrations)

### Verify Setup

```bash
# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy src/
```

## Project Structure

```
madakit/
├── src/madakit/           # Library source code
│   ├── _base.py           # BaseAgentClient ABC
│   ├── _types.py          # Core types (AgentRequest, AgentResponse, etc.)
│   ├── _errors.py         # Error hierarchy
│   ├── middleware/        # 16 middleware implementations
│   ├── providers/         # 21 provider implementations
│   │   ├── cloud/         # Cloud providers (OpenAI, Anthropic, etc.)
│   │   ├── local_server/  # Local server providers (Ollama, vLLM, etc.)
│   │   └── native/        # Native providers (Transformers, llama.cpp)
│   ├── config/            # Configuration loader and schema
│   ├── tools/             # Tool registry and workflow engine
│   └── integrations/      # Framework integrations (LangChain, FastAPI, etc.)
├── tests/                 # Test files (mirror src/ structure)
├── docs/                  # Documentation
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # Project overview
```

## Contribution Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` — new features
- `fix/` — bug fixes
- `docs/` — documentation changes
- `refactor/` — code refactoring
- `test/` — test improvements

### 2. Make Changes

- Write your code following the [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Commit Your Changes

```bash
git add .
git commit -m "type(scope): subject"
```

Commit message format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding or updating tests
- `chore` — maintenance tasks

**Scopes:**
- `types`, `errors`, `base` — core components
- `retry`, `cache`, `tracking`, etc. — specific middleware
- `openai`, `anthropic`, `ollama`, etc. — specific providers
- `config`, `tools`, `integrations` — other modules
- `packaging`, `docs`, `tests` — project-level changes

**Examples:**
```
feat(middleware): add RateLimitMiddleware with token bucket algorithm

fix(openai): handle empty content in streaming response

docs(tutorial): add FastAPI integration example

test(transformers): add lazy loading tests
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Open a Pull Request

- Go to the [madakit repository](https://github.com/madahub/madakit)
- Click "New Pull Request"
- Select your fork and branch
- Fill out the PR template with:
  - Description of changes
  - Related issues (if any)
  - Testing performed
  - Documentation updates

## Code Standards

### Python Style

- **Python version:** 3.11+
- **Linter:** ruff (line-length=100)
- **Type checker:** mypy (strict mode)
- **Async-first:** Use `async def` for all I/O operations

### Code Quality Rules

1. **Zero core dependencies** — `_types.py`, `_errors.py`, `_base.py`, and middleware must use only stdlib
2. **Type annotations** — all public functions and methods must have type hints
3. **Docstrings** — required at every level:
   - **Module** — purpose and constraints
   - **Class** — one-line description
   - **Public methods** — description, params, returns, raises
   - **Test methods** — one-sentence assertion
4. **No `print()`** — use logging for library code
5. **TLS enforcement** — cloud providers must enforce HTTPS
6. **API key redaction** — mask secrets in `__repr__`
7. **No `eval`/`exec`** — zero dynamic code execution

### Formatting

Run before committing:
```bash
# Auto-fix formatting issues
ruff check --fix .

# Check types
mypy src/
```

## Testing Guidelines

### Test Structure

- Tests mirror the `src/` structure in `tests/`
- One test file per source file (e.g., `src/madakit/middleware/retry.py` → `tests/middleware/test_retry.py`)
- Test classes group related test methods

### Test Requirements

Every contribution must include tests:
1. **Module exports** — verify `__all__` and importability
2. **Constructor validation** — test parameter validation and defaults
3. **Core functionality** — test primary behavior
4. **Error handling** — test exception paths
5. **Integration** — test with real/mock dependencies

### Writing Tests

```python
"""Tests for FooMiddleware."""

import pytest
from madakit.middleware.foo import FooMiddleware


class TestModuleExports:
    """Verify module exports."""

    def test_foo_middleware_in_all(self) -> None:
        """__all__ contains FooMiddleware."""
        from madakit.middleware import foo
        assert "FooMiddleware" in foo.__all__


class TestFooMiddlewareConstructor:
    """Test FooMiddleware initialization."""

    def test_client_required(self) -> None:
        """Constructor requires client parameter."""
        with pytest.raises(TypeError):
            FooMiddleware()  # type: ignore


@pytest.mark.asyncio
class TestSendRequest:
    """Test send_request behavior."""

    async def test_returns_agent_response(self) -> None:
        """send_request returns AgentResponse."""
        # ... test implementation
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/middleware/test_retry.py

# Run with coverage
pytest --cov=madakit --cov-report=html

# Run specific test
pytest tests/middleware/test_retry.py::TestRetryMiddleware::test_exponential_backoff
```

### Test Coverage

- Aim for >90% coverage on new code
- All public APIs must be tested
- Edge cases and error paths must be covered

## Documentation

### Docstring Format

```python
def send_request(self, request: AgentRequest) -> AgentResponse:
    """Send a request and return the response.

    Args:
        request: The agent request containing prompt and parameters.

    Returns:
        AgentResponse with content and metadata.

    Raises:
        ProviderError: If the API call fails.
        MiddlewareError: If middleware processing fails.
    """
```

### Documentation Updates

When adding features:
1. **API Reference** — update `docs/api-reference.md`
2. **User Guide** — add examples to `docs/user-guide.md`
3. **Tutorial** — add recipes to `docs/tutorial.md` if applicable
4. **README** — update feature list and examples if applicable
5. **CHANGELOG** — add entry under `[Unreleased]`

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (under `[Unreleased]`)
- [ ] Commit messages follow convention

### PR Review Process

1. **Automated checks** — CI runs tests, linting, type checking
2. **Code review** — maintainer reviews code quality and design
3. **Discussion** — address feedback and questions
4. **Approval** — maintainer approves when ready
5. **Merge** — maintainer merges to `main`

### Review Criteria

- Code follows project standards
- Tests are comprehensive
- Documentation is clear and complete
- Changes are backward-compatible (unless major version bump)
- No unnecessary dependencies added
- Performance impact is acceptable

## Community

### Getting Help

- **GitHub Issues** — bug reports and feature requests
- **Discussions** — questions and community support
- **Documentation** — comprehensive guides in `docs/`

### Reporting Bugs

When reporting bugs, include:
1. **Description** — what happened vs. what you expected
2. **Reproduction** — minimal code to reproduce the issue
3. **Environment** — Python version, OS, madakit version
4. **Traceback** — full error message and stack trace

### Feature Requests

When requesting features:
1. **Use case** — why do you need this feature?
2. **Proposed API** — how should it work?
3. **Alternatives** — what workarounds exist today?

### Questions

For questions about usage:
- Check the [documentation](docs/)
- Search [existing issues](https://github.com/madahub/madakit/issues)
- Open a [discussion](https://github.com/madahub/madakit/discussions)

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes for their contributions
- Special thanks in major releases

---

Thank you for contributing to madakit! Your efforts help make AI development more composable and accessible for everyone.
