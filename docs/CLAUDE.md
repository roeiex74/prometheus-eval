# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prometheus-Eval** is an academic research framework for rigorous evaluation of LLM prompt effectiveness. It transforms prompt engineering from an intuitive art into a rigorous science through comprehensive quantitative metrics, Docker-based code execution sandboxes, and academic-standard evaluation techniques.

The framework evaluates prompts across three dimensions:
- **Lexical Metrics** (BLEU, ROUGE, METEOR): N-gram overlap and surface-level similarity
- **Semantic Metrics** (BERTScore, Stability, Tone): Contextual embeddings and consistency
- **Logic-Based Metrics** (Pass@k, Perplexity): Code correctness and token probabilities

## Development Commands

### Environment Setup
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Download NLTK data required for tokenization
python -c "import nltk; nltk.download('punkt')"

# Build Docker sandbox for Pass@k evaluation
cd docker-images/python-sandbox
docker build -t prometheus-eval-sandbox:latest .
cd ../..
```

### Testing
```bash
# Run all tests with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific metric test suites
pytest tests/test_metrics/test_bleu.py -v
pytest tests/test_metrics/test_bertscore.py -v
pytest tests/test_metrics/test_pass_at_k.py -v

# Run inference engine tests
pytest tests/test_inference/ -v

# Run single test with debug logging
pytest tests/test_metrics/test_bleu.py::test_basic_bleu -v --log-cli-level=DEBUG

# Run tests with specific markers (if defined)
pytest -m "not slow" -v
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

## Architecture

### Core Module Structure

```
src/
├── inference/          # LLM provider implementations
│   ├── base.py        # Abstract provider interface with retry logic & rate limiting
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── config_loader.py
├── metrics/           # Evaluation metric implementations
│   ├── lexical/       # BLEU, ROUGE, METEOR
│   ├── semantic/      # BERTScore, Stability, Tone
│   └── logic/         # Pass@k, Perplexity
├── evaluator/         # Docker-based code execution sandbox
│   └── executor.py    # CodeExecutor class for Pass@k
├── variator/          # Prompt variation generators (Phase 2)
└── visualization/     # Interactive dashboard (Phase 3)
```

### Key Design Patterns

#### 1. LLM Provider Architecture
All LLM providers inherit from `AbstractLLMProvider` (`src/inference/base.py`) which provides:
- Automatic retry logic with exponential backoff via tenacity
- Rate limiting via asyncio-throttle (requests per minute)
- Error handling with custom exceptions (`RateLimitError`, `AuthenticationError`, etc.)
- Request/response logging and statistics tracking

**Implementation Pattern:**
```python
class CustomProvider(AbstractLLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        # 1. Validate parameters
        self.validate_parameters(temperature, max_tokens)

        # 2. Log request
        self._log_request(prompt, model)

        # 3. Make API call with retry decorator
        @self.get_retry_decorator()
        async def _request():
            # API call here
            pass

        # 4. Log response and return
        self._log_response(response, tokens_used)
        return response
```

#### 2. Metric Implementation Pattern
All metrics follow a consistent interface:
- Constructor accepts configuration parameters (e.g., `max_n`, `smoothing` for BLEU)
- `compute()` method takes hypothesis and reference(s), returns dict with scores
- Return format includes primary score plus detailed breakdown

**Example from BLEU** (`src/metrics/lexical/bleu.py`):
```python
result = {
    'bleu': final_score,
    'precisions': [p1, p2, p3, p4],
    'bp': brevity_penalty,
    'length_ratio': c/r
}
```

#### 3. Docker-Based Code Execution
`CodeExecutor` (`src/evaluator/executor.py`) provides secure sandboxed execution for Pass@k:
- **Security**: No network access, resource limits, non-root execution
- **Test Harness**: Automatically wraps code with test cases and assertions
- **Function Extraction**: Uses regex to extract function names from code
- **Cleanup**: Automatic container cleanup via context manager

**Critical Implementation Details:**
- Container runs with `network_mode="none"` and `pids_limit=50`
- Timeout enforced via `container.wait(timeout=self.timeout)`
- Test code preparation in `_prepare_test_code()` handles dict/list/scalar inputs
- Always use context manager or manual cleanup to prevent orphaned containers

## Configuration Management

Environment variables are loaded via `.env` file (copy from `.env.example`):

**Required:**
- `OPENAI_API_KEY` - For OpenAI provider
- `ANTHROPIC_API_KEY` - For Anthropic provider

**Important Settings:**
- `DOCKER_TIMEOUT=10` - Code execution timeout (seconds)
- `DOCKER_MEMORY_LIMIT=512m` - Container memory limit
- `EMBEDDING_DEVICE=cpu` - Set to "cuda" or "mps" for GPU acceleration
- `LOG_LEVEL=INFO` - Logging verbosity

**Testing:**
- `USE_MOCK_LLM=false` - Set to true to use fixtures instead of real API calls
- `DISABLE_SANDBOX=false` - NEVER set true in production (security risk)

## Testing Philosophy

The test suite follows academic rigor standards:

1. **Comprehensive Coverage**: All metrics have extensive test coverage (target: >80%)
2. **Mathematical Validation**: Tests verify metric calculations against known ground truth
3. **Edge Cases**: Tests cover zero-length inputs, empty references, Unicode, etc.
4. **Fixtures**: Use `tests/fixtures/` for reusable test data
5. **Mocking**: Mock LLM providers in tests to avoid API costs and ensure determinism

**Test File Naming:**
- `test_<module>.py` - Tests for `src/<module>.py`
- `conftest.py` - Pytest fixtures (in both `tests/` and subdirectories)

## Important Patterns and Conventions

### 1. Token Counting
Different providers use different tokenizers:
- OpenAI: Uses `tiktoken` library with model-specific encodings
- Anthropic: Uses `anthropic.count_tokens()` method
- Always use provider's `count_tokens()` method, not generic tokenizers

### 2. Mathematical Metric Implementation
Follow PRD mathematical specifications exactly:
- BLEU: Section 3.1.1 - Modified n-gram precision with brevity penalty
- BERTScore: Section 3.2.1 - Token-level greedy matching with contextual embeddings
- Pass@k: Section 3.4.1 - Unbiased estimator using binomial coefficient

Reference papers are cited in docstrings.

### 3. Async vs Sync
- Providers implement async methods for batch processing
- `SyncProviderMixin` provides synchronous wrappers via `run_async()`
- Use async for batch operations to leverage concurrency
- Use sync for single requests or when integration doesn't support async

### 4. Logging
Uses `loguru` for structured logging:
- `logger.debug()` - Detailed request/response info
- `logger.info()` - High-level operation status
- `logger.warning()` - Recoverable errors (e.g., retry attempts)
- `logger.error()` - Failures requiring attention

### 5. Error Handling
Custom exception hierarchy in `src/inference/base.py`:
- `LLMProviderError` - Base exception
- `RateLimitError` - Triggers automatic retry
- `AuthenticationError` - Fatal, no retry
- `TimeoutError` - Triggers retry
- `InvalidRequestError` - Fatal, no retry

## Docker Sandbox Architecture

The Pass@k metric requires Docker for secure code execution:

**Image Location:** `docker-images/python-sandbox/Dockerfile`
**Image Name:** `prometheus-eval-sandbox:latest`

**Security Layers:**
1. Minimal Python base image (python:3.11-slim)
2. No network access in container runtime
3. Resource limits (CPU, memory, PIDs)
4. Timeout enforcement
5. Automatic cleanup

**When modifying:**
- Rebuild image after Dockerfile changes
- Test with malicious code patterns (infinite loops, fork bombs, etc.)
- Never disable sandbox in production (`DISABLE_SANDBOX=false`)

## Phase Status

- **Phase 1 (Complete)**: Core metrics (BLEU, BERTScore, Pass@k), inference engine, Docker sandbox
- **Phase 2 (In Progress)**: Additional metrics (ROUGE, METEOR, Stability, Perplexity, Tone)
- **Phase 3 (Planned)**: Visualization dashboard, prompt variator engine

Check `README.md` roadmap section for detailed status.

## Project-Specific Guidelines

### Adding New Metrics
1. Create metric class in appropriate category (`lexical/`, `semantic/`, `logic/`)
2. Follow PRD mathematical specifications (Section 3.x)
3. Include detailed docstring with formula and references
4. Implement `compute()` method with consistent return format
5. Add comprehensive test suite in `tests/test_metrics/`
6. Verify against reference implementations or known ground truth

### Adding New LLM Providers
1. Inherit from `AbstractLLMProvider`
2. Implement `generate()`, `generate_batch()`, `count_tokens()`
3. Use `self.get_retry_decorator()` for automatic retries
4. Use `self._rate_limited_request()` for rate limiting
5. Call `self._log_request()` and `self._log_response()`
6. Add provider tests in `tests/test_inference/test_providers.py`

### Docker Sandbox Modifications
1. Update `docker-images/python-sandbox/Dockerfile`
2. Rebuild image: `docker build -t prometheus-eval-sandbox:latest .`
3. Test with `CodeExecutor` in `tests/test_metrics/test_pass_at_k.py`
4. Verify security constraints (no network, resource limits)
5. Update image name in `.env.example` if changed

## Common Pitfalls

1. **NLTK Data Missing**: Run `nltk.download('punkt')` before using lexical metrics
2. **Docker Image Not Built**: Pass@k tests fail if sandbox image doesn't exist
3. **API Keys Not Set**: Provider tests fail without `.env` configuration
4. **GPU/MPS Device Mismatch**: BERTScore requires correct `EMBEDDING_DEVICE` setting
5. **Token Count Mismatch**: Always use provider-specific tokenizer, not generic ones
6. **Container Cleanup**: Always use `CodeExecutor` as context manager or call `cleanup()`

## Reference Documentation

- **PRD**: `PRD.md` - Mathematical foundations and metric specifications
- **README**: `README.md` - Installation, quick start, architecture overview
- **Environment**: `.env.example` - All configuration options with descriptions
- **Project Status**: `docs/PLAN-LATEST.md` - Current development plan
