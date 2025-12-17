# Prometheus-Eval Architecture Documentation

**Version:** 1.0
**Last Updated:** 2025-12-17
**Status:** Active Development

---

## Table of Contents

1. [System Overview](#system-overview)
2. [C4 Model Diagrams](#c4-model-diagrams)
3. [Component Architecture](#component-architecture)
4. [Data Flow](#data-flow)
5. [Key Design Decisions](#key-design-decisions)
6. [Extension Points](#extension-points)

---

## System Overview

Prometheus-Eval is designed as a modular, extensible framework for evaluating prompt effectiveness through systematic comparison of prompting techniques. The architecture follows the **Building Blocks Design Pattern** with clear separation of concerns and well-defined interfaces.

### Core Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new variators, metrics, or providers
3. **Type Safety**: Pydantic models for configuration validation
4. **Performance**: Multiprocessing for parallel execution
5. **Testability**: 70%+ test coverage with comprehensive unit tests

### Technology Stack

- **Language**: Python 3.11+
- **LLM Integration**: OpenAI, Anthropic SDKs
- **Metrics**: NLTK, SacreBLEU, BERTScore, transformers
- **Execution**: Docker for sandboxed code execution
- **Testing**: pytest, pytest-cov
- **Configuration**: Pydantic, python-dotenv

---

## C4 Model Diagrams

### Level 1: System Context

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                  Prometheus-Eval Framework                  │
│                                                             │
│  Evaluates LLM prompts through systematic comparison        │
│  of baseline vs. improved prompting techniques             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                    ↑                         ↑
                    │                         │
                    │                         │
        ┌───────────┴──────────┐  ┌──────────┴────────────┐
        │                      │  │                       │
        │  Researcher/User     │  │  LLM Provider APIs    │
        │                      │  │  (OpenAI, Anthropic)  │
        │  • Runs experiments  │  │                       │
        │  • Analyzes results  │  │  • GPT-3.5/4         │
        │  • Views graphs      │  │  • Claude            │
        │                      │  │                       │
        └──────────────────────┘  └───────────────────────┘
```

**Description**: The system serves researchers who want to rigorously evaluate prompt effectiveness by running experiments that compare different prompting techniques and analyzing results.

### Level 2: Container Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                     Prometheus-Eval System                            │
│                                                                       │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │              │      │              │      │              │      │
│  │   Variator   │─────▶│  Inference   │─────▶│   Metrics    │      │
│  │   Module     │      │   Engine     │      │   Engine     │      │
│  │              │      │              │      │              │      │
│  │ • Baseline   │      │ • OpenAI     │      │ • Accuracy   │      │
│  │ • Few-Shot   │      │ • Anthropic  │      │ • BLEU       │      │
│  │ • CoT        │      │ • Retry      │      │ • BERTScore  │      │
│  │ • CoT++      │      │ • Rate Limit │      │              │      │
│  │              │      │              │      │              │      │
│  └──────────────┘      └──────────────┘      └──────────────┘      │
│         │                     │                     │               │
│         └─────────────────────┼─────────────────────┘               │
│                               ↓                                     │
│                    ┌──────────────────┐                            │
│                    │                  │                            │
│                    │   Experiment     │                            │
│                    │   Framework      │                            │
│                    │                  │                            │
│                    │ • Runner         │                            │
│                    │ • Evaluator      │                            │
│                    │ • Multiprocessing│                            │
│                    │ • Results        │                            │
│                    │                  │                            │
│                    └──────────────────┘                            │
│                               │                                     │
│                               ↓                                     │
│                    ┌──────────────────┐                            │
│                    │                  │                            │
│                    │  Visualization   │                            │
│                    │    Module        │                            │
│                    │                  │                            │
│                    │ • Jupyter        │                            │
│                    │ • Matplotlib     │                            │
│                    │ • 300 DPI Export │                            │
│                    │                  │                            │
│                    └──────────────────┘                            │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

        External Systems:

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │              │
│  OpenAI API  │    │Anthropic API │    │   File       │
│              │    │              │    │   System     │
│ GPT-3.5/4    │    │ Claude       │    │              │
│              │    │              │    │ • Datasets   │
│              │    │              │    │ • Results    │
│              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

**Description**: The system consists of five main containers that work together to run experiments and generate results.

### Level 3: Component Diagram

#### Variator Module Components

```
┌─────────────────────────────────────────────────────────┐
│                   Variator Module                       │
│                                                         │
│  ┌────────────────────────────────────────────┐        │
│  │          BaseVariator (Abstract)           │        │
│  │  • generate_prompt()                       │        │
│  │  • get_metadata()                          │        │
│  │  • _validate_config()                      │        │
│  └────────────────────────────────────────────┘        │
│                      ↑                                  │
│          ┌───────────┼───────────┬───────────┐        │
│          │           │           │           │         │
│  ┌───────┴──┐ ┌──────┴───┐ ┌────┴─────┐ ┌───┴──────┐│
│  │Baseline  │ │FewShot   │ │   CoT    │ │  CoT++   ││
│  │Variator  │ │Variator  │ │Variator  │ │Variator  ││
│  │          │ │          │ │          │ │          ││
│  │Simple    │ │1-3       │ │Step-by-  │ │Majority  ││
│  │prompts   │ │examples  │ │step      │ │voting    ││
│  │          │ │          │ │reasoning │ │          ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Experiment Framework Components

```
┌─────────────────────────────────────────────────────────┐
│              Experiment Framework                       │
│                                                         │
│  ┌────────────────────────────────────────────┐        │
│  │         ExperimentRunner                   │        │
│  │  • load_dataset()                          │        │
│  │  • run_experiment()                        │        │
│  │  • _run_single_variator()                  │        │
│  │  • _run_parallel_inference() [MULTIPROC]  │        │
│  │  • _create_comparison()                    │        │
│  │  • _save_results()                         │        │
│  └────────────────────────────────────────────┘        │
│                      │                                  │
│                      │ uses                             │
│                      ↓                                  │
│  ┌────────────────────────────────────────────┐        │
│  │        AccuracyEvaluator                   │        │
│  │  • evaluate()                              │        │
│  │  • _compare()                              │        │
│  │  • _fuzzy_compare()                        │        │
│  │  • _validate_inputs()                      │        │
│  │                                            │        │
│  │  Configuration:                            │        │
│  │  • case_sensitive: bool                    │        │
│  │  • normalize_whitespace: bool              │        │
│  │  • fuzzy_match: bool                       │        │
│  │  • fuzzy_threshold: float                  │        │
│  └────────────────────────────────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Level 4: Code-Level Design

#### Building Block Pattern (Example: AccuracyEvaluator)

```python
class AccuracyEvaluator:
    """
    Building Block: Accuracy Evaluation

    Input Data:
        - predictions: List[str] - Model predictions
        - ground_truth: List[str] - Expected answers
        - dataset_items: Optional[List[Dict]] - Original data with metadata

    Output Data:
        - accuracy: float - Overall accuracy (0-1)
        - correct_count: int - Number correct
        - total_count: int - Total predictions
        - per_category_accuracy: Dict[str, float] - Breakdown by category
        - errors: List[Dict] - Error details

    Setup Data:
        - case_sensitive: bool - Case matching (default: False)
        - normalize_whitespace: bool - Whitespace handling (default: True)
        - fuzzy_match: bool - Allow fuzzy matching (default: False)
        - fuzzy_threshold: float - Fuzzy threshold (default: 0.8)
    """
```

This pattern is applied consistently across all building blocks:
- **BaseVariator** and subclasses
- **AbstractLLMProvider** and implementations
- **Metric classes** (BLEUMetric, BERTScoreMetric, etc.)
- **ExperimentRunner**

---

## Component Architecture

### 1. Variator Module (`src/variator/`)

**Purpose**: Generate systematic prompt variations for comparative experiments

**Components**:
- `base.py`: Abstract base class defining the variator interface
- `baseline.py`: Simple prompt wrapper (control group)
- `few_shot.py`: Adds 1-3 demonstration examples
- `cot.py`: Chain-of-Thought reasoning prompts
- `cot_plus.py`: CoT with self-consistency (majority voting)

**Key Interfaces**:
```python
class BaseVariator(ABC):
    @abstractmethod
    def generate_prompt(self, base_prompt: str, **kwargs) -> Dict[str, Any]:
        """Returns: {prompt: str, metadata: dict}"""
        pass
```

### 2. Inference Engine (`src/inference/`)

**Purpose**: Integrate with LLM providers (OpenAI, Anthropic)

**Components**:
- `base.py`: Abstract provider with retry logic and rate limiting
- `openai_provider.py`: OpenAI GPT-3.5/4 integration
- `anthropic_provider.py`: Anthropic Claude integration
- `config_loader.py`: Environment-based configuration

**Key Features**:
- Automatic retry with exponential backoff (tenacity)
- Rate limiting (asyncio-throttle)
- Token counting and usage tracking
- Async batch processing

### 3. Metrics Engine (`src/metrics/`)

**Purpose**: Evaluate prompt effectiveness quantitatively

**Structure**:
```
metrics/
├── lexical/
│   ├── bleu.py       # N-gram overlap
│   ├── rouge.py      # Recall-oriented n-gram
│   └── meteor.py     # Alignment-based
├── semantic/
│   ├── bertscore.py  # Contextual embeddings
│   ├── stability.py  # Consistency across runs
│   └── tone.py       # Emotional analysis
└── logic/
    ├── pass_at_k.py  # Code correctness
    └── perplexity.py # Token probability
```

### 4. Experiment Framework (`src/experiments/`)

**Purpose**: Orchestrate experiments with parallel processing

**Components**:
- `runner.py`: Main experiment orchestration
  - Loads datasets
  - Runs variators in parallel
  - Collects and aggregates results
  - Generates comparison reports

- `evaluator.py`: Accuracy evaluation
  - Case-insensitive matching
  - Fuzzy matching for close answers
  - Per-category accuracy breakdown

**Multiprocessing Implementation**:
```python
def _run_parallel_inference(self, prompts: List[str]) -> List[str]:
    """Uses multiprocessing.Pool for CPU-bound LLM calls"""
    with Pool(processes=self.num_workers) as pool:
        results = pool.map(self._process_single_prompt, prompts)
    return results
```

### 5. Visualization (`notebooks/`)

**Purpose**: Generate publication-quality figures (300 DPI)

**Components**:
- `results_analysis.ipynb`: Jupyter notebook for analysis
  - Bar charts comparing techniques
  - Statistical significance tests (t-tests)
  - Heatmaps for multi-metric comparison
  - LaTeX table generation

---

## Data Flow

### Experiment Execution Flow

```
1. User invokes run_experiments.py
        │
        ↓
2. ExperimentRunner initializes
   - Loads dataset (JSON)
   - Initializes LLM provider
   - Creates variators
        │
        ↓
3. For each variator:
   a. Generate prompts using variator
   b. Run parallel inference (multiprocessing)
        │
        ├─→ Worker 1: Process prompts 1-25
        ├─→ Worker 2: Process prompts 26-50
        ├─→ Worker 3: Process prompts 51-75
        └─→ Worker 4: Process prompts 76-100
        │
   c. Collect responses
   d. Evaluate accuracy
        │
        ↓
4. Create comparison summary
   - Calculate improvements
   - Identify best variator
   - Per-category breakdown
        │
        ↓
5. Save results to JSON
   - Individual variator results
   - Comparison summary
   - Metadata (timestamp, config)
        │
        ↓
6. Display summary in terminal
```

### Data Formats

**Dataset Format** (`data/datasets/*.json`):
```json
[
  {
    "id": 1,
    "input": "What is 2+2?",
    "expected": "4",
    "category": "math",
    "subcategory": "arithmetic"
  }
]
```

**Results Format** (`results/experiments/*/results.json`):
```json
{
  "experiment_id": "sentiment_20251217_123456",
  "dataset": "data/datasets/sentiment_analysis.json",
  "timestamp": "2025-12-17T12:34:56",
  "results": {
    "BaselineVariator": {
      "accuracy": 0.65,
      "correct_count": 39,
      "total_count": 60,
      "total_time": 12.5,
      "per_category_accuracy": {...}
    },
    "ChainOfThoughtVariator": {
      "accuracy": 0.85,
      ...
    }
  },
  "comparison": {
    "best_variator": "ChainOfThoughtVariator",
    "improvements": {
      "ChainOfThoughtVariator": "+30.8%"
    }
  }
}
```

---

## Key Design Decisions

### ADR-001: Building Blocks Design Pattern

**Status**: Accepted
**Date**: 2025-12-17

**Context**: Need clear, reusable components with well-defined interfaces for prompt evaluation framework.

**Decision**: Adopt Building Blocks Design Pattern with mandatory Input/Output/Setup documentation.

**Rationale**:
- **Clarity**: Each component has explicit data contracts
- **Testability**: Clear inputs/outputs make unit testing straightforward
- **Reusability**: Components can be used independently
- **Validation**: Setup data validation prevents runtime errors

**Consequences**:
- All classes must document Input/Output/Setup in docstrings
- Configuration validation required in constructors
- Consistent error handling patterns across components

### ADR-002: Multiprocessing for Experiment Execution

**Status**: Accepted
**Date**: 2025-12-17

**Context**: Running experiments sequentially is too slow (minutes for 60 samples).

**Decision**: Use `multiprocessing.Pool` for parallel LLM API calls.

**Rationale**:
- **Performance**: 4x speedup with 4 workers
- **CPU-Bound**: LLM API calls are CPU-bound (encoding/decoding)
- **Independence**: Each prompt is independent, perfect for parallelization

**Consequences**:
- Must handle serialization (no lambdas in map)
- Rate limiting still required to avoid API limits
- Workers capped at 4 to balance performance and API constraints

**Alternative Considered**: `threading` - Rejected because GIL limits effectiveness for CPU-bound work.

### ADR-003: Fuzzy Matching for Accuracy Evaluation

**Status**: Accepted
**Date**: 2025-12-17

**Context**: Strict exact matching penalizes semantically correct but stylistically different answers.

**Decision**: Implement optional fuzzy matching with configurable threshold.

**Rationale**:
- **Flexibility**: Handles "42" vs "The answer is 42"
- **Real-world**: LLMs often add context to answers
- **Configurable**: Can disable for strict evaluation

**Implementation**: Substring matching + character overlap ratio

---

## Extension Points

### Adding New Variator

1. Create new file in `src/variator/`
2. Inherit from `BaseVariator`
3. Implement `generate_prompt()` method
4. Add Input/Output/Setup documentation
5. Write tests in `tests/test_variator/`
6. Export in `src/variator/__init__.py`

**Example**:
```python
class TreeOfThoughtsVariator(BaseVariator):
    """
    Input Data:
        - base_prompt: str
        - num_branches: int

    Output Data:
        - prompt: str (with ToT instructions)
        - metadata: dict

    Setup Data:
        - max_depth: int
        - branching_factor: int
    """

    def generate_prompt(self, base_prompt: str, **kwargs) -> Dict[str, Any]:
        # Implementation
        pass
```

### Adding New LLM Provider

1. Create file in `src/inference/`
2. Inherit from `AbstractLLMProvider`
3. Implement:
   - `generate()` - Single request
   - `generate_batch()` - Batch requests
   - `count_tokens()` - Token counting
4. Add retry decorator and rate limiting
5. Write provider tests
6. Update configuration

### Adding New Metric

1. Create file in appropriate `src/metrics/` subdirectory
2. Implement `compute()` method
3. Add Building Block documentation
4. Write comprehensive tests
5. Export in metrics `__init__.py`

---

## Performance Considerations

### Multiprocessing Optimization

- **Worker Count**: Set to `min(cpu_count(), 4)` to balance performance and API rate limits
- **Batch Size**: Process full dataset in single batch when possible
- **Overhead**: For <10 samples, run sequentially to avoid multiprocessing overhead

### Memory Management

- **Streaming**: Process results as they arrive
- **Cleanup**: Use context managers for resource management
- **Limits**: Cap error lists at 10 to prevent memory bloat

### API Cost Optimization

- **Model Selection**: Use GPT-3.5-turbo for testing, GPT-4 for final runs
- **Token Limits**: Set max_tokens=256 for short answers
- **Caching**: Save results to avoid re-running experiments

---

## Security Considerations

### API Key Management

- ✅ Keys in `.env` file (git-ignored)
- ✅ `.env.example` with dummy values
- ✅ No hardcoded secrets in source code
- ✅ Environment variable validation on startup

### Input Validation

- ✅ All user inputs validated before processing
- ✅ Type checking with isinstance()
- ✅ Range checking for numeric parameters
- ✅ Path traversal prevention for file operations

### Docker Sandbox (Pass@k)

- ✅ Network isolation (no internet access)
- ✅ Resource limits (CPU, memory, timeout)
- ✅ Non-root execution
- ✅ Automatic cleanup

---

## Future Architecture Enhancements

1. **Async Everywhere**: Convert experiment runner to fully async
2. **Distributed Execution**: Support for Celery/Ray for large-scale experiments
3. **Real-time Dashboard**: Web UI for monitoring experiments
4. **Prompt Optimization Loop**: Automatic variator selection based on results
5. **A/B Testing Framework**: Statistical power analysis and sample size calculation

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-17 | Initial architecture documentation |

**Maintainers**: Prometheus-Eval Development Team
**Last Review**: 2025-12-17
