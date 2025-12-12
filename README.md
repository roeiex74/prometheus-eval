# Prometheus-Eval

**A Comprehensive Framework for Rigorous Evaluation of Prompt Effectiveness**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-Academic-green.svg)]()
[![Phase](https://img.shields.io/badge/phase-1%20Complete-success.svg)]()

---

## Overview

Prometheus-Eval is an academic research framework designed to transform prompt engineering from an intuitive art into a rigorous science. The framework addresses the fundamental challenge of evaluating Large Language Model (LLM) prompts through a comprehensive suite of quantitative metrics, Docker-based code execution sandboxes, and advanced visualization tools.

Unlike traditional "trial-and-error" approaches to prompt engineering, Prometheus-Eval provides:

- **Multi-Dimensional Evaluation**: Assess prompt effectiveness across lexical, semantic, and logical dimensions
- **Rigorous Metrics**: Implementation of BLEU, BERTScore, Pass@k, and other academic-standard metrics
- **Isolated Execution**: Docker-sandboxed code execution for secure Pass@k evaluation
- **Production-Ready**: Type-safe configuration, async batch processing, and comprehensive error handling
- **Research-Oriented**: Detailed mathematical foundations and references to peer-reviewed research

### What Makes It Unique

- **Comprehensive Metric Suite**: From n-gram overlap (BLEU) to contextual embeddings (BERTScore) to code execution (Pass@k)
- **Safety-First Design**: Docker-based sandboxing ensures secure code execution without compromising your system
- **Academic Rigor**: All metrics implemented per peer-reviewed papers with full mathematical specifications
- **Extensible Architecture**: Modular design allows easy integration of new metrics and LLM providers

---

## Features (Phase 1 - Complete)

### LLM Inference Engine
- Multi-provider support (OpenAI, Anthropic)
- Async batch processing with rate limiting
- Automatic retry logic with exponential backoff
- Type-safe configuration management
- Comprehensive error handling and logging

### Core Metrics

#### Lexical Metrics
- **BLEU**: N-gram overlap precision with brevity penalty
- Configurable smoothing (epsilon, add-k)
- Multi-reference support
- Corpus-level computation

#### Semantic Metrics
- **BERTScore**: Contextual embedding-based similarity
- Token-level alignment with greedy matching
- Support for multiple BERT variants (BERT, RoBERTa, DeBERTa)
- Precision, Recall, and F1 computation

#### Logic-Based Metrics
- **Pass@k**: Code correctness evaluation
- Docker-sandboxed Python execution
- Configurable timeout and resource limits
- Batch execution support

### Code Execution Sandbox
- Docker-based isolation
- Configurable memory and CPU limits
- Secure execution environment
- Automatic cleanup and resource management

---

## Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Docker**: Required for Pass@k code execution
- **Operating System**: macOS, Linux, or Windows with WSL2

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/prometheus-eval.git
cd prometheus-eval
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for inference engine
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: HuggingFace token for gated models
HUGGINGFACE_API_TOKEN=your_token_here
```

### Step 4: Build Docker Image (for Pass@k)

```bash
cd docker-images/python-sandbox
docker build -t prometheus-sandbox:latest .
cd ../..
```

### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## Quick Start

### Example 1: Using the Inference Engine

```python
from src.inference.openai_provider import OpenAIProvider
import os

# Initialize provider
provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    default_model="gpt-4-turbo-preview",
    temperature=0.7
)

# Generate response
response = provider.generate(
    prompt="Explain quantum entanglement in simple terms.",
    max_tokens=256
)

print(response['text'])
print(f"Tokens used: {response['usage']['total_tokens']}")
```

### Example 2: Computing BLEU Score

```python
from src.metrics.lexical.bleu import BLEUMetric

# Initialize BLEU metric
metric = BLEUMetric(max_n=4, smoothing="epsilon")

# Compute score
result = metric.compute(
    hypothesis="The cat is sitting on the mat",
    reference="The cat is on the mat"
)

print(f"BLEU Score: {result['bleu']:.4f}")
print(f"Precision by n-gram: {result['precisions']}")
print(f"Brevity Penalty: {result['bp']}")
```

### Example 3: Computing BERTScore

```python
from src.metrics.semantic.bertscore import BERTScoreMetric

# Initialize BERTScore metric
metric = BERTScoreMetric(
    model_name="microsoft/deberta-base-mnli",
    device="cpu"
)

# Compute semantic similarity
result = metric.compute(
    hypothesis="The feline sat on the rug",
    reference="The cat is on the mat"
)

print(f"Precision: {result['precision']:.4f}")
print(f"Recall: {result['recall']:.4f}")
print(f"F1 Score: {result['f1']:.4f}")
```

### Example 4: Computing Pass@k

```python
from src.metrics.logic.pass_at_k import PassAtKMetric
from src.evaluator.executor import DockerExecutor

# Initialize executor and metric
executor = DockerExecutor(
    image="prometheus-sandbox:latest",
    timeout=10
)
metric = PassAtKMetric(executor=executor)

# Define test cases
test_cases = [
    {"input": "5", "expected_output": "120"},  # factorial(5)
    {"input": "3", "expected_output": "6"},
]

# Generate code samples (from LLM)
code_samples = [
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
    "def factorial(n):\n    result = 1\n    for i in range(1, n+1):\n        result *= i\n    return result",
]

# Compute Pass@k
result = metric.compute(
    code_samples=code_samples,
    test_cases=test_cases,
    k_values=[1, 2]
)

print(f"Pass@1: {result['pass@1']:.2%}")
print(f"Pass@2: {result['pass@2']:.2%}")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Prometheus-Eval System                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
        │   Inference  │ │  Metrics  │ │  Evaluator  │
        │    Engine    │ │  Engine   │ │   (Pass@k)  │
        └──────────────┘ └───────────┘ └─────────────┘
                │               │               │
        ┌───────┴───────┐       │       ┌───────┴────────┐
        │               │       │       │                │
    ┌───▼───┐     ┌────▼───┐   │   ┌───▼────┐   ┌──────▼──────┐
    │OpenAI │     │Anthropic│   │   │ Docker │   │Test Harness │
    └───────┘     └─────────┘   │   │Sandbox │   └─────────────┘
                                │   └────────┘
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼────┐ ┌───▼────┐ ┌───▼────┐
              │ Lexical  │ │Semantic│ │ Logic  │
              │  (BLEU)  │ │(BERT)  │ │(Pass@k)│
              └──────────┘ └────────┘ └────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼────┐ ┌───▼────┐ ┌───▼────┐
              │ Variator │ │Analysis│ │  Viz   │
              │ (Phase 2)│ │(Phase 2)│ │(Phase 3)│
              └──────────┘ └────────┘ └────────┘
```

### Component Descriptions

#### Inference Engine (`src/inference/`)
Handles LLM API interactions with support for multiple providers. Includes:
- Abstract base class for provider interface
- OpenAI and Anthropic implementations
- Rate limiting and retry logic
- Async batch processing
- Token usage tracking

#### Metrics Engine (`src/metrics/`)
Implements evaluation metrics across three categories:

**Lexical Metrics** (`src/metrics/lexical/`)
- BLEU: N-gram precision with brevity penalty
- ROUGE: Recall-oriented metrics (planned)
- METEOR: Synonym-aware evaluation (planned)

**Semantic Metrics** (`src/metrics/semantic/`)
- BERTScore: Contextual embedding similarity
- Semantic Stability: Multi-run consistency (planned)

**Logic-Based Metrics** (`src/metrics/logic/`)
- Pass@k: Code correctness evaluation
- G-Eval: LLM-based evaluation (planned Phase 2)

#### Evaluator (`src/evaluator/`)
Manages secure code execution for Pass@k:
- Docker-based sandboxing
- Resource limits (CPU, memory, timeout)
- Test case execution
- Result validation

#### Variator (Phase 2 - Planned)
Generates prompt variations for comparative analysis:
- Paraphrasing with LLMs
- Emotional prompting intensity scaling
- Chain-of-Thought augmentation
- Few-shot example selection

#### Visualization (Phase 3 - Planned)
Interactive dashboard for metric analysis:
- Multi-dimensional metric plots
- Prompt comparison heatmaps
- Trade-off analysis (Pareto frontiers)
- Export functionality

---

## Metrics Reference

### BLEU (Bilingual Evaluation Understudy)

**Purpose**: Measures n-gram overlap precision between candidate and reference texts.

**Formula**:
```
BLEU = BP × exp(Σ(w_n × log p_n))

Where:
- p_n: modified n-gram precision
- w_n: uniform weights (1/4 for 4-grams)
- BP: brevity penalty = min(1, exp(1 - r/c))
```

**Use Cases**:
- Structured data generation (SQL, regex)
- Translation quality
- Template adherence

**Parameters**:
- `max_n`: Maximum n-gram order (default: 4)
- `smoothing`: Zero-count handling ("none", "epsilon", "add-k")

**Reference**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"

---

### BERTScore

**Purpose**: Measures semantic similarity using contextual embeddings.

**Formula**:
```
R_BERT = (1/|x|) × Σ max(x_i^T × x̂_j)
         x_i∈x  x̂_j∈x̂

Where x_i and x̂_j are token embeddings from BERT.
```

**Use Cases**:
- Paraphrase detection
- Semantic consistency
- Content preservation in style transfer

**Parameters**:
- `model_name`: BERT variant (default: "microsoft/deberta-base-mnli")
- `device`: Computation device ("cpu", "cuda", "mps")
- `batch_size`: Batch size for embedding computation

**Reference**: Zhang et al. (2020) - "BERTScore: Evaluating Text Generation with BERT"

---

### Pass@k

**Purpose**: Evaluates functional correctness of generated code.

**Formula**:
```
Pass@k = 1 - (C(n-c, k) / C(n, k))

Where:
- n: total code samples
- c: correct samples
- k: samples selected
- C(n, k): binomial coefficient
```

**Use Cases**:
- Code generation quality
- Programming benchmark evaluation
- LLM coding capability assessment

**Parameters**:
- `k_values`: List of k values to compute (e.g., [1, 5, 10])
- `timeout`: Maximum execution time per test (seconds)
- `memory_limit`: Docker container memory limit

**Reference**: Chen et al. (2021) - "Evaluating Large Language Models Trained on Code"

---

## Configuration

### Environment Variables

All configuration is managed through environment variables (see `.env.example`):

**LLM API Keys**:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_token_here  # Optional
```

**Inference Settings**:
```bash
DEFAULT_OPENAI_MODEL=gpt-4-turbo-preview
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048
OPENAI_RPM_LIMIT=60  # Requests per minute
```

**Embedding Models**:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu  # Options: cpu, cuda, mps
```

**Docker Sandbox**:
```bash
DOCKER_TIMEOUT=10  # Seconds
DOCKER_MEMORY_LIMIT=512m
DOCKER_CPU_QUOTA=50000  # 0.5 CPU core
```

**Logging**:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=./logs/prometheus_eval.log
```

### Security Best Practices

1. **API Key Management**:
   - Never commit `.env` file to version control
   - Use separate keys for development and production
   - Rotate keys regularly

2. **Code Execution**:
   - Always use Docker sandbox for untrusted code
   - Set appropriate resource limits
   - Review `DISABLE_SANDBOX=false` setting (never disable in production)

3. **Rate Limiting**:
   - Configure appropriate RPM limits for your API tier
   - Monitor usage to avoid unexpected costs

---

## Development

### Running Tests

Execute the full test suite:

```bash
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_metrics/  # Only metric tests
pytest tests/test_inference/  # Only inference tests

# Run with verbose output
pytest tests/ -v

# Run with debug logging
pytest tests/ --log-cli-level=DEBUG
```

### Code Structure

```
prometheus-eval/
├── src/                      # Source code
│   ├── inference/           # LLM provider implementations
│   │   ├── base.py         # Abstract provider interface
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── metrics/            # Metric implementations
│   │   ├── lexical/        # BLEU, ROUGE, METEOR
│   │   ├── semantic/       # BERTScore, stability
│   │   └── logic/          # Pass@k, G-Eval
│   ├── evaluator/          # Code execution
│   │   └── executor.py     # Docker sandbox manager
│   ├── variator/           # Prompt generators (Phase 2)
│   └── visualization/      # Dashboard (Phase 3)
├── tests/                   # Test suite
│   ├── test_metrics/
│   ├── test_inference/
│   └── fixtures/           # Test data and mocks
├── data/                    # Datasets and experiments
├── docker-images/           # Sandbox Docker images
├── docs/                    # Additional documentation
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

### Contributing Guidelines

This is an academic research project. Contributions are welcome in the following areas:

1. **New Metrics**: Implement additional evaluation metrics with proper mathematical foundations
2. **LLM Providers**: Add support for new providers (Cohere, Google, local models)
3. **Optimizations**: Performance improvements for batch processing
4. **Documentation**: Improve examples, tutorials, and API documentation
5. **Bug Fixes**: Address issues found during testing

**Contribution Process**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-metric`)
3. Implement changes with tests and documentation
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request with detailed description

---

## Roadmap

### Phase 1: Foundation & Core Metrics ✅ COMPLETE
- [x] LLM Inference Engine (OpenAI, Anthropic)
- [x] BLEU metric implementation
- [x] BERTScore metric implementation
- [x] Pass@k metric with Docker sandbox
- [x] Type-safe configuration management
- [x] Comprehensive test suite
- [x] Documentation and examples

### Phase 2: Additional Metrics & Prompt Variator 🚧 IN PROGRESS
- [ ] ROUGE metric family (ROUGE-N, ROUGE-L)
- [ ] METEOR with synonym matching
- [ ] Semantic Stability Score
- [ ] Perplexity and token probability metrics
- [ ] G-Eval (LLM-as-a-judge)
- [ ] Prompt Variator engine
  - [ ] Paraphrasing module
  - [ ] Emotional intensity scaling
  - [ ] Chain-of-Thought augmentation
  - [ ] Few-shot example selection

### Phase 3: Visualization Dashboard & Advanced Features 📋 PLANNED
- [ ] Interactive web dashboard
  - [ ] Multi-metric comparison plots
  - [ ] Prompt A/B testing interface
  - [ ] Pareto frontier analysis
- [ ] Advanced features
  - [ ] Automated prompt optimization
  - [ ] Multi-objective optimization
  - [ ] Benchmark dataset integration
  - [ ] Export and reporting tools

### Future Considerations
- Local LLM support (vLLM, llama.cpp)
- GPU acceleration for metric computation
- Distributed evaluation for large-scale experiments
- Integration with MLflow/Weights & Biases

---

## License & Citation

### License

This project is developed for academic purposes as part of a Master's program in Computer Science. The code is provided as-is for educational and research use.

### How to Cite

If you use Prometheus-Eval in your research, please cite:

```bibtex
@software{prometheus_eval_2024,
  title = {Prometheus-Eval: A Comprehensive Framework for Rigorous Evaluation of Prompt Effectiveness},
  author = {[Your Name]},
  year = {2024},
  url = {https://github.com/your-org/prometheus-eval},
  note = {Academic Research Project}
}
```

### Acknowledgments

This framework builds upon the following foundational research:

- **BLEU**: Papineni et al. (2002) - Automatic evaluation of machine translation
- **BERTScore**: Zhang et al. (2020) - Text generation evaluation with BERT
- **Pass@k**: Chen et al. (2021) - Code generation evaluation methodology
- **Chain-of-Thought**: Wei et al. (2022) - Reasoning in language models
- **Emotional Prompting**: Li et al. (2023) - Affective stimuli in prompts

Special thanks to the open-source community for providing the foundational libraries that made this project possible.

---

## References

### Key Papers

1. **Papineni, K., et al. (2002)**. "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL 2002*.

2. **Zhang, T., et al. (2020)**. "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*.

3. **Chen, M., et al. (2021)**. "Evaluating Large Language Models Trained on Code." *arXiv:2107.03374*.

4. **Wei, J., et al. (2022)**. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*.

5. **Li, C., et al. (2023)**. "EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement." *arXiv:2307.11760*.

### Documentation

- **Full Product Requirements Document**: [PRD.md](./PRD.md)
- **API Documentation**: Coming in Phase 2
- **Tutorial Notebooks**: Coming in Phase 3

### External Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Docker SDK for Python](https://docker-py.readthedocs.io)

---

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/your-org/prometheus-eval/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/prometheus-eval/discussions)
- **Email**: [your.email@university.edu]

---

**Built with academic rigor for the future of prompt engineering.**
