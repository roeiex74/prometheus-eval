# Complete Plan for 100% Score - Assignment 6
## Prompt Engineering with Chain of Thought, ReAct, and Tree of Thoughts

**Version:** 1.0
**Date:** 2025-12-17
**Goal:** Achieve 100/100 on Assignment 6

---

## Table of Contents
1. [Assignment Overview](#assignment-overview)
2. [Grading Breakdown](#grading-breakdown)
3. [Academic Requirements (60%)](#academic-requirements-60)
4. [Technical Requirements (40%)](#technical-requirements-40)
5. [Complete Implementation Checklist](#complete-implementation-checklist)
6. [Timeline and Priorities](#timeline-and-priorities)

---

## Assignment Overview

### Core Task (From Lecture)
**The Assignment:** Demonstrate improvement (or degradation) in prompt effectiveness through graphs by comparing different prompt versions.

### Key Concepts to Master:
1. **Entropy** - Minimize entropy for consistent, predictable responses
2. **Atomic Prompts** - Shortest instruction that performs the defined task
3. **Chain of Thought (CoT)** - Step-by-step reasoning (improved accuracy from 18% to 58% on GSM8K)
4. **ReAct** - Combining reasoning and action with external tools
5. **Tree of Thoughts (ToT)** - Exploring multiple reasoning paths (4% to 74% success on Game of 24)
6. **Mass Production at Scale** - Prompts that work for millions of queries

### Assignment Steps (from page 8 of lecture):
1. **Create Dataset** - Question-answer pairs (sentiment analysis, math exercises, logical sentences)
2. **Baseline Measurement** - Run with basic prompt, measure distances to true answers
3. **Improve Prompts** - Try:
   - Regular prompt improvement (system wording changes)
   - Few-Shot Learning (up to 3 examples)
   - Chain of Thought ("think step by step")
   - ReAct (optional - integration with external tools)
4. **Comparison & Visualization** - Show improvement/degradation via graphs

---

## Grading Breakdown

### Total: 100 Points

#### Academic Criteria (60%):
1. **Project Documentation (PRD)** - 20%
2. **Code Documentation & README** - 15%
3. **Project Structure & Code Quality** - 15%
4. **Configuration & Security** - 10%
5. **Testing & QA** - 15%
6. **Research & Analysis** - 15%
7. **UI/UX & Extensibility** - 10%

#### Technical Criteria (40%):
1. **Package Organization (as Python package)** - Technical check A
2. **Multiprocessing/Multithreading** - Technical check B
3. **Building Blocks Design** - Technical check C

---

## Academic Requirements (60%)

### 1. Project Documentation (PRD) - 20%

#### Must Include:

**a) Product Requirements Document (PRD):**
- ✅ Clear description of user problem and project goal
- ✅ Success metrics (KPIs) - How to measure prompt improvement
- ✅ Detailed functional requirements
  - Baseline prompt execution
  - Improved prompt variations (regular, few-shot, CoT, ReAct)
  - Metric computation (distance from ground truth)
  - Visualization generation
- ✅ Non-functional requirements
  - Performance requirements
  - Scalability (can handle large datasets)
  - Token usage limits
- ✅ Constraints, assumptions, and dependencies
- ✅ Timeline with milestones

**b) Architecture Documentation:**
- ✅ C4 Model diagrams (Context, Container, Component, Code)
- ✅ Operational architecture
- ✅ Architecture Decision Records (ADRs) - Document why you chose specific approaches
- ✅ Complete API documentation with interfaces

**PRD Structure Example:**
```markdown
# Product Requirements Document: Prometheus-Eval Prompt Optimization

## Executive Summary
Problem: Current prompt engineering lacks rigorous quantitative evaluation
Solution: Multi-metric framework to evaluate and optimize prompts

## Success Metrics (KPIs)
- Prompt improvement rate (% accuracy gain)
- Entropy reduction (consistency measurement)
- Pass@k scores for code generation tasks
- BLEU/BERTScore for text quality

## Functional Requirements
FR1: Baseline prompt execution with metric computation
FR2: Few-shot learning implementation (1-3 examples)
FR3: Chain of Thought integration
FR4: ReAct framework (optional)
FR5: Comparative visualization (graphs showing improvement)

## Non-Functional Requirements
NFR1: Support for 1000+ test cases
NFR2: Token usage tracking and optimization
NFR3: Execution time < 5 minutes for full evaluation
NFR4: API cost < $5 per full experiment run

[Continue with full PRD following structure in software_submission_guidelines.pdf]
```

---

### 2. Code Documentation & README - 15%

#### Comprehensive README Must Include:

```markdown
# README.md

## Overview
[Clear description of the prompt evaluation framework]

## Installation
### Prerequisites
- Python 3.11+
- Docker (for Pass@k evaluation if doing code tasks)

### Step-by-Step Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

## Usage Instructions
### Quick Start
```python
from src.inference.openai_provider import OpenAIProvider
from src.metrics.lexical.bleu import BLEUMetric

# Initialize provider
provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

# Run baseline
baseline_results = run_baseline_experiment(dataset, provider)

# Run improved prompts
cot_results = run_cot_experiment(dataset, provider)

# Compare
visualize_comparison(baseline_results, cot_results)
```

## Configuration Guide
[Explain all .env variables]

## Troubleshooting
[Common issues and solutions]

## Project Structure
```
prometheus-eval/
├── src/
│   ├── inference/          # LLM providers
│   ├── metrics/            # Evaluation metrics
│   ├── variator/           # Prompt variations
│   └── visualization/      # Result visualization
├── tests/
├── data/
├── results/
└── docs/
```

## Screenshots/Examples
[Include example outputs, graphs]
```

#### Code Comments Quality:
- ✅ Docstrings for EVERY function/class/module
- ✅ Explain WHY for complex design decisions
- ✅ Clear, descriptive variable/function names
- ✅ Comments updated with code changes

---

### 3. Project Structure & Code Quality - 15%

#### Perfect Project Structure:

```
prometheus-eval/
├── src/
│   ├── __init__.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract provider
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── lexical/
│   │   │   ├── __init__.py
│   │   │   └── bleu.py
│   │   ├── semantic/
│   │   │   ├── __init__.py
│   │   │   └── bertscore.py
│   │   └── logic/
│   │       ├── __init__.py
│   │       └── pass_at_k.py
│   ├── variator/
│   │   ├── __init__.py
│   │   ├── baseline.py                # Regular prompts
│   │   ├── few_shot.py                # Few-shot implementation
│   │   ├── cot.py                     # Chain of Thought
│   │   └── react.py                   # ReAct framework
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── comparisons.py             # Graph generation
│   └── evaluator/
│       ├── __init__.py
│       └── executor.py                # Docker sandbox if needed
├── tests/
│   ├── test_inference/
│   ├── test_metrics/
│   └── test_variator/
├── data/
│   ├── datasets/
│   └── prompts/
├── results/
│   ├── experiments/
│   └── visualizations/
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── ADRs/
├── config/
│   └── experiment_config.yaml
├── pyproject.toml                      # Package definition
├── setup.py                            # Alternative package definition
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

#### Code Quality Standards:
- ✅ NO files over 150 lines (break into smaller modules)
- ✅ Single Responsibility Principle - each class/function does ONE thing
- ✅ DRY (Don't Repeat Yourself) - no duplicate code
- ✅ Consistent naming conventions throughout
- ✅ Clear separation: data/code/results/documentation

---

### 4. Configuration & Security - 10%

#### Configuration Files:

**`.env.example`:**
```bash
# LLM API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Model Settings
DEFAULT_MODEL=gpt-4-turbo-preview
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048

# Rate Limiting
OPENAI_RPM_LIMIT=60

# Experiment Settings
EXPERIMENT_DATA_DIR=./data/experiments
RESULTS_DIR=./results
```

**`config/experiment_config.yaml`:**
```yaml
experiments:
  baseline:
    name: "Baseline Prompt"
    temperature: 0.7
    max_tokens: 256

  few_shot:
    name: "Few-Shot Learning"
    num_examples: 3
    temperature: 0.7

  chain_of_thought:
    name: "Chain of Thought"
    cot_trigger: "Let's think step by step"
    temperature: 0.7

metrics:
  - bleu
  - bertscore
  - custom_accuracy

visualization:
  output_format: png
  dpi: 300
```

#### Security Checklist:
- ✅ NO hardcoded API keys in source code
- ✅ Use environment variables for secrets
- ✅ `.gitignore` is up to date (includes `.env`, `*.key`, etc.)
- ✅ Provide `.env.example` with dummy values
- ✅ Document all configuration parameters

---

### 5. Testing & QA - 15%

#### Testing Requirements:

**Target: 70%+ coverage on new code**

```python
# tests/test_variator/test_cot.py
import pytest
from src.variator.cot import ChainOfThoughtVariator

def test_cot_prompt_generation():
    """Test CoT prompt includes step-by-step trigger"""
    variator = ChainOfThoughtVariator()
    prompt = variator.generate_prompt("What is 2+2?")

    assert "Let's think step by step" in prompt
    assert "What is 2+2?" in prompt

def test_cot_with_custom_trigger():
    """Test CoT with custom trigger phrase"""
    variator = ChainOfThoughtVariator(trigger="Let's solve this carefully")
    prompt = variator.generate_prompt("Solve: x + 5 = 10")

    assert "Let's solve this carefully" in prompt

# Edge cases
def test_cot_empty_question():
    """Test CoT handles empty input gracefully"""
    variator = ChainOfThoughtVariator()
    with pytest.raises(ValueError, match="Question cannot be empty"):
        variator.generate_prompt("")

def test_cot_very_long_question():
    """Test CoT handles long inputs"""
    variator = ChainOfThoughtVariator()
    long_question = "What is " + " + ".join(str(i) for i in range(100)) + "?"
    prompt = variator.generate_prompt(long_question)

    assert len(prompt) > 0
    assert "Let's think step by step" in prompt
```

#### Testing Checklist:
- ✅ Unit tests for all core modules
- ✅ Edge case testing (empty inputs, boundary values, etc.)
- ✅ Error handling tests
- ✅ Coverage reports (`pytest --cov=src --cov-report=html`)
- ✅ Automated testing documentation
- ✅ Test results documented with expected outputs

---

### 6. Research & Analysis - 15%

#### Experimental Design:

**This is CRITICAL for the assignment - the graphs showing improvement!**

**Dataset Creation:**
```python
# data/datasets/sentiment_analysis.json
[
    {
        "input": "This movie was absolutely amazing!",
        "expected": "positive",
        "category": "sentiment"
    },
    {
        "input": "I hated every minute of it",
        "expected": "negative",
        "category": "sentiment"
    },
    # ... at least 50-100 examples
]

# data/datasets/math_exercises.json
[
    {
        "input": "If John has 5 apples and gives away 2, then finds 3 more, how many does he have?",
        "expected": "6",
        "category": "math",
        "steps": ["Start: 5", "Give away 2: 5-2=3", "Find 3: 3+3=6"]
    },
    # ... at least 50-100 examples
]
```

**Experimental Protocol:**

1. **Baseline Measurement:**
   - Run all test cases with basic prompt
   - Measure: accuracy, BLEU score, distance from expected
   - Record: token usage, cost, time

2. **Systematic Improvements:**
   - **Regular Prompt:** Improve wording/structure
   - **Few-Shot:** Add 1, 2, 3 examples systematically
   - **CoT:** Add "Let's think step by step"
   - **CoT++:** CoT with majority voting
   - **ReAct:** (Optional) Integration with tools

3. **Parameter Sensitivity Analysis:**
   - Test different temperatures (0.0, 0.3, 0.7, 1.0)
   - Test different models (GPT-3.5, GPT-4, Claude)
   - Identify critical parameters

**Analysis Notebook (Jupyter):**
```python
# notebooks/experiment_analysis.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
baseline = pd.read_json('results/baseline_results.json')
few_shot = pd.read_json('results/few_shot_results.json')
cot = pd.read_json('results/cot_results.json')

# Statistical Analysis
from scipy import stats

# Compare means
t_stat, p_value = stats.ttest_ind(baseline['accuracy'], cot['accuracy'])
print(f"CoT vs Baseline: t={t_stat:.3f}, p={p_value:.4f}")

# Visualization 1: Bar Chart Comparison
plt.figure(figsize=(10, 6))
methods = ['Baseline', 'Improved Wording', 'Few-Shot (3)', 'CoT', 'CoT++']
accuracies = [0.65, 0.72, 0.78, 0.85, 0.89]
plt.bar(methods, accuracies)
plt.title('Prompt Method Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.savefig('results/accuracy_comparison.png', dpi=300)

# Visualization 2: Line Chart - Parameter Sensitivity
temperatures = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
cot_accuracy_by_temp = [0.87, 0.88, 0.87, 0.85, 0.82, 0.78]
plt.figure(figsize=(10, 6))
plt.plot(temperatures, cot_accuracy_by_temp, marker='o')
plt.title('Temperature Sensitivity - CoT Prompts')
plt.xlabel('Temperature')
plt.ylabel('Accuracy')
plt.savefig('results/temperature_sensitivity.png', dpi=300)

# Visualization 3: Heatmap - Multi-metric comparison
metrics_df = pd.DataFrame({
    'Accuracy': [0.65, 0.85, 0.89],
    'BLEU': [0.42, 0.68, 0.71],
    'BERTScore': [0.71, 0.82, 0.85],
}, index=['Baseline', 'CoT', 'CoT++'])

plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df.T, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Multi-Metric Performance Heatmap')
plt.savefig('results/multi_metric_heatmap.png', dpi=300)
```

#### Research Checklist:
- ✅ Systematic experiments with parameter variations
- ✅ Sensitivity analysis documented
- ✅ Results table with experiment parameters and outcomes
- ✅ Statistical validation (t-tests, confidence intervals)
- ✅ Jupyter notebook with analysis
- ✅ Mathematical formulas in LaTeX (if relevant)
- ✅ Academic references cited

#### Required Visualizations:
- ✅ Bar charts comparing prompt methods
- ✅ Line charts showing trends over parameters
- ✅ Heatmaps for multi-dimensional comparisons
- ✅ Clear labels, legends, and high resolution (300 DPI)

---

### 7. UI/UX & Extensibility - 10%

#### User Interface:
- ✅ Clear, intuitive command-line interface
- ✅ Screenshots of workflow in README
- ✅ Accessibility considerations

**Example CLI:**
```bash
# Simple usage
python -m src.main --dataset data/sentiment.json --experiment baseline

# Advanced usage
python -m src.main \
    --dataset data/math.json \
    --experiments baseline,few-shot,cot \
    --output results/experiment_1/ \
    --visualize
```

#### Extensibility:
- ✅ Extension points documented (how to add new metrics, new prompt strategies)
- ✅ Example plugin development guide
- ✅ Clear interfaces for adding new LLM providers

**Example Extension Point:**
```python
# src/metrics/base.py
from abc import ABC, abstractmethod

class MetricBase(ABC):
    """Base class for all metrics - extend this to add new metrics"""

    @abstractmethod
    def compute(self, hypothesis: str, reference: str) -> dict:
        """Compute metric score

        Args:
            hypothesis: Generated text
            reference: Ground truth

        Returns:
            Dict with metric scores
        """
        pass

# To add new metric:
# 1. Create file in src/metrics/category/my_metric.py
# 2. Inherit from MetricBase
# 3. Implement compute() method
# 4. Register in src/metrics/__init__.py
```

---

## Technical Requirements (40%)

### Technical Check A: Package Organization (Python Package)

**CRITICAL: Must be installable as a Python package!**

#### 1. Package Definition Files:

**`pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "prometheus-eval"
version = "0.1.0"
description = "Rigorous evaluation framework for prompt effectiveness"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.12.0",
    "anthropic>=0.18.0",
    "tiktoken>=0.5.2",
    "torch>=2.1.0",
    "transformers>=4.37.0",
    "nltk>=3.8.1",
    "sacrebleu>=2.4.0",
    "numpy>=1.24.0",
    "pandas>=2.1.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "black>=23.12.0",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
exclude = ["tests*", "docs*"]
```

**Alternative `setup.py`:**
```python
from setuptools import setup, find_packages

setup(
    name="prometheus-eval",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        # ... etc
    ],
)
```

#### 2. __init__.py Files:

**`src/__init__.py`:**
```python
"""Prometheus-Eval: Rigorous prompt evaluation framework"""
__version__ = "0.1.0"

from src.inference.openai_provider import OpenAIProvider
from src.inference.anthropic_provider import AnthropicProvider
from src.metrics.lexical.bleu import BLEUMetric

__all__ = [
    "OpenAIProvider",
    "AnthropicProvider",
    "BLEUMetric",
]
```

**`src/variator/__init__.py`:**
```python
"""Prompt variation generators"""
from src.variator.baseline import BaselineVariator
from src.variator.few_shot import FewShotVariator
from src.variator.cot import ChainOfThoughtVariator

__all__ = ["BaselineVariator", "FewShotVariator", "ChainOfThoughtVariator"]
```

#### 3. Relative Imports:
```python
# CORRECT - Relative imports
from src.inference.base import AbstractLLMProvider
from src.metrics.lexical.bleu import BLEUMetric

# INCORRECT - Absolute filesystem imports
# from /Users/.../src/inference/base import AbstractLLMProvider  # NEVER DO THIS
```

#### 4. Package Organization Checklist:
- ✅ `pyproject.toml` OR `setup.py` with all metadata
- ✅ `__init__.py` in every package directory
- ✅ Exports public interfaces via `__all__`
- ✅ `__version__` defined
- ✅ All imports are relative to package name
- ✅ Can install with `pip install -e .`

---

### Technical Check B: Multiprocessing/Multithreading

**Purpose:** Speed up experiment execution by running tests in parallel

#### When to Use What:

- **Multiprocessing**: For CPU-bound operations (LLM API calls, metric computation)
- **Multithreading**: For I/O-bound operations (file reading, network requests)

#### Implementation Example:

**Multiprocessing for LLM API Calls:**
```python
# src/variator/batch_processor.py
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any

class BatchProcessor:
    """Process multiple prompts in parallel using multiprocessing"""

    def __init__(self, provider, num_workers: int = None):
        self.provider = provider
        self.num_workers = num_workers or cpu_count()

    def process_batch(self, prompts: List[str]) -> List[str]:
        """Process multiple prompts in parallel

        Args:
            prompts: List of prompts to process

        Returns:
            List of responses
        """
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single, prompts)
        return results

    def _process_single(self, prompt: str) -> str:
        """Process a single prompt (called by worker processes)"""
        return self.provider.generate(prompt)

# Usage
processor = BatchProcessor(openai_provider, num_workers=4)
results = processor.process_batch(test_prompts)
```

**Thread Safety for Shared Data:**
```python
# src/evaluator/concurrent_evaluator.py
from threading import Lock, Thread
from queue import Queue

class ConcurrentEvaluator:
    """Evaluate prompts concurrently with thread-safe result collection"""

    def __init__(self):
        self.results_lock = Lock()
        self.results = []

    def evaluate_batch(self, test_cases: List[Dict]) -> List[Dict]:
        """Evaluate test cases concurrently"""
        result_queue = Queue()
        threads = []

        for test_case in test_cases:
            thread = Thread(
                target=self._evaluate_single,
                args=(test_case, result_queue)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        return results

    def _evaluate_single(self, test_case: Dict, result_queue: Queue):
        """Evaluate single test case (thread-safe)"""
        result = self.metric.compute(
            test_case['hypothesis'],
            test_case['reference']
        )
        result_queue.put(result)
```

#### Multiprocessing/Multithreading Checklist:
- ✅ Identified CPU-bound vs I/O-bound operations
- ✅ Used multiprocessing for CPU-bound tasks
- ✅ Used multithreading for I/O-bound tasks
- ✅ Set number of workers dynamically (cpu_count())
- ✅ Proper data sharing between processes (Queue, Pipe)
- ✅ Thread safety with locks for shared data
- ✅ Prevent race conditions
- ✅ Prevent deadlocks
- ✅ Proper cleanup (context managers)

---

### Technical Check C: Building Blocks Design

**Purpose:** Modular architecture with clear input/output/setup contracts

#### Building Block Structure:

Every building block (class/module) must define:
1. **Input Data**: What data it needs to operate
2. **Output Data**: What data it produces
3. **Setup Data**: Configuration parameters

**Example Building Block:**

```python
# src/metrics/custom/accuracy.py
class AccuracyMetric:
    """
    Building block for computing accuracy metric

    Input Data:
        - predictions: List[str] - Model predictions
        - ground_truth: List[str] - Expected answers

    Output Data:
        - accuracy: float - Percentage of correct predictions (0-1)
        - correct_count: int - Number of correct predictions
        - total_count: int - Total number of predictions

    Setup Data:
        - case_sensitive: bool - Whether comparison is case-sensitive (default: False)
        - normalize_whitespace: bool - Whether to normalize whitespace (default: True)
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True
    ):
        # Setup/configuration
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self._validate_config()

    def compute(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, Any]:
        """Compute accuracy metric

        Args:
            predictions: Model predictions
            ground_truth: Expected answers

        Returns:
            Dict with accuracy metrics

        Raises:
            ValueError: If predictions and ground_truth lengths don't match
        """
        # Input validation
        self._validate_input(predictions, ground_truth)

        # Processing logic
        correct = self._count_correct(predictions, ground_truth)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        # Return output
        return {
            'accuracy': accuracy,
            'correct_count': correct,
            'total_count': total,
        }

    def _validate_config(self):
        """Validate configuration parameters"""
        # Config validation logic
        pass

    def _validate_input(self, predictions, ground_truth):
        """Validate input data"""
        if not isinstance(predictions, list):
            raise TypeError("predictions must be a list")
        if not isinstance(ground_truth, list):
            raise TypeError("ground_truth must be a list")
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(ground_truth)} ground truth"
            )

    def _count_correct(self, predictions, ground_truth):
        """Count correct predictions"""
        correct = 0
        for pred, truth in zip(predictions, ground_truth):
            if self._compare(pred, truth):
                correct += 1
        return correct

    def _compare(self, pred: str, truth: str) -> bool:
        """Compare prediction with ground truth"""
        # Normalize if needed
        if self.normalize_whitespace:
            pred = ' '.join(pred.split())
            truth = ' '.join(truth.split())

        # Case sensitivity
        if not self.case_sensitive:
            pred = pred.lower()
            truth = truth.lower()

        return pred == truth
```

#### Building Blocks Checklist:
- ✅ Every building block clearly documents Input/Output/Setup data
- ✅ Comprehensive input validation
- ✅ Separation of concerns (each block has single responsibility)
- ✅ Configuration separated from code
- ✅ Building blocks are reusable
- ✅ Building blocks are independently testable
- ✅ Dependency injection (dependencies provided via constructor)

---

## Complete Implementation Checklist

### Phase 1: Setup & Infrastructure (Day 1)
- [ ] Create project directory structure
- [ ] Set up Python package (pyproject.toml/setup.py)
- [ ] Create all __init__.py files
- [ ] Set up .env and .env.example
- [ ] Set up .gitignore
- [ ] Initialize git repository
- [ ] Create requirements.txt
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Verify package installation (`pip install -e .`)

### Phase 2: Documentation Foundation (Day 1-2)
- [ ] Write complete PRD.md
  - [ ] Problem statement
  - [ ] KPIs and success metrics
  - [ ] Functional requirements
  - [ ] Non-functional requirements
  - [ ] Constraints and dependencies
  - [ ] Timeline and milestones
- [ ] Write ARCHITECTURE.md
  - [ ] C4 diagrams
  - [ ] Component descriptions
  - [ ] Data flow
- [ ] Start ADRs (Architecture Decision Records)
- [ ] Create initial README.md structure

### Phase 3: Core Implementation (Day 2-3)

#### 3.1 LLM Inference Layer
- [ ] Create AbstractLLMProvider base class
  - [ ] Define interface (generate, generate_batch, count_tokens)
  - [ ] Add retry logic with tenacity
  - [ ] Add rate limiting
  - [ ] Add logging
- [ ] Implement OpenAIProvider
  - [ ] API integration
  - [ ] Token counting
  - [ ] Error handling
  - [ ] Tests (70%+ coverage)
- [ ] Implement AnthropicProvider (optional but recommended)
  - [ ] API integration
  - [ ] Tests

#### 3.2 Metrics Implementation
- [ ] Create MetricBase abstract class
- [ ] Implement BLEU metric
  - [ ] N-gram precision calculation
  - [ ] Brevity penalty
  - [ ] Smoothing options
  - [ ] Tests (edge cases!)
- [ ] Implement custom accuracy metric
  - [ ] Exact match
  - [ ] Case-insensitive option
  - [ ] Tests
- [ ] Optional: Implement BERTScore
  - [ ] Embedding computation
  - [ ] Greedy matching
  - [ ] Tests

#### 3.3 Prompt Variator Implementation
- [ ] Create BaseVariator abstract class
- [ ] Implement BaselineVariator
  - [ ] Simple prompt wrapper
  - [ ] Tests
- [ ] Implement FewShotVariator
  - [ ] Example selection logic
  - [ ] Format examples in prompt
  - [ ] Tests with 1, 2, 3 examples
- [ ] Implement ChainOfThoughtVariator
  - [ ] Add CoT trigger phrase
  - [ ] Optional: Parse reasoning steps
  - [ ] Tests
- [ ] Implement CoTPlusVariator (CoT++ with majority voting)
  - [ ] Multiple sampling
  - [ ] Voting logic
  - [ ] Tests
- [ ] OPTIONAL: Implement ReActVariator
  - [ ] Tool integration
  - [ ] Action/observation loop
  - [ ] Tests

### Phase 4: Dataset Creation (Day 3)
- [ ] Create sentiment analysis dataset (50-100 examples)
- [ ] Create math problem dataset (50-100 examples)
- [ ] Create logical reasoning dataset (50-100 examples)
- [ ] Validate dataset format
- [ ] Document dataset structure

### Phase 5: Experiment Framework (Day 3-4)
- [ ] Create Experiment runner
  - [ ] Load dataset
  - [ ] Run baseline
  - [ ] Run variations (few-shot, CoT, etc.)
  - [ ] Collect results
  - [ ] Save to JSON/CSV
- [ ] Implement parallel processing
  - [ ] Multiprocessing for LLM calls
  - [ ] Thread-safe result collection
  - [ ] Tests
- [ ] Implement cost tracking
  - [ ] Token usage logging
  - [ ] Cost calculation
  - [ ] Budget warnings

### Phase 6: Analysis & Visualization (Day 4-5)
- [ ] Create Jupyter analysis notebook
  - [ ] Load experiment results
  - [ ] Statistical analysis (t-tests, confidence intervals)
  - [ ] Parameter sensitivity analysis
- [ ] Create visualization functions
  - [ ] Bar chart: Method comparison
  - [ ] Line chart: Parameter sensitivity
  - [ ] Heatmap: Multi-metric comparison
  - [ ] Ensure 300 DPI, clear labels
- [ ] Generate final graphs
- [ ] Document findings

### Phase 7: Testing & QA (Day 5)
- [ ] Write unit tests for all modules
  - [ ] Inference providers
  - [ ] Metrics
  - [ ] Variators
  - [ ] Experiment runner
- [ ] Edge case testing
  - [ ] Empty inputs
  - [ ] Very long inputs
  - [ ] Invalid inputs
- [ ] Run coverage report
  - [ ] `pytest --cov=src --cov-report=html --cov-report=term`
  - [ ] Verify 70%+ coverage
  - [ ] Save coverage badge/report
- [ ] Integration testing
  - [ ] Full experiment pipeline
  - [ ] End-to-end workflow
- [ ] Error handling verification
  - [ ] API failures
  - [ ] Invalid API keys
  - [ ] Network errors
  - [ ] Rate limit handling

### Phase 8: Final Documentation (Day 5-6)
- [ ] Complete README.md
  - [ ] Installation instructions
  - [ ] Quick start examples
  - [ ] Configuration guide
  - [ ] Troubleshooting section
  - [ ] Screenshots/output examples
- [ ] Finalize PRD.md
  - [ ] All sections complete
  - [ ] Diagrams included
  - [ ] ADRs documented
- [ ] Complete ARCHITECTURE.md
  - [ ] C4 diagrams (all 4 levels)
  - [ ] Component descriptions
  - [ ] Data flow documentation
  - [ ] Extension points documented
- [ ] Code documentation
  - [ ] Docstrings for all public APIs
  - [ ] Inline comments for complex logic
  - [ ] Type hints throughout
- [ ] Create CHANGELOG.md
- [ ] License file (if needed)
- [ ] Contributing guide (optional)

### Phase 9: Final Verification & Submission (Day 6)
- [ ] **Academic Criteria Verification (60%)**
  - [ ] PRD complete (20%)
  - [ ] README & docs complete (15%)
  - [ ] Code quality check (15%)
    - [ ] No files >150 lines
    - [ ] Single responsibility
    - [ ] DRY principle
  - [ ] Security & config check (10%)
    - [ ] No hardcoded secrets
    - [ ] .env.example provided
  - [ ] Testing verified (15%)
    - [ ] 70%+ coverage
    - [ ] Edge cases covered
  - [ ] Research & analysis complete (15%)
    - [ ] Experiments run
    - [ ] Visualizations generated
    - [ ] Statistical analysis done
  - [ ] UI/UX documented (10%)
    - [ ] CLI examples
    - [ ] Extension points

- [ ] **Technical Criteria Verification (40%)**
  - [ ] Check A: Package organization
    - [ ] `pip install -e .` works
    - [ ] All __init__.py files present
    - [ ] Relative imports only
  - [ ] Check B: Multiprocessing/Multithreading
    - [ ] Parallel processing implemented
    - [ ] Thread safety verified
    - [ ] Performance improvement documented
  - [ ] Check C: Building blocks design
    - [ ] Input/Output/Setup documented for all blocks
    - [ ] Reusable components
    - [ ] Independent testability

- [ ] **Final Checks**
  - [ ] Git history clean (meaningful commits)
  - [ ] No unnecessary files in repo
  - [ ] .gitignore comprehensive
  - [ ] All tests pass (`pytest`)
  - [ ] Code quality passes (`flake8`, `black --check`)
  - [ ] Type checking passes (if using mypy)
  - [ ] Requirements.txt up to date
  - [ ] Documentation builds correctly

- [ ] **Prepare Submission**
  - [ ] Create submission archive/repository
  - [ ] Verify all required files included
  - [ ] Test installation on clean environment
  - [ ] Final README review
  - [ ] Submit!

---

## Timeline and Priorities

### Critical Path Items (MUST HAVE for 100%)

**Week 1 (Days 1-3): Foundation & Core Implementation**
- **Day 1**: Setup + Documentation foundation
  - Priority: HIGH
  - Deliverables: Project structure, pyproject.toml, PRD.md draft, __init__.py files

- **Day 2**: LLM Providers + Metrics
  - Priority: CRITICAL
  - Deliverables: Working OpenAI provider, BLEU metric, accuracy metric

- **Day 3**: Prompt Variators + Datasets
  - Priority: CRITICAL
  - Deliverables: Baseline, Few-Shot, CoT variators, 3 datasets (150-300 examples total)

**Week 2 (Days 4-6): Experiments & Documentation**
- **Day 4**: Experiment Framework + Analysis
  - Priority: CRITICAL
  - Deliverables: Experiment runner, parallel processing, Jupyter notebook with graphs

- **Day 5**: Testing + Documentation
  - Priority: HIGH
  - Deliverables: 70%+ test coverage, complete README, finalized PRD

- **Day 6**: Final Verification + Submission
  - Priority: CRITICAL
  - Deliverables: All checklists complete, submission ready

### Priority Levels

**CRITICAL (Must have for 100% - 80% of grade):**
1. Functioning prompt variations (baseline, few-shot, CoT) - 25%
2. Comparative visualizations showing improvement - 20%
3. Package organization (installable) - 15%
4. Complete PRD with architecture - 15%
5. Building blocks design with documentation - 15%

**HIGH (Important for full marks - 15% of grade):**
6. 70%+ test coverage - 10%
7. Complete README with examples - 5%
8. Multiprocessing implementation - 5%

**MEDIUM (Nice to have - 5% of grade):**
9. ReAct implementation (optional but impressive)
10. Multiple LLM providers
11. Advanced visualizations (heatmaps, etc.)
12. Statistical validation (t-tests)

### Risk Mitigation

**High-Risk Items:**
1. **API Costs**: Monitor token usage closely, use cheaper models for testing
   - Mitigation: Set budget limits, cache results, use small test datasets during development

2. **Time Pressure on Experiments**: Running full experiments might take hours
   - Mitigation: Start experiments early, use parallel processing, have backup datasets ready

3. **Multiprocessing Complexity**: Debugging parallel code is hard
   - Mitigation: Test with sequential execution first, add comprehensive logging

4. **Graph Generation**: Getting visualization right can be time-consuming
   - Mitigation: Create templates early, test with mock data

**Backup Plans:**
- If ReAct is too complex: Focus on CoT and CoT++ instead
- If BERTScore is too slow: Use BLEU and accuracy only
- If one dataset type fails: Have 2-3 backup dataset options ready

### Resource Allocation

**API Budget Estimation:**
- Development/Testing: ~50,000 tokens (~$0.50 with GPT-3.5)
- Full Experiments: ~200,000 tokens (~$2-5 depending on model)
- Buffer for iterations: ~100,000 tokens (~$1-2)
- **Total Budget Needed: $5-10**

**Time Allocation (40 hours total):**
- Setup & Infrastructure: 4 hours (10%)
- Core Implementation: 16 hours (40%)
- Documentation: 8 hours (20%)
- Testing & QA: 6 hours (15%)
- Experiments & Analysis: 4 hours (10%)
- Final Verification: 2 hours (5%)

### Success Criteria

**Minimum Viable Submission (80/100):**
- ✅ Working baseline and one improvement (Few-Shot OR CoT)
- ✅ One dataset (sentiment analysis)
- ✅ Basic visualization (bar chart)
- ✅ Package installation works
- ✅ Basic README and PRD
- ✅ 50%+ test coverage

**Target Submission (100/100):**
- ✅ Baseline + Few-Shot + CoT + CoT++
- ✅ 3 diverse datasets (sentiment, math, logic)
- ✅ Multiple visualizations (bar, line, heatmap)
- ✅ Complete PRD with C4 diagrams
- ✅ Complete README with examples
- ✅ 70%+ test coverage
- ✅ Multiprocessing implementation
- ✅ Building blocks fully documented
- ✅ Statistical analysis in Jupyter notebook

**Stretch Goals (Bonus/Impressive):**
- ✅ ReAct implementation with tool integration
- ✅ Multiple LLM providers (OpenAI + Anthropic)
- ✅ Tree of Thoughts implementation
- ✅ Real-time cost tracking dashboard
- ✅ Automated experiment pipeline

---

## Key Takeaways

### What Makes a 100/100 Submission

1. **Complete Coverage**: Address EVERY point in the self-assessment guide
2. **Rigorous Testing**: 70%+ coverage with meaningful tests, not just hitting numbers
3. **Clear Improvement**: Graphs MUST show measurable prompt improvement
4. **Professional Documentation**: PRD, README, and code docs at industry standard
5. **Technical Excellence**: Package organization, parallel processing, building blocks all implemented correctly
6. **Academic Rigor**: Statistical validation, proper experimental methodology

### Common Pitfalls to Avoid

1. ❌ Starting experiments too late (API costs, time pressure)
2. ❌ Ignoring test coverage requirements
3. ❌ Poor package organization (absolute imports, missing __init__.py)
4. ❌ Incomplete documentation (missing PRD sections, no diagrams)
5. ❌ No parallel processing (critical technical requirement)
6. ❌ Building blocks without Input/Output/Setup documentation
7. ❌ Visualizations without statistical validation
8. ❌ Hardcoded API keys or secrets

### Final Checklist Before Submission

```bash
# Installation test
cd /tmp
git clone <your-repo>
cd <repo-name>
python -m venv venv
source venv/bin/activate
pip install -e .
# Should work without errors!

# Run tests
pytest --cov=src --cov-report=term
# Should show 70%+ coverage

# Verify package structure
python -c "from src.inference.openai_provider import OpenAIProvider; print('OK')"
# Should print "OK"

# Check documentation
ls docs/
# Should show: PRD.md, ARCHITECTURE.md, ADRs/

# Verify experiments
ls results/visualizations/
# Should show: graphs at 300 DPI

# Security check
grep -r "sk-" src/
# Should return NOTHING (no hardcoded keys)
```

---

## Conclusion

This plan covers ALL requirements from the three PDFs:
- ✅ Assignment task (datasets, baseline, improvements, visualization)
- ✅ Academic criteria (60%): PRD, docs, code quality, security, testing, research, UI/UX
- ✅ Technical criteria (40%): Package organization, multiprocessing, building blocks

**Follow this plan systematically, and you WILL get 100/100.**

Good luck! 🚀