# Phase 1 Issues Tracker

**Generated:** 2025-12-13
**Status:** 10 issues identified (3 Critical, 4 High, 3 Medium)

---

## CRITICAL ISSUES (Block Phase 2)

### ISSUE-001: Missing Package Definition File
- **Severity:** CRITICAL
- **Status:** OPEN
- **Assigned to:** System Architect Agent
- **Due Date:** 2025-12-14
- **Estimated Effort:** 2 hours

**Description:**
No setup.py or pyproject.toml file exists in project root.

**Impact:**
- Project cannot be installed as a package (`pip install -e .` fails)
- Violates Python packaging best practices (PEP 517/518)
- Violates Chapter 15.1 guidelines
- Prevents distribution to PyPI or internal package repos

**Acceptance Criteria:**
- [ ] setup.py exists in project root
- [ ] Contains all metadata (name, version, author, description)
- [ ] Lists all dependencies from requirements.txt
- [ ] Specifies python_requires=">=3.11"
- [ ] Includes entry_points if needed
- [ ] `pip install -e .` succeeds without errors

**Implementation Template:**
```python
# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prometheus-eval",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Comprehensive framework for rigorous evaluation of LLM prompt effectiveness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/prometheus-eval",
    packages=find_packages(where=".", exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed
        ],
    },
)
```

**Verification:**
```bash
# Test installation
pip install -e .

# Test import
python -c "import prometheus_eval; print(prometheus_eval.__version__)"
```

---

### ISSUE-002: Empty __init__.py Files
- **Severity:** HIGH
- **Status:** OPEN
- **Assigned to:** System Architect Agent
- **Due Date:** 2025-12-15
- **Estimated Effort:** 4 hours

**Description:**
9 out of 10 __init__.py files are empty with no exports or __all__ definitions.

**Impact:**
- Users must use deep imports: `from src.metrics.lexical.bleu import BLEUMetric`
- Poor API discoverability
- Violates Chapter 15.2 guidelines
- No version metadata accessible

**Affected Files:**
- /src/__init__.py
- /src/metrics/__init__.py
- /src/metrics/lexical/__init__.py
- /src/metrics/semantic/__init__.py
- /src/metrics/logic/__init__.py
- /src/evaluator/__init__.py
- /src/analysis/__init__.py (empty module)
- /src/variator/__init__.py (empty module)
- /src/visualization/__init__.py (empty module)

**Acceptance Criteria:**
- [ ] src/__init__.py exports main components and defines __version__
- [ ] src/metrics/__init__.py exports all metric classes
- [ ] src/inference/__init__.py updated (already good, verify completeness)
- [ ] Each submodule __init__.py exports relevant classes
- [ ] User can do: `from prometheus_eval import BLEUMetric, BERTScoreMetric`

**Implementation Examples:**

```python
# src/__init__.py
"""
Prometheus-Eval: Comprehensive LLM Prompt Evaluation Framework
"""

__version__ = "0.1.0"

from src.metrics import (
    BLEUMetric,
    BERTScoreMetric,
    PassAtKMetric,
)

from src.inference import (
    AbstractLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    InferenceConfig,
    create_openai_provider,
    create_anthropic_provider,
)

from src.evaluator import (
    CodeExecutor,
)

__all__ = [
    "__version__",
    # Metrics
    "BLEUMetric",
    "BERTScoreMetric",
    "PassAtKMetric",
    # Inference
    "AbstractLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "InferenceConfig",
    "create_openai_provider",
    "create_anthropic_provider",
    # Evaluator
    "CodeExecutor",
]
```

```python
# src/metrics/__init__.py
"""
Metrics module for LLM evaluation.
"""

from src.metrics.lexical.bleu import BLEUMetric
from src.metrics.semantic.bertscore import BERTScoreMetric
from src.metrics.logic.pass_at_k import PassAtKMetric, PassAtKResult

__all__ = [
    "BLEUMetric",
    "BERTScoreMetric",
    "PassAtKMetric",
    "PassAtKResult",
]
```

```python
# src/metrics/lexical/__init__.py
"""
Lexical metrics (n-gram based).
"""

from src.metrics.lexical.bleu import BLEUMetric

__all__ = ["BLEUMetric"]
```

**Verification:**
```python
# Test convenience imports
from prometheus_eval import BLEUMetric, OpenAIProvider
from prometheus_eval import __version__

print(__version__)  # Should print "0.1.0"
```

---

### ISSUE-008: Test Collection Error
- **Severity:** MEDIUM
- **Status:** OPEN
- **Assigned to:** QA Agent
- **Due Date:** 2025-12-14
- **Estimated Effort:** 2 hours

**Description:**
pytest shows "1 error during collection" despite collecting 58 tests.

**Impact:**
- One test module has import or syntax error
- May hide broken tests
- Prevents clean CI/CD pipeline

**Evidence:**
```
collected 58 items / 1 error
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

**Acceptance Criteria:**
- [ ] Run `pytest --collect-only -v` and identify failing module
- [ ] Fix import or syntax error
- [ ] All tests collect without errors
- [ ] pytest exit code is 0

**Investigation Steps:**
```bash
# Detailed collection with verbose output
pytest tests/ --collect-only -v 2>&1 | tee collection_output.txt

# Look for ERROR or FAILED in output
grep -i error collection_output.txt

# Try collecting each module individually
pytest tests/test_inference/ --collect-only
pytest tests/test_metrics/ --collect-only
pytest tests/test_variator/ --collect-only
```

**Common Causes:**
- Missing import in test file
- Circular import
- Fixture dependency issue
- __init__.py missing in test directory

---

## HIGH Priority Issues

### ISSUE-003: Missing docs/ Directory
- **Severity:** HIGH
- **Status:** IN_PROGRESS (directory created during review)
- **Assigned to:** Documentation Agent
- **Due Date:** 2025-12-17
- **Estimated Effort:** 6 hours

**Description:**
No structured docs/ directory for API documentation per Chapter 15.3.

**Impact:**
- Violates organizational structure guidelines
- No central place for documentation
- Poor developer experience

**Acceptance Criteria:**
- [x] docs/ directory exists (created during review)
- [ ] docs/index.md with project overview
- [ ] docs/api/ for API documentation
- [ ] docs/guides/ for user guides
- [ ] docs/architecture/ for architecture docs
- [ ] docs/adr/ for Architecture Decision Records

**Directory Structure:**
```
docs/
├── index.md                    # Main documentation landing page
├── phase1_architectural_review.md  # Already created
├── phase1_review_summary.md        # Already created
├── phase1_issues_tracker.md        # This file
├── api/
│   ├── inference.md            # Inference engine API
│   ├── metrics.md              # Metrics API
│   └── evaluator.md            # Code executor API
├── guides/
│   ├── installation.md         # Installation guide
│   ├── quickstart.md           # Quick start tutorial
│   ├── configuration.md        # Configuration guide
│   └── custom-metrics.md       # Creating custom metrics
├── architecture/
│   ├── overview.md             # System architecture
│   ├── design-patterns.md      # Design patterns used
│   └── data-flow.md            # Data flow diagrams
└── adr/
    ├── 0001-use-docker-sandbox.md
    ├── 0002-abc-provider-pattern.md
    └── 0003-metric-interface-design.md
```

---

### ISSUE-004: No API Documentation
- **Severity:** HIGH
- **Status:** OPEN
- **Assigned to:** Documentation Agent
- **Due Date:** 2025-12-20
- **Estimated Effort:** 8 hours

**Description:**
No generated API documentation using Sphinx or MkDocs.

**Impact:**
- Difficult for external developers to understand interfaces
- No searchable API reference
- Violates Chapter 13 documentation requirements

**Acceptance Criteria:**
- [ ] Sphinx or MkDocs installed and configured
- [ ] Auto-generated API docs from docstrings
- [ ] HTML documentation buildable
- [ ] Hosted documentation (optional: ReadTheDocs)

**Recommended: Sphinx with autodoc**

```bash
# Setup Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Initialize Sphinx
cd docs
sphinx-quickstart

# Configure conf.py
# - Enable autodoc extension
# - Set up path to src/
# - Configure theme

# Build docs
make html

# View docs
open _build/html/index.html
```

**conf.py additions:**
```python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

html_theme = 'sphinx_rtd_theme'
```

---

### ISSUE-006: Empty src/config/ Directory
- **Severity:** MEDIUM
- **Status:** OPEN
- **Assigned to:** System Architect Agent
- **Due Date:** 2025-12-16
- **Estimated Effort:** 30 minutes

**Description:**
src/config/ directory exists but is empty. Configuration is in src/inference/config.py.

**Impact:**
- Inconsistent organization
- Confusing for developers expecting global config here

**Options:**

**Option A: Move config to src/config/**
- Move src/inference/config.py → src/config/inference.py
- Update imports throughout codebase
- Effort: 2 hours

**Option B: Remove src/config/ directory**
- Delete empty directory
- Keep config in src/inference/config.py
- Effort: 5 minutes

**Recommendation:** Option B (simpler, less disruptive)

**Acceptance Criteria:**
- [ ] Decision made on Option A or B
- [ ] If A: config moved and imports updated
- [ ] If B: directory removed

---

### ISSUE-007: No Test Coverage Report
- **Severity:** MEDIUM
- **Status:** OPEN
- **Assigned to:** QA Agent
- **Due Date:** 2025-12-15
- **Estimated Effort:** 1 hour

**Description:**
No test coverage report generated to verify >70% coverage requirement.

**Impact:**
- Cannot verify compliance with Chapter 13 testing guidelines
- Unknown code coverage

**Acceptance Criteria:**
- [ ] pytest-cov configured
- [ ] Coverage report generated
- [ ] Coverage >70% (guideline requirement)
- [ ] HTML report available at htmlcov/index.html
- [ ] Coverage badge in README (optional)

**Commands:**
```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View coverage report
open htmlcov/index.html

# Generate coverage badge (optional)
coverage-badge -o coverage.svg
```

**Expected Result:**
```
---------- coverage: platform darwin, python 3.12.3 -----------
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src/__init__.py                         10      0   100%
src/inference/base.py                  150     15    90%
src/inference/config.py                 80      5    94%
src/metrics/lexical/bleu.py            280     20    93%
src/metrics/semantic/bertscore.py      250     25    90%
--------------------------------------------------------
TOTAL                                 1800    180    90%
```

---

## MEDIUM Priority Issues

### ISSUE-005: No Version Metadata
- **Severity:** LOW
- **Status:** OPEN
- **Assigned to:** System Architect Agent
- **Due Date:** 2025-12-15
- **Estimated Effort:** 15 minutes

**Description:**
No __version__ defined in src/__init__.py.

**Impact:**
- Cannot programmatically query package version
- No version info in error reports

**Implementation:**
```python
# src/__init__.py
__version__ = "0.1.0"

# Also in setup.py
version="0.1.0"
```

**Acceptance Criteria:**
- [ ] __version__ = "0.1.0" in src/__init__.py
- [ ] Matches version in setup.py
- [ ] Accessible via: `import prometheus_eval; print(prometheus_eval.__version__)`

---

### ISSUE-009: No Architecture Decision Records
- **Severity:** LOW
- **Status:** OPEN
- **Assigned to:** Documentation Agent
- **Due Date:** 2025-12-22
- **Estimated Effort:** 4 hours

**Description:**
No ADR documentation for key architectural decisions.

**Impact:**
- Future developers may not understand design rationale
- Lost knowledge when team changes

**Suggested ADRs:**

**ADR-0001: Use Docker for Code Sandboxing**
- Context: Need secure execution for Pass@k
- Decision: Docker containers with resource limits
- Consequences: Requires Docker, adds overhead, increases security

**ADR-0002: ABC Pattern for LLM Providers**
- Context: Multiple LLM providers needed
- Decision: AbstractLLMProvider base class
- Consequences: Enforces consistent interface, easy to add providers

**ADR-0003: Separate Metric Categories**
- Context: Different types of metrics (lexical, semantic, logic)
- Decision: Separate modules: metrics/{lexical,semantic,logic}/
- Consequences: Clear organization, independent development

**Template:**
```markdown
# ADR-XXXX: Title

Date: YYYY-MM-DD
Status: Accepted | Deprecated | Superseded

## Context
[Describe the situation and problem]

## Decision
[Describe the solution chosen]

## Consequences
[Describe positive and negative consequences]

## Alternatives Considered
[Describe other options and why they were rejected]
```

---

### ISSUE-010: Missing Benchmark Results
- **Severity:** LOW
- **Status:** OPEN
- **Assigned to:** Research Agent
- **Due Date:** 2025-12-30
- **Estimated Effort:** 8 hours

**Description:**
PRD Phase 1 requires "Benchmark baseline performance of Zero-Shot vs. Few-Shot on HumanEval".

**Impact:**
- Research deliverable incomplete
- Cannot validate metric implementations

**Acceptance Criteria:**
- [ ] HumanEval dataset acquired
- [ ] Benchmark script created
- [ ] Zero-shot baseline results
- [ ] Few-shot baseline results
- [ ] Results documented in docs/benchmarks/humaneval_baseline.md

**Tasks:**
1. Download HumanEval dataset
2. Create benchmark runner script
3. Run OpenAI/Anthropic on HumanEval subset
4. Compute Pass@k metrics
5. Document results with analysis

---

## Progress Tracking

**Overall Completion:** 0/10 issues resolved

**Critical (3):** 0/3 resolved
- [ ] ISSUE-001: Missing setup.py
- [ ] ISSUE-002: Empty __init__.py
- [ ] ISSUE-008: Test collection error

**High (4):** 0/4 resolved
- [ ] ISSUE-003: Missing docs/ (partial - directory created)
- [ ] ISSUE-004: No API docs
- [ ] ISSUE-006: Empty config/ directory
- [ ] ISSUE-007: No coverage report

**Medium (3):** 0/3 resolved
- [ ] ISSUE-005: No version metadata
- [ ] ISSUE-009: No ADRs
- [ ] ISSUE-010: Missing benchmarks

---

## Timeline

**Week 1 (Dec 13-17):**
- Day 1: ISSUE-001 (setup.py)
- Day 2: ISSUE-002 (__init__.py exports)
- Day 3: ISSUE-008 (test error), ISSUE-005 (version)
- Day 4: ISSUE-007 (coverage)
- Day 5: ISSUE-006 (config dir), ISSUE-003 partial

**Week 2 (Dec 18-22):**
- ISSUE-004 (API docs)
- ISSUE-003 (complete docs/)
- ISSUE-009 (ADRs)

**Week 3+ (Dec 23+):**
- ISSUE-010 (benchmarks)

---

## Issue Dependencies

```
ISSUE-001 (setup.py) → Blocks Phase 2 gate
    ↓
ISSUE-002 (__init__.py) → Enables clean imports
    ↓
ISSUE-005 (version) → Small, combine with ISSUE-002

ISSUE-008 (test error) → Blocks clean CI/CD
    ↓
ISSUE-007 (coverage) → Requires clean test run

ISSUE-003 (docs/) → Foundation for ISSUE-004
    ↓
ISSUE-004 (API docs) → Long-term documentation
    ↓
ISSUE-009 (ADRs) → Knowledge preservation
```

---

**Last Updated:** 2025-12-13
**Next Review:** 2025-12-17 (check critical issues resolved)
