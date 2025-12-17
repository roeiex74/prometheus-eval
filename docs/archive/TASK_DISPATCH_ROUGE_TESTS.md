# TASK DISPATCH: ROUGE Test Suite Creation

**Dispatch Date**: 2025-12-14
**Dispatch By**: project-orchestrator
**Agent**: validation-submission-agent
**Priority**: CRITICAL
**Due Date**: 2025-12-15
**Estimated Effort**: 6 hours

---

## Task Summary

Create comprehensive test suite for ROUGE metric implementation to achieve 70%+ coverage and unblock Week 6 completion.

**Context**: metric-mathematician-agent completed ROUGE implementation (src/metrics/lexical/rouge.py, 118 lines) on 2025-12-13 but did not create tests. This creates a critical coverage gap (0% for 77 statements) that blocks the project's 70% overall coverage target.

**Impact**: Adding ROUGE tests with 70% coverage will push overall project coverage from 67% → 72%, exceeding the 70% target.

---

## Deliverable Specification

### Primary Deliverable
**File**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/tests/test_metrics/test_rouge.py`

**Target Metrics**:
- Coverage: 70%+ (54+ of 77 statements tested)
- Tests: 18-25 comprehensive tests
- Pass Rate: 100%
- Collection Errors: 0

### Implementation Reference
**File**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/metrics/lexical/rouge.py`

**ROUGE Implementation Details**:
- Class: `ROUGEMetric`
- Variants: rouge1, rouge2, rougeL
- Beta parameter: Configurable F-measure weighting (default: 1.0)
- Multi-reference: Selects max score across references
- LCS Algorithm: Space-optimized dynamic programming
- Lines: 118 (within 150-line limit)
- Academic Citation: Lin (2004)

**Key Methods to Test**:
1. `__init__(variants, beta)` - Initialization and validation
2. `_tokenize(text)` - Tokenization and lowercasing
3. `_get_ngrams(tokens, n)` - N-gram extraction
4. `_compute_f1(recall, precision)` - F-measure with beta
5. `_compute_rouge_n(cand, ref, n)` - ROUGE-N score
6. `_lcs_length(seq1, seq2)` - LCS computation
7. `_compute_rouge_l(cand, ref)` - ROUGE-L score
8. `compute(candidate, references)` - Main API

---

## Test Coverage Requirements

### 1. Basic ROUGE-N Computation (4-5 tests)

**Coverage Goal**: Test ROUGE-1 and ROUGE-2 calculation correctness

**Test Cases**:
- Perfect match: Identical candidate and reference (scores = 1.0)
- Complete mismatch: No overlapping tokens (scores = 0.0)
- Partial overlap: Verify precision, recall, and F1 calculation
- Empty string edge cases: Empty candidate, empty reference, both empty

**Example**:
```python
def test_perfect_match(self):
    '''ROUGE-1 and ROUGE-2 should be 1.0 for identical texts'''
    metric = ROUGEMetric(variants=['rouge1', 'rouge2'])
    candidate = "the cat sat on the mat"
    reference = "the cat sat on the mat"
    result = metric.compute(candidate, reference)

    assert result['rouge1'] == pytest.approx(1.0, abs=1e-6)
    assert result['rouge2'] == pytest.approx(1.0, abs=1e-6)
    assert result['overall'] == pytest.approx(1.0, abs=1e-6)
```

### 2. ROUGE-L (LCS) Algorithm (4-5 tests)

**Coverage Goal**: Verify LCS algorithm correctness

**Test Cases**:
- Perfect sequence match: Identical sequences (LCS = length)
- Reversed sequence: Reversed tokens (low LCS)
- Interleaved tokens: Mixed order (medium LCS)
- LCS length correctness: Manually calculated LCS examples
- Edge case: Single token, very long sequences

**Example**:
```python
def test_lcs_computation(self):
    '''ROUGE-L should compute correct LCS for interleaved sequences'''
    metric = ROUGEMetric(variants=['rougeL'])
    candidate = "a b c d e"
    reference = "a c e b d"
    result = metric.compute(candidate, reference)

    # LCS = "a c e" or "a b d" (length 3)
    # Recall = 3/5 = 0.6, Precision = 3/5 = 0.6, F1 = 0.6
    assert result['rougeL'] == pytest.approx(0.6, abs=0.01)
```

### 3. Multi-Reference Selection (3-4 tests)

**Coverage Goal**: Verify max score selection across multiple references

**Test Cases**:
- Single reference baseline: Standard computation
- Multiple references with different scores: Verify max selection
- All references score identically: Verify consistent result
- Empty reference list: Verify error handling

**Example**:
```python
def test_multi_reference_selection(self):
    '''Should select max score across multiple references'''
    metric = ROUGEMetric(variants=['rouge1'])
    candidate = "the cat sat"
    references = [
        "the cat jumped",  # ROUGE-1 = 2/3 (overlap: "the", "cat")
        "the cat sat on mat"  # ROUGE-1 = 3/3 = 1.0 (perfect overlap)
    ]
    result = metric.compute(candidate, references)

    # Should select max score (1.0 from second reference)
    assert result['rouge1'] == pytest.approx(1.0, abs=1e-6)
```

### 4. Beta Parameter for F-Measure (2-3 tests)

**Coverage Goal**: Verify beta parameter affects precision/recall weighting

**Test Cases**:
- Beta = 1.0: Balanced F1 (default)
- Beta = 0.5: Precision-weighted
- Beta = 2.0: Recall-weighted

**Example**:
```python
def test_beta_parameter_weighting(self):
    '''Beta parameter should weight precision vs recall'''
    candidate = "cat sat"
    reference = "the cat sat on the mat"

    # Beta = 1.0: Balanced F1
    metric_balanced = ROUGEMetric(variants=['rouge1'], beta=1.0)
    result_balanced = metric_balanced.compute(candidate, reference)

    # Beta = 2.0: Recall-weighted (should increase score)
    metric_recall = ROUGEMetric(variants=['rouge1'], beta=2.0)
    result_recall = metric_recall.compute(candidate, reference)

    # Recall = 2/6 = 0.33, Precision = 2/2 = 1.0
    # F1 (beta=1): (2*0.33*1.0)/(0.33+1.0) = 0.5
    # F1 (beta=2): (5*0.33*1.0)/(0.33+4*1.0) = 0.38
    # Actually beta=2 favors recall, which is low here
    assert result_balanced['rouge1'] > result_recall['rouge1']
```

### 5. Edge Cases (3-4 tests)

**Coverage Goal**: Verify robustness to unusual inputs

**Test Cases**:
- Single token inputs: "cat" vs "dog"
- Very long sequences: 100+ tokens
- Whitespace-only strings: "   " vs "  "
- Punctuation handling: Case sensitivity and tokenization
- Special characters: Numbers, symbols

**Example**:
```python
def test_empty_candidate(self):
    '''Empty candidate should return score of 0.0'''
    metric = ROUGEMetric(variants=['rouge1', 'rouge2', 'rougeL'])
    result = metric.compute("", "the cat sat")

    assert result['rouge1'] == 0.0
    assert result['rouge2'] == 0.0
    assert result['rougeL'] == 0.0
    assert result['overall'] == 0.0
```

### 6. Type Validation (2-3 tests)

**Coverage Goal**: Verify input validation and error handling

**Test Cases**:
- Invalid variant names: Should raise ValueError
- String vs list references: Both should work
- None inputs: Should raise appropriate error
- Invalid beta values: Negative or zero

**Example**:
```python
def test_invalid_variant(self):
    '''Invalid variant name should raise ValueError'''
    with pytest.raises(ValueError, match="Invalid variant"):
        metric = ROUGEMetric(variants=['rouge3'])  # Invalid
```

### 7. Known Reference Examples (1-2 tests)

**Coverage Goal**: Validate against known ROUGE scores or manual calculations

**Test Cases**:
- Manually calculated example: Verify exact score
- Symmetric case: Score(A, B) should relate to Score(B, A)

**Example**:
```python
def test_known_example(self):
    '''Verify ROUGE calculation against manual computation'''
    metric = ROUGEMetric(variants=['rouge1', 'rouge2'])

    # Manual calculation:
    # Candidate: "the cat" (2 unigrams, 1 bigram)
    # Reference: "the dog" (2 unigrams, 1 bigram)
    # ROUGE-1: overlap = "the" (1/2 unigrams) → Recall=0.5, Precision=0.5, F1=0.5
    # ROUGE-2: overlap = none (0/1 bigrams) → 0.0

    result = metric.compute("the cat", "the dog")

    assert result['rouge1'] == pytest.approx(0.5, abs=1e-6)
    assert result['rouge2'] == pytest.approx(0.0, abs=1e-6)
```

---

## Test Structure & Organization

### File Organization

```python
'''
Comprehensive test suite for ROUGE metric implementation.

Coverage: 70%+ of src/metrics/lexical/rouge.py
Tests: 18-25 tests across 7 categories
'''
import pytest
from src.metrics.lexical.rouge import ROUGEMetric


class TestROUGEBasic:
    '''Basic ROUGE-N computation tests'''
    # 4-5 tests here


class TestROUGELCS:
    '''ROUGE-L (LCS) algorithm tests'''
    # 4-5 tests here


class TestROUGEMultiReference:
    '''Multi-reference selection tests'''
    # 3-4 tests here


class TestROUGEBetaParameter:
    '''Beta parameter F-measure weighting tests'''
    # 2-3 tests here


class TestROUGEEdgeCases:
    '''Edge case robustness tests'''
    # 3-4 tests here


class TestROUGETypeValidation:
    '''Input validation and error handling tests'''
    # 2-3 tests here


class TestROUGEKnownExamples:
    '''Known reference examples and manual calculations'''
    # 1-2 tests here
```

### Best Practices

1. **Docstrings**: Every test has clear docstring explaining what it validates
2. **Naming**: Descriptive test names (`test_perfect_match` not `test_1`)
3. **Assertions**: Use `pytest.approx()` for float comparisons
4. **Tolerance**: `abs=1e-6` for exact matches, `abs=0.01` for approximate
5. **Comments**: Explain manual calculations inline
6. **Organization**: Group related tests into classes
7. **Independence**: Each test should be independent (no shared state)

---

## Validation & Success Criteria

### Required Checks

1. **Coverage Verification**:
```bash
pytest tests/test_metrics/test_rouge.py --cov=src/metrics/lexical/rouge --cov-report=term-missing -v
```
**Expected**: Coverage ≥ 70% (54+ / 77 statements)

2. **Test Execution**:
```bash
pytest tests/test_metrics/test_rouge.py -v
```
**Expected**: All tests pass (100% pass rate), 0 failures, 0 collection errors

3. **Overall Coverage Impact**:
```bash
pytest --cov=src --cov-report=term | grep "TOTAL"
```
**Expected**: Overall coverage ≥ 70% (should show ~72%)

4. **Integration Check**:
```bash
pytest tests/test_metrics/ -v
```
**Expected**: ROUGE tests integrate cleanly with existing test suite

### Success Criteria Checklist

- [ ] File created: `tests/test_metrics/test_rouge.py`
- [ ] Coverage ≥ 70% for rouge.py (54+ statements)
- [ ] Test count: 18-25 comprehensive tests
- [ ] Pass rate: 100% (0 failures, 0 errors)
- [ ] All 7 test categories covered
- [ ] Follows existing test patterns (similar to test_bleu.py)
- [ ] Google-style docstrings on all tests
- [ ] Float comparisons use pytest.approx()
- [ ] No external dependencies beyond pytest, ROUGE module
- [ ] Manual calculations documented in comments
- [ ] Edge cases handled
- [ ] Type validation tests included

---

## State Management Protocol

### Required State Updates

**1. On Task Start**:
```bash
python /Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/tools/state_manager.py \
  --agent validation-submission-agent \
  --action update \
  --phase IMPLEMENTATION \
  --status IN_PROGRESS \
  --log "Starting ROUGE test suite creation. Target: 70%+ coverage, 18-25 tests."
```

**2. Mid-Task Progress** (after completing major milestones):
```bash
python /Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/tools/state_manager.py \
  --agent validation-submission-agent \
  --action update \
  --phase IMPLEMENTATION \
  --status IN_PROGRESS \
  --artifact tests/test_metrics/test_rouge.py \
  --log "Milestone: Basic ROUGE-N tests complete (5 tests, ~25% coverage)"
```

**3. On Completion**:
```bash
python /Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/tools/state_manager.py \
  --agent validation-submission-agent \
  --action update \
  --phase VALIDATION \
  --status PENDING_REVIEW \
  --log "ROUGE test suite complete. Coverage: 72%, Tests: 22, Pass Rate: 100%. Ready for orchestrator review."
```

---

## Reference Materials

### ROUGE Implementation Location
- File: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/metrics/lexical/rouge.py`
- Lines: 118
- Academic Citation: Lin (2004). "ROUGE: Package for Automatic Evaluation of Summaries"

### Existing Test Patterns
- Reference: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/tests/test_metrics/test_bleu.py`
- Structure: Class-based organization (TestBLEUBasic, TestBLEUSmoothing, etc.)
- Style: Google docstrings, pytest.approx for floats, descriptive names

### Project Documentation
- Master Plan: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/docs/PLAN-LATEST.md`
- Status Analysis: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/docs/PROJECT_STATUS_ANALYSIS_2025-12-14.md`
- Dispatch Report: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/docs/ORCHESTRATOR_DISPATCH_2025-12-14.md`

---

## Expected Outcomes

### Coverage Impact
**Before**: 67% overall (0% for ROUGE)
**After**: 72% overall (70%+ for ROUGE)
**Gap Closed**: -3% → +2% (exceeds 70% target)

### Timeline Impact
**No delays expected**:
- Task due: 2025-12-15
- METEOR due: 2025-12-17
- Buffer: 1 day available
- Week 6 completion: On track for 2025-12-20

### Quality Impact
**Establishes test-first precedent**:
- Future metrics: Implementation + tests together
- No coverage debt accumulation
- Maintains 70%+ threshold

---

## Task Priority Justification

**Why CRITICAL?**
1. Blocks Week 6 completion
2. Blocks 70% overall coverage target
3. Largest single coverage gap (77 untested statements)
4. Prevents pattern of implementation-without-tests
5. Required for Phase 2 gate approval

**Why validation-submission-agent?**
1. Specialized in comprehensive test coverage
2. Successfully delivered inference tests (71% coverage, 95 tests)
3. Currently idle (available immediately)
4. Allows metric-mathematician to continue with METEOR
5. Proven track record of quality test suites

---

## Orchestrator Resumption Trigger

**Signal**: State update with `--status PENDING_REVIEW`

**Expected Date**: 2025-12-15 EOD

**Next Action**:
1. Verify coverage metrics (should show 72%+ overall)
2. Review test quality and patterns
3. Update PROJECT_STATE.json with new metrics
4. Dispatch metric-mathematician for METEOR implementation
5. Continue Week 6 execution

**Control Flow**:
```
validation-submission-agent (ROUGE tests)
  ↓ (completion signal)
project-orchestrator (verify & update state)
  ↓ (dispatch)
metric-mathematician-agent (METEOR implementation + tests)
```

---

## Contact & Support

**Orchestrator**: project-orchestrator
**Status Monitoring**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json`
**State Manager**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/src/tools/state_manager.py`

**Questions**: Update state with `--log` message flagged with "QUESTION:" prefix

**Blockers**: Update state with `--status BLOCKED` and detailed log message

---

**END OF TASK DISPATCH**

**Agent**: validation-submission-agent
**Status**: DISPATCHED
**Priority**: CRITICAL
**Due**: 2025-12-15
**Control**: TRANSFERRED
