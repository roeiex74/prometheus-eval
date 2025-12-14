# Phase 2 Issues Tracker

**Generated**: 2025-12-15
**Source**: Analysis of Phase 2 Week 6 Documentation

This document tracks open issues, refactoring tasks, and documentation gaps identified during the Phase 2, Week 6 validation cycle.

## 1. Technical Limitations & Bugs

| ID | Severity | Description | Status | Owner |
|----|----------|-------------|--------|-------|
| **ISSUE-QA-001** | Medium | **BERTScore Architecture Mismatch**: `torch` dependency issues on MacOS ARM64. | **DEFERRED** | system-architect |
| **ISSUE-PEP-001** | Low | **PEP8 Violations**: Minor line length violations in `src/metrics/semantic/tone.py`. | **OPEN** | validation-agent |

## 2. Refactoring Tasks (150-Line Limit Compliance)

The following files exceed the strict 150-line limit and require refactoring (splitting into submodules).

| ID | Priority | File | Current Lines | Target Action |
|----|----------|------|---------------|---------------|
| **ISSUE-REF-001** | High | `src/inference/base.py` | 380 | Split into exceptions, provider_abc, rate_limiter |
| **ISSUE-REF-002** | High | `src/inference/openai_provider.py` | 447 | Split into core, helpers, async |
| **ISSUE-REF-003** | High | `src/inference/anthropic_provider.py` | 445 | Split into core, helpers, async |
| **ISSUE-REF-004** | High | `src/evaluator/executor.py` | 353 | Split into docker_manager, test_harness, utils |
| **ISSUE-REF-005** | Medium | `src/metrics/logic/pass_at_k.py` | 448 | Split into core, calc, types |
| **ISSUE-REF-006** | Medium | `src/metrics/lexical/bleu.py` | 588 | Split into core, smoothing, ngram, utils |
| **ISSUE-REF-007** | Medium | `src/metrics/semantic/bertscore.py` | 518 | Split into core, embed, align, utils |

## 3. Documentation Gaps

| ID | Priority | Description | Affected Files |
|----|----------|-------------|----------------|
| **ISSUE-DOC-001** | High | **Missing Usage Examples**: No usage examples in module/class/method docstrings. | All Week 6 metrics (`rouge.py`, `meteor.py`, etc.) |
| **ISSUE-DOC-002** | High | **Incomplete Citations**: Missing formal academic citations/authors. | `stability.py`, `perplexity.py` |
| **ISSUE-DOC-003** | Medium | **Missing Raises Info**: `ValueError` conditions not documented in docstrings. | `rouge.py`, `meteor.py` |
| **ISSUE-DOC-004** | Low | **Missing Type Hint**: Return type for `sentiment_analyzer` property. | `tone.py` |
| **ISSUE-DOC-005** | Medium | **Missing API Docs**: No dedicated API documentation file for Week 6 metrics. | `docs/api/WEEK6_METRICS_API.md` (to be created) |
| **ISSUE-DOC-006** | Low | **README Update**: Main README missing Week 6 metrics section. | `README.md` |

## 4. Next Steps Summary

1.  **Immediate**: Address `ISSUE-DOC-001` (Usage Examples) as it impacts usability.
2.  **Short Term**: Begin high-priority refactoring (`ISSUE-REF-001` to `ISSUE-REF-004`) to maintain codebase health.
3.  **Medium Term**: Create API documentation (`ISSUE-DOC-005`).
