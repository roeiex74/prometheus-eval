# Comprehensive Fix & Validation Plan

**Date**: 2025-12-17
**Status**: Critical Fix Required before Final Experiments

## 1. Critical Technical Fix: Experiment Runner Multiprocessing
**Issue**: `TypeError: cannot pickle '_thread.RLock' object`
**Analysis**: The `ExperimentRunner` attempts to pickle the `LLMProvider` instance when dispatching tasks to worker processes. Use of `asyncio.Throttler` and potential internal locks in the provider class makes it non-picklable.
**Fix Strategy**:
- Refactor `src/experiments/runner.py` to use the `multiprocessing.Pool` **initializer pattern**.
- Move inference logic to a **top-level function** to avoid pickling `self`.
- Initialize a fresh (or global) LLM provider instance *inside* each worker process.

### Plan Steps:
1.  **Refactor** `src/experiments/runner.py`:
    - Define global `_worker_provider`.
    - Create `init_worker(provider_config)` function.
    - Create `run_inference_task(prompt)` top-level function.
    - Update `_run_parallel_inference` to pass config instead of instance.
2.  **Verify**: Run `python run_experiments.py --dataset sentiment --max-samples 5` (Smoke Test).

## 2. Submission Guidelines Evaluation
**Source**: `software_submission_guidelines.pdf` (Extracted)

### Compliance Check:
- **Heuristics & UX** (Ref 19/20/21): The CLI is robust, but the output graphs need to be clear and "publication quality" (300 DPI).
- **ISO 25010** (Ref 22):
    - *Reliability*: Addressed by the fix above (Multiprocessing stability).
    - *Maintainability*: Code quality is high (SRP, Type Hinting).
    - *Portability*: `pyproject.toml` and standard layout ensure this.
- **Engineering Practices** (Ref 24 Google): Code style and documentation (docstrings) appear compliant.

### Action Items:
- **Visualizations**: Ensure `notebooks/results_analysis.ipynb` explicitly sets DPI to 300 for saved figures.
- **Documentation**: Verify `PROJECT_SUMMARY.md` claims against actual code state after fix.

## 3. Execution Roadmap
1.  [DONE] Apply `runner.py` fix.
2.  [DONE] Re-run Smoke Test (Passed).
3.  [NEXT] Run Full Experiment Suite (Sentiment, Math, Logic).
4.  [NEXT] Execute Jupyter Notebook to generate figures.
5.  [FINAL] Update `PROJECT_STATE.json` to 100/100.
