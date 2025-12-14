# Prometheus-Eval: Detailed Execution Plan & Process Loop

**Date:** 2025-12-15
**Version:** 2.0 (Refined based on `docs` and `PRD`)
**Status:** **Phase 2 (Week 6) - IN PROGRESS**

---

## 1. The Operational Process Loop

The project follows a recurring **"Implement-Validate-Gate"** cycle for every major component.

### The Loop Structure
1.  **Dispatch**: The `project-orchestrator` parses the PRD/Plan and assigns a task to a Specialist Agent (e.g., `metric-mathematician`).
2.  **Implementation**: The Specialist Agent writes code, docstrings, and **unit tests** (Test-Driven Development is now mandatory).
3.  **Continuous Validation**:
    *   **Self-Correction**: The Specialist Agent runs basic tests.
    *   **QA Trigger**: Once implemented, the `qa-agent` is triggered to run the full suite (`pytest`).
4.  **Stage Review**: For critical milestones, specific validation agents (Security, Architect, Research) review the artifacts.
5.  **Integration**: The `system-architect` ensures the new component fits into the `InferenceEngine` or `Evaluator` scaffold.
6.  **Completion Signal**: The task is marked `COMPLETE` in `PROJECT_STATE.json` only after all checks pass.

---

## 2. Phase 1: Core Infrastructure (Completed)

> **Status**: ✅ **CONDITIONAL PASS**
> **Critical Pending Item**: ROUGE Tests (Metrics implemented, but 0% coverage).

*   **Infrastructure**: `InferenceEngine` (OpenAI/Anthropic), `DockerSandbox` ✅
*   **Metrics**: `BLEU`, `BERTScore`, `Pass@k` ✅
*   **Documentation**: README, PRD, Agent Docs ✅

---

## 3. Phase 2: Semantic & Embedding Layer (Current - Weeks 5-8)

**Objective**: Enable evaluation of open-ended text using semantic embeddings and variability testing.

### Week 6: Advanced Metrics Implementation (Current Focus)

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **2.1** | **ROUGE Metric Tests** | `metric-mathematician` (or `qa-agent`) | 🔴 **CRITICAL** | `tests/test_metrics/test_rouge.py` (Cov > 70%) |
| **2.2** | **METEOR Metric** | `metric-mathematician` | ⏳ Pending | `src/metrics/lexical/meteor.py` + Tests |
| **2.3** | **Semantic Stability** | `metric-mathematician` | ⏳ Pending | `src/metrics/semantic/stability.py` |
| **2.4** | **Perplexity Metric** | `metric-mathematician` | ⏳ Pending | `src/metrics/logic/perplexity.py` |
| **2.5** | **Tone Consistency** | `metric-mathematician` | ⏳ Pending | `src/metrics/semantic/tone.py` |

### Week 7: The Variator (Prompt Engineering Engine)

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **2.6** | **Variator Architecture** | `system-architect` | ⏳ Pending | Class design for `PromptGenerator` |
| **2.7** | **PromptGenerator** | `variator-agent` | ⏳ Pending | `src/variator/prompt_generator.py` |
| **2.8** | **Template Expansion** | `variator-agent` | ⏳ Pending | `src/variator/templates.py` |
| **2.9** | **Technique Injection** | `variator-agent` | ⏳ Pending | Injectors for CoT, ToT (Wait for Phase 3 for full ToT logic) |
| **2.10**| **Emotional Prompting** | `variator-agent` | ⏳ Pending | `src/variator/emotional.py` |

### Week 8: Research & Phase Gate

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **2.11**| **Emotional Analysis** | `research-agent` | ⏳ Pending | `notebooks/emotional_prompting.ipynb` |
| **2.12**| **Phase 2 Validation** | *All Validation Agents* | ⏳ Pending | Full Validation Reports |

---

## 4. Phase 3: Advanced Logic & Visualization (Weeks 9-12)

**Objective**: "Academic Level" Analysis Dashboard and Deep Logic Evaluation.

### Week 9-10: Visualization Dashboard

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | **Dashboard Arch** | `visualization-expert` | 🔮 Future | React App Scaffolding |
| **3.2** | **Parallel Coords** | `visualization-expert` | 🔮 Future | `parallel_coords.tsx` |
| **3.3** | **Entropy Heatmap** | `visualization-expert` | 🔮 Future | `entropy_heatmap.tsx` |
| **3.4** | **Radar Charts** | `visualization-expert` | 🔮 Future | `radar_chart.tsx` |

### Week 11: Advanced Logic (G-Eval & Auto-CoT)

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **3.5** | **G-Eval Framework** | `metric-mathematician` | 🔮 Future | `src/metrics/judge/g_eval.py` |
| **3.6** | **Auto-CoT Logic** | `metric-mathematician` | 🔮 Future | `src/metrics/judge/auto_cot.py` |

### Week 12: Final Research & Delivery

| Step | Task | Agent Responsible | Status | Delieverable |
| :--- | :--- | :--- | :--- | :--- |
| **3.7** | **CoT vs ToT Study** | `research-agent` | 🔮 Future | Comprehensive Research Paper |
| **3.8** | **Final Polish** | `project-orchestrator` | 🔮 Future | V1.0 Release |

---

## 5. Immediate Action Plan (Next 24 Hours)

1.  **Orchestrator**: Dispatch `ROUGE Test Creation` task to `metric-mathematician`.
2.  **Metric Mathematician**: Create `tests/test_metrics/test_rouge.py` to achieve >70% coverage.
3.  **QA Agent**: Verify ROUGE tests pass.
4.  **Orchestrator**: Dispatch `METEOR Metric` task.
