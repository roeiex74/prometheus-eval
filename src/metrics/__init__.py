"""
Prometheus-Eval Metrics Module

Comprehensive suite of evaluation metrics for LLM prompt effectiveness:
- Lexical: BLEU (n-gram precision)
- Semantic: BERTScore (contextual embedding similarity)
- Logic: Pass@k (code correctness)

Note: Import metrics directly from submodules to avoid loading heavy dependencies:
    from src.metrics.lexical.bleu import BLEUMetric
    from src.metrics.semantic.bertscore import BERTScoreMetric
    from src.metrics.logic.pass_at_k import PassAtKMetric
"""

__all__ = [
    "BLEUMetric",
    "BERTScoreMetric",
    "PassAtKMetric",
    "PassAtKResult",
]
