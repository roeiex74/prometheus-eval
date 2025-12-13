"""
Prometheus-Eval: Comprehensive LLM Prompt Evaluation Framework

A rigorous, metrics-driven approach to evaluating LLM prompt effectiveness across
lexical, semantic, and logic-based dimensions.

Note: To avoid importing heavy dependencies (torch, transformers) at package import time,
some classes are not imported at the top level. Use explicit imports:
    from src.metrics.lexical.bleu import BLEUMetric
    from src.metrics.semantic.bertscore import BERTScoreMetric
    from src.metrics.logic.pass_at_k import PassAtKMetric
"""

__version__ = "0.1.0"

# Define __all__ for documentation purposes
# Users should import directly from submodules to avoid loading heavy dependencies
__all__ = [
    "__version__",
]
