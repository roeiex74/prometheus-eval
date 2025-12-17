"""
Prometheus-Eval Metrics Module

Comprehensive suite of evaluation metrics for LLM prompt effectiveness:
- Lexical: BLEU (n-gram precision), ROUGE (recall-oriented), METEOR (semantic matching)
- Semantic: BERTScore (contextual embedding similarity), Semantic Stability, Tone Consistency
- Logic: Pass@k (code correctness), Perplexity (language fluency)

Note: Import metrics directly from submodules to avoid loading heavy dependencies:
    from src.metrics.lexical.bleu import BLEUMetric
    from src.metrics.lexical.rouge import ROUGEMetric
    from src.metrics.lexical.meteor import METEORMetric
    from src.metrics.semantic.bertscore import BERTScoreMetric
    from src.metrics.semantic.stability import SemanticStabilityMetric
    from src.metrics.semantic.tone import ToneConsistencyMetric
    from src.metrics.logic.pass_at_k import PassAtKMetric
    from src.metrics.logic.perplexity import PerplexityMetric
"""

__all__ = [
    "BLEUMetric",
    "ROUGEMetric",
    "METEORMetric",
    "BERTScoreMetric",
    "SemanticStabilityMetric",
    "ToneConsistencyMetric",
    "PassAtKMetric",
    "PassAtKResult",
    "PerplexityMetric",
]
