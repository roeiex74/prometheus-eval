"""
Lexical Metrics (N-gram based)

Metrics that evaluate text similarity based on surface-level token overlap.
"""

from src.metrics.lexical.bleu import BLEUMetric

__all__ = ["BLEUMetric"]
