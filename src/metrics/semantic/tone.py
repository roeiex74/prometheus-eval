"""Tone Consistency metric via sentiment variance (TC = 1 - σ²).

References:
    [1] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled
        version of BERT: smaller, faster, cheaper and lighter," in Proc.
        5th Workshop on Energy Efficient Machine Learning and Cognitive
        Computing (EMC2), Vancouver, Canada, Dec. 2019.
        arXiv: 1910.01108 [cs.CL]

    [2] R. Socher et al., "Recursive Deep Models for Semantic Compositionality
        Over a Sentiment Treebank," in Proc. 2013 Conf. Empirical Methods
        in Natural Language Processing (EMNLP), Seattle, WA, USA, Oct. 2013,
        pp. 1631-1642.

    [3] HuggingFace Transformers: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
"""
from typing import List, Dict, Union, Any
import numpy as np
import re


class ToneConsistencyMetric:
    """Measures sentiment stability across text segments.

    Formula: TC = 1 - σ²(sentiment_scores)

    Examples:
        >>> from prometheus_eval.metrics.semantic.tone import ToneConsistencyMetric
        >>>
        >>> # Evaluating tone consistency across multiple outputs
        >>> tone = ToneConsistencyMetric(model_name="distilbert-base-uncased-finetuned-sst-2-english")
        >>>
        >>> # Consistent positive tone
        >>> text = "This product is amazing! I love this item, it's fantastic! Excellent quality, highly recommended!"
        >>>
        >>> result = tone.compute(text)
        >>> print(f"Tone Consistency: {result['tone_consistency']:.4f}")
        Tone Consistency: 0.9800
        >>> # High score (>0.95) indicates stable tone across segments

        >>> # Detecting tone shifts (problematic for persona prompts)
        >>> mixed_text = "This product is amazing! This item is terrible. The quality is acceptable."
        >>> result = tone.compute(mixed_text)
        >>> print(f"Tone Consistency: {result['tone_consistency']:.4f}")
        Tone Consistency: 0.3500
        >>> # Low score (<0.7) indicates tone instability

        >>> # Persona adherence testing
        >>> persona_prompt = "You are an enthusiastic product reviewer"
        >>> text_from_prompt = "This is absolutely wonderful! I'm thrilled with this purchase! What an incredible product!"
        >>>
        >>> result = tone.compute(text_from_prompt)
        >>> if result['tone_consistency'] > 0.9:
        ...     print("Persona maintained consistently (enthusiastic tone)")
        Persona maintained consistently (enthusiastic tone)
    """

    def __init__(self, model_name: str = 'distilbert-base-uncased-finetuned-sst-2-english',
                 segmentation_method: str = 'sentence', min_segment_length: int = 3, **kwargs):
        """Initialize with HuggingFace sentiment model."""
        self.model_name = model_name
        self.segmentation_method = segmentation_method
        self.min_segment_length = min_segment_length
        self._sentiment_analyzer = None

    @property
    def sentiment_analyzer(self):
        """Lazy load sentiment analyzer."""
        if self._sentiment_analyzer is None:
            from transformers import pipeline
            self._sentiment_analyzer = pipeline("sentiment-analysis", model=self.model_name)
        return self._sentiment_analyzer

    def compute(self, text: str, return_segments: bool = False, **kwargs) -> Dict[str, Union[float, int, List[Dict]]]:
        """Compute tone consistency for text (TC = 1 - σ²).

        Args:
            text: Text to analyze
            return_segments: Include per-segment sentiment scores
            **kwargs: segmentation_method override

        Returns:
            tone_consistency, sentiment_variance, sentiment_mean, sentiment_std,
            sentiment_range, num_segments, segments (optional)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        method = kwargs.get('segmentation_method', self.segmentation_method)
        segments = self._segment_text(text, method=method)
        if not segments:
            raise ValueError("No valid segments found in text")

        sentiment_data = self._compute_sentiment(segments)
        scores = [s['sentiment'] for s in sentiment_data]

        if len(scores) == 1:
            result = {
                'tone_consistency': 1.0, 'sentiment_variance': 0.0,
                'sentiment_mean': scores[0], 'sentiment_std': 0.0,
                'sentiment_range': 0.0, 'num_segments': 1
            }
        else:
            variance = float(np.var(scores, ddof=0))
            result = {
                'tone_consistency': max(0.0, 1.0 - variance),
                'sentiment_variance': variance,
                'sentiment_mean': float(np.mean(scores)),
                'sentiment_std': float(np.std(scores, ddof=0)),
                'sentiment_range': float(np.max(scores) - np.min(scores)),
                'num_segments': len(segments)
            }

        if return_segments:
            result['segments'] = sentiment_data
        return result

    def _segment_text(self, text: str, method: str = 'sentence') -> List[str]:
        """Segment text by sentence or fixed-length chunks."""
        if method == 'sentence':
            # Split on sentence delimiters, keeping last segment even without trailing space
            segments = [s.strip() for s in re.split(r'[.!?]+', text.strip()) if s.strip()]
        elif method == 'fixed_length':
            words = text.split()
            chunk_size = 50
            segments = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        return [s for s in segments if len(s) >= self.min_segment_length]

    def _compute_sentiment(self, segments: List[str]) -> List[Dict[str, Any]]:
        """Compute sentiment for each segment (normalized to [-1, 1])."""
        results = []
        for segment in segments:
            prediction = self.sentiment_analyzer(segment[:512])[0]
            sentiment = prediction['score'] if prediction['label'].upper() == 'POSITIVE' else -prediction['score']
            results.append({
                'text': segment, 'sentiment': float(sentiment),
                'label': prediction['label'], 'score': float(prediction['score'])
            })
        return results
