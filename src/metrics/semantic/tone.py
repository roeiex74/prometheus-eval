"""Tone Consistency metric via sentiment variance (TC = 1 - σ²)."""
from typing import List, Dict, Union, Any
import numpy as np
import re


class ToneConsistencyMetric:
    """Measures sentiment stability across text segments.

    Formula: TC = 1 - σ²(sentiment_scores)

    References:
        PRD Section 4.3; Socher et al. (2013) SST; Ribeiro et al. (2020) CheckList
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
