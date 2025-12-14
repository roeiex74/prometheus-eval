"""
Comprehensive test suite for Tone Consistency metric.
Tests sentiment stability measurement via variance calculation.
"""
import pytest
import numpy as np
from src.metrics.semantic.tone import ToneConsistencyMetric


EPSILON = 1e-3


class TestToneConsistencyBasic:
    """Test basic tone consistency functionality."""

    def test_all_positive_sentiment(self):
        """All positive sentences should have high consistency."""
        metric = ToneConsistencyMetric()
        text = "This is great! I love it. Excellent work. Fantastic quality."
        result = metric.compute(text)

        assert result['tone_consistency'] > 0.5, "Consistent positive tone should have high TC"
        assert 0.0 <= result['tone_consistency'] <= 1.0
        assert result['sentiment_mean'] > 0, "Mean should be positive"
        assert result['num_segments'] == 4

    def test_all_negative_sentiment(self):
        """All negative sentences should have high consistency."""
        metric = ToneConsistencyMetric()
        text = "This is terrible. I hate it. Awful quality. Disgusting work."
        result = metric.compute(text)

        assert result['tone_consistency'] > 0.5, "Consistent negative tone should have high TC"
        assert 0.0 <= result['tone_consistency'] <= 1.0
        assert result['sentiment_mean'] < 0, "Mean should be negative"
        assert result['num_segments'] == 4

    def test_mixed_sentiment(self):
        """Mixed positive/negative should have lower consistency."""
        metric = ToneConsistencyMetric()
        text = "This is great! I love it. Wait, this is terrible. I hate it."
        result = metric.compute(text)

        assert result['tone_consistency'] < 0.8, "Mixed sentiment should have lower TC"
        assert 0.0 <= result['tone_consistency'] <= 1.0
        assert result['num_segments'] == 4

    def test_single_sentence(self):
        """Single sentence should have perfect consistency."""
        metric = ToneConsistencyMetric()
        text = "This is a great example of wonderful work."
        result = metric.compute(text)

        assert abs(result['tone_consistency'] - 1.0) < EPSILON
        assert abs(result['sentiment_variance']) < EPSILON
        assert abs(result['sentiment_std']) < EPSILON
        assert abs(result['sentiment_range']) < EPSILON
        assert result['num_segments'] == 1


class TestMathematicalCorrectness:
    """Test mathematical formula accuracy."""

    def test_formula_tc_equals_1_minus_variance(self):
        """Verify TC = 1 - σ²(sentiments)."""
        metric = ToneConsistencyMetric()
        text = "I love this! This is okay. I hate this."
        result = metric.compute(text)

        expected_tc = 1.0 - result['sentiment_variance']
        # Allow for max(0, ...) capping
        expected_tc = max(0.0, expected_tc)

        assert abs(result['tone_consistency'] - expected_tc) < EPSILON

    def test_zero_variance_perfect_consistency(self):
        """Zero variance should give TC = 1.0."""
        metric = ToneConsistencyMetric()
        # Use identical sentences
        text = "Great work. Great work. Great work."
        result = metric.compute(text)

        assert result['sentiment_variance'] < 0.01, "Identical sentiment should have ~0 variance"
        assert result['tone_consistency'] > 0.99, "Zero variance → TC ≈ 1.0"

    def test_variance_calculation(self):
        """Variance should match numpy calculation."""
        metric = ToneConsistencyMetric()
        text = "Excellent! Good. Okay. Bad. Terrible!"
        result = metric.compute(text, return_segments=True)

        scores = [seg['sentiment'] for seg in result['segments']]
        expected_variance = float(np.var(scores, ddof=0))

        assert abs(result['sentiment_variance'] - expected_variance) < EPSILON


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_raises_error(self):
        """Empty text should raise ValueError."""
        metric = ToneConsistencyMetric()
        with pytest.raises(ValueError, match="cannot be empty"):
            metric.compute("")

    def test_whitespace_only_raises_error(self):
        """Whitespace-only text should raise ValueError."""
        metric = ToneConsistencyMetric()
        with pytest.raises(ValueError, match="cannot be empty"):
            metric.compute("   \n\t  ")

    def test_very_short_text_below_min_length(self):
        """Text below min_segment_length should raise error."""
        metric = ToneConsistencyMetric(min_segment_length=50)
        with pytest.raises(ValueError, match="No valid segments"):
            metric.compute("Short.")

    def test_text_with_no_sentence_delimiters(self):
        """Text without sentence delimiters should be treated as single segment."""
        metric = ToneConsistencyMetric()
        text = "This is one long continuous text without any sentence endings"
        result = metric.compute(text)

        assert result['num_segments'] >= 1
        assert result['tone_consistency'] == 1.0  # Single segment


class TestSegmentation:
    """Test text segmentation functionality."""

    def test_sentence_segmentation(self):
        """Sentence method should split on punctuation."""
        metric = ToneConsistencyMetric(segmentation_method='sentence')
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        result = metric.compute(text)

        assert result['num_segments'] == 4

    def test_fixed_length_segmentation(self):
        """Fixed-length method should create word-based chunks."""
        metric = ToneConsistencyMetric(segmentation_method='fixed_length')
        # Create 100+ word text
        words = ["word"] * 120
        text = " ".join(words)
        result = metric.compute(text)

        # Should create ~3 chunks (50 words each)
        assert result['num_segments'] >= 2

    def test_min_segment_length_filtering(self):
        """Segments below min_length should be filtered."""
        metric = ToneConsistencyMetric(min_segment_length=20)
        text = "Hi. Short. This is a longer sentence that meets the minimum length."
        result = metric.compute(text)

        # Only the long sentence should remain
        assert result['num_segments'] == 1

    def test_multi_paragraph_text(self):
        """Multi-paragraph text should be properly segmented."""
        metric = ToneConsistencyMetric()
        text = """
        This is the first paragraph. It has multiple sentences. Each one matters.

        This is the second paragraph. Also with multiple sentences. Great work!
        """
        result = metric.compute(text)

        assert result['num_segments'] >= 5


class TestSentimentAnalysis:
    """Test sentiment analysis accuracy."""

    def test_positive_sentiment_detection(self):
        """Positive text should have positive sentiment mean."""
        metric = ToneConsistencyMetric()
        text = "Amazing! Wonderful! Excellent! Fantastic! Perfect!"
        result = metric.compute(text)

        assert result['sentiment_mean'] > 0.5, "Strong positive text should have high positive mean"

    def test_negative_sentiment_detection(self):
        """Negative text should have negative sentiment mean."""
        metric = ToneConsistencyMetric()
        text = "Terrible! Awful! Horrible! Disgusting! Worst!"
        result = metric.compute(text)

        assert result['sentiment_mean'] < -0.5, "Strong negative text should have low negative mean"

    def test_sentiment_normalization_range(self):
        """Sentiment scores should be in [-1, 1]."""
        metric = ToneConsistencyMetric()
        text = "Great! Okay. Bad. Terrible! Excellent!"
        result = metric.compute(text, return_segments=True)

        for segment in result['segments']:
            assert -1.0 <= segment['sentiment'] <= 1.0


class TestReturnValues:
    """Test return value structure and types."""

    def test_required_keys_present(self):
        """Result should contain all required keys."""
        metric = ToneConsistencyMetric()
        text = "This is a test. Another sentence."
        result = metric.compute(text)

        required_keys = [
            'tone_consistency', 'sentiment_variance', 'sentiment_mean',
            'sentiment_std', 'sentiment_range', 'num_segments'
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_return_segments_flag(self):
        """return_segments=True should include segments list."""
        metric = ToneConsistencyMetric()
        text = "First sentence. Second sentence."
        result = metric.compute(text, return_segments=True)

        assert 'segments' in result
        assert isinstance(result['segments'], list)
        assert len(result['segments']) == 2

    def test_segment_structure(self):
        """Each segment should have text, sentiment, label, score."""
        metric = ToneConsistencyMetric()
        text = "This is great. This is terrible."
        result = metric.compute(text, return_segments=True)

        for seg in result['segments']:
            assert 'text' in seg
            assert 'sentiment' in seg
            assert 'label' in seg
            assert 'score' in seg
            assert seg['label'] in ['POSITIVE', 'NEGATIVE']
            assert 0.0 <= seg['score'] <= 1.0

    def test_data_types(self):
        """All numeric values should be float or int."""
        metric = ToneConsistencyMetric()
        text = "Test sentence one. Test sentence two."
        result = metric.compute(text)

        assert isinstance(result['tone_consistency'], float)
        assert isinstance(result['sentiment_variance'], float)
        assert isinstance(result['sentiment_mean'], float)
        assert isinstance(result['sentiment_std'], float)
        assert isinstance(result['sentiment_range'], float)
        assert isinstance(result['num_segments'], int)


class TestVariancePatterns:
    """Test variance calculation patterns."""

    def test_low_variance_high_consistency(self):
        """Low sentiment variance should yield high consistency."""
        metric = ToneConsistencyMetric()
        # All similarly positive
        text = "Good work. Nice job. Well done. Great effort."
        result = metric.compute(text)

        assert result['sentiment_variance'] < 0.1, "Similar sentiments should have low variance"
        assert result['tone_consistency'] > 0.9, "Low variance should yield high consistency"

    def test_high_variance_low_consistency(self):
        """High sentiment variance should yield low consistency."""
        metric = ToneConsistencyMetric()
        # Alternating extreme sentiments
        text = "Absolutely perfect! Completely terrible. Utterly fantastic! Totally horrible."
        result = metric.compute(text)

        assert result['sentiment_variance'] > 0.3, "Extreme swings should have high variance"
        assert result['tone_consistency'] < 0.7, "High variance should yield lower consistency"

    def test_sentiment_range_calculation(self):
        """Sentiment range should be max - min."""
        metric = ToneConsistencyMetric()
        text = "Perfect! Good. Okay. Bad. Terrible!"
        result = metric.compute(text, return_segments=True)

        scores = [seg['sentiment'] for seg in result['segments']]
        expected_range = max(scores) - min(scores)

        assert abs(result['sentiment_range'] - expected_range) < EPSILON


class TestPerformance:
    """Test performance characteristics."""

    def test_long_text_processing(self):
        """Should handle moderately long text efficiently."""
        metric = ToneConsistencyMetric()
        # Create 20 sentences
        sentences = ["This is sentence number {}.".format(i) for i in range(20)]
        text = " ".join(sentences)

        result = metric.compute(text)

        assert result['num_segments'] == 20
        assert 0.0 <= result['tone_consistency'] <= 1.0

    def test_lazy_loading_model(self):
        """Model should lazy load on first compute call."""
        metric = ToneConsistencyMetric()
        assert metric._sentiment_analyzer is None, "Model should not load on init"

        text = "Test sentence."
        metric.compute(text)

        assert metric._sentiment_analyzer is not None, "Model should load after compute"


class TestConfiguration:
    """Test configuration options."""

    def test_custom_model_name(self):
        """Should accept custom model name."""
        metric = ToneConsistencyMetric(model_name='distilbert-base-uncased-finetuned-sst-2-english')
        assert metric.model_name == 'distilbert-base-uncased-finetuned-sst-2-english'

    def test_custom_min_segment_length(self):
        """Should respect custom min_segment_length."""
        metric = ToneConsistencyMetric(min_segment_length=30)
        text = "Short. This is a much longer sentence that exceeds the minimum."
        result = metric.compute(text)

        # Only long sentence should be analyzed
        assert result['num_segments'] == 1

    def test_segmentation_method_override(self):
        """Should allow runtime override of segmentation method."""
        metric = ToneConsistencyMetric(segmentation_method='sentence')
        # Create 100+ word text
        words = ["word"] * 120
        text = " ".join(words)

        # Override to fixed_length
        result = metric.compute(text, segmentation_method='fixed_length')

        # Should create multiple chunks
        assert result['num_segments'] >= 2
