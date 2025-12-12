"""
Comprehensive test suite for BERTScore metric implementation.

Tests cover:
1. Identical sentences (F1 ≈ 1.0)
2. Paraphrases (high F1)
3. Unrelated sentences (low F1)
4. Batch computation
5. Model loading and caching
6. Edge cases (empty strings, single tokens, etc.)
7. Precision vs Recall trade-offs
8. Token-level alignment validation
"""

import pytest
import torch
from src.metrics.semantic.bertscore import BERTScoreMetric

# Test tolerance for floating point comparisons
EPSILON = 1e-3  # Larger tolerance for embedding-based metrics


class TestBERTScoreBasic:
    """Basic BERTScore functionality tests."""

    @pytest.fixture(scope="class")
    def metric(self):
        """Create a shared BERTScore metric instance."""
        return BERTScoreMetric(device="cpu")

    def test_identical_sentences(self, metric):
        """Test that identical sentences yield F1 ≈ 1.0."""
        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is on the mat"
        )

        assert result['f1'] > 0.99, \
            f"Identical sentences should yield F1 ≈ 1.0, got {result['f1']}"
        assert result['precision'] > 0.99
        assert result['recall'] > 0.99

    def test_paraphrase_high_similarity(self, metric):
        """Test that paraphrases yield high F1 score."""
        result = metric.compute(
            hypothesis="A cat sits on a mat",
            reference="The cat is on the mat"
        )

        assert result['f1'] > 0.7, \
            f"Paraphrases should yield high F1 (>0.7), got {result['f1']}"
        assert 0.0 <= result['precision'] <= 1.0
        assert 0.0 <= result['recall'] <= 1.0

    def test_unrelated_sentences_low_similarity(self, metric):
        """Test that unrelated sentences yield low F1 score."""
        result = metric.compute(
            hypothesis="The weather is sunny today",
            reference="The cat is on the mat"
        )

        assert result['f1'] < 0.6, \
            f"Unrelated sentences should yield low F1 (<0.6), got {result['f1']}"

    def test_semantic_preservation(self, metric):
        """Test that semantically equivalent sentences have high similarity."""
        result = metric.compute(
            hypothesis="The quick brown fox jumps over the lazy dog",
            reference="A fast brown fox leaps over a lazy dog"
        )

        assert result['f1'] > 0.75, \
            f"Semantically equivalent sentences should have F1 > 0.75, got {result['f1']}"

    def test_partial_overlap(self, metric):
        """Test sentences with partial semantic overlap."""
        result = metric.compute(
            hypothesis="The cat is sleeping on the couch",
            reference="The cat is on the mat"
        )

        # Should have moderate F1 (shared: "cat is on the")
        assert 0.4 < result['f1'] < 0.9, \
            f"Partial overlap should yield moderate F1, got {result['f1']}"


class TestBERTScoreScoreComponents:
    """Test precision, recall, and F1 relationships."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_f1_formula(self, metric):
        """Verify F1 = 2 × (P × R) / (P + R)."""
        result = metric.compute(
            hypothesis="The cat sat on the mat",
            reference="The cat is sitting on the mat"
        )

        P = result['precision']
        R = result['recall']
        expected_f1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0.0

        assert result['f1'] == pytest.approx(expected_f1, abs=EPSILON), \
            f"F1 formula incorrect: expected {expected_f1}, got {result['f1']}"

    def test_recall_vs_precision_short_hypothesis(self, metric):
        """Test that short hypothesis has high precision, lower recall."""
        result = metric.compute(
            hypothesis="The cat",
            reference="The cat is sitting on the mat"
        )

        # Precision: All hypothesis tokens match well with reference
        # Recall: Many reference tokens not covered by hypothesis
        # Therefore: Precision >= Recall
        assert result['precision'] >= result['recall'] - EPSILON, \
            "Short hypothesis should have precision >= recall"

    def test_recall_vs_precision_long_hypothesis(self, metric):
        """Test that long hypothesis has lower precision, higher recall."""
        result = metric.compute(
            hypothesis="The cat is sitting comfortably on the blue mat in the living room",
            reference="The cat is on the mat"
        )

        # Precision: Many hypothesis tokens don't match reference well
        # Recall: Most reference tokens covered by hypothesis
        # Therefore: Recall >= Precision
        assert result['recall'] >= result['precision'] - EPSILON, \
            "Long hypothesis should have recall >= precision"

    def test_score_symmetry(self, metric):
        """Test that swapping hypothesis and reference swaps precision and recall."""
        text1 = "The cat sat on the mat"
        text2 = "The cat is sitting"

        result1 = metric.compute(hypothesis=text1, reference=text2)
        result2 = metric.compute(hypothesis=text2, reference=text1)

        # Precision and recall should swap
        assert result1['precision'] == pytest.approx(result2['recall'], abs=EPSILON), \
            "Swapping texts should swap precision and recall"
        assert result1['recall'] == pytest.approx(result2['precision'], abs=EPSILON), \
            "Swapping texts should swap precision and recall"

        # F1 should be the same (symmetric)
        assert result1['f1'] == pytest.approx(result2['f1'], abs=EPSILON), \
            "F1 should be symmetric"


class TestBERTScoreBatch:
    """Test batch computation functionality."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_batch_basic(self, metric):
        """Test basic batch computation."""
        hypotheses = [
            "The cat sat on the mat",
            "A dog is barking loudly"
        ]
        references = [
            "The cat is sitting on the mat",
            "A dog barks very loudly"
        ]

        results = metric.compute_batch(hypotheses, references)

        assert len(results['f1']) == 2
        assert len(results['precision']) == 2
        assert len(results['recall']) == 2
        assert all(0.0 <= f1 <= 1.0 for f1 in results['f1'])

    def test_batch_mean_computation(self, metric):
        """Test that batch means are computed correctly."""
        hypotheses = ["The cat sat", "A dog barked"]
        references = ["The cat is sitting", "A dog is barking"]

        results = metric.compute_batch(hypotheses, references)

        # Verify means
        import numpy as np
        assert results['mean_f1'] == pytest.approx(np.mean(results['f1']), abs=EPSILON)
        assert results['mean_precision'] == pytest.approx(np.mean(results['precision']), abs=EPSILON)
        assert results['mean_recall'] == pytest.approx(np.mean(results['recall']), abs=EPSILON)

    def test_batch_length_mismatch(self, metric):
        """Test that length mismatch raises error."""
        with pytest.raises(ValueError, match="Length mismatch"):
            metric.compute_batch(
                hypotheses=["The cat sat", "A dog barked"],
                references=["The cat is sitting"]
            )

    def test_batch_empty(self, metric):
        """Test that empty batch raises error."""
        with pytest.raises(ValueError, match="Cannot compute BERTScore on empty batch"):
            metric.compute_batch(
                hypotheses=[],
                references=[]
            )

    def test_batch_single_item(self, metric):
        """Test batch with single item."""
        results = metric.compute_batch(
            hypotheses=["The cat sat"],
            references=["The cat is sitting"]
        )

        assert len(results['f1']) == 1
        assert results['mean_f1'] == results['f1'][0]


class TestBERTScoreEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_empty_hypothesis(self, metric):
        """Test behavior with empty hypothesis."""
        result = metric.compute(
            hypothesis="",
            reference="The cat is on the mat"
        )

        # Should handle gracefully
        assert 0.0 <= result['f1'] <= 1.0
        assert result['num_hyp_tokens'] >= 0

    def test_empty_reference(self, metric):
        """Test behavior with empty reference."""
        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference=""
        )

        # Should handle gracefully
        assert 0.0 <= result['f1'] <= 1.0
        assert result['num_ref_tokens'] >= 0

    def test_both_empty(self, metric):
        """Test behavior when both texts are empty."""
        result = metric.compute(
            hypothesis="",
            reference=""
        )

        assert 0.0 <= result['f1'] <= 1.0

    def test_single_token(self, metric):
        """Test with single-token texts."""
        result = metric.compute(
            hypothesis="cat",
            reference="cat"
        )

        assert result['f1'] > 0.95, "Identical single tokens should have high F1"
        assert result['num_hyp_tokens'] >= 1
        assert result['num_ref_tokens'] >= 1

    def test_single_token_different(self, metric):
        """Test with different single tokens."""
        result = metric.compute(
            hypothesis="cat",
            reference="dog"
        )

        # Different words but both are animals, so some semantic similarity
        assert 0.0 <= result['f1'] <= 1.0

    def test_whitespace_only(self, metric):
        """Test with whitespace-only strings."""
        result = metric.compute(
            hypothesis="   ",
            reference="The cat is on the mat"
        )

        assert 0.0 <= result['f1'] <= 1.0

    def test_very_long_text(self, metric):
        """Test with very long text (truncation)."""
        # Create text longer than max_length
        long_text = " ".join(["word"] * 600)
        short_text = "The cat is on the mat"

        result = metric.compute(
            hypothesis=long_text,
            reference=short_text
        )

        assert 0.0 <= result['f1'] <= 1.0
        # Text should be truncated to max_length


class TestBERTScoreSemanticUnderstanding:
    """Test that BERTScore captures semantic meaning."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_synonym_similarity(self, metric):
        """Test that synonyms have high similarity."""
        result = metric.compute(
            hypothesis="The quick brown fox",
            reference="The fast brown fox"
        )

        # "quick" and "fast" are synonyms
        assert result['f1'] > 0.85, \
            f"Synonyms should have high similarity, got F1={result['f1']}"

    def test_antonym_lower_similarity(self, metric):
        """Test that antonyms have lower similarity."""
        result_synonym = metric.compute(
            hypothesis="The day is bright",
            reference="The day is light"
        )

        result_antonym = metric.compute(
            hypothesis="The day is bright",
            reference="The day is dark"
        )

        # Synonym should have higher F1 than antonym
        assert result_synonym['f1'] > result_antonym['f1'], \
            "Synonyms should have higher F1 than antonyms"

    def test_negation_sensitivity(self, metric):
        """Test sensitivity to negation."""
        result_positive = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is on the mat"
        )

        result_negation = metric.compute(
            hypothesis="The cat is not on the mat",
            reference="The cat is on the mat"
        )

        # Negation should lower F1
        assert result_positive['f1'] > result_negation['f1'], \
            "Negation should lower F1 score"

    def test_word_order_less_important(self, metric):
        """Test that word order is less important than lexical metrics."""
        result = metric.compute(
            hypothesis="The mat is on the cat",
            reference="The cat is on the mat"
        )

        # BERTScore should still be relatively high (same words, different order)
        assert result['f1'] > 0.7, \
            f"Word reordering should still yield high F1, got {result['f1']}"

    def test_case_insensitivity(self, metric):
        """Test that BERTScore handles different cases."""
        result_lower = metric.compute(
            hypothesis="the cat is on the mat",
            reference="the cat is on the mat"
        )

        result_mixed = metric.compute(
            hypothesis="THE CAT IS ON THE MAT",
            reference="the cat is on the mat"
        )

        # Scores should be very close (minor tokenization differences possible)
        assert abs(result_lower['f1'] - result_mixed['f1']) < 0.1, \
            "Case should not significantly affect BERTScore"


class TestBERTScoreModelConfiguration:
    """Test model configuration and device handling."""

    def test_cpu_device(self):
        """Test metric on CPU device."""
        metric = BERTScoreMetric(device="cpu")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is sitting on the mat"
        )

        assert 0.0 <= result['f1'] <= 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test metric on CUDA device."""
        metric = BERTScoreMetric(device="cuda")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is sitting on the mat"
        )

        assert 0.0 <= result['f1'] <= 1.0

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device(self):
        """Test metric on MPS device (Apple Silicon)."""
        metric = BERTScoreMetric(device="mps")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is sitting on the mat"
        )

        assert 0.0 <= result['f1'] <= 1.0

    def test_auto_device_detection(self):
        """Test automatic device detection."""
        metric = BERTScoreMetric(device=None)

        # Should auto-select a device
        assert metric.device in ["cpu", "cuda", "mps"]

    def test_model_caching(self):
        """Test that model is loaded only once."""
        metric1 = BERTScoreMetric(device="cpu")
        metric2 = BERTScoreMetric(device="cpu")

        # Both should work independently
        result1 = metric1.compute("The cat sat", "The cat is sitting")
        result2 = metric2.compute("The cat sat", "The cat is sitting")

        assert result1['f1'] == pytest.approx(result2['f1'], abs=EPSILON)

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        metric = BERTScoreMetric(max_length=128)

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is sitting on the mat"
        )

        assert 0.0 <= result['f1'] <= 1.0


class TestBERTScoreCallable:
    """Test callable interface."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_callable_interface(self, metric):
        """Test that metric can be called directly."""
        result = metric(
            hypothesis="The cat sat",
            reference="The cat is sitting"
        )

        assert 'f1' in result
        assert 'precision' in result
        assert 'recall' in result


class TestBERTScoreMetadata:
    """Test metadata returned by BERTScore."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_token_counts(self, metric):
        """Test that token counts are accurate."""
        result = metric.compute(
            hypothesis="The cat sat on the mat",  # 6 tokens
            reference="The cat is sitting"  # 4 tokens
        )

        # Token counts may vary slightly due to special token removal
        assert result['num_hyp_tokens'] >= 4, "Should have tokens from hypothesis"
        assert result['num_ref_tokens'] >= 3, "Should have tokens from reference"

    def test_return_structure(self, metric):
        """Test that return dictionary has expected structure."""
        result = metric.compute(
            hypothesis="The cat sat",
            reference="The cat is sitting"
        )

        required_keys = {'precision', 'recall', 'f1', 'num_hyp_tokens', 'num_ref_tokens'}
        assert required_keys.issubset(result.keys()), \
            f"Missing required keys. Expected {required_keys}, got {result.keys()}"

        # All scores should be floats in [0, 1]
        for key in ['precision', 'recall', 'f1']:
            assert isinstance(result[key], float)
            assert 0.0 <= result[key] <= 1.0

        # Token counts should be non-negative integers
        assert result['num_hyp_tokens'] >= 0
        assert result['num_ref_tokens'] >= 0


class TestBERTScoreValidation:
    """Validation tests for numerical stability and correctness."""

    @pytest.fixture(scope="class")
    def metric(self):
        return BERTScoreMetric(device="cpu")

    def test_numerical_stability_repeated_words(self, metric):
        """Test numerical stability with repeated words."""
        result = metric.compute(
            hypothesis="cat cat cat cat cat",
            reference="cat dog bird fish horse"
        )

        # Should handle repeated words without numerical issues
        assert not torch.isnan(torch.tensor(result['f1']))
        assert not torch.isinf(torch.tensor(result['f1']))

    def test_special_characters(self, metric):
        """Test handling of special characters."""
        result = metric.compute(
            hypothesis="Hello! How are you?",
            reference="Hello. How are you?"
        )

        # Should handle punctuation gracefully
        assert 0.0 <= result['f1'] <= 1.0

    def test_unicode_characters(self, metric):
        """Test handling of Unicode characters."""
        result = metric.compute(
            hypothesis="The café is open",
            reference="The cafe is open"
        )

        # Should handle Unicode
        assert 0.0 <= result['f1'] <= 1.0

    def test_numbers(self, metric):
        """Test handling of numbers."""
        result = metric.compute(
            hypothesis="The year 2024 is here",
            reference="The year 2023 is here"
        )

        # Should handle numbers
        assert 0.0 <= result['f1'] <= 1.0
        # Different years should lower F1
        assert result['f1'] < 0.95


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
