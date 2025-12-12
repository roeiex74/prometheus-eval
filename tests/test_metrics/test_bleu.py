"""
Comprehensive test suite for BLEU metric implementation.

Tests cover:
1. Perfect match scenarios (BLEU = 1.0)
2. Partial matches with varying n-gram overlap
3. Complete mismatches (BLEU ≈ 0)
4. Brevity penalty activation
5. Multi-reference support
6. Corpus-level computation
7. Edge cases (empty strings, single tokens, etc.)
8. Comparison with sacrebleu reference implementation
"""

import pytest
import math
from src.metrics.lexical.bleu import BLEUMetric

# Test tolerance for floating point comparisons
EPSILON = 1e-4


class TestBLEUBasic:
    """Basic BLEU functionality tests."""

    def test_perfect_match(self):
        """Test BLEU = 1.0 for identical strings."""
        metric = BLEUMetric(max_n=4, smoothing="none")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference="The cat is on the mat"
        )

        assert result['bleu'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield BLEU = 1.0"
        assert result['bp'] == 1.0, "Brevity penalty should be 1.0 for perfect match"
        assert all(p == 1.0 for p in result['precisions']), \
            "All n-gram precisions should be 1.0"

    def test_complete_mismatch(self):
        """Test BLEU ≈ 0 for completely different strings."""
        metric = BLEUMetric(max_n=4, smoothing="none")

        result = metric.compute(
            hypothesis="The weather is sunny today",
            reference="A quick brown fox jumps"
        )

        assert result['bleu'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield BLEU ≈ 0"

    def test_partial_match(self):
        """Test BLEU for partial n-gram overlap."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute(
            hypothesis="The cat sat on the mat",
            reference="The cat is sitting on the mat"
        )

        # Should have high unigram precision, lower for higher n-grams
        assert 0.0 < result['bleu'] < 1.0, "Partial match should have 0 < BLEU < 1"
        assert result['precisions'][0] > result['precisions'][1], \
            "Unigram precision should be higher than bigram"

    def test_brevity_penalty_short_hypothesis(self):
        """Test brevity penalty activation for short candidates."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        # Short hypothesis
        result_short = metric.compute(
            hypothesis="The cat",
            reference="The cat is sitting on the mat"
        )

        # Long hypothesis (same length as reference)
        result_long = metric.compute(
            hypothesis="The cat is sitting on the mat",
            reference="The cat is sitting on the mat"
        )

        assert result_short['bp'] < 1.0, \
            "Brevity penalty should be < 1.0 for short candidate"
        assert result_short['bp'] < result_long['bp'], \
            "Short candidate should have lower BP than long candidate"
        assert result_short['length_ratio'] < 1.0, \
            "Length ratio should be < 1.0 for short candidate"

    def test_brevity_penalty_formula(self):
        """Verify brevity penalty formula: BP = exp(1 - r/c) when c < r."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute(
            hypothesis="The cat sat",  # 3 tokens
            reference="The cat is sitting on the mat"  # 7 tokens
        )

        # BP = exp(1 - 7/3) = exp(1 - 2.333...) = exp(-1.333...)
        expected_bp = math.exp(1 - result['reference_length'] / result['candidate_length'])

        assert result['bp'] == pytest.approx(expected_bp, abs=EPSILON), \
            f"BP calculation incorrect: expected {expected_bp}, got {result['bp']}"

    def test_empty_hypothesis(self):
        """Test behavior with empty hypothesis."""
        metric = BLEUMetric(max_n=4)

        result = metric.compute(
            hypothesis="",
            reference="The cat is on the mat"
        )

        assert result['bleu'] == 0.0, "Empty hypothesis should yield BLEU = 0"
        assert result['bp'] == 0.0, "Empty hypothesis should have BP = 0"
        assert result['candidate_length'] == 0

    def test_empty_reference(self):
        """Test behavior with empty reference."""
        # Use no smoothing to ensure BLEU = 0 for no matches
        metric = BLEUMetric(max_n=4, smoothing="none")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference=""
        )

        # With empty reference, no n-grams match
        assert result['bleu'] == 0.0 or result['bleu'] == pytest.approx(0.0, abs=EPSILON)


class TestBLEUSmoothing:
    """Test different smoothing methods."""

    def test_no_smoothing_zero_match(self):
        """Test that no smoothing yields BLEU = 0 for no bigram matches."""
        metric = BLEUMetric(max_n=2, smoothing="none")

        result = metric.compute(
            hypothesis="a b",
            reference="c d"
        )

        assert result['bleu'] == 0.0, \
            "No smoothing with zero matches should yield BLEU = 0"

    def test_epsilon_smoothing(self):
        """Test epsilon smoothing prevents BLEU = 0."""
        metric = BLEUMetric(max_n=2, smoothing="epsilon", epsilon=0.1)

        result = metric.compute(
            hypothesis="a b",
            reference="c d"
        )

        # With epsilon smoothing, should get small non-zero score
        assert result['bleu'] > 0.0, \
            "Epsilon smoothing should prevent BLEU = 0"

    def test_add_k_smoothing(self):
        """Test add-k smoothing."""
        metric = BLEUMetric(max_n=2, smoothing="add-k", k=1.0)

        result = metric.compute(
            hypothesis="a b",
            reference="c d"
        )

        assert result['bleu'] > 0.0, \
            "Add-k smoothing should prevent BLEU = 0"


class TestBLEUMultiReference:
    """Test multi-reference BLEU computation."""

    def test_multi_reference_basic(self):
        """Test BLEU with multiple references."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference=[
                "The cat is sitting on the mat",
                "A cat is on the mat",
                "The cat sits on a mat"
            ]
        )

        # Should use max clipped counts across all references
        assert 0.0 < result['bleu'] <= 1.0
        assert 'precisions' in result

    def test_multi_reference_exact_match(self):
        """Test multi-reference with one exact match."""
        metric = BLEUMetric(max_n=4, smoothing="none")

        result = metric.compute(
            hypothesis="The cat is on the mat",
            reference=[
                "The dog is in the house",
                "The cat is on the mat",  # Exact match
                "A bird flies in the sky"
            ]
        )

        assert result['bleu'] == pytest.approx(1.0, abs=EPSILON), \
            "Multi-reference with exact match should yield BLEU = 1.0"

    def test_multi_reference_empty_list(self):
        """Test that empty reference list raises error."""
        metric = BLEUMetric(max_n=4)

        with pytest.raises(ValueError, match="Reference list cannot be empty"):
            metric.compute(
                hypothesis="The cat is on the mat",
                reference=[]
            )

    def test_multi_reference_closest_length(self):
        """Test that closest reference is used for brevity penalty."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        # Hypothesis: 6 tokens
        # References: 3, 7, 15 tokens
        # Should use 7-token reference (closest)
        result = metric.compute(
            hypothesis="The cat is on the mat",  # 6 tokens
            reference=[
                "Cat mat dog",  # 3 tokens
                "The cat is sitting on the mat today",  # 7 tokens
                "The small orange cat is sitting comfortably on the blue mat in the room"  # 15 tokens
            ]
        )

        # Length ratio should be reasonable (actual ratio depends on tokenization)
        # Just check that it's in a sensible range
        assert 0.5 < result['length_ratio'] < 1.5, \
            f"Length ratio should be sensible, got {result['length_ratio']}"


class TestBLEUCorpus:
    """Test corpus-level BLEU computation."""

    def test_corpus_basic(self):
        """Test basic corpus-level BLEU."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute_corpus(
            hypotheses=[
                "The cat sat on the mat",
                "A dog barked loudly"
            ],
            references=[
                "The cat is sitting on the mat",
                "A dog is barking loudly"
            ]
        )

        assert 0.0 < result['bleu'] <= 1.0
        assert result['num_sentences'] == 2
        assert 'precisions' in result
        assert len(result['precisions']) == 4

    def test_corpus_perfect_match(self):
        """Test corpus-level BLEU with perfect matches."""
        metric = BLEUMetric(max_n=4, smoothing="none")

        result = metric.compute_corpus(
            hypotheses=[
                "The cat is on the mat",
                "A dog is barking"
            ],
            references=[
                "The cat is on the mat",
                "A dog is barking"
            ]
        )

        assert result['bleu'] == pytest.approx(1.0, abs=EPSILON), \
            "Corpus with perfect matches should yield BLEU = 1.0"

    def test_corpus_length_mismatch(self):
        """Test that length mismatch raises error."""
        metric = BLEUMetric(max_n=4)

        with pytest.raises(ValueError, match="Length mismatch"):
            metric.compute_corpus(
                hypotheses=["The cat sat", "A dog barked"],
                references=["The cat is sitting"]
            )

    def test_corpus_empty(self):
        """Test that empty corpus raises error."""
        metric = BLEUMetric(max_n=4)

        with pytest.raises(ValueError, match="Cannot compute BLEU on empty corpus"):
            metric.compute_corpus(
                hypotheses=[],
                references=[]
            )

    def test_corpus_multi_reference(self):
        """Test corpus-level BLEU with multiple references per hypothesis."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute_corpus(
            hypotheses=[
                "The cat sat",
                "A dog barked"
            ],
            references=[
                ["The cat is sitting", "The cat sat down"],
                ["A dog is barking", "A dog barked loudly"]
            ]
        )

        assert 0.0 < result['bleu'] <= 1.0
        assert result['num_sentences'] == 2


class TestBLEUNGramOrders:
    """Test different n-gram orders."""

    def test_unigram_only(self):
        """Test BLEU with only unigrams (max_n=1)."""
        metric = BLEUMetric(max_n=1, smoothing="none")

        result = metric.compute(
            hypothesis="cat dog bird",
            reference="cat dog fish"
        )

        # 2 out of 3 unigrams match
        expected_precision = 2.0 / 3.0

        assert len(result['precisions']) == 1
        assert result['precisions'][0] == pytest.approx(expected_precision, abs=EPSILON)

    def test_different_max_n(self):
        """Test that max_n parameter controls n-gram order."""
        text_hyp = "The cat is on the mat"
        text_ref = "The cat is sitting on the mat"

        for max_n in [1, 2, 3, 4]:
            metric = BLEUMetric(max_n=max_n, smoothing="epsilon")
            result = metric.compute(hypothesis=text_hyp, reference=text_ref)

            assert len(result['precisions']) == max_n, \
                f"max_n={max_n} should produce {max_n} precisions"

    def test_invalid_max_n(self):
        """Test that invalid max_n raises error."""
        with pytest.raises(ValueError, match="max_n must be >= 1"):
            BLEUMetric(max_n=0)


class TestBLEUEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_token(self):
        """Test BLEU with single-token texts."""
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute(
            hypothesis="cat",
            reference="cat"
        )

        assert result['bleu'] > 0.0
        assert result['candidate_length'] == 1
        assert result['reference_length'] == 1

    def test_whitespace_only(self):
        """Test behavior with whitespace-only strings."""
        metric = BLEUMetric(max_n=4)

        result = metric.compute(
            hypothesis="   ",
            reference="The cat is on the mat"
        )

        assert result['bleu'] == 0.0
        assert result['candidate_length'] == 0

    def test_case_insensitivity(self):
        """Test that BLEU is case-insensitive (due to tokenization)."""
        metric = BLEUMetric(max_n=4, smoothing="none")

        result_lower = metric.compute(
            hypothesis="the cat is on the mat",
            reference="the cat is on the mat"
        )

        result_mixed = metric.compute(
            hypothesis="The Cat Is On The Mat",
            reference="the cat is on the mat"
        )

        assert result_lower['bleu'] == pytest.approx(result_mixed['bleu'], abs=EPSILON), \
            "BLEU should be case-insensitive"

    def test_punctuation_handling(self):
        """Test that punctuation is tokenized separately."""
        metric = BLEUMetric(max_n=2, smoothing="epsilon")

        result = metric.compute(
            hypothesis="Hello, world!",
            reference="Hello world"
        )

        # Punctuation creates additional tokens, affecting score
        assert result['candidate_length'] != result['reference_length']


class TestBLEUValidation:
    """Validation tests comparing with expected behavior."""

    def test_known_example_1(self):
        """Test against a known example from literature."""
        # Example: Repeated words with clipping
        metric = BLEUMetric(max_n=4, smoothing="epsilon")

        result = metric.compute(
            hypothesis="the the the the",
            reference="the cat is on the mat"
        )

        # Reference has TWO "the" tokens, so clipped count is 2 out of 4
        # Unigram precision should be 2/4 = 0.5
        # But low higher-order n-gram precision
        # Brevity penalty applies since candidate is shorter
        assert result['precisions'][0] == pytest.approx(0.5, abs=EPSILON), \
            f"Unigram precision should be 2/4=0.5, got {result['precisions'][0]}"
        assert result['bp'] < 1.0, "Brevity penalty should apply"

    def test_known_example_2(self):
        """Test repeated n-gram clipping."""
        metric = BLEUMetric(max_n=2, smoothing="none")

        result = metric.compute(
            hypothesis="the the the",
            reference="the cat"
        )

        # Reference has only one "the", so clipped count is 1
        # Candidate has 3 "the", so precision is 1/3
        # But BP applies since candidate is longer
        assert result['precisions'][0] == pytest.approx(1.0/3.0, abs=EPSILON), \
            "Clipping should limit precision to 1/3"

    def test_sacrebleu_comparison_simple(self):
        """Compare with sacrebleu on simple example."""
        try:
            import sacrebleu

            metric = BLEUMetric(max_n=4, smoothing="none")

            hypothesis = "The cat is on the mat"
            reference = "The cat is sitting on the mat"

            # Our implementation
            our_result = metric.compute(hypothesis, reference)

            # sacrebleu
            sacre_result = sacrebleu.sentence_bleu(
                hypothesis,
                [reference],
                smooth_method='none'
            )

            # Scores should be very close (may differ slightly due to tokenization)
            # We allow larger tolerance for this comparison
            assert abs(our_result['bleu'] - sacre_result.score / 100.0) < 0.1, \
                f"Our BLEU ({our_result['bleu']}) differs significantly from " \
                f"sacrebleu ({sacre_result.score / 100.0})"

        except ImportError:
            pytest.skip("sacrebleu not installed")


class TestBLEUConfiguration:
    """Test metric configuration options."""

    def test_invalid_smoothing(self):
        """Test that invalid smoothing method raises error."""
        with pytest.raises(ValueError, match="smoothing must be one of"):
            BLEUMetric(smoothing="invalid")

    def test_custom_epsilon(self):
        """Test custom epsilon value."""
        metric = BLEUMetric(max_n=2, smoothing="epsilon", epsilon=0.5)

        # With larger epsilon, zero-match cases get higher scores
        result = metric.compute(
            hypothesis="a b",
            reference="c d"
        )

        assert result['bleu'] > 0.0

    def test_custom_k(self):
        """Test custom k value for add-k smoothing."""
        metric = BLEUMetric(max_n=2, smoothing="add-k", k=2.0)

        result = metric.compute(
            hypothesis="a b",
            reference="c d"
        )

        assert result['bleu'] > 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
