"""
Comprehensive test suite for ROUGE metric implementation.

Tests cover:
1. Basic ROUGE-1, ROUGE-2, ROUGE-L functionality
2. Longest Common Subsequence (LCS) computation
3. Multi-reference support
4. Beta parameter for F-measure
5. Edge cases (empty strings, single tokens, etc.)
6. Input validation
7. Known examples with expected scores
"""

import pytest
import math
from src.metrics.lexical.rouge import ROUGEMetric

# Test tolerance for floating point comparisons
EPSILON = 1e-4


class TestROUGEBasic:
    """Test basic ROUGE-1, ROUGE-2, ROUGE-L functionality."""

    def test_perfect_match(self):
        """Test ROUGE = 1.0 for identical strings."""
        rouge = ROUGEMetric()
        candidate = "the cat sat on the mat"
        reference = "the cat sat on the mat"
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield ROUGE-1 = 1.0"
        assert scores['rouge2'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield ROUGE-2 = 1.0"
        assert scores['rougeL'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield ROUGE-L = 1.0"
        assert scores['overall'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield overall = 1.0"

    def test_complete_mismatch(self):
        """Test ROUGE = 0 for completely different strings."""
        rouge = ROUGEMetric()
        candidate = "the weather is sunny today"
        reference = "a quick brown fox jumps"
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield ROUGE-1 = 0.0"
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield ROUGE-2 = 0.0"
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield ROUGE-L = 0.0"
        assert scores['overall'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield overall = 0.0"

    def test_partial_unigram_overlap(self):
        """Test ROUGE-1 with partial unigram overlap."""
        rouge = ROUGEMetric(variants=['rouge1'])
        candidate = "the cat is on the mat"
        reference = "the dog is in the house"
        scores = rouge.compute(candidate, reference)

        # Common words: "the" (appears twice in both), "is"
        # Unique unigrams in candidate: {the, cat, is, on, mat} = 5
        # Unique unigrams in reference: {the, dog, is, in, house} = 5
        # Overlap: {the, is} = 2
        # Recall = 2/5 = 0.4, Precision = 2/5 = 0.4, F1 = 0.4
        expected_f1 = 0.4

        assert scores['rouge1'] == pytest.approx(expected_f1, abs=EPSILON), \
            f"Expected ROUGE-1 = {expected_f1}, got {scores['rouge1']}"

    def test_partial_bigram_overlap(self):
        """Test ROUGE-2 with partial bigram overlap."""
        rouge = ROUGEMetric(variants=['rouge2'])
        candidate = "the cat sat on the mat"
        reference = "the cat is on the mat"
        scores = rouge.compute(candidate, reference)

        # Candidate bigrams: {(the,cat), (cat,sat), (sat,on), (on,the), (the,mat)}
        # Reference bigrams: {(the,cat), (cat,is), (is,on), (on,the), (the,mat)}
        # Overlap: {(the,cat), (on,the), (the,mat)} = 3
        # Recall = 3/5, Precision = 3/5, F1 = 3/5 = 0.6
        expected_f1 = 0.6

        assert scores['rouge2'] == pytest.approx(expected_f1, abs=EPSILON), \
            f"Expected ROUGE-2 = {expected_f1}, got {scores['rouge2']}"

    def test_lcs_computation(self):
        """Test ROUGE-L with specific LCS example."""
        rouge = ROUGEMetric(variants=['rougeL'])
        # LCS of these two is "the cat on the mat" (length 5)
        candidate = "the cat sat on the mat"  # 6 tokens
        reference = "the cat is on the mat"   # 6 tokens
        scores = rouge.compute(candidate, reference)

        # LCS length = 5 (the, cat, on, the, mat)
        # Recall = 5/6, Precision = 5/6, F1 = 5/6
        expected_f1 = 5.0 / 6.0

        assert scores['rougeL'] == pytest.approx(expected_f1, abs=EPSILON), \
            f"Expected ROUGE-L = {expected_f1}, got {scores['rougeL']}"

    def test_variant_selection(self):
        """Test initialization with specific variants only."""
        rouge_1_only = ROUGEMetric(variants=['rouge1'])
        scores = rouge_1_only.compute("the cat", "the dog")

        assert 'rouge1' in scores
        assert 'rouge2' not in scores
        assert 'rougeL' not in scores
        assert 'overall' in scores

        rouge_2_only = ROUGEMetric(variants=['rouge2'])
        scores = rouge_2_only.compute("the cat sat", "the cat is")

        assert 'rouge1' not in scores
        assert 'rouge2' in scores
        assert 'rougeL' not in scores
        assert 'overall' in scores

    def test_all_variants(self):
        """Test that all variants are computed by default."""
        rouge = ROUGEMetric()
        scores = rouge.compute("the cat sat", "the cat is")

        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores
        assert 'overall' in scores
        assert len([k for k in scores.keys() if k != 'overall']) == 3


class TestROUGELCS:
    """Test Longest Common Subsequence computation."""

    def test_lcs_length_basic(self):
        """Test LCS with known example."""
        rouge = ROUGEMetric()
        # LCS of "ABCD" vs "ACBD" is "ABD" (length 3)
        candidate = "a b c d"
        reference = "a c b d"
        scores = rouge.compute(candidate, reference)

        # LCS = [a, b, d] or [a, c, d] = 3 tokens
        # Recall = 3/4, Precision = 3/4, F1 = 3/4 = 0.75
        assert scores['rougeL'] == pytest.approx(0.75, abs=EPSILON)

    def test_lcs_no_common_subsequence(self):
        """Test LCS with no common words."""
        rouge = ROUGEMetric(variants=['rougeL'])
        candidate = "a b c"
        reference = "x y z"
        scores = rouge.compute(candidate, reference)

        # LCS length = 0
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON)

    def test_lcs_identical_sequences(self):
        """Test LCS with identical sequences."""
        rouge = ROUGEMetric(variants=['rougeL'])
        candidate = "the quick brown fox"
        reference = "the quick brown fox"
        scores = rouge.compute(candidate, reference)

        # LCS length = 4 (all tokens)
        # Recall = 4/4 = 1.0, Precision = 4/4 = 1.0, F1 = 1.0
        assert scores['rougeL'] == pytest.approx(1.0, abs=EPSILON)

    def test_lcs_subsequence_vs_substring(self):
        """Test that LCS allows non-contiguous matches."""
        rouge = ROUGEMetric(variants=['rougeL'])
        # LCS allows skipping tokens
        candidate = "a x b y c"  # 5 tokens
        reference = "a b c"      # 3 tokens
        scores = rouge.compute(candidate, reference)

        # LCS = [a, b, c] = 3 tokens
        # Recall = 3/3 = 1.0, Precision = 3/5 = 0.6
        # F1 = 2*1.0*0.6/(1.0+0.6) = 1.2/1.6 = 0.75
        expected_f1 = 0.75

        assert scores['rougeL'] == pytest.approx(expected_f1, abs=EPSILON)

    def test_lcs_space_optimization(self):
        """Test that space-optimized DP produces correct results."""
        rouge = ROUGEMetric(variants=['rougeL'])
        # Longer sequences to test space optimization
        candidate = "the quick brown fox jumps over the lazy dog"
        reference = "the quick brown fox sleeps near the lazy cat"
        scores = rouge.compute(candidate, reference)

        # LCS should include: the, quick, brown, fox, the, lazy
        # Should have non-zero score
        assert scores['rougeL'] > 0.0
        assert scores['rougeL'] < 1.0

    def test_rouge_l_f_measure(self):
        """Verify F1 calculation from LCS recall/precision."""
        rouge = ROUGEMetric(variants=['rougeL'])
        candidate = "a b c d e"  # 5 tokens
        reference = "a b c"       # 3 tokens
        scores = rouge.compute(candidate, reference)

        # LCS = [a, b, c] = 3 tokens
        # Recall = 3/3 = 1.0
        # Precision = 3/5 = 0.6
        # F1 = 2*1.0*0.6/(1.0+0.6) = 1.2/1.6 = 0.75
        expected_recall = 1.0
        expected_precision = 0.6
        expected_f1 = (2 * expected_recall * expected_precision) / (expected_recall + expected_precision)

        assert scores['rougeL'] == pytest.approx(expected_f1, abs=EPSILON)


class TestROUGEMultiReference:
    """Test multi-reference handling."""

    def test_single_reference_string(self):
        """Test single reference as string input."""
        rouge = ROUGEMetric()
        candidate = "the cat sat on the mat"
        reference = "the cat is on the mat"
        scores = rouge.compute(candidate, reference)

        assert 'rouge1' in scores
        assert 'overall' in scores
        assert scores['overall'] > 0.0

    def test_single_reference_list(self):
        """Test single reference as list input."""
        rouge = ROUGEMetric()
        candidate = "the cat sat on the mat"
        reference = ["the cat is on the mat"]
        scores = rouge.compute(candidate, reference)

        assert 'rouge1' in scores
        assert 'overall' in scores
        assert scores['overall'] > 0.0

    def test_multiple_references_max_selection(self):
        """Test that max score is selected across multiple references."""
        rouge = ROUGEMetric()
        candidate = "the cat sat on the mat"
        references = [
            "the dog is in the house",  # Low overlap
            "the cat sat on the mat",   # Perfect match
            "a bird flies in the sky"   # No overlap
        ]
        scores = rouge.compute(candidate, references)

        # Should select perfect match (second reference)
        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(1.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(1.0, abs=EPSILON)

    def test_multiple_references_different_scores(self):
        """Verify correct max selection when references have different scores."""
        rouge = ROUGEMetric(variants=['rouge1'])
        candidate = "the cat sat"
        references = [
            "the dog is",      # Overlap: {the} = 1/3 ≈ 0.33
            "the cat sat",     # Overlap: {the, cat, sat} = 3/3 = 1.0
            "cat sat on mat"   # Overlap: {cat, sat} = 2/3 ≈ 0.67
        ]
        scores = rouge.compute(candidate, references)

        # Should select the perfect match (second reference)
        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON)

    def test_empty_reference_list(self):
        """Test that empty reference list raises ValueError."""
        rouge = ROUGEMetric()
        candidate = "the cat sat"

        with pytest.raises(ValueError, match="References cannot be empty"):
            rouge.compute(candidate, [])

    def test_multi_reference_partial_overlap(self):
        """Test multi-reference with all partial overlaps."""
        rouge = ROUGEMetric(variants=['rouge1'])
        candidate = "the cat sat on the mat"
        references = [
            "the dog is in the house",  # Overlap: {the, is}
            "a cat sat near a tree",    # Overlap: {cat, sat}
            "on the big red mat"        # Overlap: {on, the, mat}
        ]
        scores = rouge.compute(candidate, references)

        # Should pick the best match (third reference has 3 overlapping words)
        # Candidate unique: {the, cat, sat, on, mat} = 5
        # Reference 3 unique: {on, the, big, red, mat} = 5
        # Overlap: {on, the, mat} = 3
        # Recall = 3/5, Precision = 3/5, F1 = 3/5 = 0.6
        assert scores['rouge1'] >= 0.5  # Should be at least this good


class TestROUGEBetaParameter:
    """Test F-measure beta configuration."""

    def test_default_beta(self):
        """Test default beta = 1.0 (balanced F1)."""
        rouge = ROUGEMetric()
        # Verify beta_squared is set to 1.0
        assert rouge.beta_squared == 1.0

    def test_custom_beta_recall_weighted(self):
        """Test beta > 1.0 (favors recall)."""
        rouge_recall_heavy = ROUGEMetric(beta=2.0)
        rouge_balanced = ROUGEMetric(beta=1.0)

        # Create scenario where recall > precision
        candidate = "a b"           # 2 tokens
        reference = "a b c d e"     # 5 tokens
        # Recall = 2/5 = 0.4, Precision = 2/2 = 1.0

        scores_recall = rouge_recall_heavy.compute(candidate, reference)
        scores_balanced = rouge_balanced.compute(candidate, reference)

        # With beta=2.0, the score should be different from beta=1.0
        # Higher beta weights recall more, and recall < precision here
        # So beta=2 should give lower score than beta=1
        assert scores_recall['rouge1'] != scores_balanced['rouge1']

    def test_custom_beta_precision_weighted(self):
        """Test beta < 1.0 (favors precision)."""
        rouge_precision_heavy = ROUGEMetric(beta=0.5)
        rouge_balanced = ROUGEMetric(beta=1.0)

        # Create scenario where precision > recall
        candidate = "a b c d e"     # 5 tokens
        reference = "a b"           # 2 tokens
        # Recall = 2/2 = 1.0, Precision = 2/5 = 0.4

        scores_precision = rouge_precision_heavy.compute(candidate, reference)
        scores_balanced = rouge_balanced.compute(candidate, reference)

        # Scores should differ
        assert scores_precision['rouge1'] != scores_balanced['rouge1']

    def test_beta_formula_verification(self):
        """Verify F-measure formula with custom beta."""
        rouge = ROUGEMetric(beta=2.0, variants=['rouge1'])
        candidate = "a b c"      # 3 tokens
        reference = "a b c d e"  # 5 tokens
        scores = rouge.compute(candidate, reference)

        # Unique unigrams in candidate: {a, b, c} = 3
        # Unique unigrams in reference: {a, b, c, d, e} = 5
        # Overlap: {a, b, c} = 3
        # Recall = 3/5 = 0.6
        # Precision = 3/3 = 1.0
        # F_beta = (1 + beta²) * R * P / (R + beta² * P)
        # F_2 = (1 + 4) * 0.6 * 1.0 / (0.6 + 4 * 1.0)
        # F_2 = 5 * 0.6 / 4.6 = 3.0 / 4.6 ≈ 0.652

        recall = 0.6
        precision = 1.0
        beta_squared = 4.0
        expected_f = ((1 + beta_squared) * recall * precision) / (recall + beta_squared * precision)

        assert scores['rouge1'] == pytest.approx(expected_f, abs=EPSILON)


class TestROUGEEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_candidate(self):
        """Test empty candidate string."""
        rouge = ROUGEMetric()
        candidate = ""
        reference = "the cat sat on the mat"
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['overall'] == pytest.approx(0.0, abs=EPSILON)

    def test_empty_reference(self):
        """Test empty reference string."""
        rouge = ROUGEMetric()
        candidate = "the cat sat on the mat"
        reference = ""
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['overall'] == pytest.approx(0.0, abs=EPSILON)

    def test_both_empty(self):
        """Test both candidate and reference empty."""
        rouge = ROUGEMetric()
        candidate = ""
        reference = ""
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['overall'] == pytest.approx(0.0, abs=EPSILON)

    def test_whitespace_only(self):
        """Test whitespace-only strings."""
        rouge = ROUGEMetric()
        candidate = "   "
        reference = "the cat sat"
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['overall'] == pytest.approx(0.0, abs=EPSILON)

    def test_single_word(self):
        """Test single-word inputs."""
        rouge = ROUGEMetric()
        candidate = "cat"
        reference = "cat"
        scores = rouge.compute(candidate, reference)

        # ROUGE-1: perfect match
        # ROUGE-2: no bigrams (need at least 2 tokens)
        # ROUGE-L: perfect match
        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)  # No bigrams
        assert scores['rougeL'] == pytest.approx(1.0, abs=EPSILON)

    def test_single_word_mismatch(self):
        """Test single-word mismatch."""
        rouge = ROUGEMetric()
        candidate = "cat"
        reference = "dog"
        scores = rouge.compute(candidate, reference)

        assert scores['rouge1'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(0.0, abs=EPSILON)

    def test_punctuation_handling(self):
        """Test NLTK tokenization with punctuation."""
        rouge = ROUGEMetric(variants=['rouge1'])
        candidate = "Hello, world!"
        reference = "Hello world"
        scores = rouge.compute(candidate, reference)

        # Tokenization: ["hello", ",", "world", "!"] vs ["hello", "world"]
        # Overlap: {hello, world} = 2
        # Candidate unique: {hello, ,, world, !} = 4
        # Reference unique: {hello, world} = 2
        # Recall = 2/2 = 1.0, Precision = 2/4 = 0.5
        # F1 = 2*1.0*0.5/(1.0+0.5) = 1.0/1.5 ≈ 0.667
        expected_f1 = 2.0 / 3.0

        assert scores['rouge1'] == pytest.approx(expected_f1, abs=0.01)

    def test_case_insensitivity(self):
        """Test case insensitivity."""
        rouge = ROUGEMetric()
        candidate = "The Cat Sat"
        reference = "the cat sat"
        scores = rouge.compute(candidate, reference)

        # Should be perfect match due to lowercasing
        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON)
        assert scores['rouge2'] == pytest.approx(1.0, abs=EPSILON)
        assert scores['rougeL'] == pytest.approx(1.0, abs=EPSILON)

    def test_repeated_words(self):
        """Test handling of repeated words."""
        rouge = ROUGEMetric(variants=['rouge1'])
        candidate = "the the the cat"
        reference = "the cat sat"
        scores = rouge.compute(candidate, reference)

        # Unique unigrams in candidate: {the, cat} = 2
        # Unique unigrams in reference: {the, cat, sat} = 3
        # Overlap: {the, cat} = 2
        # Recall = 2/3, Precision = 2/2 = 1.0
        # F1 = 2*(2/3)*1.0/((2/3)+1.0) = (4/3)/(5/3) = 4/5 = 0.8
        expected_f1 = 0.8

        assert scores['rouge1'] == pytest.approx(expected_f1, abs=EPSILON)


class TestROUGEValidation:
    """Test input validation and error handling."""

    def test_invalid_variant(self):
        """Test invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid variant"):
            ROUGEMetric(variants=['rouge1', 'rouge99'])

    def test_invalid_variant_type(self):
        """Test invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Invalid variant"):
            ROUGEMetric(variants=['bleu'])

    def test_empty_variants_list(self):
        """Test empty variants list uses defaults."""
        rouge = ROUGEMetric(variants=[])
        # Empty list should still allow compute (may result in empty scores)
        scores = rouge.compute("test", "test")
        # Overall should be 0 if no variants
        assert scores['overall'] == 0.0

    def test_none_variants_uses_default(self):
        """Test None variants uses default set."""
        rouge = ROUGEMetric(variants=None)
        scores = rouge.compute("the cat", "the dog")

        # Should have all three default variants
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores

    def test_duplicate_variants(self):
        """Test duplicate variants in list."""
        rouge = ROUGEMetric(variants=['rouge1', 'rouge1', 'rouge2'])
        scores = rouge.compute("the cat", "the dog")

        # Should handle duplicates gracefully
        assert 'rouge1' in scores
        assert 'rouge2' in scores


class TestROUGEKnownExamples:
    """Validate against known ROUGE scores."""

    def test_known_example_1(self):
        """Test with known ROUGE scores."""
        rouge = ROUGEMetric()
        reference = "the cat sat on the mat"
        candidate = "the cat is on the mat"

        scores = rouge.compute(candidate, reference)

        # ROUGE-1:
        # Candidate unique: {the, cat, is, on, mat} = 5
        # Reference unique: {the, cat, sat, on, mat} = 5
        # Overlap: {the, cat, on, mat} = 4
        # Recall = 4/5 = 0.8, Precision = 4/5 = 0.8, F1 = 0.8
        assert scores['rouge1'] == pytest.approx(0.8, abs=EPSILON)

        # ROUGE-2:
        # Candidate bigrams: {(the,cat), (cat,is), (is,on), (on,the), (the,mat)}
        # Reference bigrams: {(the,cat), (cat,sat), (sat,on), (on,the), (the,mat)}
        # Overlap: {(the,cat), (on,the), (the,mat)} = 3
        # Recall = 3/5 = 0.6, Precision = 3/5 = 0.6, F1 = 0.6
        assert scores['rouge2'] == pytest.approx(0.6, abs=EPSILON)

        # ROUGE-L:
        # LCS: [the, cat, on, the, mat] = 5 tokens
        # Recall = 5/6, Precision = 5/6, F1 = 5/6 ≈ 0.833
        assert scores['rougeL'] == pytest.approx(5.0/6.0, abs=EPSILON)

    def test_known_example_2_summarization(self):
        """Test summarization example with known scores."""
        rouge = ROUGEMetric()
        reference = "the quick brown fox jumps over the lazy dog"
        candidate = "the brown fox jumps over the dog"

        scores = rouge.compute(candidate, reference)

        # ROUGE-1:
        # Candidate unique: {the, brown, fox, jumps, over, dog} = 6
        # Reference unique: {the, quick, brown, fox, jumps, over, lazy, dog} = 8
        # Overlap: {the, brown, fox, jumps, over, dog} = 6
        # Recall = 6/8 = 0.75, Precision = 6/6 = 1.0
        # F1 = 2*0.75*1.0/(0.75+1.0) = 1.5/1.75 ≈ 0.857
        expected_rouge1 = (2 * 0.75 * 1.0) / (0.75 + 1.0)
        assert scores['rouge1'] == pytest.approx(expected_rouge1, abs=EPSILON)

        # All scores should be > 0
        assert scores['rouge2'] > 0.0
        assert scores['rougeL'] > 0.0

    def test_overall_score_aggregation(self):
        """Verify 'overall' key is mean of all variants."""
        rouge = ROUGEMetric()
        candidate = "the cat sat"
        reference = "the dog is"

        scores = rouge.compute(candidate, reference)

        # Overall should be the mean of rouge1, rouge2, rougeL
        expected_overall = (scores['rouge1'] + scores['rouge2'] + scores['rougeL']) / 3.0

        assert scores['overall'] == pytest.approx(expected_overall, abs=EPSILON)

    def test_known_example_3_no_bigram_overlap(self):
        """Test case with unigram overlap but no bigram overlap."""
        rouge = ROUGEMetric()
        candidate = "a b c d"
        reference = "d c b a"

        scores = rouge.compute(candidate, reference)

        # ROUGE-1: All unigrams match (4/4)
        assert scores['rouge1'] == pytest.approx(1.0, abs=EPSILON)

        # ROUGE-2: No bigram matches
        # Candidate: {(a,b), (b,c), (c,d)}
        # Reference: {(d,c), (c,b), (b,a)}
        # Overlap: {} = 0
        assert scores['rouge2'] == pytest.approx(0.0, abs=EPSILON)

        # ROUGE-L: LCS could be length 1 (any single token)
        # Recall = 1/4, Precision = 1/4, F1 = 1/4 = 0.25
        assert scores['rougeL'] == pytest.approx(0.25, abs=EPSILON)

    def test_known_example_4_long_lcs(self):
        """Test with long common subsequence."""
        rouge = ROUGEMetric(variants=['rougeL'])
        candidate = "a b x y c d"      # 6 tokens
        reference = "a b c d e f"      # 6 tokens

        scores = rouge.compute(candidate, reference)

        # LCS: [a, b, c, d] = 4 tokens
        # Recall = 4/6 = 0.667, Precision = 4/6 = 0.667, F1 = 0.667
        expected_f1 = 4.0 / 6.0

        assert scores['rougeL'] == pytest.approx(expected_f1, abs=EPSILON)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
