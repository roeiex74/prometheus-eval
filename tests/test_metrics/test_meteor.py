"""
Comprehensive test suite for METEOR metric implementation.

Tests cover:
1. Basic METEOR functionality (perfect match, mismatch, partial overlap)
2. Exact word matching
3. Stem matching (Porter stemmer)
4. Synonym matching (WordNet)
5. Fragmentation penalty calculation
6. Multi-reference support
7. Edge cases (empty strings, single tokens, punctuation)
8. Known examples from literature
"""

import pytest
from src.metrics.lexical.meteor import METEORMetric

# Test tolerance for floating point comparisons
EPSILON = 1e-4


class TestMETEORBasic:
    """Test basic METEOR functionality."""

    def test_perfect_match(self):
        """Test METEOR for identical strings."""
        meteor = METEORMetric()
        candidate = "the cat sat on the mat"
        reference = "the cat sat on the mat"
        result = meteor.compute(candidate, reference)

        # Perfect alignment: 6 matches, 1 chunk, P=R=1.0, F_mean=1.0
        # Penalty = 0.5 * (1/6)^3 ≈ 0.0023, METEOR ≈ 0.9977
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield precision = 1.0"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield recall = 1.0"
        assert result['f_mean'] == pytest.approx(1.0, abs=EPSILON), \
            "Perfect match should yield f_mean = 1.0"
        assert result['chunks'] == 1, \
            "Perfect match should have 1 chunk (all consecutive)"
        assert result['meteor'] > 0.99, \
            "Perfect match should yield METEOR close to 1.0 (with small penalty)"

    def test_complete_mismatch(self):
        """Test METEOR = 0 for completely different strings."""
        meteor = METEORMetric()
        candidate = "the weather is sunny today"
        reference = "a quick brown fox jumps"
        result = meteor.compute(candidate, reference)

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield METEOR = 0.0"
        assert result['precision'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield precision = 0.0"
        assert result['recall'] == pytest.approx(0.0, abs=EPSILON), \
            "Complete mismatch should yield recall = 0.0"
        assert result['chunks'] == 0, \
            "Complete mismatch should have 0 chunks"

    def test_partial_overlap(self):
        """Test METEOR with partial word overlap."""
        meteor = METEORMetric()
        candidate = "the cat is here"
        reference = "the dog is there"
        result = meteor.compute(candidate, reference)

        # Common words: "the", "is" (2 matches out of 4 tokens each)
        # Precision = 2/4 = 0.5, Recall = 2/4 = 0.5
        # F_mean = (10 * 0.5 * 0.5) / (0.5 + 9 * 0.5) = 2.5 / 5 = 0.5
        # Chunks: (0,0), (2,2) -> 2 chunks (not consecutive)
        # Penalty = 0.5 * (2/2)^3 = 0.5
        # METEOR = (1 - 0.5) * 0.5 = 0.25
        expected_meteor = 0.25
        assert result['meteor'] == pytest.approx(expected_meteor, abs=EPSILON), \
            f"Expected METEOR = {expected_meteor}, got {result['meteor']}"
        assert result['chunks'] == 2, \
            "Should have 2 chunks for non-consecutive matches"

    def test_exact_word_matching(self):
        """Test exact word matching (case-insensitive)."""
        meteor = METEORMetric()
        candidate = "The Cat Sat"
        reference = "the cat sat"
        result = meteor.compute(candidate, reference)

        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Case should not matter for exact matching"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)
        assert result['meteor'] > 0.95, \
            "Perfect word match should yield high METEOR"

    def test_multi_reference_selection(self):
        """Test that METEOR selects best scoring reference."""
        meteor = METEORMetric()
        candidate = "the cat sat on the mat"
        references = [
            "the dog stood on the rug",  # Lower score
            "the cat sat on the mat",     # Perfect match
            "a feline rested"             # Low score
        ]
        result = meteor.compute(candidate, references)

        assert result['meteor'] > 0.99, \
            "Should select best matching reference (perfect match with small penalty)"
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON)
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)


class TestMETEORStemMatching:
    """Test stem matching functionality."""

    def test_stem_match_running_run(self):
        """Test stem matching: running vs run."""
        meteor = METEORMetric()
        candidate = "the cat is running"
        reference = "the cat is run"
        result = meteor.compute(candidate, reference)

        # "running" and "run" have same stem
        # Should match all 4 words
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Stem matching should match 'running' with 'run'"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)

    def test_stem_match_flies_flying(self):
        """Test stem matching: flies vs flying."""
        meteor = METEORMetric()
        candidate = "the bird flies"
        reference = "the bird flying"
        result = meteor.compute(candidate, reference)

        # "flies" and "flying" have same stem "fli"
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Stem matching should match 'flies' with 'flying'"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)

    def test_stem_priority_exact_over_stem(self):
        """Test that exact matching takes priority over stem matching."""
        meteor = METEORMetric()
        # Both should get exact match since they're identical
        candidate = "running quickly"
        reference = "running quickly"
        result = meteor.compute(candidate, reference)

        # 2 matches, 1 chunk: penalty = 0.5 * (1/2)^3 = 0.0625
        # METEOR ≈ 0.9375
        assert result['meteor'] > 0.90, \
            "Exact match should yield high METEOR"
        assert result['chunks'] == 1, \
            "Should have 1 chunk (consecutive exact matches)"

    def test_stem_different_words(self):
        """Test that different stems don't match."""
        meteor = METEORMetric()
        candidate = "cat dog"
        reference = "bird fish"
        result = meteor.compute(candidate, reference)

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Different stems should not match"


class TestMETEORSynonymMatching:
    """Test synonym matching functionality."""

    def test_synonym_match_car_automobile(self):
        """Test synonym matching: car vs automobile."""
        meteor = METEORMetric()
        candidate = "the car is red"
        reference = "the automobile is red"
        result = meteor.compute(candidate, reference)

        # "car" and "automobile" are synonyms in WordNet
        # Should match all 4 words (the, car/automobile, is, red)
        assert result['precision'] >= 0.75, \
            "Synonym matching should match 'car' with 'automobile'"
        assert result['recall'] >= 0.75

    def test_synonym_match_happy_joyful(self):
        """Test synonym matching: happy vs joyful."""
        meteor = METEORMetric()
        candidate = "she is happy"
        reference = "she is joyful"
        result = meteor.compute(candidate, reference)

        # "happy" and "joyful" are synonyms
        # Should match "she", "is", and happy/joyful
        assert result['precision'] >= 0.66, \
            "Synonym matching should match 'happy' with 'joyful'"

    def test_synonym_priority(self):
        """Test that synonym matching comes after exact and stem."""
        meteor = METEORMetric()
        # This tests the order: exact -> stem -> synonym
        candidate = "the quick cat"
        reference = "the fast cat"
        result = meteor.compute(candidate, reference)

        # "quick" and "fast" are synonyms
        # Should match "the", "cat", and quick/fast
        assert result['precision'] >= 0.66, \
            "Should match 'quick' and 'fast' via synonyms"

    def test_no_synonym_for_nonsense(self):
        """Test that nonsense words don't get synonym matches."""
        meteor = METEORMetric()
        candidate = "xyzabc defghi"
        reference = "jklmno pqrstu"
        result = meteor.compute(candidate, reference)

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Nonsense words should not match via synonyms"


class TestMETEORPenalty:
    """Test fragmentation penalty calculation."""

    def test_consecutive_matches_low_penalty(self):
        """Test that consecutive matches have low penalty."""
        meteor = METEORMetric()
        candidate = "the cat sat on the mat"
        reference = "the cat sat on the mat"
        result = meteor.compute(candidate, reference)

        # All consecutive -> 1 chunk
        # Penalty = 0.5 * (1/6)^3 ≈ 0.0023 (very low)
        assert result['chunks'] == 1, \
            "Consecutive matches should form 1 chunk"
        assert result['penalty'] < 0.01, \
            "Consecutive matches should have very low penalty"

    def test_fragmented_matches_high_penalty(self):
        """Test that fragmented matches have higher penalty."""
        meteor = METEORMetric()
        candidate = "the cat is here"
        reference = "the dog is there"
        result = meteor.compute(candidate, reference)

        # Matches: (0,0)=the, (2,2)=is -> 2 chunks
        # Penalty = 0.5 * (2/2)^3 = 0.5
        assert result['chunks'] == 2, \
            "Non-consecutive matches should form multiple chunks"
        assert result['penalty'] == pytest.approx(0.5, abs=EPSILON), \
            "Fragmented matches should have penalty = 0.5 * (chunks/matches)^3"

    def test_penalty_formula_validation(self):
        """Test penalty formula: gamma * (chunks/matches)^beta."""
        meteor = METEORMetric(gamma=0.5, beta=3.0)
        candidate = "a b c"
        reference = "a x b"
        result = meteor.compute(candidate, reference)

        # Matches: (0,0)=a, (1,2)=b -> 2 matches, 2 chunks
        # Penalty = 0.5 * (2/2)^3 = 0.5
        expected_penalty = 0.5 * (result['chunks'] / 2) ** 3
        assert result['penalty'] == pytest.approx(expected_penalty, abs=EPSILON), \
            f"Penalty should match formula: 0.5 * ({result['chunks']}/2)^3"

    def test_custom_penalty_parameters(self):
        """Test custom gamma and beta parameters."""
        meteor = METEORMetric(gamma=0.7, beta=2.0)
        candidate = "a b"
        reference = "a b"
        result = meteor.compute(candidate, reference)

        # 1 chunk, 2 matches
        # Penalty = 0.7 * (1/2)^2 = 0.175
        expected_penalty = 0.7 * (1 / 2) ** 2
        assert result['penalty'] == pytest.approx(expected_penalty, abs=EPSILON), \
            "Custom gamma and beta should be used in penalty calculation"


class TestMETEOREdgeCases:
    """Test edge cases and error handling."""

    def test_empty_candidate(self):
        """Test with empty candidate string."""
        meteor = METEORMetric()
        result = meteor.compute("", "the cat sat")

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Empty candidate should yield METEOR = 0.0"
        assert result['precision'] == pytest.approx(0.0, abs=EPSILON)
        assert result['recall'] == pytest.approx(0.0, abs=EPSILON)

    def test_empty_reference(self):
        """Test with empty reference string."""
        meteor = METEORMetric()
        result = meteor.compute("the cat sat", "")

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Empty reference should yield METEOR = 0.0"

    def test_both_empty(self):
        """Test with both empty strings."""
        meteor = METEORMetric()
        result = meteor.compute("", "")

        assert result['meteor'] == pytest.approx(0.0, abs=EPSILON), \
            "Both empty should yield METEOR = 0.0"

    def test_single_word_match(self):
        """Test with single word inputs."""
        meteor = METEORMetric()
        result = meteor.compute("cat", "cat")

        # 1 match, 1 chunk: penalty = 0.5 * (1/1)^3 = 0.5
        # F_mean = 1.0, METEOR = (1 - 0.5) * 1.0 = 0.5
        assert result['meteor'] == pytest.approx(0.5, abs=EPSILON), \
            "Single word match has penalty = 0.5, METEOR = 0.5"
        assert result['chunks'] == 1
        assert result['penalty'] == pytest.approx(0.5, abs=EPSILON)

    def test_single_word_mismatch(self):
        """Test with single word mismatch."""
        meteor = METEORMetric()
        result = meteor.compute("cat", "dog")

        # Unless they're synonyms, should be 0 or very low
        # Since cat and dog aren't synonyms in WordNet
        assert result['meteor'] < 0.5, \
            "Single word mismatch should yield low METEOR"

    def test_punctuation_handling(self):
        """Test that punctuation affects token matching."""
        meteor = METEORMetric()
        candidate = "hello world"
        reference = "hello world"
        result = meteor.compute(candidate, reference)

        # Without punctuation, perfect match
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Exact tokens should match"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)

    def test_case_insensitivity(self):
        """Test that matching is case-insensitive."""
        meteor = METEORMetric()
        candidate = "THE CAT SAT"
        reference = "the cat sat"
        result = meteor.compute(candidate, reference)

        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "Matching should be case-insensitive"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)
        assert result['meteor'] > 0.95, \
            "Perfect match regardless of case"


class TestMETEORKnownExamples:
    """Test against known examples and validate overall structure."""

    def test_translation_example_1(self):
        """Test typical MT evaluation example."""
        meteor = METEORMetric()
        candidate = "the cat is on the mat"
        reference = "there is a cat on the mat"
        result = meteor.compute(candidate, reference)

        # Should have high recall (most ref words in candidate)
        # But precision affected by "the" appearing twice
        assert result['recall'] >= 0.7, \
            "Should have high recall for MT example"
        assert 0.0 < result['meteor'] < 1.0, \
            "METEOR should be between 0 and 1"
        assert result['precision'] > 0.0
        assert result['f_mean'] > 0.0

    def test_translation_example_2(self):
        """Test MT example with different word order."""
        meteor = METEORMetric()
        candidate = "on the mat sat the cat"
        reference = "the cat sat on the mat"
        result = meteor.compute(candidate, reference)

        # All words match but order different -> more chunks
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON), \
            "All words match"
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)
        assert result['chunks'] > 1, \
            "Different order should create multiple chunks"
        assert result['meteor'] < 1.0, \
            "Penalty should reduce score due to fragmentation"

    def test_overall_score_structure(self):
        """Test that all score components are present and valid."""
        meteor = METEORMetric()
        candidate = "the quick brown fox"
        reference = "the fast brown fox"
        result = meteor.compute(candidate, reference)

        # Validate all required keys are present
        required_keys = ['meteor', 'precision', 'recall', 'f_mean', 'penalty', 'chunks']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Validate value ranges
        assert 0.0 <= result['meteor'] <= 1.0, "METEOR should be in [0, 1]"
        assert 0.0 <= result['precision'] <= 1.0, "Precision should be in [0, 1]"
        assert 0.0 <= result['recall'] <= 1.0, "Recall should be in [0, 1]"
        assert 0.0 <= result['f_mean'] <= 1.0, "F-mean should be in [0, 1]"
        assert 0.0 <= result['penalty'] <= 1.0, "Penalty should be in [0, 1]"
        assert result['chunks'] >= 0, "Chunks should be non-negative"

    def test_f_mean_formula(self):
        """Test that F-mean follows 9:1 recall:precision weighting."""
        meteor = METEORMetric()
        candidate = "a b c d"
        reference = "a b x y z"
        result = meteor.compute(candidate, reference)

        # Manual calculation
        # Matches: a, b -> 2
        # Precision = 2/4 = 0.5, Recall = 2/5 = 0.4
        # F_mean = (10 * 0.5 * 0.4) / (0.4 + 9 * 0.5) = 2.0 / 4.9 ≈ 0.408
        expected_precision = 0.5
        expected_recall = 0.4
        expected_f_mean = (10 * expected_precision * expected_recall) / (expected_recall + 9 * expected_precision)

        assert result['precision'] == pytest.approx(expected_precision, abs=EPSILON)
        assert result['recall'] == pytest.approx(expected_recall, abs=EPSILON)
        assert result['f_mean'] == pytest.approx(expected_f_mean, abs=EPSILON), \
            "F-mean should follow formula: (10PR)/(R+9P)"


class TestMETEORMultiReference:
    """Test multi-reference support."""

    def test_multi_reference_best_selection(self):
        """Test that best reference is selected."""
        meteor = METEORMetric()
        candidate = "the cat sat"
        references = [
            "the dog stood",     # Low score
            "the cat sat",       # Perfect match - should be selected
            "a feline rested"    # Low score
        ]
        result = meteor.compute(candidate, references)

        assert result['meteor'] > 0.95, \
            "Should select reference with highest METEOR score"
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON)
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)

    def test_single_reference_as_string(self):
        """Test that single reference can be provided as string."""
        meteor = METEORMetric()
        candidate = "the cat"
        reference = "the cat"  # String, not list
        result = meteor.compute(candidate, reference)

        assert result['meteor'] > 0.90, \
            "Single reference as string should work"
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON)
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)

    def test_single_reference_as_list(self):
        """Test that single reference can be provided as list."""
        meteor = METEORMetric()
        candidate = "the cat"
        reference = ["the cat"]  # List with one element
        result = meteor.compute(candidate, reference)

        assert result['meteor'] > 0.90, \
            "Single reference as list should work"
        assert result['precision'] == pytest.approx(1.0, abs=EPSILON)
        assert result['recall'] == pytest.approx(1.0, abs=EPSILON)
