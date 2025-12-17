"""
Unit tests for AccuracyEvaluator

Tests cover:
- Basic accuracy calculation
- Case sensitivity options
- Fuzzy matching
- Per-category accuracy
- Edge cases and validation
"""

import pytest
from src.experiments.evaluator import AccuracyEvaluator


class TestAccuracyEvaluator:
    """Test suite for AccuracyEvaluator"""

    def test_perfect_accuracy(self):
        """Test 100% accuracy case"""
        evaluator = AccuracyEvaluator()
        predictions = ["positive", "negative", "neutral"]
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 3
        assert result["total_count"] == 3
        assert result["error_count"] == 0

    def test_zero_accuracy(self):
        """Test 0% accuracy case"""
        evaluator = AccuracyEvaluator()
        predictions = ["wrong1", "wrong2", "wrong3"]
        ground_truth = ["right1", "right2", "right3"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 3
        assert result["error_count"] == 3

    def test_partial_accuracy(self):
        """Test partial accuracy"""
        evaluator = AccuracyEvaluator()
        predictions = ["correct", "wrong", "correct"]
        ground_truth = ["correct", "correct", "correct"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == pytest.approx(2/3)
        assert result["correct_count"] == 2
        assert result["total_count"] == 3

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching (default)"""
        evaluator = AccuracyEvaluator(case_sensitive=False)
        predictions = ["POSITIVE", "Negative", "nEuTrAl"]
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0

    def test_case_sensitive_matching(self):
        """Test case-sensitive matching"""
        evaluator = AccuracyEvaluator(case_sensitive=True)
        predictions = ["POSITIVE", "Negative", "neutral"]
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        # Only "neutral" matches exactly
        assert result["accuracy"] == pytest.approx(1/3)
        assert result["correct_count"] == 1

    def test_whitespace_normalization(self):
        """Test whitespace normalization (default)"""
        evaluator = AccuracyEvaluator(normalize_whitespace=True)
        predictions = ["  positive  ", "negative\n", "\tneutral"]
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0

    def test_without_whitespace_normalization(self):
        """Test without whitespace normalization"""
        evaluator = AccuracyEvaluator(normalize_whitespace=False)
        predictions = ["  positive  ", "negative", "neutral"]
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        # Only exact matches
        assert result["correct_count"] == 2  # "negative" and "neutral"

    def test_fuzzy_matching_enabled(self):
        """Test fuzzy matching for close answers"""
        evaluator = AccuracyEvaluator(fuzzy_match=True, fuzzy_threshold=0.8)
        predictions = ["positiv", "negative", "neutral"]  # Typo in first
        ground_truth = ["positive", "negative", "neutral"]

        result = evaluator.evaluate(predictions, ground_truth)

        # Fuzzy match should catch "positiv" ~ "positive"
        assert result["accuracy"] >= 0.9  # Should be high

    def test_fuzzy_matching_substring(self):
        """Test fuzzy matching with substring"""
        evaluator = AccuracyEvaluator(fuzzy_match=True)
        predictions = ["The answer is 42", "yes", "no"]
        ground_truth = ["42", "yes", "no"]

        result = evaluator.evaluate(predictions, ground_truth)

        # "42" is contained in first prediction
        assert result["correct_count"] >= 1

    def test_per_category_accuracy(self):
        """Test per-category accuracy calculation"""
        evaluator = AccuracyEvaluator()
        predictions = ["pos", "neg", "neu", "pos", "neg"]
        ground_truth = ["pos", "neg", "neu", "neg", "neg"]
        dataset_items = [
            {"category": "sentiment", "input": "test1"},
            {"category": "sentiment", "input": "test2"},
            {"category": "sentiment", "input": "test3"},
            {"category": "logic", "input": "test4"},
            {"category": "logic", "input": "test5"},
        ]

        result = evaluator.evaluate(predictions, ground_truth, dataset_items)

        assert "per_category_accuracy" in result
        assert "sentiment" in result["per_category_accuracy"]
        assert "logic" in result["per_category_accuracy"]

        # Sentiment: 3/3 correct
        assert result["per_category_accuracy"]["sentiment"] == 1.0
        # Logic: 1/2 correct (second one is correct)
        assert result["per_category_accuracy"]["logic"] == 0.5

    def test_errors_list_populated(self):
        """Test that errors list contains error details"""
        evaluator = AccuracyEvaluator()
        predictions = ["wrong", "correct", "wrong"]
        ground_truth = ["correct", "correct", "correct"]
        dataset_items = [
            {"input": "q1", "category": "test"},
            {"input": "q2", "category": "test"},
            {"input": "q3", "category": "test"},
        ]

        result = evaluator.evaluate(predictions, ground_truth, dataset_items)

        assert len(result["errors"]) == 2
        assert result["errors"][0]["prediction"] == "wrong"
        assert result["errors"][0]["expected"] == "correct"
        assert "input" in result["errors"][0]

    def test_errors_limited_to_ten(self):
        """Test that errors list is limited to first 10"""
        evaluator = AccuracyEvaluator()
        predictions = ["wrong"] * 20
        ground_truth = ["correct"] * 20

        result = evaluator.evaluate(predictions, ground_truth)

        assert len(result["errors"]) == 10  # Limited
        assert result["total_errors"] == 20  # But total is correct

    def test_empty_lists_raise_error(self):
        """Test that empty lists raise ValueError"""
        evaluator = AccuracyEvaluator()

        with pytest.raises(ValueError, match="Cannot evaluate empty predictions"):
            evaluator.evaluate([], [])

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched lengths raise ValueError"""
        evaluator = AccuracyEvaluator()

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(
                ["a", "b", "c"],
                ["a", "b"]  # Different length
            )

    def test_non_list_predictions_raise_error(self):
        """Test that non-list predictions raise TypeError"""
        evaluator = AccuracyEvaluator()

        with pytest.raises(TypeError, match="predictions must be a list"):
            evaluator.evaluate("not a list", ["a", "b"])

    def test_non_list_ground_truth_raise_error(self):
        """Test that non-list ground_truth raises TypeError"""
        evaluator = AccuracyEvaluator()

        with pytest.raises(TypeError, match="ground_truth must be a list"):
            evaluator.evaluate(["a", "b"], "not a list")

    def test_dataset_items_length_mismatch(self):
        """Test that dataset_items length mismatch raises error"""
        evaluator = AccuracyEvaluator()
        predictions = ["a", "b", "c"]
        ground_truth = ["a", "b", "c"]
        dataset_items = [{"input": "q1"}]  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch.*dataset items"):
            evaluator.evaluate(predictions, ground_truth, dataset_items)

    def test_invalid_case_sensitive_type(self):
        """Test invalid case_sensitive configuration"""
        with pytest.raises(TypeError, match="case_sensitive must be a boolean"):
            AccuracyEvaluator(case_sensitive="yes")

    def test_invalid_normalize_whitespace_type(self):
        """Test invalid normalize_whitespace configuration"""
        with pytest.raises(TypeError, match="normalize_whitespace must be a boolean"):
            AccuracyEvaluator(normalize_whitespace=1)

    def test_invalid_fuzzy_match_type(self):
        """Test invalid fuzzy_match configuration"""
        with pytest.raises(TypeError, match="fuzzy_match must be a boolean"):
            AccuracyEvaluator(fuzzy_match="true")

    def test_invalid_fuzzy_threshold_type(self):
        """Test invalid fuzzy_threshold type"""
        with pytest.raises(TypeError, match="fuzzy_threshold must be a number"):
            AccuracyEvaluator(fuzzy_threshold="0.8")

    def test_fuzzy_threshold_out_of_range_low(self):
        """Test fuzzy_threshold below 0"""
        with pytest.raises(ValueError, match="fuzzy_threshold must be between"):
            AccuracyEvaluator(fuzzy_threshold=-0.1)

    def test_fuzzy_threshold_out_of_range_high(self):
        """Test fuzzy_threshold above 1"""
        with pytest.raises(ValueError, match="fuzzy_threshold must be between"):
            AccuracyEvaluator(fuzzy_threshold=1.5)

    def test_single_prediction(self):
        """Test with single prediction"""
        evaluator = AccuracyEvaluator()
        result = evaluator.evaluate(["correct"], ["correct"])

        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1

    def test_large_dataset(self):
        """Test with large dataset"""
        evaluator = AccuracyEvaluator()
        predictions = ["answer"] * 1000
        ground_truth = ["answer"] * 1000

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0
        assert result["total_count"] == 1000

    def test_unicode_predictions(self):
        """Test with unicode characters"""
        evaluator = AccuracyEvaluator()
        predictions = ["你好", "世界", "café"]
        ground_truth = ["你好", "世界", "café"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0

    def test_special_characters(self):
        """Test with special characters"""
        evaluator = AccuracyEvaluator()
        predictions = ["a\nb", "c\td", "e f"]
        ground_truth = ["a\nb", "c\td", "e f"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0

    def test_numeric_strings(self):
        """Test with numeric string answers"""
        evaluator = AccuracyEvaluator()
        predictions = ["42", "3.14", "100"]
        ground_truth = ["42", "3.14", "100"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result["accuracy"] == 1.0

    def test_mixed_correctness(self):
        """Test mixed correct and incorrect answers"""
        evaluator = AccuracyEvaluator()
        predictions = ["yes", "no", "maybe", "yes", "no"]
        ground_truth = ["yes", "yes", "maybe", "no", "no"]

        result = evaluator.evaluate(predictions, ground_truth)

        # Correct: positions 0, 2, 4 = 3/5
        assert result["accuracy"] == 0.6
        assert result["correct_count"] == 3
        assert result["error_count"] == 2

    def test_empty_strings(self):
        """Test with empty string predictions"""
        evaluator = AccuracyEvaluator()
        predictions = ["", "correct", ""]
        ground_truth = ["correct", "correct", "correct"]

        result = evaluator.evaluate(predictions, ground_truth)

        # Only middle one correct
        assert result["correct_count"] == 1
        assert result["error_count"] == 2
