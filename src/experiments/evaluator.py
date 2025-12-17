"""
Accuracy Evaluator

Building Block for evaluating prompt effectiveness.

Input Data:
    - predictions: List[str] - Model predictions/responses
    - ground_truth: List[str] - Expected correct answers
    - dataset_items: List[Dict] - Original dataset items with metadata

Output Data:
    - accuracy: float - Overall accuracy score (0-1)
    - correct_count: int - Number of correct predictions
    - total_count: int - Total number of predictions
    - per_category_accuracy: Dict[str, float] - Accuracy broken down by category
    - errors: List[Dict] - Details of incorrect predictions

Setup Data:
    - case_sensitive: bool - Whether comparison is case-sensitive (default: False)
    - normalize_whitespace: bool - Whether to normalize whitespace (default: True)
    - fuzzy_match: bool - Allow fuzzy matching for close answers (default: False)
"""

from typing import List, Dict, Any, Optional
import re


class AccuracyEvaluator:
    """
    Evaluates accuracy of LLM responses against ground truth.

    Provides flexible matching with options for case sensitivity,
    whitespace normalization, and fuzzy matching.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
        fuzzy_match: bool = False,
        fuzzy_threshold: float = 0.8
    ):
        """
        Initialize accuracy evaluator.

        Args:
            case_sensitive: Whether to match case exactly
            normalize_whitespace: Whether to normalize whitespace
            fuzzy_match: Whether to allow fuzzy matching
            fuzzy_threshold: Threshold for fuzzy matching (0-1)
        """
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self.fuzzy_match = fuzzy_match
        self.fuzzy_threshold = fuzzy_threshold
        self._validate_config()

    def evaluate(
        self,
        predictions: List[str],
        ground_truth: List[str],
        dataset_items: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Model predictions
            ground_truth: Expected answers
            dataset_items: Optional original dataset items for detailed analysis

        Returns:
            Dictionary with accuracy metrics

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        # Validate inputs
        self._validate_inputs(predictions, ground_truth, dataset_items)

        # Compare predictions with ground truth
        correct = 0
        errors = []
        category_results = {}  # Track per-category accuracy

        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            is_correct = self._compare(pred, truth)

            if is_correct:
                correct += 1
            else:
                error_detail = {
                    "index": i,
                    "prediction": pred,
                    "expected": truth,
                }
                if dataset_items and i < len(dataset_items):
                    error_detail["input"] = dataset_items[i].get("input", "")
                    error_detail["category"] = dataset_items[i].get("category", "unknown")
                errors.append(error_detail)

            # Track per-category results
            if dataset_items and i < len(dataset_items):
                category = dataset_items[i].get("category", "unknown")
                if category not in category_results:
                    category_results[category] = {"correct": 0, "total": 0}
                category_results[category]["total"] += 1
                if is_correct:
                    category_results[category]["correct"] += 1

        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-category accuracy
        per_category_accuracy = {}
        for category, results in category_results.items():
            per_category_accuracy[category] = (
                results["correct"] / results["total"]
                if results["total"] > 0 else 0.0
            )

        return {
            "accuracy": accuracy,
            "correct_count": correct,
            "total_count": total,
            "error_count": len(errors),
            "per_category_accuracy": per_category_accuracy,
            "errors": errors[:10],  # Limit errors in output to first 10
            "total_errors": len(errors),
        }

    def _compare(self, prediction: str, ground_truth: str) -> bool:
        """
        Compare prediction with ground truth.

        Args:
            prediction: Model's prediction
            ground_truth: Correct answer

        Returns:
            True if match, False otherwise
        """
        # Normalize whitespace if enabled
        if self.normalize_whitespace:
            prediction = " ".join(prediction.split())
            ground_truth = " ".join(ground_truth.split())

        # Handle case sensitivity
        if not self.case_sensitive:
            prediction = prediction.lower()
            ground_truth = ground_truth.lower()

        # Exact match
        if prediction == ground_truth:
            return True

        # Try fuzzy match if enabled
        if self.fuzzy_match:
            return self._fuzzy_compare(prediction, ground_truth)

        return False

    def _fuzzy_compare(self, pred: str, truth: str) -> bool:
        """
        Fuzzy comparison using Levenshtein-like approach.

        Args:
            pred: Prediction string
            truth: Ground truth string

        Returns:
            True if strings are sufficiently similar
        """
        if pred.lower().startswith("error:") and not truth.lower().startswith("error:"):
            return False

        # Simple fuzzy matching: check if prediction contains ground truth
        # or ground truth contains prediction (only for sufficiently long strings and non-empty pred)
        if pred and len(truth) > 3 and (truth in pred or pred in truth):
            return True

        # Calculate simple character overlap ratio
        pred_chars = set(pred)
        truth_chars = set(truth)

        if not truth_chars:
            return False

        overlap = len(pred_chars & truth_chars)
        ratio = overlap / len(truth_chars)

        return ratio >= self.fuzzy_threshold

    def _validate_inputs(
        self,
        predictions: List[str],
        ground_truth: List[str],
        dataset_items: Optional[List[Dict]]
    ):
        """
        Validate input parameters.

        Args:
            predictions: Predictions list
            ground_truth: Ground truth list
            dataset_items: Optional dataset items

        Raises:
            TypeError: If inputs have wrong type
            ValueError: If inputs have invalid values
        """
        if not isinstance(predictions, list):
            raise TypeError(f"predictions must be a list, got {type(predictions)}")

        if not isinstance(ground_truth, list):
            raise TypeError(f"ground_truth must be a list, got {type(ground_truth)}")

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions "
                f"vs {len(ground_truth)} ground truth"
            )

        if not predictions:
            raise ValueError("Cannot evaluate empty predictions list")

        if dataset_items is not None:
            if not isinstance(dataset_items, list):
                raise TypeError(f"dataset_items must be a list, got {type(dataset_items)}")

            if len(dataset_items) != len(predictions):
                raise ValueError(
                    f"Length mismatch: {len(dataset_items)} dataset items "
                    f"vs {len(predictions)} predictions"
                )

    def _validate_config(self):
        """Validate configuration parameters."""
        if not isinstance(self.case_sensitive, bool):
            raise TypeError("case_sensitive must be a boolean")

        if not isinstance(self.normalize_whitespace, bool):
            raise TypeError("normalize_whitespace must be a boolean")

        if not isinstance(self.fuzzy_match, bool):
            raise TypeError("fuzzy_match must be a boolean")

        if not isinstance(self.fuzzy_threshold, (int, float)):
            raise TypeError("fuzzy_threshold must be a number")

        if not 0.0 <= self.fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be between 0.0 and 1.0")
