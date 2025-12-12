"""
Pass@k Metric Implementation

Implements the unbiased Pass@k estimator for code generation evaluation.
Uses combinatorial formula to estimate probability of generating at least one
correct solution in k attempts.

Formula: Pass@k = 1 - ((n - c) choose k) / (n choose k)
Where:
    n = total number of generated solutions
    c = number of correct solutions
    k = number of samples

Reference: "Evaluating Large Language Models Trained on Code" (Chen et al., 2021)

Author: System Architect Agent
Date: 2025-12-13
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger

from src.evaluator.executor import CodeExecutor


@dataclass
class PassAtKResult:
    """
    Result container for Pass@k metric computation.

    Attributes:
        pass_at_k: Estimated Pass@k probability (0.0 to 1.0)
        n_total: Total number of solutions evaluated
        n_correct: Number of correct solutions
        k: Value of k used in computation
        individual_results: List of boolean results for each solution
        execution_times: List of execution times for each solution
        confidence_interval: Optional tuple (lower, upper) confidence bounds
    """
    pass_at_k: float
    n_total: int
    n_correct: int
    k: int
    individual_results: List[bool]
    execution_times: List[float]
    confidence_interval: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'pass_at_k': self.pass_at_k,
            'n_total': self.n_total,
            'n_correct': self.n_correct,
            'k': self.k,
            'individual_results': self.individual_results,
            'execution_times': self.execution_times,
            'confidence_interval': self.confidence_interval
        }


class PassAtKMetric:
    """
    Pass@k metric for evaluating code generation models.

    Implements the unbiased estimator using combinatorial formula:
        Pass@k = 1 - ((n - c) choose k) / (n choose k)

    This estimates the probability that at least one correct solution
    is generated when sampling k solutions from n total solutions.

    Attributes:
        executor: CodeExecutor instance for running generated code
        k_values: List of k values to compute (default: [1, 5, 10])
    """

    def __init__(
        self,
        executor: CodeExecutor,
        k_values: List[int] = None
    ):
        """
        Initialize Pass@k metric with code executor.

        Args:
            executor: CodeExecutor instance for evaluating code
            k_values: List of k values to compute (default: [1, 5, 10])
        """
        self.executor = executor
        self.k_values = k_values or [1, 5, 10]

        logger.info(
            f"PassAtKMetric initialized with k_values={self.k_values}"
        )

    def compute(
        self,
        generated_solutions: List[str],
        test_cases: List[Dict[str, Any]],
        k: int = 1
    ) -> PassAtKResult:
        """
        Compute Pass@k for generated code solutions.

        Args:
            generated_solutions: List of generated code strings
            test_cases: List of test case dictionaries with 'input' and 'expected'
            k: Number of samples (must be <= len(generated_solutions))

        Returns:
            PassAtKResult containing metric value and detailed results

        Raises:
            ValueError: If k > n or k < 1
        """
        n = len(generated_solutions)

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > n:
            raise ValueError(
                f"k ({k}) cannot be greater than number of solutions ({n})"
            )

        logger.info(f"Computing Pass@{k} for {n} solutions with {len(test_cases)} test cases")

        # Evaluate all solutions
        individual_results = []
        execution_times = []

        for idx, solution in enumerate(generated_solutions):
            logger.debug(f"Evaluating solution {idx + 1}/{n}")

            try:
                result = self.executor.execute(
                    code=solution,
                    test_cases=test_cases,
                    language="python"
                )

                is_correct = result['success']
                individual_results.append(is_correct)
                execution_times.append(result['execution_time'])

                logger.debug(
                    f"Solution {idx + 1}: {'PASS' if is_correct else 'FAIL'} "
                    f"({result['passed_tests']}/{result['total_tests']} tests)"
                )

            except Exception as e:
                logger.error(f"Error evaluating solution {idx + 1}: {e}")
                individual_results.append(False)
                execution_times.append(0.0)

        # Count correct solutions
        c = sum(individual_results)

        logger.info(f"Evaluation complete: {c}/{n} solutions correct")

        # Compute Pass@k using unbiased estimator
        pass_at_k_value = self._compute_pass_at_k(n, c, k)

        result = PassAtKResult(
            pass_at_k=pass_at_k_value,
            n_total=n,
            n_correct=c,
            k=k,
            individual_results=individual_results,
            execution_times=execution_times
        )

        logger.info(f"Pass@{k} = {pass_at_k_value:.4f}")

        return result

    def compute_multiple_k(
        self,
        generated_solutions: List[str],
        test_cases: List[Dict[str, Any]],
        k_values: Optional[List[int]] = None
    ) -> Dict[int, PassAtKResult]:
        """
        Compute Pass@k for multiple k values efficiently.

        Evaluates all solutions once and computes Pass@k for each k value.

        Args:
            generated_solutions: List of generated code strings
            test_cases: List of test case dictionaries
            k_values: List of k values to compute (uses self.k_values if None)

        Returns:
            Dictionary mapping k -> PassAtKResult

        Raises:
            ValueError: If any k value is invalid
        """
        k_vals = k_values or self.k_values
        n = len(generated_solutions)

        # Validate k values
        for k in k_vals:
            if k < 1 or k > n:
                raise ValueError(
                    f"Invalid k={k}: must be between 1 and {n}"
                )

        logger.info(
            f"Computing Pass@k for k={k_vals} with {n} solutions"
        )

        # Evaluate all solutions once
        individual_results = []
        execution_times = []

        for idx, solution in enumerate(generated_solutions):
            logger.debug(f"Evaluating solution {idx + 1}/{n}")

            try:
                result = self.executor.execute(
                    code=solution,
                    test_cases=test_cases,
                    language="python"
                )

                is_correct = result['success']
                individual_results.append(is_correct)
                execution_times.append(result['execution_time'])

            except Exception as e:
                logger.error(f"Error evaluating solution {idx + 1}: {e}")
                individual_results.append(False)
                execution_times.append(0.0)

        c = sum(individual_results)

        logger.info(f"Evaluation complete: {c}/{n} solutions correct")

        # Compute Pass@k for each k value
        results = {}

        for k in k_vals:
            pass_at_k_value = self._compute_pass_at_k(n, c, k)

            results[k] = PassAtKResult(
                pass_at_k=pass_at_k_value,
                n_total=n,
                n_correct=c,
                k=k,
                individual_results=individual_results.copy(),
                execution_times=execution_times.copy()
            )

            logger.info(f"Pass@{k} = {pass_at_k_value:.4f}")

        return results

    def _compute_pass_at_k(self, n: int, c: int, k: int) -> float:
        """
        Compute Pass@k using unbiased combinatorial estimator.

        Formula: Pass@k = 1 - ((n - c) choose k) / (n choose k)

        This is equivalent to:
            Pass@k = 1 - (probability of choosing k samples with none correct)

        Args:
            n: Total number of solutions
            c: Number of correct solutions
            k: Number of samples

        Returns:
            Pass@k probability (0.0 to 1.0)

        Edge cases:
            - If c >= k: Pass@k = 1.0 (guaranteed at least one correct)
            - If c == 0: Pass@k = 0.0 (no correct solutions)
            - If n == c: Pass@k = 1.0 (all solutions correct)
        """
        # Edge case: no correct solutions
        if c == 0:
            return 0.0

        # Edge case: all solutions correct
        if n == c:
            return 1.0

        # Edge case: k == n (sampling all solutions)
        if k == n:
            return 1.0  # Guaranteed to get at least one correct if c > 0

        # Edge case: not enough incorrect solutions to fill k samples
        # If n - c < k, we must sample at least one correct solution
        if n - c < k:
            return 1.0

        # General case: compute using combinatorial formula
        # Pass@k = 1 - C(n-c, k) / C(n, k)
        #        = 1 - [product_{i=0}^{k-1} (n-c-i) / (n-i)]

        try:
            # Compute using log space for numerical stability
            log_prob_fail = 0.0

            for i in range(k):
                log_prob_fail += math.log(n - c - i) - math.log(n - i)

            prob_fail = math.exp(log_prob_fail)
            pass_at_k = 1.0 - prob_fail

            # Clamp to [0, 1] due to floating point errors
            return max(0.0, min(1.0, pass_at_k))

        except (ValueError, ZeroDivisionError, OverflowError) as e:
            logger.error(
                f"Error computing Pass@{k} with n={n}, c={c}: {e}"
            )
            # Fallback: simple estimate
            return c / n

    @staticmethod
    def compute_confidence_interval(
        n: int,
        c: int,
        k: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for Pass@k using Wilson score interval.

        Args:
            n: Total number of solutions
            c: Number of correct solutions
            k: Number of samples
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        import scipy.stats as stats

        # Simple binomial proportion confidence interval
        p = c / n  # Point estimate
        z = stats.norm.ppf((1 + confidence) / 2)

        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        margin = z * math.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

        return (lower, upper)

    def evaluate_dataset(
        self,
        problems: List[Dict[str, Any]],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset of problems with multiple solutions.

        Args:
            problems: List of problem dictionaries with:
                - 'solutions': List[str] - Generated code solutions
                - 'test_cases': List[Dict] - Test cases for the problem
                - 'problem_id': str - Unique problem identifier
            k_values: List of k values to compute

        Returns:
            Dictionary with aggregate statistics across all problems
        """
        k_vals = k_values or self.k_values

        logger.info(f"Evaluating {len(problems)} problems for k={k_vals}")

        all_results = []

        for idx, problem in enumerate(problems):
            problem_id = problem.get('problem_id', f'problem_{idx}')
            solutions = problem['solutions']
            test_cases = problem['test_cases']

            logger.info(f"Evaluating problem {problem_id} ({idx + 1}/{len(problems)})")

            try:
                results = self.compute_multiple_k(
                    generated_solutions=solutions,
                    test_cases=test_cases,
                    k_values=k_vals
                )

                all_results.append({
                    'problem_id': problem_id,
                    'results': results
                })

            except Exception as e:
                logger.error(f"Error evaluating problem {problem_id}: {e}")

        # Aggregate statistics
        aggregate_stats = self._aggregate_results(all_results, k_vals)

        return aggregate_stats

    def _aggregate_results(
        self,
        all_results: List[Dict[str, Any]],
        k_values: List[int]
    ) -> Dict[str, Any]:
        """
        Aggregate Pass@k results across multiple problems.

        Args:
            all_results: List of problem results
            k_values: List of k values

        Returns:
            Dictionary with mean, median, std for each k value
        """
        import numpy as np

        aggregate = {
            'n_problems': len(all_results),
            'k_values': k_values,
            'per_k_stats': {}
        }

        for k in k_values:
            pass_at_k_values = [
                problem['results'][k].pass_at_k
                for problem in all_results
                if k in problem['results']
            ]

            if pass_at_k_values:
                aggregate['per_k_stats'][k] = {
                    'mean': float(np.mean(pass_at_k_values)),
                    'median': float(np.median(pass_at_k_values)),
                    'std': float(np.std(pass_at_k_values)),
                    'min': float(np.min(pass_at_k_values)),
                    'max': float(np.max(pass_at_k_values))
                }

        return aggregate
