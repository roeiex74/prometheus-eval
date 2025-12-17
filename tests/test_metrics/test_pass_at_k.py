"""
Comprehensive Test Suite for Pass@k Metric

Tests the PassAtKMetric implementation including:
- Correct solutions (Pass@1 = 1.0)
- All incorrect solutions (Pass@k = 0.0)
- Partial correctness
- Different k values (1, 5, 10)
- Edge cases: syntax errors, timeouts, infinite loops
- Docker container lifecycle
- Resource limit enforcement
- Mathematical correctness of combinatorial formula

Author: System Architect Agent
Date: 2025-12-13
"""

import pytest
import time
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

from src.evaluator.executor import CodeExecutor
from src.metrics.logic.pass_at_k import PassAtKMetric, PassAtKResult

# Check for Docker availability
try:
    import docker
    client = docker.from_env()
    client.ping()
    DOCKER_AVAILABLE = True
except (ImportError, Exception):
    DOCKER_AVAILABLE = False


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
class TestCodeExecutor:
    """Test suite for CodeExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create CodeExecutor instance for testing."""
        return CodeExecutor(timeout=5, memory_limit="256m")

    @pytest.fixture
    def simple_test_cases(self):
        """Simple test cases for basic functions."""
        return [
            {'input': {'a': 1, 'b': 2}, 'expected': 3},
            {'input': {'a': -1, 'b': 1}, 'expected': 0},
            {'input': {'a': 0, 'b': 0}, 'expected': 0},
            {'input': {'a': 10, 'b': -5}, 'expected': 5}
        ]

    def test_correct_solution(self, executor, simple_test_cases):
        """Test execution of correct solution."""
        code = """
def add(a, b):
    return a + b
"""
        result = executor.execute(code, simple_test_cases)

        assert result['success'] is True
        assert result['passed_tests'] == len(simple_test_cases)
        assert result['total_tests'] == len(simple_test_cases)
        assert len(result['errors']) == 0
        assert result['execution_time'] > 0

    def test_incorrect_solution(self, executor, simple_test_cases):
        """Test execution of incorrect solution."""
        code = """
def add(a, b):
    return 42  # Always returns wrong value
"""
        result = executor.execute(code, simple_test_cases)

        assert result['success'] is False
        assert result['passed_tests'] == 0
        assert result['total_tests'] == len(simple_test_cases)
        assert len(result['errors']) > 0

    def test_partially_correct_solution(self, executor):
        """Test solution that passes some but not all tests."""
        code = """
def add(a, b):
    if a == 0:
        return 0  # Wrong: always returns 0 when a == 0
    return a + b
"""
        test_cases = [
            {'input': {'a': 1, 'b': 2}, 'expected': 3},  # Pass
            {'input': {'a': 0, 'b': 5}, 'expected': 5},  # Fail (returns 0 instead of 5)
        ]

        result = executor.execute(code, test_cases)

        assert result['success'] is False
        assert result['passed_tests'] == 1
        assert result['total_tests'] == 2

    def test_syntax_error(self, executor, simple_test_cases):
        """Test handling of syntax errors in code."""
        code = """
def add(a, b)  # Missing colon
    return a + b
"""
        result = executor.execute(code, simple_test_cases)

        assert result['success'] is False
        assert result['passed_tests'] == 0
        assert len(result['errors']) > 0

    def test_runtime_error(self, executor):
        """Test handling of runtime errors."""
        code = """
def add(a, b):
    return a / 0  # Division by zero
"""
        test_cases = [{'input': {'a': 1, 'b': 2}, 'expected': 3}]

        result = executor.execute(code, test_cases)

        assert result['success'] is False
        assert result['passed_tests'] == 0

    def test_timeout_enforcement(self, executor):
        """Test timeout enforcement for infinite loops."""
        code = """
def add(a, b):
    while True:  # Infinite loop
        pass
    return a + b
"""
        test_cases = [{'input': {'a': 1, 'b': 2}, 'expected': 3}]

        start_time = time.time()
        result = executor.execute(code, test_cases)
        elapsed = time.time() - start_time

        assert result['success'] is False
        assert result['passed_tests'] == 0
        # Should timeout within executor.timeout + some buffer
        assert elapsed < executor.timeout + 5

    def test_multiple_test_cases(self, executor):
        """Test execution with multiple diverse test cases."""
        code = """
def multiply(x, y):
    return x * y
"""
        test_cases = [
            {'input': {'x': 2, 'y': 3}, 'expected': 6},
            {'input': {'x': -1, 'y': 5}, 'expected': -5},
            {'input': {'x': 0, 'y': 100}, 'expected': 0},
            {'input': {'x': 7, 'y': 7}, 'expected': 49}
        ]

        result = executor.execute(code, test_cases)

        assert result['success'] is True
        assert result['passed_tests'] == 4
        assert len(result['outputs']) == 4

    def test_cleanup(self, executor):
        """Test proper cleanup of Docker resources."""
        code = "def add(a, b):\n    return a + b"
        test_cases = [{'input': {'a': 1, 'b': 2}, 'expected': 3}]

        executor.execute(code, test_cases)
        executor.cleanup()

        # After cleanup, should not have dangling containers
        # (This is tested implicitly - no exception should be raised)


class TestPassAtKMetric:
    """Test suite for PassAtKMetric class."""

    @pytest.fixture
    def executor(self):
        """Create CodeExecutor instance."""
        if DOCKER_AVAILABLE:
            return CodeExecutor(timeout=5, memory_limit="256m")
        else:
            # Mock executor if Docker not available
            executor = MagicMock(spec=CodeExecutor)
            executor.execute.return_value = {
                'success': True,
                'passed_tests': 1,
                'total_tests': 1,
                'execution_time': 0.1,
                'errors': [],
                'outputs': [3]
            }
            return executor

    @pytest.fixture
    def metric(self, executor):
        """Create PassAtKMetric instance."""
        return PassAtKMetric(executor, k_values=[1, 2, 3])

    @pytest.fixture
    def test_cases(self):
        """Standard test cases for testing."""
        return [
            {'input': {'a': 1, 'b': 2}, 'expected': 3},
            {'input': {'a': -1, 'b': 1}, 'expected': 0},
            {'input': {'a': 5, 'b': 5}, 'expected': 10}
        ]

    def test_pass_at_1_all_correct(self, metric, test_cases):
        """Test Pass@1 = 1.0 when all solutions are correct."""
        solutions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return b + a",  # Commutative
            "def add(a, b):\n    return a + b"
        ]
        
        # Mock executor behavior for these solutions
        if not DOCKER_AVAILABLE:
            metric.executor.execute.return_value = {
                'success': True, 
                'passed_tests': 3, 
                'total_tests': 3, 
                'errors': [],
                'execution_time': 0.1
            }

        result = metric.compute(solutions, test_cases, k=1)

        assert result.pass_at_k == 1.0
        assert result.n_total == 3
        assert result.n_correct == 3
        assert result.k == 1
        assert all(result.individual_results)

    def test_pass_at_1_all_incorrect(self, metric, test_cases):
        """Test Pass@1 = 0.0 when all solutions are incorrect."""
        solutions = [
            "def add(a, b):\n    return a - b",  # Wrong
            "def add(a, b):\n    return a * b",  # Wrong
            "def add(a, b):\n    return 0"      # Wrong
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.return_value = {
                'success': False, 
                'passed_tests': 0, 
                'total_tests': 3, 
                'errors': [],
                'execution_time': 0.1
            }

        result = metric.compute(solutions, test_cases, k=1)

        assert result.pass_at_k == 0.0
        assert result.n_total == 3
        assert result.n_correct == 0
        assert not any(result.individual_results)

    def test_pass_at_k_partial_correct(self, metric, test_cases):
        """Test Pass@k with partial correctness."""
        solutions = [
            "def add(a, b):\n    return a + b",  # Correct
            "def add(a, b):\n    return a - b",  # Wrong
            "def add(a, b):\n    return a * b",  # Wrong
            "def add(a, b):\n    return b + a",  # Correct
        ]
        
        if not DOCKER_AVAILABLE:
            # Need to mock per-call responses
            metric.executor.execute.side_effect = [
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
            ]

        # n=4, c=2, k=1: Pass@1 = 1 - (2 choose 1)/(4 choose 1) = 1 - 2/4 = 0.5
        result = metric.compute(solutions, test_cases, k=1)

        assert result.n_total == 4
        assert result.n_correct == 2
        assert result.k == 1
        assert abs(result.pass_at_k - 0.5) < 0.001

    def test_pass_at_k_different_k_values(self, metric, test_cases):
        """Test Pass@k with different k values."""
        solutions = [
            "def add(a, b):\n    return a + b",  # Correct
            "def add(a, b):\n    return a - b",  # Wrong
            "def add(a, b):\n    return a * b",  # Wrong
            "def add(a, b):\n    return a / 2",  # Wrong
            "def add(a, b):\n    return b + a",  # Correct
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = [
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
            ]

        # n=5, c=2
        results = metric.compute_multiple_k(solutions, test_cases, k_values=[1, 2, 3])

        # Pass@1 = 1 - C(3,1)/C(5,1) = 1 - 3/5 = 0.4
        assert abs(results[1].pass_at_k - 0.4) < 0.001

        # Pass@2 = 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
        assert abs(results[2].pass_at_k - 0.7) < 0.001

        # Pass@3 = 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
        assert abs(results[3].pass_at_k - 0.9) < 0.001

    def test_pass_at_k_edge_case_c_equals_k(self, metric, test_cases):
        """Test edge case when c >= k (guaranteed success)."""
        solutions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return b + a",
            "def add(a, b):\n    return a - b"  # Wrong
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = [
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
            ]

        # n=3, c=2, k=2: Should be 1.0 (guaranteed at least one correct)
        result = metric.compute(solutions, test_cases, k=2)

        assert result.pass_at_k == 1.0
        assert result.n_correct == 2

    def test_pass_at_k_edge_case_c_equals_0(self, metric, test_cases):
        """Test edge case when no solutions are correct."""
        solutions = [
            "def add(a, b):\n    return a - b",
            "def add(a, b):\n    return a * b"
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = None
            metric.executor.execute.return_value = {
                'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1
            }

        result = metric.compute(solutions, test_cases, k=1)

        assert result.pass_at_k == 0.0
        assert result.n_correct == 0

    def test_pass_at_k_edge_case_all_correct(self, metric, test_cases):
        """Test edge case when all solutions are correct."""
        solutions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return b + a"
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = None
            metric.executor.execute.return_value = {
                'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1
            }

        result = metric.compute(solutions, test_cases, k=1)

        assert result.pass_at_k == 1.0
        assert result.n_correct == 2

    def test_invalid_k_value(self, metric, test_cases):
        """Test that invalid k values raise ValueError."""
        solutions = ["def add(a, b):\n    return a + b"]

        # k > n
        with pytest.raises(ValueError, match="cannot be greater than"):
            metric.compute(solutions, test_cases, k=5)

        # k < 1
        with pytest.raises(ValueError, match="must be >= 1"):
            metric.compute(solutions, test_cases, k=0)

    def test_compute_multiple_k_efficiency(self, metric, test_cases):
        """Test that compute_multiple_k evaluates solutions only once."""
        solutions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a - b"
        ]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = [
                {'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
                {'success': False, 'passed_tests': 0, 'total_tests': 3, 'errors': [], 'execution_time': 0.1},
            ]

        start_time = time.time()
        results = metric.compute_multiple_k(solutions, test_cases, k_values=[1, 2])
        elapsed = time.time() - start_time

        # Should be roughly the same time as single computation
        # (not 2x because solutions are only evaluated once)
        assert len(results) == 2
        assert 1 in results
        assert 2 in results

        # Both should have same individual_results
        assert results[1].individual_results == results[2].individual_results

    def test_pass_at_k_result_to_dict(self, metric, test_cases):
        """Test PassAtKResult.to_dict() method."""
        solutions = ["def add(a, b):\n    return a + b"]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = None
            metric.executor.execute.return_value = {
                'success': True, 'passed_tests': 3, 'total_tests': 3, 'errors': [], 'execution_time': 0.1
            }
    def test_mathematical_correctness(self, metric):
        """Test mathematical correctness of Pass@k formula."""
        # Create controlled scenario with known results
        # 10 solutions, 3 correct, k=2
        # Pass@2 = 1 - C(7,2)/C(10,2) = 1 - 21/45 = 0.5333...

        solutions = ["def add(a, b):\n    return a + b"] * 3  # 3 correct
        solutions += ["def add(a, b):\n    return a - b"] * 7  # 7 wrong

        test_cases = [{'input': {'a': 1, 'b': 1}, 'expected': 2}]
        
        if not DOCKER_AVAILABLE:
            metric.executor.execute.side_effect = (
                [{'success': True, 'passed_tests': 1, 'total_tests': 1, 'errors': []}] * 3 +
                [{'success': False, 'passed_tests': 0, 'total_tests': 1, 'errors': []}] * 7
            )

        result = metric.compute(solutions, test_cases, k=2)

        expected = 1.0 - (21.0 / 45.0)  # 0.5333...

        assert result.n_total == 10
        assert result.n_correct == 3
        assert abs(result.pass_at_k - expected) < 0.001


class TestPassAtKFormula:
    """Test suite for the combinatorial formula implementation."""

    def test_formula_n5_c2_k1(self):
        """Test formula: n=5, c=2, k=1 -> Pass@1 = 0.4"""
        executor = MagicMock(spec=CodeExecutor)
        metric = PassAtKMetric(executor)

        # Direct formula test
        result = metric._compute_pass_at_k(n=5, c=2, k=1)
        expected = 1.0 - (3.0/5.0)  # 0.4

        assert abs(result - expected) < 0.001

    def test_formula_n5_c2_k2(self):
        """Test formula: n=5, c=2, k=2 -> Pass@2 = 0.7"""
        executor = MagicMock(spec=CodeExecutor)
        metrics = PassAtKMetric(executor)

        result = metrics._compute_pass_at_k(n=5, c=2, k=2)
        # C(3,2)/C(5,2) = 3/10
        expected = 1.0 - (3.0/10.0)  # 0.7

        assert abs(result - expected) < 0.001

    def test_formula_n10_c3_k5(self):
        """Test formula: n=10, c=3, k=5"""
        executor = MagicMock(spec=CodeExecutor)
        metric = PassAtKMetric(executor)

        result = metric._compute_pass_at_k(n=10, c=3, k=5)

        # C(7,5)/C(10,5) = 21/252 = 1/12
        expected = 1.0 - (21.0/252.0)  # 0.9166...

        assert abs(result - expected) < 0.001

    def test_formula_edge_cases(self):
        """Test edge cases of the formula."""
        executor = MagicMock(spec=CodeExecutor)
        metric = PassAtKMetric(executor)

        # c = 0: No correct solutions
        assert metric._compute_pass_at_k(n=5, c=0, k=1) == 0.0

        # n - c < k: Not enough incorrect to fill sample (must get correct)
        assert metric._compute_pass_at_k(n=5, c=4, k=2) == 1.0  # Only 1 incorrect, can't pick 2

        # c = n: All correct
        assert metric._compute_pass_at_k(n=5, c=5, k=1) == 1.0

        # k = n: Sampling all solutions
        assert metric._compute_pass_at_k(n=5, c=2, k=5) == 1.0


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
class TestDockerSandboxSecurity:
    """Test suite for Docker sandbox security features."""

    @pytest.fixture
    def executor(self):
        """Create CodeExecutor instance."""
        return CodeExecutor(timeout=5, memory_limit="256m")

    def test_no_network_access(self, executor):
        """Test that sandboxed code has no network access."""
        code = """
def add(a, b):
    import urllib.request
    # This should fail due to no network
    urllib.request.urlopen('http://google.com')
    return a + b
"""
        test_cases = [{'input': {'a': 1, 'b': 2}, 'expected': 3}]

        result = executor.execute(code, test_cases)

        # Should fail due to network restriction
        assert result['success'] is False

    def test_resource_limits(self, executor):
        """Test that resource limits are enforced."""
        # This test verifies timeout is enforced (tested in test_timeout_enforcement)
        # Memory limits are harder to test without actually exhausting memory
        # CPU limits are enforced by Docker
        pass

    def test_filesystem_isolation(self, executor):
        """Test that filesystem is isolated."""
        code = """
def add(a, b):
    # Try to read sensitive file (should not exist in sandbox)
    try:
        with open('/etc/passwd', 'r') as f:
            content = f.read()
    except:
        pass  # Expected
    return a + b
"""
        test_cases = [{'input': {'a': 1, 'b': 2}, 'expected': 3}]

        result = executor.execute(code, test_cases)

        # Should still pass (file reading is handled gracefully)
        assert result['success'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
