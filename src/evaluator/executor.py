"""
Code Executor Module

Secure Docker-based code execution environment for evaluating generated code.
Implements sandboxed execution with resource limits and timeout enforcement.

Author: System Architect Agent
Date: 2025-12-13
"""

import os
import time
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

import docker
from docker.errors import DockerException, ContainerError, ImageNotFound
from loguru import logger


class CodeExecutor:
    """
    Docker-based secure code executor for evaluating generated code solutions.

    Provides isolated execution environment with:
    - No network access
    - Resource limits (CPU, memory, timeout)
    - Non-root execution
    - Automatic cleanup

    Attributes:
        timeout: Maximum execution time in seconds
        memory_limit: Memory limit (e.g., "512m", "1g")
        cpu_quota: CPU quota in microseconds (50000 = 50% of one core)
        image_name: Docker image name for sandbox
        client: Docker client instance
    """

    def __init__(
        self,
        timeout: int = 10,
        memory_limit: str = "512m",
        cpu_quota: int = 50000,
        image_name: str = "prometheus-eval-sandbox:latest"
    ):
        """
        Initialize Docker-based code executor.

        Args:
            timeout: Maximum execution time in seconds (default: 10)
            memory_limit: Memory limit string (default: "512m")
            cpu_quota: CPU quota in microseconds (default: 50000 = 50%)
            image_name: Docker image name (default: "prometheus-eval-sandbox:latest")

        Raises:
            DockerException: If Docker daemon is not available
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.image_name = image_name

        try:
            self.client = docker.from_env()
            logger.info(f"Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise

        self._ensure_image_exists()

    def _ensure_image_exists(self) -> None:
        """
        Verify that the sandbox Docker image exists.

        Raises:
            ImageNotFound: If the required Docker image is not built
        """
        try:
            self.client.images.get(self.image_name)
            logger.debug(f"Docker image '{self.image_name}' found")
        except ImageNotFound:
            logger.warning(
                f"Docker image '{self.image_name}' not found. "
                f"Please build it using: docker build -t {self.image_name} docker/python-sandbox/"
            )
            raise

    def execute(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Execute code with provided test cases in sandboxed environment.

        Args:
            code: Source code to execute
            test_cases: List of test case dictionaries with 'input' and 'expected' keys
            language: Programming language (currently only "python" supported)

        Returns:
            Dictionary containing:
                - success: bool - Overall execution success
                - passed_tests: int - Number of passed test cases
                - total_tests: int - Total number of test cases
                - outputs: List[str] - Output for each test case
                - errors: List[str] - Error messages for failed tests
                - execution_time: float - Total execution time in seconds

        Raises:
            ValueError: If language is not supported
        """
        if language != "python":
            raise ValueError(f"Unsupported language: {language}. Only 'python' is supported.")

        logger.info(f"Executing code with {len(test_cases)} test cases")

        start_time = time.time()
        results = {
            'success': False,
            'passed_tests': 0,
            'total_tests': len(test_cases),
            'outputs': [],
            'errors': [],
            'execution_time': 0.0
        }

        for idx, test_case in enumerate(test_cases):
            logger.debug(f"Running test case {idx + 1}/{len(test_cases)}")

            test_result = self._execute_single_test(code, test_case)

            results['outputs'].append(test_result.get('output', ''))

            if test_result['passed']:
                results['passed_tests'] += 1
            else:
                results['errors'].append(
                    test_result.get('error', f"Test {idx + 1} failed")
                )

        results['execution_time'] = time.time() - start_time
        results['success'] = results['passed_tests'] == results['total_tests']

        logger.info(
            f"Execution complete: {results['passed_tests']}/{results['total_tests']} "
            f"tests passed in {results['execution_time']:.2f}s"
        )

        return results

    def _execute_single_test(
        self,
        code: str,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute code with a single test case.

        Args:
            code: Source code to execute
            test_case: Dictionary with 'input' and 'expected' keys

        Returns:
            Dictionary with 'passed', 'output', and optional 'error' keys
        """
        container = None

        try:
            # Prepare test code with input and assertion
            test_code = self._prepare_test_code(code, test_case)

            # Create and run container
            container = self.client.containers.run(
                self.image_name,
                command=["python3", "-c", test_code],
                detach=True,
                network_mode="none",  # No network access
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                pids_limit=50,  # Limit number of processes
                remove=False  # Manual cleanup for better error handling
            )

            # Wait for execution with timeout
            try:
                exit_code = container.wait(timeout=self.timeout)

                # Handle different Docker SDK versions
                if isinstance(exit_code, dict):
                    exit_code = exit_code.get('StatusCode', 1)

                output = container.logs().decode('utf-8', errors='replace')

                if exit_code == 0:
                    return {
                        'passed': True,
                        'output': output.strip()
                    }
                else:
                    return {
                        'passed': False,
                        'output': output.strip(),
                        'error': f"Execution failed with exit code {exit_code}: {output.strip()}"
                    }

            except Exception as e:
                # Timeout or execution error
                logger.warning(f"Test execution error: {e}")
                return {
                    'passed': False,
                    'output': '',
                    'error': f"Timeout or execution error: {str(e)}"
                }

        except ContainerError as e:
            logger.error(f"Container error: {e}")
            return {
                'passed': False,
                'output': '',
                'error': f"Container error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error during execution: {e}")
            return {
                'passed': False,
                'output': '',
                'error': f"Unexpected error: {str(e)}"
            }
        finally:
            # Cleanup container
            if container:
                try:
                    container.stop(timeout=1)
                    container.remove()
                except Exception as e:
                    logger.warning(f"Failed to cleanup container: {e}")

    def _prepare_test_code(
        self,
        code: str,
        test_case: Dict[str, Any]
    ) -> str:
        """
        Prepare executable test code with input and assertion.

        Args:
            code: Original function/class code
            test_case: Test case with 'input' and 'expected' keys

        Returns:
            Complete Python code string ready for execution
        """
        test_input = test_case.get('input', {})
        expected = test_case.get('expected')

        # Build test code
        test_code_parts = [
            code,  # Original code
            "",
            "# Test execution",
        ]

        # Handle different input formats
        if isinstance(test_input, dict):
            # Named arguments
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in test_input.items())
        elif isinstance(test_input, (list, tuple)):
            # Positional arguments
            args_str = ', '.join(repr(arg) for arg in test_input)
        else:
            # Single argument
            args_str = repr(test_input)

        # Extract function name (simple heuristic)
        function_name = self._extract_function_name(code)

        if function_name:
            test_code_parts.extend([
                f"result = {function_name}({args_str})",
                f"expected = {repr(expected)}",
                "assert result == expected, f'Expected {{expected}}, got {{result}}'",
                "print(f'PASS: {{result}}')"
            ])
        else:
            # If no function found, try to execute code directly
            test_code_parts.extend([
                f"# Direct execution test",
                f"expected = {repr(expected)}",
                f"# Code should define 'result' variable",
                "assert 'result' in locals(), 'Code must define result variable'",
                "assert result == expected, f'Expected {{expected}}, got {{result}}'",
                "print(f'PASS: {{result}}')"
            ])

        return '\n'.join(test_code_parts)

    def _extract_function_name(self, code: str) -> Optional[str]:
        """
        Extract function name from code (simple heuristic).

        Args:
            code: Python source code

        Returns:
            Function name or None if not found
        """
        import re

        # Match: def function_name(...)
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)

        return None

    def cleanup(self) -> None:
        """
        Clean up Docker resources and close client connection.

        Removes any dangling containers and closes the Docker client.
        """
        try:
            # Remove dangling containers from this executor
            containers = self.client.containers.list(
                all=True,
                filters={'ancestor': self.image_name}
            )

            for container in containers:
                try:
                    container.stop(timeout=1)
                    container.remove()
                    logger.debug(f"Removed container {container.short_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.short_id}: {e}")

            self.client.close()
            logger.info("Docker client cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
