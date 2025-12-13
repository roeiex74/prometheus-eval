"""Global pytest configuration for Prometheus-Eval tests."""

import pytest
from pathlib import Path

# Check if torch is available before collecting torch-dependent tests
def pytest_ignore_collect(collection_path, config):
    """
    Skip collection of torch-dependent tests if torch is not available.
    This handles architecture mismatches (e.g., x86_64 torch on arm64 Mac).
    """
    if "test_bertscore.py" in str(collection_path):
        try:
            import torch
            return False  # Don't ignore, torch is available
        except (ImportError, OSError):
            return True  # Ignore this file, torch not available
    return False
