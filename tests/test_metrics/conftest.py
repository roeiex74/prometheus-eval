"""Configuration for metrics tests."""

import pytest
import sys

def pytest_collection_modifyitems(config, items):
    """
    Skip bertscore tests if torch is not properly installed.
    This handles architecture mismatches (e.g., x86_64 torch on arm64 Mac).
    """
    try:
        import torch
    except (ImportError, OSError) as e:
        skip_torch = pytest.mark.skip(reason=f"torch not available: {e}")
        for item in items:
            if "bertscore" in item.nodeid.lower():
                item.add_marker(skip_torch)
