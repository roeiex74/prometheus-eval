"""
Inference Engine Configuration Management.

This module handles configuration loading and validation for LLM providers
using pydantic for type safety and python-dotenv for environment management.

Note: This module has been refactored into smaller components:
- config_model.py: Configuration data model (InferenceConfig)
- config_loader.py: Configuration loading utilities
"""

# Re-export all public APIs for backward compatibility
from .config_model import InferenceConfig
from .config_loader import load_config, get_config, reset_config

__all__ = [
    "InferenceConfig",
    "load_config",
    "get_config",
    "reset_config",
]
