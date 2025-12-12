"""
Prometheus-Eval Inference Engine.

This module provides a unified interface for LLM inference across multiple providers
(OpenAI, Anthropic) with built-in retry logic, rate limiting, and error handling.

Example usage:
    >>> from src.inference import create_openai_provider, load_config
    >>> config = load_config()
    >>> provider = create_openai_provider(config)
    >>> response = provider.generate("What is AI?")
    >>> print(response)

    >>> # Or use Anthropic
    >>> from src.inference import create_anthropic_provider
    >>> provider = create_anthropic_provider(config)
    >>> response = provider.generate("Explain machine learning")
"""

from .config import (
    InferenceConfig,
    load_config,
    get_config,
    reset_config,
)

from .base import (
    AbstractLLMProvider,
    LLMProviderError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    InvalidRequestError,
)

from .openai_provider import (
    OpenAIProvider,
    create_openai_provider,
)

from .anthropic_provider import (
    AnthropicProvider,
    create_anthropic_provider,
)

__all__ = [
    # Configuration
    "InferenceConfig",
    "load_config",
    "get_config",
    "reset_config",
    # Base classes and exceptions
    "AbstractLLMProvider",
    "LLMProviderError",
    "RateLimitError",
    "AuthenticationError",
    "TimeoutError",
    "InvalidRequestError",
    # OpenAI
    "OpenAIProvider",
    "create_openai_provider",
    # Anthropic
    "AnthropicProvider",
    "create_anthropic_provider",
]
