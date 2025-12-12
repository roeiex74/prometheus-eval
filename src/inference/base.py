"""
Abstract Base Class for LLM Providers.

This module defines the interface that all LLM provider implementations must follow.
It includes retry logic, rate limiting, error handling, and logging functionality.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import time
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger
from asyncio_throttle import Throttler


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class RateLimitError(LLMProviderError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class TimeoutError(LLMProviderError):
    """Raised when request times out."""
    pass


class InvalidRequestError(LLMProviderError):
    """Raised when request is invalid."""
    pass


class AbstractLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations (OpenAI, Anthropic, etc.) must inherit from this class
    and implement the required methods.

    Attributes:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        api_key: API key for authentication
        default_model: Default model to use for generation
        temperature: Default temperature for generation
        max_tokens: Default maximum tokens to generate
        timeout: Request timeout in seconds
        retry_attempts: Number of retry attempts for failed requests
        rpm_limit: Requests per minute limit
    """

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        default_model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        retry_attempts: int = 3,
        rpm_limit: int = 60,
    ):
        """
        Initialize the LLM provider.

        Args:
            provider_name: Name of the provider
            api_key: API key for authentication
            default_model: Default model to use
            temperature: Default temperature (0.0-2.0)
            max_tokens: Default maximum tokens
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            rpm_limit: Requests per minute limit
        """
        self.provider_name = provider_name
        self.api_key = api_key
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.rpm_limit = rpm_limit

        # Initialize rate limiter (rpm_limit / 60 = requests per second)
        self.throttler = Throttler(rate_limit=rpm_limit, period=60.0)

        # Track request statistics
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0

        logger.info(
            f"Initialized {provider_name} provider with model {default_model}, "
            f"RPM limit: {rpm_limit}, timeout: {timeout}s"
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate a response for a single prompt.

        Args:
            prompt: Input prompt text
            model: Model to use (defaults to default_model)
            temperature: Temperature for generation (defaults to default_temperature)
            max_tokens: Maximum tokens to generate (defaults to default_max_tokens)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            model: Model to use (defaults to default_model)
            temperature: Temperature for generation (defaults to default_temperature)
            max_tokens: Maximum tokens to generate (defaults to default_max_tokens)
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated text responses

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text to count tokens
            model: Model to use for tokenization (defaults to default_model)

        Returns:
            Number of tokens in the text

        Raises:
            LLMProviderError: If token counting fails
        """
        pass

    def get_retry_decorator(self):
        """
        Get a tenacity retry decorator configured for this provider.

        Returns:
            Configured retry decorator
        """
        return retry(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((RateLimitError, TimeoutError)),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )

    async def _rate_limited_request(self, request_func, *args, **kwargs):
        """
        Execute a request with rate limiting.

        Args:
            request_func: Async function to execute
            *args: Positional arguments for request_func
            **kwargs: Keyword arguments for request_func

        Returns:
            Result from request_func
        """
        async with self.throttler:
            return await request_func(*args, **kwargs)

    def _log_request(self, prompt: str, model: str, **kwargs):
        """
        Log an API request.

        Args:
            prompt: Request prompt
            model: Model used
            **kwargs: Additional parameters
        """
        self.request_count += 1
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.debug(
            f"[{self.provider_name}] Request #{self.request_count} | "
            f"Model: {model} | Prompt: {prompt_preview}"
        )

    def _log_response(self, response: str, tokens_used: Optional[int] = None):
        """
        Log an API response.

        Args:
            response: Response text
            tokens_used: Number of tokens used
        """
        response_preview = response[:100] + "..." if len(response) > 100 else response
        log_msg = f"[{self.provider_name}] Response: {response_preview}"

        if tokens_used is not None:
            self.total_tokens += tokens_used
            log_msg += f" | Tokens: {tokens_used}"

        logger.debug(log_msg)

    def _log_error(self, error: Exception):
        """
        Log an error.

        Args:
            error: Exception that occurred
        """
        self.error_count += 1
        logger.error(
            f"[{self.provider_name}] Error #{self.error_count}: "
            f"{type(error).__name__} - {str(error)}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get provider usage statistics.

        Returns:
            Dictionary containing usage statistics
        """
        return {
            "provider": self.provider_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_tokens": self.total_tokens,
            "error_rate": self.error_count / max(self.request_count, 1),
        }

    def reset_stats(self):
        """Reset usage statistics."""
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        logger.info(f"[{self.provider_name}] Statistics reset")

    def validate_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Validate generation parameters.

        Args:
            temperature: Temperature value to validate
            max_tokens: Max tokens value to validate

        Raises:
            InvalidRequestError: If parameters are invalid
        """
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise InvalidRequestError(
                    f"Temperature must be between 0.0 and 2.0, got {temperature}"
                )

        if max_tokens is not None:
            if max_tokens <= 0:
                raise InvalidRequestError(
                    f"max_tokens must be positive, got {max_tokens}"
                )

    def _merge_kwargs(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge provided kwargs with defaults.

        Args:
            model: Model override
            temperature: Temperature override
            max_tokens: Max tokens override
            **kwargs: Additional parameters

        Returns:
            Merged parameters dictionary
        """
        params = {
            "model": model or self.default_model,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        params.update(kwargs)
        return params

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"{self.__class__.__name__}(provider={self.provider_name}, "
            f"model={self.default_model}, rpm_limit={self.rpm_limit})"
        )


class SyncProviderMixin:
    """
    Mixin to provide synchronous wrappers for async methods.

    This allows providers to offer both sync and async interfaces.
    """

    @staticmethod
    def run_async(coro):
        """
        Run an async coroutine in a synchronous context.

        Args:
            coro: Coroutine to run

        Returns:
            Result from coroutine
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)
