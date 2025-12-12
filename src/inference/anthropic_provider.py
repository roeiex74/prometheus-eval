"""
Anthropic LLM Provider Implementation.

This module provides an implementation of the AbstractLLMProvider for Anthropic's Claude API,
supporting Claude models with async/await, retry logic, and rate limiting.
"""

from typing import List, Dict, Any, Optional
import asyncio
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from loguru import logger

from .base import (
    AbstractLLMProvider,
    LLMProviderError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    InvalidRequestError,
)
from .config import InferenceConfig


class AnthropicProvider(AbstractLLMProvider):
    """
    Anthropic Claude implementation of AbstractLLMProvider.

    Supports Claude models with features including:
    - Async and sync generation
    - Batch processing with concurrent requests
    - Approximate token counting
    - Automatic retry on transient failures
    - Rate limiting to prevent API quota exhaustion
    - Comprehensive error handling

    Example:
        >>> provider = AnthropicProvider(
        ...     api_key="sk-ant-...",
        ...     default_model="claude-3-sonnet-20240229"
        ... )
        >>> response = provider.generate("What is the capital of France?")
        >>> print(response)
        The capital of France is Paris.
    """

    # Approximate token ratios for Claude models (chars per token)
    # Claude uses a similar tokenizer to GPT, roughly 4 chars per token
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        api_key: str,
        default_model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        retry_attempts: int = 3,
        rpm_limit: int = 50,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            default_model: Default model to use (e.g., 'claude-3-sonnet-20240229', 'claude-3-opus-20240229')
            temperature: Default temperature (0.0-2.0)
            max_tokens: Default maximum tokens to generate
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            rpm_limit: Requests per minute limit
        """
        super().__init__(
            provider_name="anthropic",
            api_key=api_key,
            default_model=default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_attempts=retry_attempts,
            rpm_limit=rpm_limit,
        )

        # Initialize Anthropic clients
        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.async_client = AsyncAnthropic(api_key=api_key, timeout=timeout)

        logger.info(f"Anthropic provider initialized with model: {default_model}")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text (approximate for Anthropic).

        Since Anthropic doesn't provide a public tokenizer, we use an approximation
        based on character count. This is roughly accurate but not exact.

        Args:
            text: Text to count tokens for
            model: Model to use (not used for Anthropic, kept for interface compatibility)

        Returns:
            Approximate number of tokens

        Example:
            >>> provider = AnthropicProvider(api_key="sk-ant-...")
            >>> token_count = provider.count_tokens("Hello, world!")
            >>> print(token_count)
            3
        """
        try:
            # Use Anthropic's token counting API if available
            # Falls back to approximation
            try:
                # Anthropic SDK provides count_tokens method
                count_result = self.client.count_tokens(text)
                return count_result
            except (AttributeError, Exception):
                # Fallback to character-based approximation
                # Claude uses roughly 4 characters per token
                return max(1, len(text) // self.CHARS_PER_TOKEN)
        except Exception as e:
            logger.warning(f"Error counting tokens, using approximation: {e}")
            return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _handle_anthropic_error(self, error: Exception) -> Exception:
        """
        Convert Anthropic errors to provider-specific exceptions.

        Args:
            error: Original Anthropic exception

        Returns:
            Mapped provider exception
        """
        if isinstance(error, anthropic.RateLimitError):
            return RateLimitError(f"Anthropic rate limit exceeded: {error}")
        elif isinstance(error, anthropic.AuthenticationError):
            return AuthenticationError(f"Anthropic authentication failed: {error}")
        elif isinstance(error, anthropic.APITimeoutError):
            return TimeoutError(f"Anthropic request timed out: {error}")
        elif isinstance(error, anthropic.BadRequestError):
            return InvalidRequestError(f"Invalid Anthropic request: {error}")
        else:
            return LLMProviderError(f"Anthropic error: {error}")

    async def _async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Async generation with Anthropic API.

        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            **kwargs: Additional Anthropic parameters

        Returns:
            Generated text

        Raises:
            LLMProviderError: On generation failure
        """
        params = self._merge_kwargs(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

        # Validate parameters
        self.validate_parameters(
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )

        self._log_request(prompt, params["model"])

        # Build request parameters
        request_params = {
            "model": params["model"],
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add optional parameters
        if params.get("top_p") is not None:
            request_params["top_p"] = params["top_p"]

        # Add system prompt if provided
        if kwargs.get("system"):
            request_params["system"] = kwargs["system"]

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in request_params and key != "system":
                request_params[key] = value

        try:
            # Make API request with retry logic
            retry_decorator = self.get_retry_decorator()

            @retry_decorator
            async def _make_request():
                return await self._rate_limited_request(
                    self.async_client.messages.create,
                    **request_params,
                )

            response = await _make_request()

            # Extract generated text from response
            # Anthropic responses have a different structure
            generated_text = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        generated_text += block.text

            # Log response
            tokens_used = None
            if hasattr(response, "usage"):
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            self._log_response(generated_text, tokens_used)

            return generated_text

        except Exception as e:
            error = self._handle_anthropic_error(e)
            self._log_error(error)
            raise error

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate response for a single prompt (synchronous).

        Args:
            prompt: Input prompt text
            model: Model to use (defaults to default_model)
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            system: Optional system prompt for Claude
            **kwargs: Additional Anthropic parameters

        Returns:
            Generated text response

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out

        Example:
            >>> provider = AnthropicProvider(api_key="sk-ant-...")
            >>> response = provider.generate(
            ...     "Explain quantum computing",
            ...     temperature=0.5,
            ...     system="You are a physics expert."
            ... )
            >>> print(response)
        """
        # Run async method synchronously
        return asyncio.run(
            self._async_generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                system=system,
                **kwargs,
            )
        )

    async def _async_generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Async batch generation with concurrent requests.

        Args:
            prompts: List of input prompts
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            system: Optional system prompt for Claude
            **kwargs: Additional Anthropic parameters

        Returns:
            List of generated texts
        """
        logger.info(f"Generating batch of {len(prompts)} prompts")

        # Create tasks for all prompts
        tasks = [
            self._async_generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                system=system,
                **kwargs,
            )
            for prompt in prompts
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                raise result  # Re-raise first exception encountered
            responses.append(result)

        logger.info(f"Batch generation completed: {len(responses)} responses")
        return responses

    def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts (synchronous).

        Uses async/await internally for concurrent requests.

        Args:
            prompts: List of input prompts
            model: Model to use (defaults to default_model)
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            system: Optional system prompt for Claude
            **kwargs: Additional Anthropic parameters

        Returns:
            List of generated text responses

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded

        Example:
            >>> provider = AnthropicProvider(api_key="sk-ant-...")
            >>> prompts = ["Explain AI", "Explain ML", "Explain DL"]
            >>> responses = provider.generate_batch(
            ...     prompts,
            ...     system="You are a helpful AI assistant."
            ... )
            >>> print(len(responses))
            3
        """
        return asyncio.run(
            self._async_generate_batch(
                prompts=prompts,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                system=system,
                **kwargs,
            )
        )


def create_anthropic_provider(
    config: Optional[InferenceConfig] = None,
) -> AnthropicProvider:
    """
    Factory function to create Anthropic provider from config.

    Args:
        config: Optional InferenceConfig instance. If not provided, loads from environment.

    Returns:
        Configured AnthropicProvider instance

    Raises:
        ValueError: If Anthropic API key is not configured

    Example:
        >>> from .config import load_config
        >>> config = load_config()
        >>> provider = create_anthropic_provider(config)
    """
    if config is None:
        from .config import get_config
        config = get_config()

    if not config.validate_provider_credentials("anthropic"):
        raise ValueError(
            "Anthropic API key not found. Please set ANTHROPIC_API_KEY in .env file."
        )

    provider_config = config.get_provider_config("anthropic")

    return AnthropicProvider(
        api_key=provider_config["api_key"],
        default_model=provider_config["default_model"],
        temperature=config.default_temperature,
        max_tokens=config.default_max_tokens,
        timeout=provider_config["timeout"],
        retry_attempts=provider_config["retry_attempts"],
        rpm_limit=provider_config["rpm_limit"],
    )
