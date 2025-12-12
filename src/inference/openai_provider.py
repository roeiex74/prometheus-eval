"""
OpenAI LLM Provider Implementation.

This module provides an implementation of the AbstractLLMProvider for OpenAI's API,
supporting GPT models with async/await, retry logic, and rate limiting.
"""

from typing import List, Dict, Any, Optional
import asyncio
import openai
from openai import AsyncOpenAI, OpenAI
import tiktoken
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


class OpenAIProvider(AbstractLLMProvider):
    """
    OpenAI implementation of AbstractLLMProvider.

    Supports GPT models with features including:
    - Async and sync generation
    - Batch processing with concurrent requests
    - Token counting with tiktoken
    - Automatic retry on transient failures
    - Rate limiting to prevent API quota exhaustion
    - Comprehensive error handling

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...", default_model="gpt-4-turbo-preview")
        >>> response = provider.generate("What is the capital of France?")
        >>> print(response)
        The capital of France is Paris.
    """

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4-turbo-preview",
        org_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30,
        retry_attempts: int = 3,
        rpm_limit: int = 60,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            default_model: Default model to use (e.g., 'gpt-4-turbo-preview', 'gpt-3.5-turbo')
            org_id: Optional OpenAI organization ID
            temperature: Default temperature (0.0-2.0)
            max_tokens: Default maximum tokens to generate
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            rpm_limit: Requests per minute limit
        """
        super().__init__(
            provider_name="openai",
            api_key=api_key,
            default_model=default_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_attempts=retry_attempts,
            rpm_limit=rpm_limit,
        )

        self.org_id = org_id

        # Initialize OpenAI clients
        client_kwargs = {"api_key": api_key, "timeout": timeout}
        if org_id:
            client_kwargs["organization"] = org_id

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        # Cache for tiktoken encodings
        self._encodings: Dict[str, tiktoken.Encoding] = {}

        logger.info(f"OpenAI provider initialized with model: {default_model}")

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """
        Get tiktoken encoding for a model (with caching).

        Args:
            model: Model name

        Returns:
            Tiktoken encoding instance
        """
        if model not in self._encodings:
            try:
                self._encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models (GPT-4, GPT-3.5-turbo default)
                logger.warning(
                    f"Model {model} not found in tiktoken, using cl100k_base encoding"
                )
                self._encodings[model] = tiktoken.get_encoding("cl100k_base")

        return self._encodings[model]

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model to use for encoding (defaults to default_model)

        Returns:
            Number of tokens

        Example:
            >>> provider = OpenAIProvider(api_key="sk-...")
            >>> token_count = provider.count_tokens("Hello, world!")
            >>> print(token_count)
            4
        """
        model = model or self.default_model
        try:
            encoding = self._get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise LLMProviderError(f"Token counting failed: {e}")

    def _handle_openai_error(self, error: Exception) -> Exception:
        """
        Convert OpenAI errors to provider-specific exceptions.

        Args:
            error: Original OpenAI exception

        Returns:
            Mapped provider exception
        """
        if isinstance(error, openai.RateLimitError):
            return RateLimitError(f"OpenAI rate limit exceeded: {error}")
        elif isinstance(error, openai.AuthenticationError):
            return AuthenticationError(f"OpenAI authentication failed: {error}")
        elif isinstance(error, openai.APITimeoutError):
            return TimeoutError(f"OpenAI request timed out: {error}")
        elif isinstance(error, openai.BadRequestError):
            return InvalidRequestError(f"Invalid OpenAI request: {error}")
        else:
            return LLMProviderError(f"OpenAI error: {error}")

    async def _async_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Async generation with OpenAI API.

        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI parameters

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
            seed=seed,
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
        }

        # Add optional parameters
        if params.get("top_p") is not None:
            request_params["top_p"] = params["top_p"]
        if params.get("seed") is not None:
            request_params["seed"] = params["seed"]

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in request_params:
                request_params[key] = value

        try:
            # Make API request with retry logic
            retry_decorator = self.get_retry_decorator()

            @retry_decorator
            async def _make_request():
                return await self._rate_limited_request(
                    self.async_client.chat.completions.create,
                    **request_params,
                )

            response = await _make_request()

            # Extract generated text
            generated_text = response.choices[0].message.content

            # Log response
            tokens_used = response.usage.total_tokens if response.usage else None
            self._log_response(generated_text, tokens_used)

            return generated_text

        except Exception as e:
            error = self._handle_openai_error(e)
            self._log_error(error)
            raise error

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
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
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI parameters

        Returns:
            Generated text response

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If authentication fails
            TimeoutError: If request times out

        Example:
            >>> provider = OpenAIProvider(api_key="sk-...")
            >>> response = provider.generate("Explain quantum computing", temperature=0.5)
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
                seed=seed,
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
        seed: Optional[int] = None,
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
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI parameters

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
                seed=seed,
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
        seed: Optional[int] = None,
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
            seed: Random seed for reproducibility
            **kwargs: Additional OpenAI parameters

        Returns:
            List of generated text responses

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded

        Example:
            >>> provider = OpenAIProvider(api_key="sk-...")
            >>> prompts = ["Explain AI", "Explain ML", "Explain DL"]
            >>> responses = provider.generate_batch(prompts)
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
                seed=seed,
                **kwargs,
            )
        )


def create_openai_provider(config: Optional[InferenceConfig] = None) -> OpenAIProvider:
    """
    Factory function to create OpenAI provider from config.

    Args:
        config: Optional InferenceConfig instance. If not provided, loads from environment.

    Returns:
        Configured OpenAIProvider instance

    Raises:
        ValueError: If OpenAI API key is not configured

    Example:
        >>> from .config import load_config
        >>> config = load_config()
        >>> provider = create_openai_provider(config)
    """
    if config is None:
        from .config import get_config
        config = get_config()

    if not config.validate_provider_credentials("openai"):
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in .env file."
        )

    provider_config = config.get_provider_config("openai")

    return OpenAIProvider(
        api_key=provider_config["api_key"],
        org_id=provider_config.get("org_id"),
        default_model=provider_config["default_model"],
        temperature=config.default_temperature,
        max_tokens=config.default_max_tokens,
        timeout=provider_config["timeout"],
        retry_attempts=provider_config["retry_attempts"],
        rpm_limit=provider_config["rpm_limit"],
    )
