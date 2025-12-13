"""
Tests for abstract LLM provider base class.

Tests retry logic, rate limiting, error handling, and statistics tracking.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.inference.base import (
    AbstractLLMProvider,
    LLMProviderError,
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    InvalidRequestError,
)


class TestLLMProviderExceptions:
    """Test custom exception classes."""

    def test_exception_hierarchy(self):
        """Test that custom exceptions inherit from LLMProviderError."""
        assert issubclass(RateLimitError, LLMProviderError)
        assert issubclass(AuthenticationError, LLMProviderError)
        assert issubclass(TimeoutError, LLMProviderError)
        assert issubclass(InvalidRequestError, LLMProviderError)

    def test_exception_messages(self):
        """Test exception messages are preserved."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"

        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"


class MockProvider(AbstractLLMProvider):
    """Mock provider for testing base class functionality."""

    def generate(self, prompt, model=None, temperature=None, max_tokens=None, **kwargs):
        """Mock generate method."""
        return "Mock response"

    def generate_batch(self, prompts, model=None, temperature=None, max_tokens=None, **kwargs):
        """Mock generate_batch method."""
        return ["Mock response"] * len(prompts)

    def count_tokens(self, text, model=None):
        """Mock count_tokens method."""
        return len(text.split())


class TestAbstractLLMProvider:
    """Test AbstractLLMProvider base class."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = MockProvider(
            provider_name="test",
            api_key="test-key",
            default_model="test-model",
            temperature=0.5,
            max_tokens=1024,
            timeout=60,
            retry_attempts=5,
            rpm_limit=100,
        )

        assert provider.provider_name == "test"
        assert provider.api_key == "test-key"
        assert provider.default_model == "test-model"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024
        assert provider.timeout == 60
        assert provider.retry_attempts == 5
        assert provider.rpm_limit == 100
        assert provider.request_count == 0
        assert provider.error_count == 0
        assert provider.total_tokens == 0

    def test_parameter_validation_valid(self):
        """Test parameter validation with valid values."""
        provider = MockProvider("test", "key", "model")

        # Should not raise
        provider.validate_parameters(temperature=0.0, max_tokens=1)
        provider.validate_parameters(temperature=2.0, max_tokens=1000)
        provider.validate_parameters(temperature=1.0, max_tokens=500)

    def test_parameter_validation_invalid_temperature(self):
        """Test parameter validation with invalid temperature."""
        provider = MockProvider("test", "key", "model")

        with pytest.raises(InvalidRequestError) as exc_info:
            provider.validate_parameters(temperature=-0.1)
        assert "Temperature must be between" in str(exc_info.value)

        with pytest.raises(InvalidRequestError) as exc_info:
            provider.validate_parameters(temperature=2.5)
        assert "Temperature must be between" in str(exc_info.value)

    def test_parameter_validation_invalid_max_tokens(self):
        """Test parameter validation with invalid max_tokens."""
        provider = MockProvider("test", "key", "model")

        with pytest.raises(InvalidRequestError) as exc_info:
            provider.validate_parameters(max_tokens=0)
        assert "max_tokens must be positive" in str(exc_info.value)

        with pytest.raises(InvalidRequestError) as exc_info:
            provider.validate_parameters(max_tokens=-100)
        assert "max_tokens must be positive" in str(exc_info.value)

    def test_merge_kwargs(self):
        """Test merging kwargs with defaults."""
        provider = MockProvider(
            "test", "key", "default-model",
            temperature=0.7,
            max_tokens=2048
        )

        # No overrides
        params = provider._merge_kwargs()
        assert params["model"] == "default-model"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 2048

        # With overrides
        params = provider._merge_kwargs(
            model="custom-model",
            temperature=0.9,
            max_tokens=512
        )
        assert params["model"] == "custom-model"
        assert params["temperature"] == 0.9
        assert params["max_tokens"] == 512

        # Partial overrides
        params = provider._merge_kwargs(temperature=0.5)
        assert params["model"] == "default-model"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 2048


class TestProviderStatistics:
    """Test provider statistics tracking."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        provider = MockProvider("test", "key", "model")
        stats = provider.get_stats()

        assert stats["provider"] == "test"
        assert stats["request_count"] == 0
        assert stats["error_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["error_rate"] == 0.0

    def test_log_request(self):
        """Test logging requests updates count."""
        provider = MockProvider("test", "key", "model")

        provider._log_request("Test prompt", "model")
        assert provider.request_count == 1

        provider._log_request("Another prompt", "model")
        assert provider.request_count == 2

    def test_log_response(self):
        """Test logging responses updates token count."""
        provider = MockProvider("test", "key", "model")

        provider._log_response("Response", tokens_used=100)
        assert provider.total_tokens == 100

        provider._log_response("Another response", tokens_used=50)
        assert provider.total_tokens == 150

    def test_log_error(self):
        """Test logging errors updates error count."""
        provider = MockProvider("test", "key", "model")

        provider._log_error(Exception("Test error"))
        assert provider.error_count == 1

        provider._log_error(RateLimitError("Rate limit"))
        assert provider.error_count == 2

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        provider = MockProvider("test", "key", "model")

        provider._log_request("Test", "model")
        provider._log_request("Test", "model")
        provider._log_error(Exception("Error"))

        stats = provider.get_stats()
        assert stats["request_count"] == 2
        assert stats["error_count"] == 1
        assert stats["error_rate"] == 0.5

    def test_reset_stats(self):
        """Test resetting statistics."""
        provider = MockProvider("test", "key", "model")

        provider._log_request("Test", "model")
        provider._log_response("Response", 100)
        provider._log_error(Exception("Error"))

        assert provider.request_count == 1
        assert provider.total_tokens == 100
        assert provider.error_count == 1

        provider.reset_stats()

        assert provider.request_count == 0
        assert provider.total_tokens == 0
        assert provider.error_count == 0


class TestProviderLogging:
    """Test provider logging methods."""

    def test_log_request_with_long_prompt(self):
        """Test logging request truncates long prompts."""
        provider = MockProvider("test", "key", "model")

        long_prompt = "x" * 200
        provider._log_request(long_prompt, "model")

        assert provider.request_count == 1

    def test_log_response_with_long_response(self):
        """Test logging response truncates long responses."""
        provider = MockProvider("test", "key", "model")

        long_response = "x" * 200
        provider._log_response(long_response, tokens_used=100)

        assert provider.total_tokens == 100


class TestProviderRepresentation:
    """Test provider string representation."""

    def test_repr(self):
        """Test provider __repr__ method."""
        provider = MockProvider("test", "key", "test-model", rpm_limit=100)
        repr_str = repr(provider)

        assert "MockProvider" in repr_str
        assert "provider=test" in repr_str
        assert "model=test-model" in repr_str
        assert "rpm_limit=100" in repr_str
