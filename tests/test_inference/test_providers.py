"""
Tests for OpenAI and Anthropic LLM providers.

Tests provider implementations with mocked API responses.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio
from src.inference.openai_provider import OpenAIProvider, create_openai_provider
from src.inference.anthropic_provider import AnthropicProvider, create_anthropic_provider
from src.inference.base import (
    RateLimitError,
    AuthenticationError,
    TimeoutError,
    InvalidRequestError,
    LLMProviderError,
)
from src.inference.config_model import InferenceConfig


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_initialization(self, mock_async_client, mock_sync_client):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(
            api_key="sk-test-key",
            default_model="gpt-4",
            org_id="org-test",
            temperature=0.5,
            max_tokens=1024,
        )

        assert provider.provider_name == "openai"
        assert provider.api_key == "sk-test-key"
        assert provider.default_model == "gpt-4"
        assert provider.org_id == "org-test"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024

    @patch('src.inference.openai_provider.tiktoken')
    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_count_tokens(self, mock_async, mock_sync, mock_tiktoken):
        """Test token counting with tiktoken."""
        # Mock encoding
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        provider = OpenAIProvider(api_key="sk-test", default_model="gpt-4")
        count = provider.count_tokens("Test text")

        assert count == 5
        mock_encoding.encode.assert_called_once_with("Test text")

    @patch('src.inference.openai_provider.tiktoken')
    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_count_tokens_caching(self, mock_async, mock_sync, mock_tiktoken):
        """Test that encodings are cached."""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        provider = OpenAIProvider(api_key="sk-test", default_model="gpt-4")

        provider.count_tokens("Text 1")
        provider.count_tokens("Text 2")

        # Should only call encoding_for_model once due to caching
        assert mock_tiktoken.encoding_for_model.call_count == 1

    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_generate_sync(self, mock_async_client_class, mock_sync_client_class, mock_openai_response):
        """Test synchronous generation."""
        # Mock the async client
        mock_async_instance = Mock()
        mock_async_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_async_client_class.return_value = mock_async_instance

        provider = OpenAIProvider(api_key="sk-test", default_model="gpt-4")

        # Patch asyncio.run to handle the async call
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = "Test response from OpenAI"
            response = provider.generate("Test prompt")

            assert response == "Test response from OpenAI"
            assert provider.request_count == 0  # Not incremented yet due to mocking

    @patch('src.inference.openai_provider.openai')
    def test_handle_openai_errors(self, mock_openai_module):
        """Test error mapping from OpenAI exceptions."""
        # Setup mock error classes
        mock_openai_module.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai_module.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai_module.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai_module.BadRequestError = type('BadRequestError', (Exception,), {})

        with patch('src.inference.openai_provider.OpenAI'), \
             patch('src.inference.openai_provider.AsyncOpenAI'):
            provider = OpenAIProvider(api_key="sk-test")

            # Test rate limit error
            error = mock_openai_module.RateLimitError("Rate limit")
            mapped = provider._handle_openai_error(error)
            assert isinstance(mapped, RateLimitError)

            # Test auth error
            error = mock_openai_module.AuthenticationError("Invalid key")
            mapped = provider._handle_openai_error(error)
            assert isinstance(mapped, AuthenticationError)

            # Test timeout error
            error = mock_openai_module.APITimeoutError("Timeout")
            mapped = provider._handle_openai_error(error)
            assert isinstance(mapped, TimeoutError)

            # Test bad request error
            error = mock_openai_module.BadRequestError("Bad request")
            mapped = provider._handle_openai_error(error)
            assert isinstance(mapped, InvalidRequestError)


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_initialization(self, mock_async_client, mock_sync_client):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            default_model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=1024,
        )

        assert provider.provider_name == "anthropic"
        assert provider.api_key == "sk-ant-test"
        assert provider.default_model == "claude-3-opus-20240229"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_count_tokens_approximation(self, mock_async, mock_sync):
        """Test approximate token counting."""
        mock_sync_instance = Mock()
        mock_sync_instance.count_tokens.side_effect = AttributeError("Not available")
        mock_sync.return_value = mock_sync_instance

        provider = AnthropicProvider(api_key="sk-ant-test")

        # Test with string of known length
        # Should use approximation: len(text) // CHARS_PER_TOKEN
        text = "x" * 40  # 40 chars / 4 chars_per_token = 10 tokens
        count = provider.count_tokens(text)

        assert count == 10

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_count_tokens_with_sdk(self, mock_async, mock_sync):
        """Test token counting using Anthropic SDK method."""
        mock_sync_instance = Mock()
        mock_sync_instance.count_tokens.return_value = 42
        mock_sync.return_value = mock_sync_instance

        provider = AnthropicProvider(api_key="sk-ant-test")
        count = provider.count_tokens("Test text")

        assert count == 42

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_generate_sync(self, mock_async_client_class, mock_sync_client_class, mock_anthropic_response):
        """Test synchronous generation."""
        mock_async_instance = Mock()
        mock_async_instance.messages.create = AsyncMock(return_value=mock_anthropic_response)
        mock_async_client_class.return_value = mock_async_instance

        provider = AnthropicProvider(api_key="sk-ant-test")

        with patch('asyncio.run') as mock_run:
            mock_run.return_value = "Test response from Anthropic"
            response = provider.generate("Test prompt")

            assert response == "Test response from Anthropic"

    @patch('src.inference.anthropic_provider.anthropic')
    def test_handle_anthropic_errors(self, mock_anthropic_module):
        """Test error mapping from Anthropic exceptions."""
        # Setup mock error classes
        mock_anthropic_module.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_anthropic_module.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_anthropic_module.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_anthropic_module.BadRequestError = type('BadRequestError', (Exception,), {})

        with patch('src.inference.anthropic_provider.Anthropic'), \
             patch('src.inference.anthropic_provider.AsyncAnthropic'):
            provider = AnthropicProvider(api_key="sk-ant-test")

            # Test rate limit error
            error = mock_anthropic_module.RateLimitError("Rate limit")
            mapped = provider._handle_anthropic_error(error)
            assert isinstance(mapped, RateLimitError)

            # Test auth error
            error = mock_anthropic_module.AuthenticationError("Invalid key")
            mapped = provider._handle_anthropic_error(error)
            assert isinstance(mapped, AuthenticationError)

            # Test timeout error
            error = mock_anthropic_module.APITimeoutError("Timeout")
            mapped = provider._handle_anthropic_error(error)
            assert isinstance(mapped, TimeoutError)


class TestProviderFactories:
    """Test provider factory functions."""

    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_create_openai_provider_with_config(self, mock_async, mock_sync):
        """Test creating OpenAI provider from config."""
        config = InferenceConfig(
            openai_api_key="sk-test-key",
            openai_org_id="org-test",
            default_openai_model="gpt-4",
            default_temperature=0.5,
            default_max_tokens=1024,
        )

        provider = create_openai_provider(config)

        assert provider.api_key == "sk-test-key"
        assert provider.org_id == "org-test"
        assert provider.default_model == "gpt-4"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024

    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_create_openai_provider_missing_key(self, mock_async, mock_sync):
        """Test creating OpenAI provider without API key raises error."""
        config = InferenceConfig(openai_api_key=None)

        with pytest.raises(ValueError) as exc_info:
            create_openai_provider(config)
        assert "OpenAI API key not found" in str(exc_info.value)

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_create_anthropic_provider_with_config(self, mock_async, mock_sync):
        """Test creating Anthropic provider from config."""
        config = InferenceConfig(
            anthropic_api_key="sk-ant-test",
            default_anthropic_model="claude-3-opus-20240229",
            default_temperature=0.5,
            default_max_tokens=1024,
        )

        provider = create_anthropic_provider(config)

        assert provider.api_key == "sk-ant-test"
        assert provider.default_model == "claude-3-opus-20240229"
        assert provider.temperature == 0.5
        assert provider.max_tokens == 1024

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_create_anthropic_provider_missing_key(self, mock_async, mock_sync):
        """Test creating Anthropic provider without API key raises error."""
        config = InferenceConfig(anthropic_api_key=None)

        with pytest.raises(ValueError) as exc_info:
            create_anthropic_provider(config)
        assert "Anthropic API key not found" in str(exc_info.value)


class TestOpenAIBatchGeneration:
    """Test OpenAI batch generation functionality."""

    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_generate_batch(self, mock_async_client_class, mock_sync_client_class):
        """Test batch generation."""
        provider = OpenAIProvider(api_key="sk-test", default_model="gpt-4")

        with patch('asyncio.run') as mock_run:
            mock_run.return_value = ["Response 1", "Response 2", "Response 3"]
            responses = provider.generate_batch(["Prompt 1", "Prompt 2", "Prompt 3"])

            assert len(responses) == 3
            assert responses == ["Response 1", "Response 2", "Response 3"]


class TestAnthropicBatchGeneration:
    """Test Anthropic batch generation functionality."""

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_generate_batch(self, mock_async_client_class, mock_sync_client_class):
        """Test batch generation."""
        provider = AnthropicProvider(api_key="sk-ant-test", default_model="claude-3")

        with patch('asyncio.run') as mock_run:
            mock_run.return_value = ["Response 1", "Response 2"]
            responses = provider.generate_batch(["Prompt 1", "Prompt 2"])

            assert len(responses) == 2
            assert responses == ["Response 1", "Response 2"]


class TestOpenAIEdgeCases:
    """Test OpenAI provider edge cases."""

    @patch('src.inference.openai_provider.tiktoken')
    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_count_tokens_unknown_model(self, mock_async, mock_sync, mock_tiktoken):
        """Test token counting with unknown model uses fallback."""
        # Simulate unknown model
        mock_tiktoken.encoding_for_model.side_effect = KeyError("Unknown model")
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.get_encoding.return_value = mock_encoding

        provider = OpenAIProvider(api_key="sk-test", default_model="unknown-model")
        count = provider.count_tokens("Test")

        assert count == 3
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    @patch('src.inference.openai_provider.tiktoken')
    @patch('src.inference.openai_provider.OpenAI')
    @patch('src.inference.openai_provider.AsyncOpenAI')
    def test_count_tokens_error_handling(self, mock_async, mock_sync, mock_tiktoken):
        """Test token counting error handling."""
        mock_tiktoken.encoding_for_model.side_effect = Exception("Unexpected error")

        provider = OpenAIProvider(api_key="sk-test")

        with pytest.raises(LLMProviderError) as exc_info:
            provider.count_tokens("Test")
        assert "Token counting failed" in str(exc_info.value)


class TestAnthropicEdgeCases:
    """Test Anthropic provider edge cases."""

    @patch('src.inference.anthropic_provider.Anthropic')
    @patch('src.inference.anthropic_provider.AsyncAnthropic')
    def test_count_tokens_error_fallback(self, mock_async, mock_sync):
        """Test token counting falls back to approximation on error."""
        mock_sync_instance = Mock()
        mock_sync_instance.count_tokens.side_effect = Exception("API error")
        mock_sync.return_value = mock_sync_instance

        provider = AnthropicProvider(api_key="sk-ant-test")

        # Should fall back to approximation
        text = "x" * 20  # 20 chars / 4 = 5 tokens
        count = provider.count_tokens(text)

        assert count == 5
