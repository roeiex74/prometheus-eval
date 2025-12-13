"""
Tests for inference configuration data model.

Tests the Pydantic InferenceConfig model including validation,
default values, and provider-specific configuration.
"""

import pytest
from pydantic import ValidationError
from src.inference.config_model import InferenceConfig


class TestInferenceConfigValidation:
    """Test InferenceConfig field validation."""

    def test_valid_config_with_defaults(self, sample_config_dict):
        """Test creating config with valid data."""
        config = InferenceConfig(**sample_config_dict)
        assert config.openai_api_key == "test-openai-key"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048

    def test_minimal_config(self):
        """Test config with only required fields (all are optional)."""
        config = InferenceConfig()
        assert config.default_openai_model == "gpt-4-turbo-preview"
        assert config.default_anthropic_model == "claude-3-sonnet-20240229"
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048

    def test_temperature_validation_valid(self):
        """Test temperature within valid range."""
        config = InferenceConfig(default_temperature=0.0)
        assert config.default_temperature == 0.0

        config = InferenceConfig(default_temperature=2.0)
        assert config.default_temperature == 2.0

        config = InferenceConfig(default_temperature=1.0)
        assert config.default_temperature == 1.0

    def test_temperature_validation_invalid(self):
        """Test temperature outside valid range raises error."""
        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(default_temperature=-0.1)
        assert "default_temperature" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(default_temperature=2.5)
        assert "default_temperature" in str(exc_info.value)

    def test_max_tokens_validation(self):
        """Test max_tokens must be positive."""
        config = InferenceConfig(default_max_tokens=1)
        assert config.default_max_tokens == 1

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(default_max_tokens=0)
        assert "default_max_tokens" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(default_max_tokens=-100)
        assert "default_max_tokens" in str(exc_info.value)

    def test_rpm_limits_validation(self):
        """Test RPM limits must be positive."""
        config = InferenceConfig(openai_rpm_limit=100, anthropic_rpm_limit=50)
        assert config.openai_rpm_limit == 100
        assert config.anthropic_rpm_limit == 50

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(openai_rpm_limit=0)
        assert "openai_rpm_limit" in str(exc_info.value)

    def test_retry_attempts_validation(self):
        """Test retry attempts within valid range."""
        config = InferenceConfig(llm_retry_attempts=1)
        assert config.llm_retry_attempts == 1

        config = InferenceConfig(llm_retry_attempts=10)
        assert config.llm_retry_attempts == 10

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(llm_retry_attempts=0)
        assert "llm_retry_attempts" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(llm_retry_attempts=11)
        assert "llm_retry_attempts" in str(exc_info.value)


class TestLogLevelValidation:
    """Test log level validation."""

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = InferenceConfig(log_level=level)
            assert config.log_level == level

    def test_log_level_case_insensitive(self):
        """Test log level is case insensitive."""
        config = InferenceConfig(log_level="info")
        assert config.log_level == "INFO"

        config = InferenceConfig(log_level="DeBuG")
        assert config.log_level == "DEBUG"

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValidationError) as exc_info:
            InferenceConfig(log_level="INVALID")
        assert "Log level must be one of" in str(exc_info.value)


class TestProviderConfig:
    """Test provider-specific configuration retrieval."""

    def test_get_openai_config(self, sample_config_dict):
        """Test retrieving OpenAI configuration."""
        config = InferenceConfig(**sample_config_dict)
        openai_config = config.get_provider_config("openai")

        assert openai_config["api_key"] == "test-openai-key"
        assert openai_config["default_model"] == "gpt-4-turbo-preview"
        assert openai_config["rpm_limit"] == 60
        assert openai_config["timeout"] == 30
        assert openai_config["retry_attempts"] == 3

    def test_get_anthropic_config(self, sample_config_dict):
        """Test retrieving Anthropic configuration."""
        config = InferenceConfig(**sample_config_dict)
        anthropic_config = config.get_provider_config("anthropic")

        assert anthropic_config["api_key"] == "test-anthropic-key"
        assert anthropic_config["default_model"] == "claude-3-sonnet-20240229"
        assert anthropic_config["rpm_limit"] == 50
        assert anthropic_config["timeout"] == 30
        assert anthropic_config["retry_attempts"] == 3

    def test_get_provider_config_case_insensitive(self, sample_config_dict):
        """Test provider name is case insensitive."""
        config = InferenceConfig(**sample_config_dict)

        openai_config = config.get_provider_config("OpenAI")
        assert openai_config["api_key"] == "test-openai-key"

        anthropic_config = config.get_provider_config("ANTHROPIC")
        assert anthropic_config["api_key"] == "test-anthropic-key"

    def test_unsupported_provider(self, sample_config_dict):
        """Test unsupported provider raises error."""
        config = InferenceConfig(**sample_config_dict)

        with pytest.raises(ValueError) as exc_info:
            config.get_provider_config("unsupported")
        assert "Unsupported provider" in str(exc_info.value)


class TestProviderCredentials:
    """Test provider credential validation."""

    def test_validate_openai_credentials_valid(self):
        """Test validating OpenAI credentials."""
        config = InferenceConfig(openai_api_key="sk-test-key")
        assert config.validate_provider_credentials("openai") is True

    def test_validate_openai_credentials_missing(self):
        """Test validating missing OpenAI credentials."""
        config = InferenceConfig(openai_api_key=None)
        assert config.validate_provider_credentials("openai") is False

        config = InferenceConfig(openai_api_key="")
        assert config.validate_provider_credentials("openai") is False

    def test_validate_anthropic_credentials_valid(self):
        """Test validating Anthropic credentials."""
        config = InferenceConfig(anthropic_api_key="sk-ant-test-key")
        assert config.validate_provider_credentials("anthropic") is True

    def test_validate_anthropic_credentials_missing(self):
        """Test validating missing Anthropic credentials."""
        config = InferenceConfig(anthropic_api_key=None)
        assert config.validate_provider_credentials("anthropic") is False

    def test_validate_unsupported_provider(self):
        """Test validating unsupported provider returns False."""
        config = InferenceConfig()
        assert config.validate_provider_credentials("unsupported") is False
