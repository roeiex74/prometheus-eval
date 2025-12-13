"""
Tests for configuration loading utilities.

Tests loading configuration from .env files and environment variables.
"""

import pytest
import os
from pathlib import Path
from src.inference.config_loader import load_config, get_config, reset_config


class TestLoadConfig:
    """Test configuration loading from .env files."""

    def test_load_config_from_env_file(self, temp_env_file):
        """Test loading configuration from .env file."""
        config = load_config(env_file=temp_env_file)

        assert config.openai_api_key == "test-openai-key"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.default_temperature == 0.5
        assert config.default_max_tokens == 1024
        assert config.openai_rpm_limit == 100
        assert config.log_level == "DEBUG"

    def test_load_config_from_env_vars(self, mock_env_vars):
        """Test loading configuration from environment variables."""
        # Clear any existing .env file path
        config = load_config(env_file="/nonexistent/.env")

        assert config.openai_api_key == "sk-test-openai-key"
        assert config.anthropic_api_key == "sk-ant-test-anthropic-key"
        assert config.default_temperature == 0.8
        assert config.default_max_tokens == 512
        assert config.openai_rpm_limit == 120
        assert config.anthropic_rpm_limit == 80

    def test_load_config_missing_env_file(self, monkeypatch):
        """Test loading config when .env file is missing."""
        # Clear all env vars to ensure clean state
        for key in ["DEFAULT_TEMPERATURE", "DEFAULT_MAX_TOKENS", "OPENAI_RPM_LIMIT", "LOG_LEVEL"]:
            monkeypatch.delenv(key, raising=False)

        # Set some env vars so config can be created
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Should not raise error, just use defaults
        config = load_config(env_file="/nonexistent/.env")

        # Should use default values
        assert config.default_openai_model == "gpt-4-turbo-preview"
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048

    def test_load_config_with_defaults(self, temp_env_file):
        """Test that defaults are applied for missing values."""
        config = load_config(env_file=temp_env_file)

        # These should use defaults since not in temp_env_file
        assert config.default_openai_model == "gpt-4-turbo-preview"
        assert config.default_anthropic_model == "claude-3-sonnet-20240229"

    def test_load_config_type_conversion(self, monkeypatch):
        """Test that environment variables are properly converted to types."""
        monkeypatch.setenv("DEFAULT_TEMPERATURE", "0.9")
        monkeypatch.setenv("DEFAULT_MAX_TOKENS", "4096")
        monkeypatch.setenv("OPENAI_RPM_LIMIT", "200")
        monkeypatch.setenv("ENABLE_CACHE", "false")

        config = load_config(env_file="/nonexistent/.env")

        assert isinstance(config.default_temperature, float)
        assert config.default_temperature == 0.9
        assert isinstance(config.default_max_tokens, int)
        assert config.default_max_tokens == 4096
        assert isinstance(config.openai_rpm_limit, int)
        assert config.openai_rpm_limit == 200
        assert isinstance(config.enable_cache, bool)
        assert config.enable_cache is False

    def test_load_config_boolean_conversion(self, monkeypatch):
        """Test boolean environment variable conversion."""
        # Test various truthy values
        for value in ["true", "True", "1", "yes"]:
            monkeypatch.setenv("ENABLE_CACHE", value)
            config = load_config(env_file="/nonexistent/.env")
            assert config.enable_cache is True

        # Test falsy values
        for value in ["false", "False", "0", "no"]:
            monkeypatch.setenv("ENABLE_CACHE", value)
            config = load_config(env_file="/nonexistent/.env")
            assert config.enable_cache is False


class TestGetConfig:
    """Test global configuration singleton."""

    def test_get_config_singleton(self, mock_env_vars):
        """Test that get_config returns same instance."""
        reset_config()  # Ensure clean state

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_loads_from_env(self, mock_env_vars):
        """Test that get_config loads from environment."""
        reset_config()  # Ensure clean state

        config = get_config()

        assert config.openai_api_key == "sk-test-openai-key"
        assert config.default_temperature == 0.8

    def test_reset_config(self, mock_env_vars):
        """Test resetting global configuration."""
        reset_config()

        config1 = get_config()
        assert config1.openai_api_key == "sk-test-openai-key"

        reset_config()

        # After reset, get_config should create new instance
        config2 = get_config()
        assert config2 is not config1
        assert config2.openai_api_key == "sk-test-openai-key"


class TestConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_all_env_vars(self, monkeypatch):
        """Test config creation with no environment variables."""
        # Clear all relevant env vars
        all_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEFAULT_TEMPERATURE",
                    "DEFAULT_MAX_TOKENS", "OPENAI_RPM_LIMIT", "LOG_LEVEL"]
        for key in all_keys:
            monkeypatch.delenv(key, raising=False)

        config = load_config(env_file="/nonexistent/.env")

        # Should still create config with defaults
        assert config.default_temperature == 0.7
        assert config.default_max_tokens == 2048
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None

    def test_invalid_temperature_from_env(self, monkeypatch):
        """Test that invalid temperature from env raises validation error."""
        monkeypatch.setenv("DEFAULT_TEMPERATURE", "3.0")

        with pytest.raises(Exception):  # Should raise validation error
            load_config(env_file="/nonexistent/.env")

    def test_invalid_max_tokens_from_env(self, monkeypatch):
        """Test that invalid max_tokens from env raises validation error."""
        monkeypatch.setenv("DEFAULT_MAX_TOKENS", "-100")

        with pytest.raises(Exception):  # Should raise validation error
            load_config(env_file="/nonexistent/.env")
