"""
Shared fixtures for inference engine tests.

Provides mock objects and test utilities for testing LLM providers.
"""

from typing import Dict, Any
from unittest.mock import Mock, MagicMock
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    response = Mock()
    response.choices = [Mock(message=Mock(content="Test response from OpenAI"))]
    response.usage = Mock(total_tokens=42, prompt_tokens=10, completion_tokens=32)
    return response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client for testing."""
    client = Mock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response."""
    response = Mock()
    # Anthropic returns content blocks
    text_block = Mock()
    text_block.text = "Test response from Anthropic"
    response.content = [text_block]
    response.usage = Mock(input_tokens=10, output_tokens=32)
    return response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """Mock Anthropic client for testing."""
    client = Mock()
    client.messages.create.return_value = mock_anthropic_response
    # Add count_tokens method that returns a simple count
    client.count_tokens.return_value = 10
    return client


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("OPENAI_API_KEY=test-openai-key\n")
        f.write("ANTHROPIC_API_KEY=test-anthropic-key\n")
        f.write("DEFAULT_TEMPERATURE=0.5\n")
        f.write("DEFAULT_MAX_TOKENS=1024\n")
        f.write("OPENAI_RPM_LIMIT=100\n")
        f.write("LOG_LEVEL=DEBUG\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "sk-test-openai-key",
        "ANTHROPIC_API_KEY": "sk-ant-test-anthropic-key",
        "DEFAULT_TEMPERATURE": "0.8",
        "DEFAULT_MAX_TOKENS": "512",
        "OPENAI_RPM_LIMIT": "120",
        "ANTHROPIC_RPM_LIMIT": "80",
        "LOG_LEVEL": "INFO",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "openai_api_key": "test-openai-key",
        "anthropic_api_key": "test-anthropic-key",
        "default_openai_model": "gpt-4-turbo-preview",
        "default_anthropic_model": "claude-3-sonnet-20240229",
        "default_temperature": 0.7,
        "default_max_tokens": 2048,
        "openai_rpm_limit": 60,
        "anthropic_rpm_limit": 50,
        "llm_request_timeout": 30,
        "llm_retry_attempts": 3,
        "log_level": "INFO",
        "enable_cache": True,
        "cache_dir": "./.cache",
    }
