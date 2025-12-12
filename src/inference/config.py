"""
Inference Engine Configuration Management.

This module handles configuration loading and validation for LLM providers
using pydantic for type safety and python-dotenv for environment management.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import os
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from loguru import logger


class InferenceConfig(BaseModel):
    """
    Configuration for LLM inference engine.

    Loads settings from environment variables and provides validation
    for API keys, model defaults, rate limiting, and retry settings.

    Attributes:
        openai_api_key: OpenAI API key for authentication
        openai_org_id: Optional OpenAI organization ID
        anthropic_api_key: Anthropic API key for authentication
        huggingface_api_token: Optional HuggingFace API token
        default_openai_model: Default OpenAI model to use
        default_anthropic_model: Default Anthropic model to use
        default_temperature: Default temperature for generation (0.0-2.0)
        default_max_tokens: Default maximum tokens to generate
        openai_rpm_limit: OpenAI requests per minute limit
        anthropic_rpm_limit: Anthropic requests per minute limit
        llm_request_timeout: Timeout in seconds for LLM API requests
        llm_retry_attempts: Number of retry attempts for failed requests
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_cache: Whether to enable response caching
        cache_dir: Directory for caching LLM responses
    """

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_org_id: Optional[str] = Field(default=None, description="OpenAI organization ID")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    huggingface_api_token: Optional[str] = Field(default=None, description="HuggingFace API token")

    # Default Model Settings
    default_openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Default OpenAI model"
    )
    default_anthropic_model: str = Field(
        default="claude-3-sonnet-20240229",
        description="Default Anthropic model"
    )
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for generation"
    )
    default_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Default maximum tokens to generate"
    )

    # Rate Limiting Settings
    openai_rpm_limit: int = Field(
        default=60,
        gt=0,
        description="OpenAI requests per minute limit"
    )
    anthropic_rpm_limit: int = Field(
        default=50,
        gt=0,
        description="Anthropic requests per minute limit"
    )

    # Timeout and Retry Settings
    llm_request_timeout: int = Field(
        default=30,
        gt=0,
        description="Timeout in seconds for LLM requests"
    )
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed requests"
    )

    # Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Caching Settings
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_dir: str = Field(
        default="./.cache",
        description="Directory for caching responses"
    )

    class Config:
        """Pydantic model configuration."""
        env_file = ".env"
        case_sensitive = False
        extra = "allow"  # Allow extra fields for future extensibility

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the allowed values."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v_upper

    @validator("default_temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration specific to a provider.

        Args:
            provider: Provider name ('openai' or 'anthropic')

        Returns:
            Dictionary containing provider-specific configuration

        Raises:
            ValueError: If provider is not supported
        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return {
                "api_key": self.openai_api_key,
                "org_id": self.openai_org_id,
                "default_model": self.default_openai_model,
                "rpm_limit": self.openai_rpm_limit,
                "timeout": self.llm_request_timeout,
                "retry_attempts": self.llm_retry_attempts,
            }
        elif provider_lower == "anthropic":
            return {
                "api_key": self.anthropic_api_key,
                "default_model": self.default_anthropic_model,
                "rpm_limit": self.anthropic_rpm_limit,
                "timeout": self.llm_request_timeout,
                "retry_attempts": self.llm_retry_attempts,
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def validate_provider_credentials(self, provider: str) -> bool:
        """
        Validate that credentials exist for a specific provider.

        Args:
            provider: Provider name ('openai' or 'anthropic')

        Returns:
            True if credentials are present, False otherwise
        """
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return self.openai_api_key is not None and len(self.openai_api_key) > 0
        elif provider_lower == "anthropic":
            return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0
        else:
            return False


def load_config(env_file: Optional[str] = None) -> InferenceConfig:
    """
    Load inference configuration from environment variables.

    Args:
        env_file: Optional path to .env file. If not provided, looks for .env in current directory.

    Returns:
        Validated InferenceConfig instance

    Example:
        >>> config = load_config()
        >>> print(config.default_openai_model)
        gpt-4-turbo-preview
    """
    # Determine .env file path
    if env_file is None:
        # Look for .env in current directory and parent directories
        current_dir = Path.cwd()
        env_path = current_dir / ".env"

        # Search up to 3 parent directories
        if not env_path.exists():
            for _ in range(3):
                current_dir = current_dir.parent
                env_path = current_dir / ".env"
                if env_path.exists():
                    break
    else:
        env_path = Path(env_file)

    # Load .env file if it exists
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.debug(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}, using environment variables only")

    # Build config from environment variables
    config_data = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_org_id": os.getenv("OPENAI_ORG_ID"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "huggingface_api_token": os.getenv("HUGGINGFACE_API_TOKEN"),
        "default_openai_model": os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4-turbo-preview"),
        "default_anthropic_model": os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
        "default_temperature": float(os.getenv("DEFAULT_TEMPERATURE", "0.7")),
        "default_max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", "2048")),
        "openai_rpm_limit": int(os.getenv("OPENAI_RPM_LIMIT", "60")),
        "anthropic_rpm_limit": int(os.getenv("ANTHROPIC_RPM_LIMIT", "50")),
        "llm_request_timeout": int(os.getenv("LLM_REQUEST_TIMEOUT", "30")),
        "llm_retry_attempts": int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "enable_cache": os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes"),
        "cache_dir": os.getenv("CACHE_DIR", "./.cache"),
    }

    config = InferenceConfig(**config_data)
    logger.info(f"Loaded inference configuration with log level: {config.log_level}")

    return config


# Global config instance (lazy-loaded)
_global_config: Optional[InferenceConfig] = None


def get_config() -> InferenceConfig:
    """
    Get the global configuration instance (singleton pattern).

    Returns:
        Global InferenceConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reset_config() -> None:
    """
    Reset the global configuration instance.

    Useful for testing or when environment variables change.
    """
    global _global_config
    _global_config = None
