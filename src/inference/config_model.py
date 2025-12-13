"""
Inference Configuration Data Model.

Defines the pydantic model for LLM inference configuration.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class InferenceConfig(BaseModel):
    """
    Configuration for LLM inference engine.

    Loads settings from environment variables and provides validation
    for API keys, model defaults, rate limiting, and retry settings.
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
        extra = "allow"

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
        """Get configuration specific to a provider."""
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
        """Validate that credentials exist for a specific provider."""
        provider_lower = provider.lower()

        if provider_lower == "openai":
            return self.openai_api_key is not None and len(self.openai_api_key) > 0
        elif provider_lower == "anthropic":
            return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0
        else:
            return False
