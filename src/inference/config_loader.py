"""
Configuration Loading Utilities.

Handles loading configuration from environment variables and .env files.
"""

from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

from .config_model import InferenceConfig


def load_config(env_file: Optional[str] = None) -> InferenceConfig:
    """
    Load inference configuration from environment variables.

    Args:
        env_file: Optional path to .env file

    Returns:
        Validated InferenceConfig instance
    """
    # Determine .env file path
    if env_file is None:
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
        "openai_org_id": os.getenv("OPENAI_ORG_ID") if os.getenv("OPENAI_ORG_ID") != "your_openai_org_id_here" else None,
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
    """Get the global configuration instance (singleton pattern)."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _global_config
    _global_config = None
