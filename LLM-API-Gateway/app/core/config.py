"""
Application configuration for the LLM API Gateway.

Settings are loaded from environment variables and the local .env file.
Sensitive values are represented with SecretStr so they are not exposed
through ordinary string representations.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General application settings.
    app_name: str = Field(
        default="LLM API Gateway",
        min_length=1,
    )

    app_version: str = Field(
        default="0.1.0",
        min_length=1,
    )

    app_environment: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    # Authentication settings.
    gateway_api_key: SecretStr

    # LLM provider settings.
    llm_provider: Literal["ollama"] = "ollama"

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
    )

    ollama_model: str = Field(
        min_length=1,
    )

    # Redis and cache settings.
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        min_length=1,
    )

    cache_ttl_seconds: int = Field(
        default=3_600,
        ge=1,
        le=86_400,
    )

    # HTTP timeout settings.
    connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=60,
    )

    read_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
    )

    write_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=600,
    )

    pool_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=60,
    )

    # Reliability and usage-control settings.
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    retry_max_wait_seconds: float = Field(
        default=8.0,
        gt=0,
        le=60,
    )

    rate_limit_per_minute: int = Field(
        default=30,
        ge=1,
        le=10_000,
    )


@field_validator("gateway_api_key")
@classmethod
def validate_gateway_api_key(cls, value):
    """
    Ensures that the configured gateway API key is not trivially short.

    Args:
        value: Secret API key loaded from the environment.

    Returns:
        Validated SecretStr value.
    """
    if len(value.get_secret_value()) < 16:
        raise ValueError("GATEWAY_API_KEY must contain at least 16 characters.")

    return value


@field_validator("ollama_base_url")
@classmethod
def normalize_ollama_base_url(cls, value):
    """
    Removes a trailing slash from the Ollama base URL.

    Args:
        value: Configured Ollama URL.

    Returns:
        Normalized URL.
    """
    return value.rstrip("/")


@lru_cache
def get_settings():
    """
    Loads and caches the application settings.

    Returns:
        Validated Settings instance.
    """
    return Settings()
