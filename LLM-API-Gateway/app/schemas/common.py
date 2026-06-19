"""
Shared Pydantic models for the LLM API Gateway.

These models define behaviour and response structures that are reused
across multiple API endpoints.
"""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """
    Base class for API request and response models.

    Unknown fields are rejected and surrounding whitespace is removed
    from string values before validation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ServiceStatus(str, Enum):
    """Supported overall health states for the application."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class APIInfoResponse(APIModel):
    """Response returned by the root endpoint."""

    service: str
    version: str
    environment: str
    message: str
    documentation: str


class HealthResponse(APIModel):
    """Response returned by the health-check endpoint."""

    status: ServiceStatus
    service: str
    version: str
    checks: dict[str, str] = Field(default_factory=dict)


class ErrorResponse(APIModel):
    """Standard error response used by the gateway."""

    detail: str
    request_id: UUID | None = None
