"""
HTTP routes for the LLM API Gateway.

This module currently contains only the root and health-check endpoints.
Additional generation and extraction routes will be added later.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.schemas.common import APIInfoResponse, HealthResponse, ServiceStatus
from app.schemas.extraction import (
    ExtractionRequest,
    ExtractionResponse,
)
from app.schemas.generation import GenerateRequest, GenerateResponse

# API Router groups related endpoints so they can be registered
# with the main FastAPI application.
router = APIRouter()

# Reusable FastAPI dependency for obtaining application settings
SettingsDependency = Annotated[Settings, Depends(get_settings)]


@router.get("/", response_model=APIInfoResponse, tags=["System"])
async def read_root(
    settings: SettingsDependency,
):
    """
    Returns basic information about the API

    Returns:
        Dictionary containing the service name and documentation path.
    """
    return APIInfoResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_environment,
        message="The API is running.",
        documentation="/docs",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check(
    settings: SettingsDependency,
):
    """
    Returns the current application health status.

    Redis and Ollama checks will be added later.

    Args:
        settings: Validated application configuration.

    Returns:
        Current gateway health information.
    """
    return HealthResponse(
        status=ServiceStatus.HEALTHY,
        service=settings.app_name,
        version=settings.app_version,
        checks={
            "api": "available",
        },
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    tags=["Generation"],
)
async def generate_text(
    request: GenerateRequest,
):
    """
    Validates a text-generation request.

    The provider implementation will be added in Hour 3.

    Args:
        request: Validated generation request.

    Raises:
        HTTPException: Always raised until the provider is implemented.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Text generation is not implemented yet. "
            "The Ollama provider will be added in Hour 3."
        ),
    )


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Extraction"],
)
async def extract_information(
    request: ExtractionRequest,
):
    """
    Validates a structured extraction request.

    The provider implementation will be added in Hour 3.

    Args:
        request: Validated extraction request.

    Raises:
        HTTPException: Always raised until the provider is implemented.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Structured extraction is not implemented yet. "
            "The Ollama provider will be added in Hour 3."
        ),
    )
