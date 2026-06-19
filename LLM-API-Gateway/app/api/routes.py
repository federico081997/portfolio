"""
HTTP routes for the LLM API Gateway.

This module currently contains only the root and health-check endpoints.
Additional generation and extraction routes will be added later.
"""

from fastapi import APIRouter

# API ROuter groups related endpoints so they can be registered
# with the main FastAPI application.
router = APIRouter()


@router.get("/")
async def read_root():
    """
    Returns basic information about the API

    Returns:
        Dictionary containing the service name and documentation path.
    """
    return {
        "service": "LLM API Gateway",
        "message": "The API is running.",
        "documentation": "/docs",
    }


@router.get("/health")
async def health_check():
    """
    Returns the current application health status.

    This is only a placeholder health check. Redis, Ollama, and model
    availablility checks will be added later.

    Returns:
        Dictionary describing the current API status.
    """
    return {
        "status": "healthy",
        "service": "llm-api-gateway",
        "checks": {
            "api": "available",
        },
    }
