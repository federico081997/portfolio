"""
Main application module for the LLM API Gateway.

This module creates the FastAPI application and registers the API routes.
"""

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings

# Load and validate application configuration
settings = get_settings()

# Create the central FastAPI application object
app = FastAPI(
    title=settings.app_name,
    description=(
        "A provider-independent gateway for text generation"
        "and structured information extraction."
    ),
    version=settings.app_version,
)

# Register the endpoints depined by app/api/routes.py
app.include_router(router)
