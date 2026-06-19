"""
Main application module for the LLM API Gateway.

This module creates the FastAPI application and registers the API routes.
"""

from fastapi import FastAPI

from app.api.routes import router

# Create the central FastAPI application object
app = FastAPI(
    title="LLM API Gateway",
    description=(
        "A provider-independent gateway for text generation"
        "and structured information extraction."
    ),
    version="0.1.0",
)

# Register the endpoints depined by app/api/routes.py
app.include_router(router)
