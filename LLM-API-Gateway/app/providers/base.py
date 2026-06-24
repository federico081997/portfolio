"""
Provider-independent interfaces and result models.

Every LLM provider must implement this interface so the rest of the
gateway can use providers without knowing their API-specific details.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.extraction import ExtractionRequest
from app.schemas.generation import GenerateRequest


class ProviderResult(APIModel):
    """Normalised result returned by an LLM provider."""

    provider: str = Field(
        min_length=1,
        description="Name of the provider that produced the response.",
    )

    model: str = Field(
        min_length=1,
        description="Model that produced the response.",
    )

    content: str = Field(
        min_length=1,
        description="Raw generated text returned by the provider.",
    )

    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of input tokens processed.",
    )

    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of output tokens generated.",
    )

    provider_latency_ms: float = Field(
        ge=0,
        description="Observed provider call duration in milliseconds.",
    )


class StructuredProviderResult(ProviderResult):
    """Normalised provider result containing validated structured data."""

    data: dict[str, Any] = Field(
        description="Structured data validated against a Pydantic model.",
    )


class LLMProvider(ABC):
    """Abstract interface implemented by every LLM provider."""

    @abstractmethod
    async def generate(self, request: GenerateRequest):
        """
        Generates an ordinary text response.

        Args:
            request: Validated gateway generation request.

        Returns:
            Normalised provider result.
        """

    @abstractmethod
    async def generate_structured(
        self,
        request: ExtractionRequest,
        response_model: type[APIModel],
    ):
        """
        Generates and validates structured JSON.

        Args:
            request: Validated extraction request.
            response_model: Pydantic model used to validate the output.

        Returns:
            Normalised provider result containing structured data.
        """

    @abstractmethod
    async def healthcheck(self):
        """
        Checks whether the provider API is reachable.

        Returns:
            True when the provider is available; otherwise False.
        """
