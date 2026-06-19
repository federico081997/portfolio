"""
Request and response models for text generation.

These schemas define the public API contract for the /generate endpoint.
"""

from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class GenerateRequest(APIModel):
    """Input accepted by the text-generation endpoint."""

    prompt: str = Field(
        min_length=1,
        max_length=20_000,
        description="User prompt sent to the language model.",
        examples=["Explain Newton-Raphson iteration."],
    )

    system_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=5_000,
        description="Optional instruction controlling model behaviour.",
        examples=["You are a concise technical assistant."],
    )

    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Optional model override. The configured default model is used "
            "when this field is omitted."
        ),
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Controls randomness in model generation.",
    )

    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4_096,
        description="Maximum number of tokens requested from the model.",
    )

    use_cache: bool = Field(
        default=True,
        description="Whether an existing identical response may be reused.",
    )


class GenerateResponse(APIModel):
    """Validated response returned by the text-generation endpoint."""

    request_id: UUID

    provider: str = Field(
        min_length=1,
        description="LLM provider that generated the response.",
    )

    model: str = Field(
        min_length=1,
        description="Model that generated the response.",
    )

    text: str = Field(
        min_length=1,
        description="Generated text returned by the language model.",
    )

    prompt_tokens: int = Field(
        ge=0,
        description="Number of tokens processed from the input.",
    )

    completion_tokens: int = Field(
        ge=0,
        description="Number of tokens generated in the response.",
    )

    latency_ms: float = Field(
        ge=0,
        description="Total gateway processing time in milliseconds.",
    )

    cached: bool = Field(
        description="Whether the response was returned from the cache.",
    )
