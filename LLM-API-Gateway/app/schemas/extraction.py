"""
Schemas for structured information extraction.

The gateway will use these Pydantic models both to request structured
output from the LLM and to validate the generated JSON.
"""

from enum import Enum
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel


class ExtractionSchemaName(str, Enum):
    """Structured extraction schemas supported by the API."""

    CONTACT = "contact"
    INCIDENT = "incident"


class IncidentSeverity(str, Enum):
    """Allowed severity levels for extracted incident reports."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContactExtraction(APIModel):
    """Structured contact information extracted from unstructured text."""

    people: list[str] = Field(
        description="Names of people found in the text.",
    )

    organisations: list[str] = Field(
        description="Organisation names found in the text.",
    )

    email_addresses: list[str] = Field(
        description="Email addresses found in the text.",
    )

    phone_numbers: list[str] = Field(
        description="Phone numbers found in the text.",
    )


class IncidentExtraction(APIModel):
    """Structured incident information extracted from unstructured text."""

    title: str = Field(
        min_length=1,
        max_length=300,
        description="Concise title describing the incident.",
    )

    severity: IncidentSeverity = Field(
        description="Estimated incident severity.",
    )

    affected_components: list[str] = Field(
        description="Systems or components affected by the incident.",
    )

    symptoms: list[str] = Field(
        description="Observed errors or abnormal behaviour.",
    )

    probable_causes: list[str] = Field(
        description="Possible causes supported by the source text.",
    )

    recommended_actions: list[str] = Field(
        description="Recommended investigation or remediation actions.",
    )


class ExtractionRequest(APIModel):
    """Input accepted by the structured extraction endpoint."""

    text: str = Field(
        min_length=1,
        max_length=30_000,
        description="Unstructured text from which information is extracted.",
    )

    schema_name: ExtractionSchemaName = Field(
        description="Schema used to validate the extracted information.",
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
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness during structured extraction.",
    )

    max_tokens: int = Field(
        default=1_024,
        ge=1,
        le=4_096,
        description="Maximum number of output tokens requested.",
    )

    use_cache: bool = Field(
        default=True,
        description="Whether an existing identical extraction may be reused.",
    )


class ExtractionResponse(APIModel):
    """Validated response returned by the extraction endpoint."""

    request_id: UUID

    provider: str = Field(
        min_length=1,
        description="LLM provider that generated the extraction.",
    )

    model: str = Field(
        min_length=1,
        description="Model that generated the extraction.",
    )

    schema_name: ExtractionSchemaName

    data: ContactExtraction | IncidentExtraction = Field(
        description="Structured and validated extracted information.",
    )

    prompt_tokens: int = Field(
        ge=0,
        description="Number of input tokens processed.",
    )

    completion_tokens: int = Field(
        ge=0,
        description="Number of output tokens generated.",
    )

    latency_ms: float = Field(
        ge=0,
        description="Total gateway processing time in milliseconds.",
    )

    cached: bool = Field(
        description="Whether the result was returned from the cache.",
    )


EXTRACTION_SCHEMA_REGISTRY = {
    ExtractionSchemaName.CONTACT: ContactExtraction,
    ExtractionSchemaName.INCIDENT: IncidentExtraction,
}
