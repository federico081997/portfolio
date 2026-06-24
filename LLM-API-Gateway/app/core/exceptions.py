"""
Custom exceptions used by the LLM API Gateway.

Provider-specific failures are converted into application exceptions so
the rest of the gateway does not depend directly on HTTPX exceptions.
"""


class GatewayError(Exception):
    """Base exception for gateway-specific failures."""


class ProviderError(GatewayError):
    """Base exception for failures involving an LLM provider."""


class ProviderTimeoutError(ProviderError):
    """Raised when the provider does not respond within the timeout."""


class ProviderUnavailableError(ProviderError):
    """Raised when the provider cannot be reached or remains unavailable."""


class ProviderRequestError(ProviderError):
    """Raised when the provider rejects a non-retryable request."""

    def __init__(self, message, status_code=None):
        """
        Initialises a provider request error.

        Args:
            message: Description of the provider failure.
            status_code: Optional upstream HTTP status code.
        """
        super().__init__(message)
        self.status_code = status_code


class ProviderModelNotFoundError(ProviderRequestError):
    """Raised when the selected provider model is unavailable."""


class InvalidProviderResponseError(ProviderError):
    """Raised when the provider returns malformed or invalid content."""


class RetryableProviderError(ProviderError):
    """
    Internal exception representing a retryable HTTP response.

    This is used for responses such as HTTP 429 and temporary HTTP 5xx
    failures. After all retries are exhausted, it is converted into a
    ProviderUnavailableError.
    """

    def __init__(self, message, status_code=None):
        """
        Initialises a retryable provider error.

        Args:
            message: Description of the temporary provider failure.
            status_code: Optional upstream HTTP status code.
        """
        super().__init__(message)
        self.status_code = status_code
