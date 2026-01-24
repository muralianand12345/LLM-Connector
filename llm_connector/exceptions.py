class ProviderNotSupportedError(ValueError):
    """Raised when an unsupported provider is requested."""

    pass


class ProviderImportError(ImportError):
    """Raised when a provider's required package is not installed."""

    pass


class AuthenticationError(Exception):
    """Raised when API key is invalid or missing."""

    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(Exception):
    """Generic API error from provider."""

    def __init__(
        self, message: str, status_code: int | None = None, response: dict | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class InvalidRequestError(ValueError):
    """Raised when request parameters are invalid."""

    pass


class ContentFilterError(Exception):
    """Raised when content is blocked by safety filters."""

    pass


class ContextLengthExceededError(InvalidRequestError):
    """Raised when input exceeds model's context length."""

    pass


class BatchError(Exception):
    """Raised when batch processing fails."""

    pass


class FileError(Exception):
    """Raised when file operations fail."""

    pass
