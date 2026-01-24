import pytest

from llm_connector.exceptions import (
    ProviderNotSupportedError,
    ProviderImportError,
    AuthenticationError,
    RateLimitError,
    APIError,
    InvalidRequestError,
    ContentFilterError,
    ContextLengthExceededError,
    BatchError,
    FileError,
)


class TestExceptions:
    """Tests for custom exceptions."""

    def test_provider_not_supported_error(self):
        """Test ProviderNotSupportedError."""
        with pytest.raises(ProviderNotSupportedError):
            raise ProviderNotSupportedError("Provider not supported")

    def test_provider_import_error(self):
        """Test ProviderImportError."""
        with pytest.raises(ProviderImportError):
            raise ProviderImportError("Package not installed")

    def test_authentication_error(self):
        """Test AuthenticationError."""
        with pytest.raises(AuthenticationError):
            raise AuthenticationError("Invalid API key")

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Rate limited", retry_after=30.0)
        assert error.retry_after == 30.0
        assert "Rate limited" in str(error)

    def test_rate_limit_error_no_retry_after(self):
        """Test RateLimitError without retry_after."""
        error = RateLimitError()
        assert error.retry_after is None
        assert "Rate limit exceeded" in str(error)

    def test_api_error(self):
        """Test APIError with status code and response."""
        error = APIError(
            "Server error", status_code=500, response={"error": "Internal"}
        )
        assert error.status_code == 500
        assert error.response == {"error": "Internal"}
        assert "Server error" in str(error)

    def test_api_error_minimal(self):
        """Test APIError with minimal info."""
        error = APIError("Error occurred")
        assert error.status_code is None
        assert error.response is None

    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        with pytest.raises(InvalidRequestError):
            raise InvalidRequestError("Invalid parameter")

    def test_content_filter_error(self):
        """Test ContentFilterError."""
        with pytest.raises(ContentFilterError):
            raise ContentFilterError("Content blocked")

    def test_context_length_exceeded_error(self):
        """Test ContextLengthExceededError is InvalidRequestError."""
        error = ContextLengthExceededError("Too many tokens")
        assert isinstance(error, InvalidRequestError)

    def test_batch_error(self):
        """Test BatchError."""
        with pytest.raises(BatchError):
            raise BatchError("Batch failed")

    def test_file_error(self):
        """Test FileError."""
        with pytest.raises(FileError):
            raise FileError("File not found")
