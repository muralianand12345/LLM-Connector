import pytest
from unittest.mock import MagicMock, patch, mock_open

from llm_connector.providers.anthropic.fileapi import AnthropicFileAPI
from llm_connector.exceptions import FileError


class TestAnthropicFileAPI:
    """Tests for AnthropicFileAPI."""

    def test_upload_from_path(self, sample_file_object):
        """Test uploading file from path."""
        mock_client = MagicMock()
        mock_client.beta.files.upload.return_value = sample_file_object

        file_api = AnthropicFileAPI(mock_client)

        # Use patch.object to mock Path behavior
        with patch("llm_connector.providers.anthropic.fileapi.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path_class.return_value = mock_path
            file_id = file_api.upload(file="test.pdf", purpose="user_data")

        assert file_id == "file-abc123"

    def test_upload_from_bytes(self, sample_file_object):
        """Test uploading file from bytes."""
        mock_client = MagicMock()
        mock_client.beta.files.upload.return_value = sample_file_object

        file_api = AnthropicFileAPI(mock_client)
        file_id = file_api.upload(file=b"file content", purpose="user_data")

        assert file_id == "file-abc123"
        mock_client.beta.files.upload.assert_called_once()
        # Verify betas parameter is passed
        call_kwargs = mock_client.beta.files.upload.call_args.kwargs
        assert "betas" in call_kwargs
        assert "files-api-2025-04-14" in call_kwargs["betas"]

    def test_upload_from_file_object(self, sample_file_object):
        """Test uploading from file-like object."""
        mock_client = MagicMock()
        mock_client.beta.files.upload.return_value = sample_file_object

        file_api = AnthropicFileAPI(mock_client)
        mock_file = MagicMock()
        file_id = file_api.upload(file=mock_file, purpose="user_data")

        assert file_id == "file-abc123"

    def test_retrieve(self, sample_file_metadata):
        """Test retrieving file metadata."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata.return_value = sample_file_metadata

        file_api = AnthropicFileAPI(mock_client)
        result = file_api.retrieve(file_id="file-abc123")

        assert result.id == "file-abc123"
        assert result.filename == "document.pdf"
        assert result.bytes == 2048
        mock_client.beta.files.retrieve_metadata.assert_called_with(
            file_id="file-abc123", betas=["files-api-2025-04-14"]
        )

    def test_download(self):
        """Test downloading file content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.read.return_value = b"file content"
        mock_client.beta.files.download.return_value = mock_response

        file_api = AnthropicFileAPI(mock_client)
        content = file_api.download(file_id="file-abc123")

        assert content == b"file content"
        mock_client.beta.files.download.assert_called_with(
            file_id="file-abc123", betas=["files-api-2025-04-14"]
        )

    def test_delete(self):
        """Test deleting a file."""
        mock_client = MagicMock()

        file_api = AnthropicFileAPI(mock_client)
        file_api.delete(file_id="file-abc123")

        mock_client.beta.files.delete.assert_called_with(
            file_id="file-abc123", betas=["files-api-2025-04-14"]
        )

    def test_list_all(self, sample_file_metadata):
        """Test listing all files."""
        mock_client = MagicMock()
        mock_client.beta.files.list.return_value = iter([sample_file_metadata])

        file_api = AnthropicFileAPI(mock_client)
        files = file_api.list()

        assert len(files) == 1
        assert files[0].id == "file-abc123"

    def test_list_ignores_purpose_filter(self, sample_file_metadata):
        """Test listing files ignores purpose filter (Anthropic doesn't support it)."""
        mock_client = MagicMock()
        mock_client.beta.files.list.return_value = iter([sample_file_metadata])

        file_api = AnthropicFileAPI(mock_client)
        # Purpose is passed but should be ignored by Anthropic
        files = file_api.list(purpose="user_data")

        assert len(files) == 1
        # Verify that list was called without purpose parameter
        mock_client.beta.files.list.assert_called_with(betas=["files-api-2025-04-14"])

    def test_error_handling_generic_exception(self):
        """Test FileError on generic exception."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata.side_effect = Exception(
            "Something went wrong"
        )

        file_api = AnthropicFileAPI(mock_client)

        with pytest.raises(FileError) as exc_info:
            file_api.retrieve(file_id="nonexistent")

        assert "Something went wrong" in str(exc_info.value)

    def test_file_object_has_default_values(self, sample_file_metadata):
        """Test file object has correct default values for Anthropic."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata.return_value = sample_file_metadata

        file_api = AnthropicFileAPI(mock_client)
        result = file_api.retrieve(file_id="file-abc123")

        # Anthropic doesn't have purpose concept, so it defaults to "user_data"
        assert result.purpose == "user_data"
        # Files are immediately available
        assert result.status == "processed"


@pytest.fixture
def sample_file_object():
    """Create a sample Anthropic file upload response."""
    file_obj = MagicMock()
    file_obj.id = "file-abc123"
    return file_obj


@pytest.fixture
def sample_file_metadata():
    """Create a sample Anthropic file metadata response."""
    metadata = MagicMock()
    metadata.id = "file-abc123"
    metadata.filename = "document.pdf"
    metadata.size_bytes = 2048
    metadata.created_at = 1700000000
    return metadata
