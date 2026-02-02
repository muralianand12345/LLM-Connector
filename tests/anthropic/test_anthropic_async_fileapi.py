import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_connector.providers.anthropic.fileapi import AnthropicAsyncFileAPI
from llm_connector.exceptions import FileError


class TestAnthropicAsyncFileAPI:
    """Tests for AnthropicAsyncFileAPI."""

    @pytest.mark.asyncio
    async def test_upload_from_bytes(self, sample_file_object):
        """Test uploading file from bytes asynchronously."""
        mock_client = MagicMock()
        mock_client.beta.files.upload = AsyncMock(return_value=sample_file_object)

        file_api = AnthropicAsyncFileAPI(mock_client)
        file_id = await file_api.upload(file=b"file content", purpose="user_data")

        assert file_id == "file-abc123"
        mock_client.beta.files.upload.assert_called_once()
        # Verify betas parameter is passed
        call_kwargs = mock_client.beta.files.upload.call_args.kwargs
        assert "betas" in call_kwargs
        assert "files-api-2025-04-14" in call_kwargs["betas"]

    @pytest.mark.asyncio
    async def test_upload_from_path(self, sample_file_object):
        """Test uploading file from path asynchronously."""
        mock_client = MagicMock()
        mock_client.beta.files.upload = AsyncMock(return_value=sample_file_object)

        file_api = AnthropicAsyncFileAPI(mock_client)

        with patch("llm_connector.providers.anthropic.fileapi.Path") as mock_path_class:
            mock_path = MagicMock()
            mock_path_class.return_value = mock_path
            file_id = await file_api.upload(file="test.pdf", purpose="user_data")

        assert file_id == "file-abc123"

    @pytest.mark.asyncio
    async def test_upload_from_file_object(self, sample_file_object):
        """Test uploading from file-like object asynchronously."""
        mock_client = MagicMock()
        mock_client.beta.files.upload = AsyncMock(return_value=sample_file_object)

        file_api = AnthropicAsyncFileAPI(mock_client)
        mock_file = MagicMock()
        file_id = await file_api.upload(file=mock_file, purpose="user_data")

        assert file_id == "file-abc123"

    @pytest.mark.asyncio
    async def test_retrieve(self, sample_file_metadata):
        """Test retrieving file metadata asynchronously."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata = AsyncMock(
            return_value=sample_file_metadata
        )

        file_api = AnthropicAsyncFileAPI(mock_client)
        result = await file_api.retrieve(file_id="file-abc123")

        assert result.id == "file-abc123"
        assert result.filename == "document.pdf"
        assert result.bytes == 2048

    @pytest.mark.asyncio
    async def test_download(self):
        """Test downloading file content asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.read = AsyncMock(return_value=b"file content")
        mock_client.beta.files.download = AsyncMock(return_value=mock_response)

        file_api = AnthropicAsyncFileAPI(mock_client)
        content = await file_api.download(file_id="file-abc123")

        assert content == b"file content"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a file asynchronously."""
        mock_client = MagicMock()
        mock_client.beta.files.delete = AsyncMock()

        file_api = AnthropicAsyncFileAPI(mock_client)
        await file_api.delete(file_id="file-abc123")

        mock_client.beta.files.delete.assert_called_with(
            file_id="file-abc123", betas=["files-api-2025-04-14"]
        )

    @pytest.mark.asyncio
    async def test_list_all(self, sample_file_metadata):
        """Test listing all files asynchronously."""
        mock_client = MagicMock()

        # Create async iterator for listing
        async def async_list():
            yield sample_file_metadata

        mock_client.beta.files.list.return_value = async_list()

        file_api = AnthropicAsyncFileAPI(mock_client)
        files = await file_api.list()

        assert len(files) == 1
        assert files[0].id == "file-abc123"

    @pytest.mark.asyncio
    async def test_list_ignores_purpose_filter(self, sample_file_metadata):
        """Test listing files ignores purpose filter asynchronously."""
        mock_client = MagicMock()

        async def async_list():
            yield sample_file_metadata

        mock_client.beta.files.list.return_value = async_list()

        file_api = AnthropicAsyncFileAPI(mock_client)
        files = await file_api.list(purpose="user_data")

        assert len(files) == 1
        mock_client.beta.files.list.assert_called_with(betas=["files-api-2025-04-14"])

    @pytest.mark.asyncio
    async def test_error_handling_generic_exception(self):
        """Test FileError on generic exception."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata = AsyncMock(
            side_effect=Exception("Something went wrong")
        )

        file_api = AnthropicAsyncFileAPI(mock_client)

        with pytest.raises(FileError) as exc_info:
            await file_api.retrieve(file_id="nonexistent")

        assert "Something went wrong" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_file_object_has_default_values(self, sample_file_metadata):
        """Test file object has correct default values for Anthropic."""
        mock_client = MagicMock()
        mock_client.beta.files.retrieve_metadata = AsyncMock(
            return_value=sample_file_metadata
        )

        file_api = AnthropicAsyncFileAPI(mock_client)
        result = await file_api.retrieve(file_id="file-abc123")

        # Anthropic doesn't have purpose concept
        assert result.purpose == "user_data"
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
