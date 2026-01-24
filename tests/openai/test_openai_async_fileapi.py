import pytest
from unittest.mock import MagicMock, AsyncMock

from llm_connector.providers.openai.fileapi import OpenAIAsyncFileAPI
from llm_connector.exceptions import FileError


class TestOpenAIAsyncFileAPI:
    """Tests for OpenAIAsyncFileAPI."""

    @pytest.mark.asyncio
    async def test_upload_from_bytes(self, sample_file_object):
        """Test uploading file from bytes asynchronously."""
        mock_client = MagicMock()
        mock_client.files.create = AsyncMock(return_value=sample_file_object)

        file_api = OpenAIAsyncFileAPI(mock_client)
        file_id = await file_api.upload(file=b'{"test": "data"}', purpose="batch")

        assert file_id == "file-abc123"
        mock_client.files.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve(self, sample_file_object):
        """Test retrieving file metadata asynchronously."""
        mock_client = MagicMock()
        mock_client.files.retrieve = AsyncMock(return_value=sample_file_object)

        file_api = OpenAIAsyncFileAPI(mock_client)
        result = await file_api.retrieve(file_id="file-abc123")

        assert result.id == "file-abc123"
        assert result.filename == "test.jsonl"
        assert result.purpose == "batch"
        assert result.bytes == 1024

    @pytest.mark.asyncio
    async def test_download(self):
        """Test downloading file content asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b'{"test": "data"}'
        mock_client.files.content = AsyncMock(return_value=mock_response)

        file_api = OpenAIAsyncFileAPI(mock_client)
        content = await file_api.download(file_id="file-abc123")

        assert content == b'{"test": "data"}'

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting a file asynchronously."""
        mock_client = MagicMock()
        mock_client.files.delete = AsyncMock()

        file_api = OpenAIAsyncFileAPI(mock_client)
        await file_api.delete(file_id="file-abc123")

        mock_client.files.delete.assert_called_with("file-abc123")

    @pytest.mark.asyncio
    async def test_list_all(self, sample_file_object):
        """Test listing all files asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_file_object]
        mock_client.files.list = AsyncMock(return_value=mock_response)

        file_api = OpenAIAsyncFileAPI(mock_client)
        files = await file_api.list()

        assert len(files) == 1
        assert files[0].id == "file-abc123"

    @pytest.mark.asyncio
    async def test_list_by_purpose(self, sample_file_object):
        """Test listing files filtered by purpose asynchronously."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_file_object]
        mock_client.files.list = AsyncMock(return_value=mock_response)

        file_api = OpenAIAsyncFileAPI(mock_client)
        await file_api.list(purpose="batch")

        mock_client.files.list.assert_called_with(purpose="batch")

    @pytest.mark.asyncio
    async def test_error_handling_generic_exception(self):
        """Test FileError on generic exception."""
        mock_client = MagicMock()
        mock_client.files.retrieve = AsyncMock(
            side_effect=Exception("Something went wrong")
        )

        file_api = OpenAIAsyncFileAPI(mock_client)

        with pytest.raises(FileError) as exc_info:
            await file_api.retrieve(file_id="nonexistent")

        assert "Something went wrong" in str(exc_info.value)
