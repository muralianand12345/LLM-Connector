import pytest
from unittest.mock import MagicMock, patch, mock_open

from llm_connector.providers.openai.fileapi import OpenAIFileAPI
from llm_connector.exceptions import FileError


class TestOpenAIFileAPI:
    """Tests for OpenAIFileAPI."""

    def test_upload_from_path(self, sample_file_object):
        """Test uploading file from path."""
        mock_client = MagicMock()
        mock_client.files.create.return_value = sample_file_object

        file_api = OpenAIFileAPI(mock_client)

        with patch("builtins.open", mock_open(read_data=b'{"test": "data"}')):
            file_id = file_api.upload(file="test.jsonl", purpose="batch")

        assert file_id == "file-abc123"

    def test_upload_from_bytes(self, sample_file_object):
        """Test uploading file from bytes."""
        mock_client = MagicMock()
        mock_client.files.create.return_value = sample_file_object

        file_api = OpenAIFileAPI(mock_client)
        file_id = file_api.upload(file=b'{"test": "data"}', purpose="batch")

        assert file_id == "file-abc123"
        mock_client.files.create.assert_called_once()

    def test_upload_from_file_object(self, sample_file_object):
        """Test uploading from file-like object."""
        mock_client = MagicMock()
        mock_client.files.create.return_value = sample_file_object

        file_api = OpenAIFileAPI(mock_client)
        mock_file = MagicMock()
        file_id = file_api.upload(file=mock_file, purpose="fine-tune")

        assert file_id == "file-abc123"

    def test_retrieve(self, sample_file_object):
        """Test retrieving file metadata."""
        mock_client = MagicMock()
        mock_client.files.retrieve.return_value = sample_file_object

        file_api = OpenAIFileAPI(mock_client)
        result = file_api.retrieve(file_id="file-abc123")

        assert result.id == "file-abc123"
        assert result.filename == "test.jsonl"
        assert result.purpose == "batch"
        assert result.bytes == 1024

    def test_download(self):
        """Test downloading file content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = b'{"test": "data"}'
        mock_client.files.content.return_value = mock_response

        file_api = OpenAIFileAPI(mock_client)
        content = file_api.download(file_id="file-abc123")

        assert content == b'{"test": "data"}'

    def test_delete(self):
        """Test deleting a file."""
        mock_client = MagicMock()

        file_api = OpenAIFileAPI(mock_client)
        file_api.delete(file_id="file-abc123")

        mock_client.files.delete.assert_called_with("file-abc123")

    def test_list_all(self, sample_file_object):
        """Test listing all files."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_file_object]
        mock_client.files.list.return_value = mock_response

        file_api = OpenAIFileAPI(mock_client)
        files = file_api.list()

        assert len(files) == 1
        assert files[0].id == "file-abc123"

    def test_list_by_purpose(self, sample_file_object):
        """Test listing files filtered by purpose."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [sample_file_object]
        mock_client.files.list.return_value = mock_response

        file_api = OpenAIFileAPI(mock_client)
        file_api.list(purpose="batch")

        mock_client.files.list.assert_called_with(purpose="batch")

    def test_error_handling_generic_exception(self):
        """Test FileError on generic exception."""
        mock_client = MagicMock()
        mock_client.files.retrieve.side_effect = Exception("Something went wrong")

        file_api = OpenAIFileAPI(mock_client)

        with pytest.raises(FileError) as exc_info:
            file_api.retrieve(file_id="nonexistent")

        assert "Something went wrong" in str(exc_info.value)
