import pytest
from unittest.mock import MagicMock, patch, mock_open

from llm_connector.base import BatchStatus
from llm_connector.providers.anthropic.batch import AnthropicBatchProcess
from llm_connector.exceptions import BatchError


class TestAnthropicBatchProcess:
    """Tests for AnthropicBatchProcess."""

    def test_create_batch_from_requests_list(self, sample_batch_response):
        """Test creating batch from requests list."""
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)
        requests = [
            {
                "custom_id": "req-1",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            }
        ]
        result = batch.create(requests=requests)

        assert result.id == "msgbatch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.messages.batches.create.assert_called_once()

    def test_create_batch_from_bytes(self, sample_batch_response):
        """Test creating batch from JSONL bytes."""
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)
        jsonl_content = b'{"custom_id": "req-1", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}}'
        result = batch.create(file=jsonl_content)

        assert result.id == "msgbatch_123"
        mock_client.messages.batches.create.assert_called_once()

    def test_create_batch_from_file_path(self, sample_batch_response):
        """Test creating batch from file path."""
        mock_client = MagicMock()
        mock_client.messages.batches.create.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)
        jsonl_content = '{"custom_id": "req-1", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}}'

        with patch("builtins.open", mock_open(read_data=jsonl_content)):
            result = batch.create(file="test.jsonl")

        assert result.id == "msgbatch_123"

    def test_create_batch_no_requests_raises(self):
        """Test creating batch without requests raises error."""
        mock_client = MagicMock()

        batch = AnthropicBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            batch.create()
        assert "No requests provided" in str(exc_info.value)

    def test_create_batch_invalid_json_raises(self):
        """Test creating batch with invalid JSON raises error."""
        mock_client = MagicMock()

        batch = AnthropicBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            batch.create(file=b"{invalid json")
        assert "Invalid JSON" in str(exc_info.value)

    def test_status(self, sample_batch_response):
        """Test getting batch status."""
        mock_client = MagicMock()
        mock_client.messages.batches.retrieve.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)
        result = batch.status("msgbatch_123")

        assert result.id == "msgbatch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.messages.batches.retrieve.assert_called_with("msgbatch_123")

    def test_result_completed(self, sample_batch_response, sample_batch_results):
        """Test getting results of completed batch."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "ended"
        mock_client.messages.batches.retrieve.return_value = sample_batch_response
        mock_client.messages.batches.results.return_value = iter(sample_batch_results)

        batch = AnthropicBatchProcess(mock_client)
        result = batch.result("msgbatch_123")

        assert result.job_id == "msgbatch_123"
        assert result.output_file_id is None  # Anthropic doesn't use file IDs
        assert len(result.records) == 2

    def test_result_not_completed(self, sample_batch_response):
        """Test getting results of non-completed batch raises error."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "in_progress"
        mock_client.messages.batches.retrieve.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            batch.result("msgbatch_123")
        assert "not completed" in str(exc_info.value)

    def test_cancel(self, sample_batch_response):
        """Test cancelling a batch."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "canceling"
        mock_client.messages.batches.cancel.return_value = sample_batch_response

        batch = AnthropicBatchProcess(mock_client)
        result = batch.cancel("msgbatch_123")

        assert result.status == BatchStatus.CANCELLED
        mock_client.messages.batches.cancel.assert_called_with("msgbatch_123")

    def test_list(self, sample_batch_response):
        """Test listing batches."""
        mock_client = MagicMock()
        mock_client.messages.batches.list.return_value = iter([sample_batch_response])

        batch = AnthropicBatchProcess(mock_client)
        results = batch.list(limit=10)

        assert len(results) == 1
        assert results[0].id == "msgbatch_123"

    def test_list_with_pagination(self, sample_batch_response):
        """Test listing batches with pagination."""
        mock_client = MagicMock()
        mock_client.messages.batches.list.return_value = iter([sample_batch_response])

        batch = AnthropicBatchProcess(mock_client)
        batch.list(limit=10, after="msgbatch_122")

        mock_client.messages.batches.list.assert_called_with(
            limit=10, after_id="msgbatch_122"
        )

    def test_batch_status_mapping(self):
        """Test batch status mapping from Anthropic processing_status."""
        mock_client = MagicMock()
        batch = AnthropicBatchProcess(mock_client)

        status_mappings = [
            ("in_progress", BatchStatus.IN_PROGRESS),
            ("canceling", BatchStatus.CANCELLED),
            ("ended", BatchStatus.COMPLETED),
        ]

        for anthropic_status, expected_status in status_mappings:
            mock_response = MagicMock()
            mock_response.id = "msgbatch_123"
            mock_response.processing_status = anthropic_status
            mock_response.created_at = "2024-01-01T00:00:00Z"
            mock_response.cancel_initiated_at = None
            mock_response.ended_at = None
            mock_response.expires_at = None
            mock_response.request_counts = None

            result = batch._to_batch_request(mock_response)
            assert result.status == expected_status

    def test_format_result_entry_succeeded(self):
        """Test formatting a successful result entry."""
        mock_client = MagicMock()
        batch = AnthropicBatchProcess(mock_client)

        mock_result = MagicMock()
        mock_result.type = "succeeded"
        mock_result.message = MagicMock()
        mock_result.message.id = "msg-123"
        mock_result.message.type = "message"
        mock_result.message.role = "assistant"

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello!"
        mock_result.message.content = [text_block]

        mock_result.message.model = "claude-sonnet-4-20250514"
        mock_result.message.stop_reason = "end_turn"
        mock_result.message.usage = MagicMock()
        mock_result.message.usage.input_tokens = 10
        mock_result.message.usage.output_tokens = 5

        formatted = batch._format_result_entry(mock_result)

        assert formatted["type"] == "succeeded"
        assert "message" in formatted
        assert formatted["message"]["id"] == "msg-123"

    def test_format_result_entry_errored(self):
        """Test formatting an errored result entry."""
        mock_client = MagicMock()
        batch = AnthropicBatchProcess(mock_client)

        mock_result = MagicMock()
        mock_result.type = "errored"
        mock_result.error = MagicMock()
        mock_result.error.type = "invalid_request"
        mock_result.error.message = "Invalid parameters"

        formatted = batch._format_result_entry(mock_result)

        assert formatted["type"] == "errored"
        assert "error" in formatted
        assert formatted["error"]["type"] == "invalid_request"


@pytest.fixture
def sample_batch_response():
    """Create a sample Anthropic batch response."""
    response = MagicMock()
    response.id = "msgbatch_123"
    response.processing_status = "ended"
    response.created_at = "2024-01-01T00:00:00Z"
    response.cancel_initiated_at = None
    response.ended_at = "2024-01-01T01:00:00Z"
    response.expires_at = "2024-01-08T00:00:00Z"
    response.request_counts = MagicMock()
    response.request_counts.processing = 0
    response.request_counts.succeeded = 10
    response.request_counts.errored = 0
    response.request_counts.canceled = 0
    response.request_counts.expired = 0
    return response


@pytest.fixture
def sample_batch_results():
    """Create sample batch result entries."""
    results = []

    # Successful result
    result1 = MagicMock()
    result1.custom_id = "req-1"
    result1.result = MagicMock()
    result1.result.type = "succeeded"
    result1.result.message = MagicMock()
    result1.result.message.id = "msg-1"
    result1.result.message.type = "message"
    result1.result.message.role = "assistant"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Response 1"
    result1.result.message.content = [text_block]
    result1.result.message.model = "claude-sonnet-4-20250514"
    result1.result.message.stop_reason = "end_turn"
    result1.result.message.usage = MagicMock()
    result1.result.message.usage.input_tokens = 10
    result1.result.message.usage.output_tokens = 5
    results.append(result1)

    # Another successful result
    result2 = MagicMock()
    result2.custom_id = "req-2"
    result2.result = MagicMock()
    result2.result.type = "succeeded"
    result2.result.message = MagicMock()
    result2.result.message.id = "msg-2"
    result2.result.message.type = "message"
    result2.result.message.role = "assistant"
    text_block2 = MagicMock()
    text_block2.type = "text"
    text_block2.text = "Response 2"
    result2.result.message.content = [text_block2]
    result2.result.message.model = "claude-sonnet-4-20250514"
    result2.result.message.stop_reason = "end_turn"
    result2.result.message.usage = MagicMock()
    result2.result.message.usage.input_tokens = 12
    result2.result.message.usage.output_tokens = 8
    results.append(result2)

    return results
