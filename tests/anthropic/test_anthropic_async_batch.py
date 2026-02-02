import pytest
from unittest.mock import MagicMock, AsyncMock, patch, mock_open

from llm_connector.base import BatchStatus
from llm_connector.providers.anthropic.batch import AnthropicAsyncBatchProcess
from llm_connector.exceptions import BatchError


class TestAnthropicAsyncBatchProcess:
    """Tests for AnthropicAsyncBatchProcess."""

    @pytest.mark.asyncio
    async def test_create_batch_from_requests_list(self, sample_batch_response):
        """Test creating batch from requests list asynchronously."""
        mock_client = MagicMock()
        mock_client.messages.batches.create = AsyncMock(
            return_value=sample_batch_response
        )

        batch = AnthropicAsyncBatchProcess(mock_client)
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
        result = await batch.create(requests=requests)

        assert result.id == "msgbatch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.messages.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_from_bytes(self, sample_batch_response):
        """Test creating batch from JSONL bytes asynchronously."""
        mock_client = MagicMock()
        mock_client.messages.batches.create = AsyncMock(
            return_value=sample_batch_response
        )

        batch = AnthropicAsyncBatchProcess(mock_client)
        jsonl_content = b'{"custom_id": "req-1", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello"}]}}'
        result = await batch.create(file=jsonl_content)

        assert result.id == "msgbatch_123"
        mock_client.messages.batches.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_no_requests_raises(self):
        """Test creating batch without requests raises error."""
        mock_client = MagicMock()

        batch = AnthropicAsyncBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            await batch.create()
        assert "No requests provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_status(self, sample_batch_response):
        """Test getting batch status asynchronously."""
        mock_client = MagicMock()
        mock_client.messages.batches.retrieve = AsyncMock(
            return_value=sample_batch_response
        )

        batch = AnthropicAsyncBatchProcess(mock_client)
        result = await batch.status("msgbatch_123")

        assert result.id == "msgbatch_123"
        assert result.status == BatchStatus.COMPLETED
        mock_client.messages.batches.retrieve.assert_called_with("msgbatch_123")

    @pytest.mark.asyncio
    async def test_result_completed(self, sample_batch_response, sample_batch_results):
        """Test getting results of completed batch asynchronously."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "ended"
        mock_client.messages.batches.retrieve = AsyncMock(
            return_value=sample_batch_response
        )

        # Create async iterator for results
        async def async_results():
            for result in sample_batch_results:
                yield result

        mock_client.messages.batches.results = AsyncMock(return_value=async_results())

        batch = AnthropicAsyncBatchProcess(mock_client)
        result = await batch.result("msgbatch_123")

        assert result.job_id == "msgbatch_123"
        assert result.output_file_id is None
        assert len(result.records) == 2

    @pytest.mark.asyncio
    async def test_result_not_completed(self, sample_batch_response):
        """Test getting results of non-completed batch raises error."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "in_progress"
        mock_client.messages.batches.retrieve = AsyncMock(
            return_value=sample_batch_response
        )

        batch = AnthropicAsyncBatchProcess(mock_client)

        with pytest.raises(BatchError) as exc_info:
            await batch.result("msgbatch_123")
        assert "not completed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel(self, sample_batch_response):
        """Test cancelling a batch asynchronously."""
        mock_client = MagicMock()
        sample_batch_response.processing_status = "canceling"
        mock_client.messages.batches.cancel = AsyncMock(
            return_value=sample_batch_response
        )

        batch = AnthropicAsyncBatchProcess(mock_client)
        result = await batch.cancel("msgbatch_123")

        assert result.status == BatchStatus.CANCELLED
        mock_client.messages.batches.cancel.assert_called_with("msgbatch_123")

    @pytest.mark.asyncio
    async def test_list(self, sample_batch_response):
        """Test listing batches asynchronously."""
        mock_client = MagicMock()

        # Create async iterator for listing
        async def async_list():
            yield sample_batch_response

        mock_client.messages.batches.list.return_value = async_list()

        batch = AnthropicAsyncBatchProcess(mock_client)
        results = await batch.list(limit=10)

        assert len(results) == 1
        assert results[0].id == "msgbatch_123"

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, sample_batch_response):
        """Test listing batches with pagination asynchronously."""
        mock_client = MagicMock()

        async def async_list():
            yield sample_batch_response

        mock_client.messages.batches.list.return_value = async_list()

        batch = AnthropicAsyncBatchProcess(mock_client)
        await batch.list(limit=10, after="msgbatch_122")

        mock_client.messages.batches.list.assert_called_with(
            limit=10, after_id="msgbatch_122"
        )

    @pytest.mark.asyncio
    async def test_list_respects_limit(self, sample_batch_response):
        """Test listing batches respects limit."""
        mock_client = MagicMock()

        # Create async iterator that yields multiple items
        async def async_list():
            for i in range(5):
                response = MagicMock()
                response.id = f"msgbatch_{i}"
                response.processing_status = "ended"
                response.created_at = "2024-01-01T00:00:00Z"
                response.cancel_initiated_at = None
                response.ended_at = None
                response.expires_at = None
                response.request_counts = None
                yield response

        mock_client.messages.batches.list.return_value = async_list()

        batch = AnthropicAsyncBatchProcess(mock_client)
        results = await batch.list(limit=3)

        assert len(results) == 3


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
