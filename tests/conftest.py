import pytest
from unittest.mock import MagicMock, patch, AsyncMock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("llm_connector.providers.openai.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_async_openai_client():
    """Create a mock AsyncOpenAI client."""
    with patch("llm_connector.providers.openai.AsyncOpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def sample_chat_response():
    """Create a sample chat completion response."""
    response = MagicMock()
    response.id = "chatcmpl-123"
    response.model = "gpt-4o-mini"
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello! How can I help you?"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 18
    return response


@pytest.fixture
def sample_stream_chunks():
    """Create sample streaming chunks."""
    chunks = []

    # First chunk
    chunk1 = MagicMock()
    chunk1.id = "chatcmpl-123"
    chunk1.model = "gpt-4o-mini"
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.choices[0].delta.tool_calls = None
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None
    chunks.append(chunk1)

    # Second chunk
    chunk2 = MagicMock()
    chunk2.id = "chatcmpl-123"
    chunk2.model = "gpt-4o-mini"
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " World!"
    chunk2.choices[0].delta.tool_calls = None
    chunk2.choices[0].finish_reason = "stop"
    chunk2.usage = MagicMock()
    chunk2.usage.prompt_tokens = 5
    chunk2.usage.completion_tokens = 2
    chunk2.usage.total_tokens = 7
    chunks.append(chunk2)

    return chunks


@pytest.fixture
def sample_tool_call_response():
    """Create a sample response with tool calls."""
    response = MagicMock()
    response.id = "chatcmpl-456"
    response.model = "gpt-4o-mini"
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [MagicMock()]
    response.choices[0].message.tool_calls[0].id = "call_abc123"
    response.choices[0].message.tool_calls[0].function.name = "get_weather"
    response.choices[0].message.tool_calls[
        0
    ].function.arguments = '{"location": "Tokyo"}'
    response.choices[0].finish_reason = "tool_calls"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 15
    response.usage.total_tokens = 35
    return response


@pytest.fixture
def sample_batch_response():
    """Create a sample batch response."""
    response = MagicMock()
    response.id = "batch_123"
    response.status = "completed"
    response.created_at = 1700000000
    response.in_progress_at = 1700000100
    response.completed_at = 1700000200
    response.cancelled_at = None
    response.expired_at = None
    response.failed_at = None
    response.finalizing_at = 1700000150
    response.completion_window = "24h"
    response.input_file_id = "file-input-123"
    response.output_file_id = "file-output-456"
    response.error_file_id = None
    response.endpoint = "/v1/chat/completions"
    response.request_counts = MagicMock()
    response.request_counts.model_dump.return_value = {
        "total": 10,
        "completed": 10,
        "failed": 0,
    }
    return response


@pytest.fixture
def sample_file_object():
    """Create a sample file object."""
    file_obj = MagicMock()
    file_obj.id = "file-abc123"
    file_obj.filename = "test.jsonl"
    file_obj.purpose = "batch"
    file_obj.bytes = 1024
    file_obj.created_at = 1700000000
    file_obj.status = "processed"
    return file_obj
