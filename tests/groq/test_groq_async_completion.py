import pytest
from unittest.mock import MagicMock, AsyncMock

from llm_connector import (
    TextBlock,
    ImageBlock,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    Role,
)
from llm_connector.providers.groq.completion import (
    GroqAsyncChatCompletion,
    GroqAsyncChatResponses,
    GroqAsyncChatStreamChunks,
)


class TestGroqAsyncChatResponses:
    """Tests for GroqAsyncChatResponses."""

    def test_response_properties(self, sample_chat_response):
        """Test response properties are accessible."""
        response = GroqAsyncChatResponses(sample_chat_response)

        assert response.id == "chatcmpl-123"
        assert response.model == "llama-3.3-70b-versatile"
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 18

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = GroqAsyncChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Tokyo"}

    def test_raw_response_access(self, sample_chat_response):
        """Test raw response is accessible."""
        response = GroqAsyncChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestGroqAsyncChatStreamChunks:
    """Tests for GroqAsyncChatStreamChunks."""

    def test_chunk_properties(self, sample_stream_chunks):
        """Test chunk properties."""
        chunk = GroqAsyncChatStreamChunks(sample_stream_chunks[0])

        assert chunk.id == "chatcmpl-123"
        assert chunk.model == "llama-3.3-70b-versatile"
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_final_chunk_with_usage(self, sample_stream_chunks):
        """Test final chunk has usage info."""
        chunk = GroqAsyncChatStreamChunks(sample_stream_chunks[1])

        assert chunk.delta_content == " World!"
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 7


class TestGroqAsyncChatCompletion:
    """Tests for GroqAsyncChatCompletion."""

    @pytest.mark.asyncio
    async def test_invoke_with_string(self, sample_chat_response):
        """Test async invoke with string message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        response = await completion.invoke(messages="Hello!")

        assert isinstance(response, GroqAsyncChatResponses)
        assert response.content == "Hello! How can I help you?"

        # Verify API call
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello!"}]

    @pytest.mark.asyncio
    async def test_invoke_with_message_object(self, sample_chat_response):
        """Test async invoke with Message object."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        message = UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")])
        response = await completion.invoke(messages=message)

        assert isinstance(response, GroqAsyncChatResponses)

    @pytest.mark.asyncio
    async def test_invoke_with_message_list(self, sample_chat_response):
        """Test async invoke with list of messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        messages = [
            SystemMessage(
                role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]
            ),
            UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")]),
        ]
        response = await completion.invoke(messages=messages)

        assert isinstance(response, GroqAsyncChatResponses)

        # Verify messages were formatted correctly
        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"]
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_invoke_with_model_override(self, sample_chat_response):
        """Test async invoke with custom model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        await completion.invoke(messages="Hello!", model="llama-3.1-70b-versatile")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "llama-3.1-70b-versatile"

    @pytest.mark.asyncio
    async def test_invoke_with_temperature(self, sample_chat_response):
        """Test async invoke with temperature."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        await completion.invoke(messages="Hello!", temperature=0.5)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_invoke_with_tools(self, sample_tool_call_response):
        """Test async invoke with tools."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_tool_call_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        response = await completion.invoke(messages="What's the weather?", tools=tools)

        assert response.tool_calls is not None

        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_invoke_streaming(self, sample_stream_chunks):
        """Test async streaming invoke."""
        mock_client = MagicMock()

        # Create async iterator for streaming
        async def async_stream():
            for chunk in sample_stream_chunks:
                yield chunk

        mock_client.chat.completions.create = AsyncMock(return_value=async_stream())

        completion = GroqAsyncChatCompletion(mock_client)
        stream = await completion.invoke(messages="Hello!", stream=True)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " World!"

    @pytest.mark.asyncio
    async def test_format_multimodal_message(self, sample_chat_response):
        """Test formatting multimodal user message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = GroqAsyncChatCompletion(mock_client)
        message = UserMessage(
            role=Role.USER,
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(url="https://example.com/image.png", detail="high"),
            ],
        )
        await completion.invoke(messages=message)

        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"][0]
        assert formatted["role"] == "user"
        assert len(formatted["content"]) == 2
        assert formatted["content"][0]["type"] == "text"
        assert formatted["content"][1]["type"] == "image_url"

    def test_default_model(self):
        """Test default model is set."""
        mock_client = MagicMock()
        completion = GroqAsyncChatCompletion(mock_client)
        assert completion.DEFAULT_MODEL == "llama-3.3-70b-versatile"


# Fixtures shared with sync tests
@pytest.fixture
def sample_chat_response():
    """Create a sample Groq chat completion response."""
    response = MagicMock()
    response.id = "chatcmpl-123"
    response.model = "llama-3.3-70b-versatile"
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
    """Create sample Groq streaming chunks."""
    chunks = []

    # First chunk
    chunk1 = MagicMock()
    chunk1.id = "chatcmpl-123"
    chunk1.model = "llama-3.3-70b-versatile"
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.choices[0].delta.tool_calls = None
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None
    # Ensure x_groq doesn't exist or is None
    del chunk1.x_groq
    chunks.append(chunk1)

    # Second chunk
    chunk2 = MagicMock()
    chunk2.id = "chatcmpl-123"
    chunk2.model = "llama-3.3-70b-versatile"
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
    """Create a sample Groq response with tool calls."""
    response = MagicMock()
    response.id = "chatcmpl-456"
    response.model = "llama-3.3-70b-versatile"
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
