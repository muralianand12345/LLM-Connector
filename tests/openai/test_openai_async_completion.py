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
from llm_connector.providers.openai.completion import (
    OpenAIAsyncChatCompletion,
    OpenAIAsyncChatResponses,
    OpenAIAsyncChatStreamChunks,
)


class TestOpenAIAsyncChatResponses:
    """Tests for OpenAIAsyncChatResponses."""

    def test_response_properties(self, sample_chat_response):
        """Test response properties are accessible."""
        response = OpenAIAsyncChatResponses(sample_chat_response)

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o-mini"
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 18

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = OpenAIAsyncChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Tokyo"}

    def test_raw_response_access(self, sample_chat_response):
        """Test raw response is accessible."""
        response = OpenAIAsyncChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestOpenAIAsyncChatStreamChunks:
    """Tests for OpenAIAsyncChatStreamChunks."""

    def test_chunk_properties(self, sample_stream_chunks):
        """Test chunk properties."""
        chunk = OpenAIAsyncChatStreamChunks(sample_stream_chunks[0])

        assert chunk.id == "chatcmpl-123"
        assert chunk.model == "gpt-4o-mini"
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_final_chunk_with_usage(self, sample_stream_chunks):
        """Test final chunk has usage info."""
        chunk = OpenAIAsyncChatStreamChunks(sample_stream_chunks[1])

        assert chunk.delta_content == " World!"
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 7


class TestOpenAIAsyncChatCompletion:
    """Tests for OpenAIAsyncChatCompletion."""

    @pytest.mark.asyncio
    async def test_invoke_with_string(self, sample_chat_response):
        """Test async invoke with string message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = OpenAIAsyncChatCompletion(mock_client)
        response = await completion.invoke(messages="Hello!")

        assert isinstance(response, OpenAIAsyncChatResponses)
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

        completion = OpenAIAsyncChatCompletion(mock_client)
        message = UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")])
        response = await completion.invoke(messages=message)

        assert isinstance(response, OpenAIAsyncChatResponses)

    @pytest.mark.asyncio
    async def test_invoke_with_message_list(self, sample_chat_response):
        """Test async invoke with list of messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = OpenAIAsyncChatCompletion(mock_client)
        messages = [
            SystemMessage(
                role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]
            ),
            UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")]),
        ]
        response = await completion.invoke(messages=messages)

        assert isinstance(response, OpenAIAsyncChatResponses)

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

        completion = OpenAIAsyncChatCompletion(mock_client)
        await completion.invoke(messages="Hello!", model="gpt-4")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_invoke_with_temperature(self, sample_chat_response):
        """Test async invoke with temperature."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=sample_chat_response
        )

        completion = OpenAIAsyncChatCompletion(mock_client)
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

        completion = OpenAIAsyncChatCompletion(mock_client)
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

        completion = OpenAIAsyncChatCompletion(mock_client)
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

        completion = OpenAIAsyncChatCompletion(mock_client)
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
        completion = OpenAIAsyncChatCompletion(mock_client)
        assert completion.DEFAULT_MODEL == "gpt-4o-mini"
