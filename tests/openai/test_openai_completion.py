import pytest
from unittest.mock import MagicMock, patch

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
    OpenAIChatCompletion,
    OpenAIChatResponses,
    OpenAIChatStreamChunks,
)


class TestOpenAIChatResponses:
    """Tests for OpenAIChatResponses."""

    def test_response_properties(self, sample_chat_response):
        """Test response properties are accessible."""
        response = OpenAIChatResponses(sample_chat_response)

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o-mini"
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 18

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = OpenAIChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Tokyo"}

    def test_raw_response_access(self, sample_chat_response):
        """Test raw response is accessible."""
        response = OpenAIChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestOpenAIChatStreamChunks:
    """Tests for OpenAIChatStreamChunks."""

    def test_chunk_properties(self, sample_stream_chunks):
        """Test chunk properties."""
        chunk = OpenAIChatStreamChunks(sample_stream_chunks[0])

        assert chunk.id == "chatcmpl-123"
        assert chunk.model == "gpt-4o-mini"
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_final_chunk_with_usage(self, sample_stream_chunks):
        """Test final chunk has usage info."""
        chunk = OpenAIChatStreamChunks(sample_stream_chunks[1])

        assert chunk.delta_content == " World!"
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 7


class TestOpenAIChatCompletion:
    """Tests for OpenAIChatCompletion."""

    def test_invoke_with_string(self, sample_chat_response):
        """Test invoke with string message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        response = completion.invoke(messages="Hello!")

        assert isinstance(response, OpenAIChatResponses)
        assert response.content == "Hello! How can I help you?"

        # Verify API call
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_invoke_with_message_object(self, sample_chat_response):
        """Test invoke with Message object."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        message = UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")])
        response = completion.invoke(messages=message)

        assert isinstance(response, OpenAIChatResponses)

    def test_invoke_with_message_list(self, sample_chat_response):
        """Test invoke with list of messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        messages = [
            SystemMessage(
                role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]
            ),
            UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")]),
        ]
        response = completion.invoke(messages=messages)

        assert isinstance(response, OpenAIChatResponses)

        # Verify messages were formatted correctly
        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"]
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"

    def test_invoke_with_model_override(self, sample_chat_response):
        """Test invoke with custom model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        completion.invoke(messages="Hello!", model="gpt-4")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"

    def test_invoke_with_temperature(self, sample_chat_response):
        """Test invoke with temperature."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        completion.invoke(messages="Hello!", temperature=0.5)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5

    def test_invoke_with_max_tokens(self, sample_chat_response):
        """Test invoke with max_tokens."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        completion.invoke(messages="Hello!", max_tokens=100)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 100

    def test_invoke_with_tools(self, sample_tool_call_response):
        """Test invoke with tools."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_tool_call_response

        completion = OpenAIChatCompletion(mock_client)
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        response = completion.invoke(messages="What's the weather?", tools=tools)

        assert response.tool_calls is not None

        call_args = mock_client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs

    def test_invoke_streaming(self, sample_stream_chunks):
        """Test streaming invoke."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(sample_stream_chunks)

        completion = OpenAIChatCompletion(mock_client)
        stream = completion.invoke(messages="Hello!", stream=True)

        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " World!"

    def test_format_multimodal_message(self, sample_chat_response):
        """Test formatting multimodal user message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        message = UserMessage(
            role=Role.USER,
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(url="https://example.com/image.png", detail="high"),
            ],
        )
        completion.invoke(messages=message)

        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"][0]
        assert formatted["role"] == "user"
        assert len(formatted["content"]) == 2
        assert formatted["content"][0]["type"] == "text"
        assert formatted["content"][1]["type"] == "image_url"
        assert formatted["content"][1]["image_url"]["detail"] == "high"

    def test_format_assistant_message_with_tool_calls(self, sample_chat_response):
        """Test formatting assistant message with tool calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        messages = [
            AssistantMessage(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", arguments={"loc": "NYC"})
                ],
            ),
            ToolMessage(
                role=Role.TOOL,
                tool_call_id="call_1",
                content=[TextBlock(text='{"temp": 72}')],
            ),
        ]
        completion.invoke(messages=messages)

        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"]

        assert formatted[0]["role"] == "assistant"
        assert "tool_calls" in formatted[0]
        assert formatted[1]["role"] == "tool"
        assert formatted[1]["tool_call_id"] == "call_1"

    def test_format_developer_role(self, sample_chat_response):
        """Test developer role is formatted correctly."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = OpenAIChatCompletion(mock_client)
        message = SystemMessage(
            role=Role.DEVELOPER, content=[TextBlock(text="Dev instructions")]
        )
        completion.invoke(messages=message)

        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"][0]
        assert formatted["role"] == "developer"

    def test_default_model(self):
        """Test default model is set."""
        mock_client = MagicMock()
        completion = OpenAIChatCompletion(mock_client)
        assert completion.DEFAULT_MODEL == "gpt-4o-mini"
