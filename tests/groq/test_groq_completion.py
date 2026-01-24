import pytest
from unittest.mock import MagicMock

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
    GroqChatCompletion,
    GroqChatResponses,
    GroqChatStreamChunks,
)


class TestGroqChatResponses:
    """Tests for GroqChatResponses."""

    def test_response_properties(self, sample_chat_response):
        """Test response properties are accessible."""
        response = GroqChatResponses(sample_chat_response)

        assert response.id == "chatcmpl-123"
        assert response.model == "llama-3.3-70b-versatile"
        assert response.content == "Hello! How can I help you?"
        assert response.tool_calls is None
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 18

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = GroqChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"location": "Tokyo"}

    def test_response_with_invalid_json_tool_call(self):
        """Test response handles invalid JSON in tool calls."""
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-456"
        mock_response.model = "llama-3.3-70b-versatile"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [MagicMock()]
        mock_response.choices[0].message.tool_calls[0].id = "call_123"
        mock_response.choices[0].message.tool_calls[0].function.name = "test_func"
        mock_response.choices[0].message.tool_calls[
            0
        ].function.arguments = "{invalid json"
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        response = GroqChatResponses(mock_response)

        # Should not crash, should return empty dict for invalid JSON
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {}

    def test_raw_response_access(self, sample_chat_response):
        """Test raw response is accessible."""
        response = GroqChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestGroqChatStreamChunks:
    """Tests for GroqChatStreamChunks."""

    def test_chunk_properties(self, sample_stream_chunks):
        """Test chunk properties."""
        chunk = GroqChatStreamChunks(sample_stream_chunks[0])

        assert chunk.id == "chatcmpl-123"
        assert chunk.model == "llama-3.3-70b-versatile"
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_final_chunk_with_usage(self, sample_stream_chunks):
        """Test final chunk has usage info."""
        chunk = GroqChatStreamChunks(sample_stream_chunks[1])

        assert chunk.delta_content == " World!"
        assert chunk.finish_reason == "stop"
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 7

    def test_chunk_with_x_groq_usage(self):
        """Test chunk with Groq-specific usage field."""
        mock_chunk = MagicMock()
        mock_chunk.id = "chatcmpl-123"
        mock_chunk.model = "llama-3.3-70b-versatile"
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Test"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = None
        mock_chunk.usage = None

        # Groq-specific usage field
        mock_chunk.x_groq = MagicMock()
        mock_chunk.x_groq.usage = MagicMock()
        mock_chunk.x_groq.usage.prompt_tokens = 5
        mock_chunk.x_groq.usage.completion_tokens = 3
        mock_chunk.x_groq.usage.total_tokens = 8

        chunk = GroqChatStreamChunks(mock_chunk)

        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 8


class TestGroqChatCompletion:
    """Tests for GroqChatCompletion."""

    def test_invoke_with_string(self, sample_chat_response):
        """Test invoke with string message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        response = completion.invoke(messages="Hello!")

        assert isinstance(response, GroqChatResponses)
        assert response.content == "Hello! How can I help you?"

        # Verify API call
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"] == [{"role": "user", "content": "Hello!"}]

    def test_invoke_with_message_object(self, sample_chat_response):
        """Test invoke with Message object."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        message = UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")])
        response = completion.invoke(messages=message)

        assert isinstance(response, GroqChatResponses)

    def test_invoke_with_message_list(self, sample_chat_response):
        """Test invoke with list of messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        messages = [
            SystemMessage(
                role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]
            ),
            UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")]),
        ]
        response = completion.invoke(messages=messages)

        assert isinstance(response, GroqChatResponses)

        # Verify messages were formatted correctly
        call_args = mock_client.chat.completions.create.call_args
        formatted = call_args.kwargs["messages"]
        assert formatted[0]["role"] == "system"
        assert formatted[1]["role"] == "user"

    def test_invoke_with_model_override(self, sample_chat_response):
        """Test invoke with custom model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        completion.invoke(messages="Hello!", model="llama-3.1-70b-versatile")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "llama-3.1-70b-versatile"

    def test_invoke_with_temperature(self, sample_chat_response):
        """Test invoke with temperature."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        completion.invoke(messages="Hello!", temperature=0.5)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.5

    def test_invoke_with_max_tokens(self, sample_chat_response):
        """Test invoke with max_tokens."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
        completion.invoke(messages="Hello!", max_tokens=100)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 100

    def test_invoke_with_tools(self, sample_tool_call_response):
        """Test invoke with tools."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_tool_call_response

        completion = GroqChatCompletion(mock_client)
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

        completion = GroqChatCompletion(mock_client)
        stream = completion.invoke(messages="Hello!", stream=True)

        chunks = list(stream)
        assert len(chunks) == 2
        assert chunks[0].delta_content == "Hello"
        assert chunks[1].delta_content == " World!"

    def test_format_multimodal_message(self, sample_chat_response):
        """Test formatting multimodal user message."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = sample_chat_response

        completion = GroqChatCompletion(mock_client)
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

        completion = GroqChatCompletion(mock_client)
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

    def test_default_model(self):
        """Test default model is set."""
        mock_client = MagicMock()
        completion = GroqChatCompletion(mock_client)
        assert completion.DEFAULT_MODEL == "llama-3.3-70b-versatile"

    def test_error_handling(self):
        """Test error handling for Groq exceptions."""
        pytest.importorskip("groq")  # Skip test if groq is not installed

        import groq

        mock_client = MagicMock()

        # Mock Groq exception
        mock_client.chat.completions.create.side_effect = groq.RateLimitError(
            "Rate limited"
        )

        completion = GroqChatCompletion(mock_client)

        from llm_connector.exceptions import RateLimitError

        with pytest.raises(RateLimitError):
            completion.invoke(messages="Hello!")


# Fixtures for Groq tests
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
    chunk1 = MagicMock(spec=['id', 'model', 'choices', 'usage'])
    chunk1.id = "chatcmpl-123"
    chunk1.model = "llama-3.3-70b-versatile"
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.choices[0].delta.tool_calls = None
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None
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
