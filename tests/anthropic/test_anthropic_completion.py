"""Tests for Anthropic chat completion functionality."""

import pytest
from unittest.mock import MagicMock, patch

from llm_connector.providers.anthropic.completion import (
    AnthropicChatCompletion,
    AnthropicChatResponses,
)
from llm_connector.exceptions import (
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    APIError,
)


@pytest.fixture
def sample_chat_response():
    """Create a sample Anthropic message response."""
    response = MagicMock()
    response.id = "msg_01XFDUDYJgAACzvnptvVoYEL"
    response.model = "claude-sonnet-4-20250514"
    response.type = "message"
    response.role = "assistant"
    response.stop_reason = "end_turn"

    # Text content block
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello! How can I help you today?"
    response.content = [text_block]

    # Usage
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20

    return response


@pytest.fixture
def sample_tool_call_response():
    """Create a sample response with tool calls."""
    response = MagicMock()
    response.id = "msg_01XFDUDYJgAACzvnptvVoYEL"
    response.model = "claude-sonnet-4-20250514"
    response.type = "message"
    response.role = "assistant"
    response.stop_reason = "tool_use"

    # Tool use content block - no text attribute
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_01A09q90qw90lq917835lgs"
    tool_block.name = "get_weather"
    tool_block.input = {"location": "San Francisco", "unit": "celsius"}
    # Explicitly remove text attribute so hasattr returns False
    del tool_block.text
    response.content = [tool_block]

    # Usage
    response.usage = MagicMock()
    response.usage.input_tokens = 15
    response.usage.output_tokens = 25

    return response


class TestAnthropicChatResponses:
    """Tests for AnthropicChatResponses class."""

    def test_response_properties(self, sample_chat_response):
        """Test basic response properties."""
        response = AnthropicChatResponses(sample_chat_response)

        assert response.id == "msg_01XFDUDYJgAACzvnptvVoYEL"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.content == "Hello! How can I help you today?"
        assert response.finish_reason == "stop"  # end_turn maps to stop

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = AnthropicChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.finish_reason == "tool_calls"  # tool_use maps to tool_calls
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "toolu_01A09q90qw90lq917835lgs"
        assert response.tool_calls[0]["function"]["name"] == "get_weather"

    def test_usage_properties(self, sample_chat_response):
        """Test usage token counts."""
        response = AnthropicChatResponses(sample_chat_response)

        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.total_tokens == 30

    def test_finish_reason_mapping(self):
        """Test all finish reason mappings."""
        mappings = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }

        for anthropic_reason, expected_reason in mappings.items():
            response = MagicMock()
            response.stop_reason = anthropic_reason
            response.content = []
            response.usage = MagicMock()
            response.usage.input_tokens = 0
            response.usage.output_tokens = 0

            wrapped = AnthropicChatResponses(response)
            assert wrapped.finish_reason == expected_reason

    def test_raw_response(self, sample_chat_response):
        """Test raw response access."""
        response = AnthropicChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestAnthropicChatCompletion:
    """Tests for AnthropicChatCompletion class."""

    def test_create_basic(self, sample_chat_response):
        """Test basic message creation."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        response = chat.create(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, AnthropicChatResponses)
        assert response.content == "Hello! How can I help you today?"
        mock_client.messages.create.assert_called_once()

    def test_create_with_model(self, sample_chat_response):
        """Test message creation with specific model."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-opus-4-20250514",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    def test_create_with_default_model(self, sample_chat_response):
        """Test default model is applied."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_create_with_default_max_tokens(self, sample_chat_response):
        """Test default max_tokens is applied."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    def test_system_message_extraction(self, sample_chat_response):
        """Test system message is extracted to separate parameter."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."
        # System message should not be in messages list
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    def test_streaming(self):
        """Test streaming response."""
        mock_client = MagicMock()

        # Create mock stream events
        event1 = MagicMock()
        event1.type = "message_start"
        event1.message = MagicMock()
        event1.message.id = "msg_123"
        event1.message.model = "claude-sonnet-4-20250514"
        event1.message.usage = MagicMock()
        event1.message.usage.input_tokens = 10
        event1.message.usage.output_tokens = 0

        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta = MagicMock()
        event2.delta.type = "text_delta"
        event2.delta.text = "Hello"

        event3 = MagicMock()
        event3.type = "content_block_delta"
        event3.delta = MagicMock()
        event3.delta.type = "text_delta"
        event3.delta.text = " world"

        event4 = MagicMock()
        event4.type = "message_delta"
        event4.delta = MagicMock()
        event4.delta.stop_reason = "end_turn"
        event4.usage = MagicMock()
        event4.usage.output_tokens = 5

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.__iter__ = MagicMock(
            return_value=iter([event1, event2, event3, event4])
        )
        mock_client.messages.stream.return_value = mock_stream

        chat = AnthropicChatCompletion(mock_client)
        chunks = list(
            chat.create(messages=[{"role": "user", "content": "Hi"}], stream=True)
        )

        assert len(chunks) >= 2
        # Check that content deltas are yielded
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) >= 2

    def test_tool_format_conversion(self, sample_chat_response):
        """Test OpenAI tool format is converted to Anthropic format."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)

        # OpenAI format tools
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        chat.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=openai_tools,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        anthropic_tools = call_kwargs["tools"]

        # Verify conversion to Anthropic format
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[0]["description"] == "Get the weather"
        assert "input_schema" in anthropic_tools[0]

    def test_native_anthropic_tool_format(self, sample_chat_response):
        """Test native Anthropic tool format passes through."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)

        # Native Anthropic format
        anthropic_tools = [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ]

        chat.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=anthropic_tools,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == anthropic_tools

    def test_multimodal_message(self, sample_chat_response):
        """Test multimodal message with image."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ]
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        # Should have converted to Anthropic's image format
        content = messages[0]["content"]
        assert isinstance(content, list)

    def test_assistant_message_with_tool_results(self, sample_chat_response):
        """Test handling of assistant messages with tool results."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": "72°F and sunny",
                },
            ]
        )

        mock_client.messages.create.assert_called_once()

    def test_error_authentication(self):
        """Test authentication error handling."""
        pytest.importorskip("anthropic")
        import anthropic

        mock_client = MagicMock()

        # Create a proper mock for AuthenticationError
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )

        chat = AnthropicChatCompletion(mock_client)

        with pytest.raises(AuthenticationError):
            chat.create(messages=[{"role": "user", "content": "Hello"}])

    def test_error_rate_limit(self):
        """Test rate limit error handling."""
        pytest.importorskip("anthropic")
        import anthropic

        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}

        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body={"error": {"message": "Rate limited"}},
        )

        chat = AnthropicChatCompletion(mock_client)

        with pytest.raises(RateLimitError):
            chat.create(messages=[{"role": "user", "content": "Hello"}])

    def test_error_invalid_request(self):
        """Test invalid request error handling."""
        pytest.importorskip("anthropic")
        import anthropic

        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}

        mock_client.messages.create.side_effect = anthropic.BadRequestError(
            message="Invalid request",
            response=mock_response,
            body={"error": {"message": "Invalid request"}},
        )

        chat = AnthropicChatCompletion(mock_client)

        with pytest.raises(InvalidRequestError):
            chat.create(messages=[{"role": "user", "content": "Hello"}])

    def test_error_api_error(self):
        """Test API error handling."""
        pytest.importorskip("anthropic")
        import anthropic

        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}

        mock_client.messages.create.side_effect = anthropic.APIStatusError(
            message="Internal error",
            response=mock_response,
            body={"error": {"message": "Internal error"}},
        )

        chat = AnthropicChatCompletion(mock_client)

        with pytest.raises(APIError):
            chat.create(messages=[{"role": "user", "content": "Hello"}])

    def test_string_message_format(self, sample_chat_response):
        """Test that string messages are properly formatted."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        chat = AnthropicChatCompletion(mock_client)
        chat.create(messages="Hello, Claude!")

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, Claude!"

    def test_message_object_format(self, sample_chat_response):
        """Test Message object format is handled."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = sample_chat_response

        from llm_connector import Message

        chat = AnthropicChatCompletion(mock_client)
        chat.create(messages=Message(role="user", content="Hello"))

        mock_client.messages.create.assert_called_once()
