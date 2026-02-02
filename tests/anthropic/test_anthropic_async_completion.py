"""Tests for Anthropic async chat completion functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from llm_connector.providers.anthropic.completion import (
    AnthropicAsyncChatCompletion,
    AnthropicAsyncChatResponses,
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


class TestAnthropicAsyncChatResponses:
    """Tests for AnthropicAsyncChatResponses class."""

    def test_response_properties(self, sample_chat_response):
        """Test basic response properties."""
        response = AnthropicAsyncChatResponses(sample_chat_response)

        assert response.id == "msg_01XFDUDYJgAACzvnptvVoYEL"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.content == "Hello! How can I help you today?"
        assert response.finish_reason == "stop"  # end_turn maps to stop

    def test_response_with_tool_calls(self, sample_tool_call_response):
        """Test response with tool calls."""
        response = AnthropicAsyncChatResponses(sample_tool_call_response)

        assert response.content is None
        assert response.finish_reason == "tool_calls"  # tool_use maps to tool_calls
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["id"] == "toolu_01A09q90qw90lq917835lgs"
        assert response.tool_calls[0]["function"]["name"] == "get_weather"

    def test_usage_properties(self, sample_chat_response):
        """Test usage token counts."""
        response = AnthropicAsyncChatResponses(sample_chat_response)

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

            wrapped = AnthropicAsyncChatResponses(response)
            assert wrapped.finish_reason == expected_reason

    def test_raw_response(self, sample_chat_response):
        """Test raw response access."""
        response = AnthropicAsyncChatResponses(sample_chat_response)
        assert response.raw == sample_chat_response


class TestAnthropicAsyncChatCompletion:
    """Tests for AnthropicAsyncChatCompletion class."""

    @pytest.mark.asyncio
    async def test_create_basic(self, sample_chat_response):
        """Test basic async message creation."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)
        response = await chat.create(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, AnthropicAsyncChatResponses)
        assert response.content == "Hello! How can I help you today?"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_model(self, sample_chat_response):
        """Test message creation with specific model."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)
        await chat.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-opus-4-20250514",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_create_with_default_model(self, sample_chat_response):
        """Test default model is applied."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)
        await chat.create(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_create_with_default_max_tokens(self, sample_chat_response):
        """Test default max_tokens is applied."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)
        await chat.create(messages=[{"role": "user", "content": "Hello"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_system_message_extraction(self, sample_chat_response):
        """Test system message is extracted to separate parameter."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)
        await chat.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ]
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test async streaming response."""
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

        # Create async context manager mock
        async def async_stream_iter():
            for event in [event1, event2, event3, event4]:
                yield event

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: async_stream_iter()
        mock_client.messages.stream.return_value = mock_stream

        chat = AnthropicAsyncChatCompletion(mock_client)
        chunks = []
        async for chunk in await chat.create(
            messages=[{"role": "user", "content": "Hi"}], stream=True
        ):
            chunks.append(chunk)

        assert len(chunks) >= 2
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) >= 2

    @pytest.mark.asyncio
    async def test_tool_format_conversion(self, sample_chat_response):
        """Test OpenAI tool format is converted to Anthropic format."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=sample_chat_response)

        chat = AnthropicAsyncChatCompletion(mock_client)

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

        await chat.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=openai_tools,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        anthropic_tools = call_kwargs["tools"]

        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[0]["description"] == "Get the weather"
        assert "input_schema" in anthropic_tools[0]

    @pytest.mark.asyncio
    async def test_error_authentication(self):
        """Test authentication error handling."""
        pytest.importorskip("anthropic")
        import anthropic

        mock_client = MagicMock()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}

        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )
        )

        chat = AnthropicAsyncChatCompletion(mock_client)

        with pytest.raises(AuthenticationError):
            await chat.create(messages=[{"role": "user", "content": "Hello"}])
