import pytest
from pydantic import ValidationError

from llm_connector import (
    Role,
    TextBlock,
    ImageBlock,
    DocumentBlock,
    ToolCall,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Conversation,
)


class TestTextBlock:
    """Tests for TextBlock."""

    def test_create_text_block(self):
        """Test creating a text block."""
        block = TextBlock(text="Hello, world!")
        assert block.type == "text"
        assert block.text == "Hello, world!"

    def test_text_block_default_type(self):
        """Test text block has default type."""
        block = TextBlock(text="Test")
        assert block.type == "text"


class TestImageBlock:
    """Tests for ImageBlock."""

    def test_create_image_block(self):
        """Test creating an image block."""
        block = ImageBlock(url="https://example.com/image.png")
        assert block.type == "image"
        assert block.url == "https://example.com/image.png"
        assert block.detail == "auto"

    def test_image_block_with_detail(self):
        """Test creating image block with detail level."""
        block = ImageBlock(url="https://example.com/image.png", detail="high")
        assert block.detail == "high"


class TestDocumentBlock:
    """Tests for DocumentBlock."""

    def test_create_document_block(self):
        """Test creating a document block."""
        block = DocumentBlock(data={"key": "value"})
        assert block.type == "document"
        assert block.data == {"key": "value"}
        assert block.id is None

    def test_document_block_with_id(self):
        """Test creating document block with ID."""
        block = DocumentBlock(data={"key": "value"}, id="doc-123")
        assert block.id == "doc-123"


class TestToolCall:
    """Tests for ToolCall."""

    def test_create_tool_call(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            id="call_123", name="get_weather", arguments={"location": "Tokyo"}
        )
        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Tokyo"}


class TestSystemMessage:
    """Tests for SystemMessage."""

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = SystemMessage(
            role=Role.SYSTEM, content=[TextBlock(text="You are a helpful assistant.")]
        )
        assert msg.role == Role.SYSTEM
        assert len(msg.content) == 1
        assert msg.content[0].text == "You are a helpful assistant."

    def test_create_developer_message(self):
        """Test creating a developer message."""
        msg = SystemMessage(
            role=Role.DEVELOPER, content=[TextBlock(text="Developer instructions.")]
        )
        assert msg.role == Role.DEVELOPER

    def test_system_message_empty_content_fails(self):
        """Test system message requires content."""
        with pytest.raises(ValidationError):
            SystemMessage(role=Role.SYSTEM, content=[])


class TestUserMessage:
    """Tests for UserMessage."""

    def test_create_user_message_text(self):
        """Test creating a user message with text."""
        msg = UserMessage(role=Role.USER, content=[TextBlock(text="Hello!")])
        assert msg.role == Role.USER
        assert msg.content[0].text == "Hello!"

    def test_create_user_message_multimodal(self):
        """Test creating a multimodal user message."""
        msg = UserMessage(
            role=Role.USER,
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(url="https://example.com/image.png"),
            ],
        )
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image"

    def test_user_message_empty_content_fails(self):
        """Test user message requires content."""
        with pytest.raises(ValidationError):
            UserMessage(role=Role.USER, content=[])


class TestAssistantMessage:
    """Tests for AssistantMessage."""

    def test_create_assistant_message_content(self):
        """Test creating assistant message with content."""
        msg = AssistantMessage(
            role=Role.ASSISTANT, content=[TextBlock(text="I can help with that!")]
        )
        assert msg.role == Role.ASSISTANT
        assert msg.content[0].text == "I can help with that!"
        assert msg.tool_calls is None

    def test_create_assistant_message_tool_calls(self):
        """Test creating assistant message with tool calls."""
        msg = AssistantMessage(
            role=Role.ASSISTANT,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"})
            ],
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_assistant_message_both_content_and_tools_fails(self):
        """Test assistant message cannot have both content and tool calls."""
        with pytest.raises(ValidationError):
            AssistantMessage(
                role=Role.ASSISTANT,
                content=[TextBlock(text="Text")],
                tool_calls=[ToolCall(id="call_1", name="func", arguments={})],
            )

    def test_assistant_message_neither_content_nor_tools_fails(self):
        """Test assistant message must have content or tool calls."""
        with pytest.raises(ValidationError):
            AssistantMessage(role=Role.ASSISTANT)


class TestToolMessage:
    """Tests for ToolMessage."""

    def test_create_tool_message(self):
        """Test creating a tool message."""
        msg = ToolMessage(
            role=Role.TOOL,
            tool_call_id="call_123",
            content=[TextBlock(text='{"temperature": 72}')],
        )
        assert msg.role == Role.TOOL
        assert msg.tool_call_id == "call_123"
        assert msg.content[0].text == '{"temperature": 72}'

    def test_tool_message_empty_content_fails(self):
        """Test tool message requires content."""
        with pytest.raises(ValidationError):
            ToolMessage(role=Role.TOOL, tool_call_id="call_123", content=[])


class TestConversation:
    """Tests for Conversation."""

    def test_create_conversation(self):
        """Test creating a conversation."""
        conv = Conversation(
            messages=[
                SystemMessage(
                    role=Role.SYSTEM, content=[TextBlock(text="System prompt")]
                ),
                UserMessage(role=Role.USER, content=[TextBlock(text="Hello")]),
            ]
        )
        assert len(conv.messages) == 2

    def test_conversation_append(self):
        """Test appending to a conversation."""
        conv = Conversation(messages=[])
        conv.append(UserMessage(role=Role.USER, content=[TextBlock(text="Hello")]))
        assert len(conv.messages) == 1

    def test_empty_conversation(self):
        """Test creating an empty conversation."""
        conv = Conversation(messages=[])
        assert len(conv.messages) == 0
