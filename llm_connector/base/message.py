from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, model_validator
from typing import List, Union, Optional, Literal, Dict


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextBlock(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image content block."""

    type: Literal["image"] = "image"
    url: str
    detail: Literal["low", "high", "auto"] = "auto"


class DocumentBlock(BaseModel):
    """Document content block."""

    type: Literal["document"] = "document"
    data: Dict
    id: Optional[str] = None


ContentBlock = Union[
    TextBlock,
    ImageBlock,
    DocumentBlock,
]


class ToolCall(BaseModel):
    """Represents a tool call within a message."""

    id: str
    name: str
    arguments: Dict


class SystemMessage(BaseModel):
    """System or Developer message."""

    role: Literal[Role.SYSTEM, Role.DEVELOPER]
    content: List[TextBlock]

    @model_validator(mode="after")
    def validate_content(self):
        if not self.content:
            raise ValueError("System/Developer message must have text content")
        return self


class UserMessage(BaseModel):
    """User message."""

    role: Literal[Role.USER]
    content: List[ContentBlock]

    @model_validator(mode="after")
    def validate_content(self):
        if not self.content:
            raise ValueError("User message must have content")
        return self


class AssistantMessage(BaseModel):
    """Assistant message."""

    role: Literal[Role.ASSISTANT]
    content: Optional[List[TextBlock]] = None
    tool_calls: Optional[List[ToolCall]] = None

    @model_validator(mode="after")
    def validate_assistant(self):
        if self.content and self.tool_calls:
            raise ValueError(
                "Assistant message cannot have both content and tool_calls"
            )
        if not self.content and not self.tool_calls:
            raise ValueError("Assistant message must have content or tool_calls")
        return self


class ToolMessage(BaseModel):
    """Tool message."""

    role: Literal[Role.TOOL]
    tool_call_id: str
    content: List[TextBlock]

    @model_validator(mode="after")
    def validate_tool(self):
        if not self.content:
            raise ValueError("Tool message must have content")
        return self


Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
]


class Conversation(BaseModel):
    """Represents a conversation consisting of multiple messages."""

    messages: List[Message]

    def append(self, message: Message) -> None:
        self.messages.append(message)
