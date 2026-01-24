from __future__ import annotations

from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional, List, Union

from .message import Message, ToolCall


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ToolCallDelta(BaseModel):
    """Represents a partial tool call during streaming."""

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatResponses(ABC):
    """Abstract base class for chat completion responses."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the completion."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model used for the completion."""
        pass

    @property
    @abstractmethod
    def content(self) -> Optional[str]:
        """Text content of the response."""
        pass

    @property
    @abstractmethod
    def tool_calls(self) -> Optional[List[ToolCall]]:
        """Tool calls made by the model."""
        pass

    @property
    @abstractmethod
    def finish_reason(self) -> Optional[str]:
        """Reason the model stopped generating."""
        pass

    @property
    @abstractmethod
    def usage(self) -> Optional[Usage]:
        """Token usage statistics."""
        pass


class ChatStreamChunks(ABC):
    """Abstract base class for streaming chat completion chunks."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the completion."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model used for the completion."""
        pass

    @property
    @abstractmethod
    def delta_content(self) -> Optional[str]:
        """Text content delta in this chunk."""
        pass

    @property
    @abstractmethod
    def delta_tool_calls(self) -> Optional[List[ToolCallDelta]]:
        """Tool call deltas in this chunk."""
        pass

    @property
    @abstractmethod
    def finish_reason(self) -> Optional[str]:
        """Reason the model stopped generating (only on final chunk)."""
        pass

    @property
    @abstractmethod
    def usage(self) -> Optional[Usage]:
        """Token usage (only available with stream_options)."""
        pass


class ChatCompletion(ABC):
    """Abstract base class for chat completion API."""

    @abstractmethod
    def invoke(
        self,
        *,
        messages: Union[str, Message, List[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ChatResponses, Iterator[ChatStreamChunks]]:
        """
        Send a chat completion request.

        Args:
            messages: Input messages - can be a string, single Message, or list of Messages
            tools: List of tool definitions for function calling
            model: Model to use (provider-specific default if not specified)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponses if stream=False, Iterator[ChatStreamChunks] if stream=True
        """
        pass


class AsyncChatCompletion(ABC):
    """Abstract base class for async chat completion API."""

    @abstractmethod
    async def invoke(
        self,
        *,
        messages: Union[str, Message, List[Message]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ChatResponses, AsyncIterator[ChatStreamChunks]]:
        """
        Send an async chat completion request.

        Args:
            messages: Input messages - can be a string, single Message, or list of Messages
            tools: List of tool definitions for function calling
            model: Model to use (provider-specific default if not specified)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ChatResponses if stream=False, AsyncIterator[ChatStreamChunks] if stream=True
        """
        pass
