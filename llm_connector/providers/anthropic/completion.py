from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    AsyncGenerator,
    Optional,
    List,
    Union,
    Generator,
    TYPE_CHECKING,
)

from ...exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ContentFilterError,
    ContextLengthExceededError,
)
from ...base import (
    ChatCompletion,
    AsyncChatCompletion,
    ChatResponses,
    ChatStreamChunks,
    Usage,
    ToolCallDelta,
    Message,
    ToolCall,
    TextBlock,
    ImageBlock,
    DocumentBlock,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Role,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic import AsyncAnthropic


class AnthropicChatResponses(ChatResponses):
    """Anthropic chat completion response wrapper."""

    def __init__(self, response) -> None:
        self._response = response

    @property
    def id(self) -> str:
        return self._response.id

    @property
    def model(self) -> str:
        return self._response.model

    @property
    def content(self) -> Optional[str]:
        """Extract text content from response."""
        if self._response.content:
            text_parts = []
            for block in self._response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts) if text_parts else None
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        """Extract tool use blocks from response."""
        if self._response.content:
            tool_calls = []
            for block in self._response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=(
                                block.input if isinstance(block.input, dict) else {}
                            ),
                        )
                    )
            return tool_calls if tool_calls else None
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        """Map Anthropic stop_reason to standard finish_reason."""
        stop_reason = self._response.stop_reason
        if stop_reason == "end_turn":
            return "stop"
        elif stop_reason == "tool_use":
            return "tool_calls"
        elif stop_reason == "max_tokens":
            return "length"
        elif stop_reason == "stop_sequence":
            return "stop"
        return stop_reason

    @property
    def usage(self) -> Optional[Usage]:
        if self._response.usage:
            return Usage(
                prompt_tokens=self._response.usage.input_tokens,
                completion_tokens=self._response.usage.output_tokens,
                total_tokens=(
                    self._response.usage.input_tokens
                    + self._response.usage.output_tokens
                ),
            )
        return None

    @property
    def raw(self):
        """Access the raw Anthropic response."""
        return self._response


class AnthropicAsyncChatResponses(ChatResponses):
    """Anthropic async chat completion response wrapper."""

    def __init__(self, response) -> None:
        self._response = response

    @property
    def id(self) -> str:
        return self._response.id

    @property
    def model(self) -> str:
        return self._response.model

    @property
    def content(self) -> Optional[str]:
        """Extract text content from response."""
        if self._response.content:
            text_parts = []
            for block in self._response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts) if text_parts else None
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        """Extract tool use blocks from response."""
        if self._response.content:
            tool_calls = []
            for block in self._response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=(
                                block.input if isinstance(block.input, dict) else {}
                            ),
                        )
                    )
            return tool_calls if tool_calls else None
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        """Map Anthropic stop_reason to standard finish_reason."""
        stop_reason = self._response.stop_reason
        if stop_reason == "end_turn":
            return "stop"
        elif stop_reason == "tool_use":
            return "tool_calls"
        elif stop_reason == "max_tokens":
            return "length"
        elif stop_reason == "stop_sequence":
            return "stop"
        return stop_reason

    @property
    def usage(self) -> Optional[Usage]:
        if self._response.usage:
            return Usage(
                prompt_tokens=self._response.usage.input_tokens,
                completion_tokens=self._response.usage.output_tokens,
                total_tokens=(
                    self._response.usage.input_tokens
                    + self._response.usage.output_tokens
                ),
            )
        return None

    @property
    def raw(self):
        """Access the raw Anthropic response."""
        return self._response


class AnthropicChatStreamChunks(ChatStreamChunks):
    """Anthropic streaming chunk wrapper.

    Anthropic streaming events include:
    - message_start: Contains message metadata and usage.input_tokens
    - content_block_start: Start of a content block (text or tool_use)
    - content_block_delta: Delta content for text or tool input
    - content_block_stop: End of a content block
    - message_delta: Contains stop_reason and usage.output_tokens
    - message_stop: End of message
    """

    def __init__(
        self,
        event,
        message_id: str = "",
        model: str = "",
        accumulated_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        self._event = event
        self._message_id = message_id
        self._model = model
        self._accumulated_usage = accumulated_usage or {}

    @property
    def id(self) -> str:
        return self._message_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def delta_content(self) -> Optional[str]:
        """Extract text delta from content_block_delta event."""
        if hasattr(self._event, "type"):
            if self._event.type == "content_block_delta":
                delta = self._event.delta
                if hasattr(delta, "type") and delta.type == "text_delta":
                    return delta.text
        return None

    @property
    def delta_tool_calls(self) -> Optional[List[ToolCallDelta]]:
        """Extract tool call deltas from streaming events."""
        if hasattr(self._event, "type"):
            # Handle content_block_start for tool_use
            if self._event.type == "content_block_start":
                content_block = self._event.content_block
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    return [
                        ToolCallDelta(
                            index=self._event.index,
                            id=content_block.id,
                            name=content_block.name,
                            arguments=None,
                        )
                    ]
            # Handle content_block_delta for tool input
            elif self._event.type == "content_block_delta":
                delta = self._event.delta
                if hasattr(delta, "type") and delta.type == "input_json_delta":
                    return [
                        ToolCallDelta(
                            index=self._event.index,
                            id=None,
                            name=None,
                            arguments=delta.partial_json,
                        )
                    ]
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        """Extract finish reason from message_delta event."""
        if hasattr(self._event, "type") and self._event.type == "message_delta":
            stop_reason = self._event.delta.stop_reason
            if stop_reason == "end_turn":
                return "stop"
            elif stop_reason == "tool_use":
                return "tool_calls"
            elif stop_reason == "max_tokens":
                return "length"
            elif stop_reason == "stop_sequence":
                return "stop"
            return stop_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        """Return usage if we have complete token counts."""
        if (
            "input_tokens" in self._accumulated_usage
            and "output_tokens" in self._accumulated_usage
        ):
            return Usage(
                prompt_tokens=self._accumulated_usage["input_tokens"],
                completion_tokens=self._accumulated_usage["output_tokens"],
                total_tokens=(
                    self._accumulated_usage["input_tokens"]
                    + self._accumulated_usage["output_tokens"]
                ),
            )
        return None

    @property
    def raw(self):
        """Access the raw Anthropic event."""
        return self._event


class AnthropicAsyncChatStreamChunks(ChatStreamChunks):
    """Anthropic async streaming chunk wrapper."""

    def __init__(
        self,
        event,
        message_id: str = "",
        model: str = "",
        accumulated_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        self._event = event
        self._message_id = message_id
        self._model = model
        self._accumulated_usage = accumulated_usage or {}

    @property
    def id(self) -> str:
        return self._message_id

    @property
    def model(self) -> str:
        return self._model

    @property
    def delta_content(self) -> Optional[str]:
        """Extract text delta from content_block_delta event."""
        if hasattr(self._event, "type"):
            if self._event.type == "content_block_delta":
                delta = self._event.delta
                if hasattr(delta, "type") and delta.type == "text_delta":
                    return delta.text
        return None

    @property
    def delta_tool_calls(self) -> Optional[List[ToolCallDelta]]:
        """Extract tool call deltas from streaming events."""
        if hasattr(self._event, "type"):
            if self._event.type == "content_block_start":
                content_block = self._event.content_block
                if hasattr(content_block, "type") and content_block.type == "tool_use":
                    return [
                        ToolCallDelta(
                            index=self._event.index,
                            id=content_block.id,
                            name=content_block.name,
                            arguments=None,
                        )
                    ]
            elif self._event.type == "content_block_delta":
                delta = self._event.delta
                if hasattr(delta, "type") and delta.type == "input_json_delta":
                    return [
                        ToolCallDelta(
                            index=self._event.index,
                            id=None,
                            name=None,
                            arguments=delta.partial_json,
                        )
                    ]
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        """Extract finish reason from message_delta event."""
        if hasattr(self._event, "type") and self._event.type == "message_delta":
            stop_reason = self._event.delta.stop_reason
            if stop_reason == "end_turn":
                return "stop"
            elif stop_reason == "tool_use":
                return "tool_calls"
            elif stop_reason == "max_tokens":
                return "length"
            elif stop_reason == "stop_sequence":
                return "stop"
            return stop_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        """Return usage if we have complete token counts."""
        if (
            "input_tokens" in self._accumulated_usage
            and "output_tokens" in self._accumulated_usage
        ):
            return Usage(
                prompt_tokens=self._accumulated_usage["input_tokens"],
                completion_tokens=self._accumulated_usage["output_tokens"],
                total_tokens=(
                    self._accumulated_usage["input_tokens"]
                    + self._accumulated_usage["output_tokens"]
                ),
            )
        return None

    @property
    def raw(self):
        """Access the raw Anthropic event."""
        return self._event


class AnthropicChatCompletion(ChatCompletion):
    """Anthropic Chat Completion API implementation."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, client: "Anthropic") -> None:
        self._client = client

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
    ) -> Union[
        AnthropicChatResponses, Generator[AnthropicChatStreamChunks, None, None]
    ]:
        """
        Send a chat completion request to Anthropic.

        Args:
            messages: Input messages
            tools: Tool definitions for function calling
            model: Model to use (defaults to claude-sonnet-4-20250514)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (required by Anthropic, defaults to 4096)
            stream: Whether to stream the response
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            AnthropicChatResponses or generator of AnthropicChatStreamChunks
        """
        system_prompt, formatted_messages = self._format_messages(messages)

        request_params: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._format_tools(tools)

        if temperature is not None:
            request_params["temperature"] = temperature

        # Handle additional kwargs, but filter out OpenAI-specific ones
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("stream_options",)
        }
        request_params.update(filtered_kwargs)

        try:
            if stream:
                return self._stream(request_params)
            else:
                response = self._client.messages.create(**request_params)
                return AnthropicChatResponses(response)
        except Exception as e:
            raise self._handle_exception(e)

    def _stream(
        self, request_params: Dict[str, Any]
    ) -> Generator[AnthropicChatStreamChunks, None, None]:
        """Generate streaming response chunks."""
        try:
            message_id = ""
            model = ""
            accumulated_usage: Dict[str, int] = {}

            with self._client.messages.stream(**request_params) as stream:
                for event in stream:
                    # Extract message metadata from message_start
                    if hasattr(event, "type") and event.type == "message_start":
                        if hasattr(event, "message"):
                            message_id = event.message.id
                            model = event.message.model
                            if event.message.usage:
                                accumulated_usage["input_tokens"] = (
                                    event.message.usage.input_tokens
                                )

                    # Extract output tokens from message_delta
                    if hasattr(event, "type") and event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            accumulated_usage["output_tokens"] = (
                                event.usage.output_tokens
                            )

                    yield AnthropicChatStreamChunks(
                        event,
                        message_id=message_id,
                        model=model,
                        accumulated_usage=accumulated_usage.copy(),
                    )
        except Exception as e:
            raise self._handle_exception(e)

    def _format_messages(
        self, messages: Union[str, Message, List[Message]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, formatted_messages)
            System messages are extracted and returned separately.
        """
        if isinstance(messages, str):
            return None, [{"role": "user", "content": messages}]

        if not isinstance(messages, list):
            messages = [messages]

        system_parts: List[str] = []
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Collect system messages separately
                system_text = " ".join(block.text for block in msg.content)
                system_parts.append(system_text)
            else:
                formatted.append(self._format_single_message(msg))

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, formatted

    def _format_single_message(self, msg: Message) -> Dict[str, Any]:
        """Convert a single message to Anthropic format."""
        if isinstance(msg, SystemMessage):
            # This shouldn't be called for system messages, but handle it
            content = " ".join(block.text for block in msg.content)
            return {"role": "user", "content": content}

        elif isinstance(msg, UserMessage):
            content = self._format_content_blocks(msg.content)
            # If single text, simplify
            if len(content) == 1 and content[0].get("type") == "text":
                return {"role": "user", "content": content[0]["text"]}
            return {"role": "user", "content": content}

        elif isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                # Assistant message with tool use
                content: List[Dict[str, Any]] = []
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                return {"role": "assistant", "content": content}
            elif msg.content:
                text = " ".join(block.text for block in msg.content)
                return {"role": "assistant", "content": text}
            else:
                return {"role": "assistant", "content": ""}

        elif isinstance(msg, ToolMessage):
            # Tool results in Anthropic format
            content = " ".join(block.text for block in msg.content)
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content,
                    }
                ],
            }

        else:
            raise InvalidRequestError(f"Unknown message type: {type(msg)}")

    def _format_content_blocks(
        self, content: List[Union[TextBlock, ImageBlock, DocumentBlock]]
    ) -> List[Dict[str, Any]]:
        """Convert content blocks to Anthropic format."""
        formatted = []

        for block in content:
            if isinstance(block, TextBlock):
                formatted.append({"type": "text", "text": block.text})

            elif isinstance(block, ImageBlock):
                # Anthropic image format
                if block.url.startswith("data:"):
                    # Base64 data URL
                    # Parse: data:image/jpeg;base64,<data>
                    try:
                        header, data = block.url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        formatted.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    except (ValueError, IndexError):
                        # Fallback: treat as URL
                        formatted.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": block.url,
                                },
                            }
                        )
                else:
                    # Regular URL
                    formatted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": block.url,
                            },
                        }
                    )

            elif isinstance(block, DocumentBlock):
                # Convert document to text representation
                formatted.append(
                    {
                        "type": "text",
                        "text": f"[Document: {json.dumps(block.data)}]",
                    }
                )

        return formatted

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        formatted = []
        for tool in tools:
            # Handle OpenAI-style tool format
            if "type" in tool and tool["type"] == "function":
                func = tool["function"]
                formatted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            elif "parameters" in tool:
                # Simple format with parameters
                formatted.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool["parameters"],
                    }
                )
            elif "input_schema" in tool:
                # Already in Anthropic format
                formatted.append(tool)
            else:
                # Minimal tool definition
                formatted.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": {"type": "object", "properties": {}},
                    }
                )
        return formatted

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return APIError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))

        elif isinstance(e, anthropic.RateLimitError):
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            return RateLimitError(str(e), retry_after=retry_after)

        elif isinstance(e, anthropic.BadRequestError):
            error_msg = str(e).lower()
            if "context_length" in error_msg or "too many tokens" in error_msg:
                return ContextLengthExceededError(str(e))
            if "content" in error_msg and "blocked" in error_msg:
                return ContentFilterError(str(e))
            return InvalidRequestError(str(e))

        elif isinstance(e, anthropic.APIError):
            return APIError(
                str(e),
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            )

        else:
            return APIError(str(e))


class AnthropicAsyncChatCompletion(AsyncChatCompletion):
    """Anthropic Async Chat Completion API implementation."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, client: "AsyncAnthropic") -> None:
        self._client = client

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
    ) -> Union[
        AnthropicAsyncChatResponses,
        AsyncGenerator[AnthropicAsyncChatStreamChunks, None],
    ]:
        """
        Send an async chat completion request to Anthropic.

        Args:
            messages: Input messages
            tools: Tool definitions for function calling
            model: Model to use (defaults to claude-sonnet-4-20250514)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (required by Anthropic, defaults to 4096)
            stream: Whether to stream the response
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            AnthropicAsyncChatResponses or async generator of AnthropicAsyncChatStreamChunks
        """
        system_prompt, formatted_messages = self._format_messages(messages)

        request_params: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }

        if system_prompt:
            request_params["system"] = system_prompt

        if tools:
            request_params["tools"] = self._format_tools(tools)

        if temperature is not None:
            request_params["temperature"] = temperature

        # Handle additional kwargs, but filter out OpenAI-specific ones
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("stream_options",)
        }
        request_params.update(filtered_kwargs)

        try:
            if stream:
                return self._stream(request_params)
            else:
                response = await self._client.messages.create(**request_params)
                return AnthropicAsyncChatResponses(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def _stream(
        self, request_params: Dict[str, Any]
    ) -> AsyncGenerator[AnthropicAsyncChatStreamChunks, None]:
        """Generate async streaming response chunks."""
        try:
            message_id = ""
            model = ""
            accumulated_usage: Dict[str, int] = {}

            async with self._client.messages.stream(**request_params) as stream:
                async for event in stream:
                    # Extract message metadata from message_start
                    if hasattr(event, "type") and event.type == "message_start":
                        if hasattr(event, "message"):
                            message_id = event.message.id
                            model = event.message.model
                            if event.message.usage:
                                accumulated_usage["input_tokens"] = (
                                    event.message.usage.input_tokens
                                )

                    # Extract output tokens from message_delta
                    if hasattr(event, "type") and event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            accumulated_usage["output_tokens"] = (
                                event.usage.output_tokens
                            )

                    yield AnthropicAsyncChatStreamChunks(
                        event,
                        message_id=message_id,
                        model=model,
                        accumulated_usage=accumulated_usage.copy(),
                    )
        except Exception as e:
            raise self._handle_exception(e)

    def _format_messages(
        self, messages: Union[str, Message, List[Message]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, formatted_messages)
            System messages are extracted and returned separately.
        """
        if isinstance(messages, str):
            return None, [{"role": "user", "content": messages}]

        if not isinstance(messages, list):
            messages = [messages]

        system_parts: List[str] = []
        formatted: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_text = " ".join(block.text for block in msg.content)
                system_parts.append(system_text)
            else:
                formatted.append(self._format_single_message(msg))

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, formatted

    def _format_single_message(self, msg: Message) -> Dict[str, Any]:
        """Convert a single message to Anthropic format."""
        if isinstance(msg, SystemMessage):
            content = " ".join(block.text for block in msg.content)
            return {"role": "user", "content": content}

        elif isinstance(msg, UserMessage):
            content = self._format_content_blocks(msg.content)
            if len(content) == 1 and content[0].get("type") == "text":
                return {"role": "user", "content": content[0]["text"]}
            return {"role": "user", "content": content}

        elif isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                content: List[Dict[str, Any]] = []
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                return {"role": "assistant", "content": content}
            elif msg.content:
                text = " ".join(block.text for block in msg.content)
                return {"role": "assistant", "content": text}
            else:
                return {"role": "assistant", "content": ""}

        elif isinstance(msg, ToolMessage):
            content = " ".join(block.text for block in msg.content)
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content,
                    }
                ],
            }

        else:
            raise InvalidRequestError(f"Unknown message type: {type(msg)}")

    def _format_content_blocks(
        self, content: List[Union[TextBlock, ImageBlock, DocumentBlock]]
    ) -> List[Dict[str, Any]]:
        """Convert content blocks to Anthropic format."""
        formatted = []

        for block in content:
            if isinstance(block, TextBlock):
                formatted.append({"type": "text", "text": block.text})

            elif isinstance(block, ImageBlock):
                if block.url.startswith("data:"):
                    try:
                        header, data = block.url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        formatted.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    except (ValueError, IndexError):
                        formatted.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": block.url,
                                },
                            }
                        )
                else:
                    formatted.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": block.url,
                            },
                        }
                    )

            elif isinstance(block, DocumentBlock):
                formatted.append(
                    {
                        "type": "text",
                        "text": f"[Document: {json.dumps(block.data)}]",
                    }
                )

        return formatted

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to Anthropic format."""
        formatted = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                func = tool["function"]
                formatted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            elif "parameters" in tool:
                formatted.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool["parameters"],
                    }
                )
            elif "input_schema" in tool:
                formatted.append(tool)
            else:
                formatted.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": {"type": "object", "properties": {}},
                    }
                )
        return formatted

    def _handle_exception(self, e: Exception) -> Exception:
        """Convert Anthropic exceptions to our custom exceptions."""
        try:
            import anthropic
        except ImportError:
            return APIError(str(e))

        if isinstance(e, anthropic.AuthenticationError):
            return AuthenticationError(str(e))

        elif isinstance(e, anthropic.RateLimitError):
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        pass
            return RateLimitError(str(e), retry_after=retry_after)

        elif isinstance(e, anthropic.BadRequestError):
            error_msg = str(e).lower()
            if "context_length" in error_msg or "too many tokens" in error_msg:
                return ContextLengthExceededError(str(e))
            if "content" in error_msg and "blocked" in error_msg:
                return ContentFilterError(str(e))
            return InvalidRequestError(str(e))

        elif isinstance(e, anthropic.APIError):
            return APIError(
                str(e),
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            )

        else:
            return APIError(str(e))
