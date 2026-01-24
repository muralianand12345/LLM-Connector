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
    from groq import Groq
    from groq import AsyncGroq


class GroqChatResponses(ChatResponses):
    """Groq chat completion response wrapper."""

    def __init__(self, response) -> None:
        self._response = response
        self._choice = response.choices[0] if response.choices else None

    @property
    def id(self) -> str:
        return self._response.id

    @property
    def model(self) -> str:
        return self._response.model

    @property
    def content(self) -> Optional[str]:
        if self._choice and self._choice.message:
            return self._choice.message.content
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        if self._choice and self._choice.message and self._choice.message.tool_calls:
            result = []
            for tc in self._choice.message.tool_calls:
                try:
                    arguments = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    arguments = {}

                result.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )
            return result
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        if self._choice:
            return self._choice.finish_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        if self._response.usage:
            return Usage(
                prompt_tokens=self._response.usage.prompt_tokens,
                completion_tokens=self._response.usage.completion_tokens,
                total_tokens=self._response.usage.total_tokens,
            )
        return None

    @property
    def raw(self):
        """Access the raw Groq response."""
        return self._response


class GroqAsyncChatResponses(ChatResponses):
    """Groq async chat completion response wrapper."""

    def __init__(self, response) -> None:
        self._response = response
        self._choice = response.choices[0] if response.choices else None

    @property
    def id(self) -> str:
        return self._response.id

    @property
    def model(self) -> str:
        return self._response.model

    @property
    def content(self) -> Optional[str]:
        if self._choice and self._choice.message:
            return self._choice.message.content
        return None

    @property
    def tool_calls(self) -> Optional[List[ToolCall]]:
        if self._choice and self._choice.message and self._choice.message.tool_calls:
            return [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=(
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    ),
                )
                for tc in self._choice.message.tool_calls
            ]
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        if self._choice:
            return self._choice.finish_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        if self._response.usage:
            return Usage(
                prompt_tokens=self._response.usage.prompt_tokens,
                completion_tokens=self._response.usage.completion_tokens,
                total_tokens=self._response.usage.total_tokens,
            )
        return None

    @property
    def raw(self):
        """Access the raw Groq response."""
        return self._response


class GroqChatStreamChunks(ChatStreamChunks):
    """Groq streaming chunk wrapper."""

    def __init__(self, chunk) -> None:
        self._chunk = chunk
        self._choice = chunk.choices[0] if chunk.choices else None

    @property
    def id(self) -> str:
        return self._chunk.id

    @property
    def model(self) -> str:
        return self._chunk.model

    @property
    def delta_content(self) -> Optional[str]:
        if self._choice and self._choice.delta:
            return self._choice.delta.content
        return None

    @property
    def delta_tool_calls(self) -> Optional[List[ToolCallDelta]]:
        if self._choice and self._choice.delta and self._choice.delta.tool_calls:
            return [
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    name=tc.function.name if tc.function else None,
                    arguments=tc.function.arguments if tc.function else None,
                )
                for tc in self._choice.delta.tool_calls
            ]
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        if self._choice:
            return self._choice.finish_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        if hasattr(self._chunk, "usage") and self._chunk.usage:
            return Usage(
                prompt_tokens=self._chunk.usage.prompt_tokens,
                completion_tokens=self._chunk.usage.completion_tokens,
                total_tokens=self._chunk.usage.total_tokens,
            )
        if (
            hasattr(self._chunk, "x_groq")
            and self._chunk.x_groq
            and hasattr(self._chunk.x_groq, "usage")
            and self._chunk.x_groq.usage
        ):
            usage_data = self._chunk.x_groq.usage
            return Usage(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        return None

    @property
    def raw(self):
        """Access the raw Groq chunk."""
        return self._chunk


class GroqAsyncChatStreamChunks(ChatStreamChunks):
    """Groq async streaming chunk wrapper."""

    def __init__(self, chunk) -> None:
        self._chunk = chunk
        self._choice = chunk.choices[0] if chunk.choices else None

    @property
    def id(self) -> str:
        return self._chunk.id

    @property
    def model(self) -> str:
        return self._chunk.model

    @property
    def delta_content(self) -> Optional[str]:
        if self._choice and self._choice.delta:
            return self._choice.delta.content
        return None

    @property
    def delta_tool_calls(self) -> Optional[List[ToolCallDelta]]:
        if self._choice and self._choice.delta and self._choice.delta.tool_calls:
            return [
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    name=tc.function.name if tc.function else None,
                    arguments=tc.function.arguments if tc.function else None,
                )
                for tc in self._choice.delta.tool_calls
            ]
        return None

    @property
    def finish_reason(self) -> Optional[str]:
        if self._choice:
            return self._choice.finish_reason
        return None

    @property
    def usage(self) -> Optional[Usage]:
        if hasattr(self._chunk, "usage") and self._chunk.usage:
            return Usage(
                prompt_tokens=self._chunk.usage.prompt_tokens,
                completion_tokens=self._chunk.usage.completion_tokens,
                total_tokens=self._chunk.usage.total_tokens,
            )
        if (
            hasattr(self._chunk, "x_groq")
            and self._chunk.x_groq
            and hasattr(self._chunk.x_groq, "usage")
        ):
            usage_data = self._chunk.x_groq.usage
            return Usage(
                prompt_tokens=usage_data.prompt_tokens,
                completion_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
            )
        return None

    @property
    def raw(self):
        """Access the raw Groq chunk."""
        return self._chunk


class GroqChatCompletion(ChatCompletion):
    """Groq Chat Completion API implementation."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, client: "Groq") -> None:
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
    ) -> Union[GroqChatResponses, Generator[GroqChatStreamChunks, None, None]]:
        formatted_messages = self._format_messages(messages)
        request_params: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
        }

        if tools:
            request_params["tools"] = self._format_tools(tools)
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if stream:
            request_params["stream"] = True

        request_params.update(kwargs)

        try:
            if stream:
                return self._stream(request_params)
            else:
                response = self._client.chat.completions.create(**request_params)
                return GroqChatResponses(response)
        except Exception as e:
            raise self._handle_exception(e)

    def _stream(
        self, request_params: Dict[str, Any]
    ) -> Generator[GroqChatStreamChunks, None, None]:
        try:
            response = self._client.chat.completions.create(**request_params)
            for chunk in response:
                yield GroqChatStreamChunks(chunk)
        except Exception as e:
            raise self._handle_exception(e)

    def _format_messages(
        self, messages: Union[str, Message, List[Message]]
    ) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list):
            messages = [messages]
        return [self._format_single_message(msg) for msg in messages]

    def _format_single_message(self, msg: Message) -> Dict[str, Any]:
        if isinstance(msg, SystemMessage):
            content = " ".join(block.text for block in msg.content)
            return {"role": "system", "content": content}
        elif isinstance(msg, UserMessage):
            content = self._format_content_blocks(msg.content)
            if len(content) == 1 and content[0].get("type") == "text":
                return {"role": "user", "content": content[0]["text"]}
            return {"role": "user", "content": content}
        elif isinstance(msg, AssistantMessage):
            result: Dict[str, Any] = {"role": "assistant"}
            if msg.content:
                result["content"] = " ".join(block.text for block in msg.content)
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return result
        elif isinstance(msg, ToolMessage):
            content = " ".join(block.text for block in msg.content)
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            }
        else:
            raise InvalidRequestError(f"Unknown message type: {type(msg)}")

    def _format_content_blocks(
        self, content: List[Union[TextBlock, ImageBlock, DocumentBlock]]
    ) -> List[Dict[str, Any]]:
        formatted = []
        for block in content:
            if isinstance(block, TextBlock):
                formatted.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                formatted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": block.url, "detail": block.detail},
                    }
                )
            elif isinstance(block, DocumentBlock):
                formatted.append(
                    {"type": "text", "text": f"[Document: {json.dumps(block.data)}]"}
                )
        return formatted

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for tool in tools:
            if "type" not in tool:
                formatted.append({"type": "function", "function": tool})
            else:
                formatted.append(tool)
        return formatted

    def _handle_exception(self, e: Exception) -> Exception:
        try:
            import groq
        except ImportError:
            return APIError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.RateLimitError):
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    retry_after = float(retry_after)
            return RateLimitError(str(e), retry_after=retry_after)
        elif isinstance(e, groq.BadRequestError):
            error_msg = str(e).lower()
            if "context_length" in error_msg or "maximum context length" in error_msg:
                return ContextLengthExceededError(str(e))
            if (
                "content_filter" in error_msg
                or "content management policy" in error_msg
            ):
                return ContentFilterError(str(e))
            return InvalidRequestError(str(e))
        elif isinstance(e, groq.APIError):
            return APIError(
                str(e),
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            )
        else:
            return APIError(str(e))


class GroqAsyncChatCompletion(AsyncChatCompletion):
    """Groq Async Chat Completion API implementation."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, client: "AsyncGroq") -> None:
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
    ) -> Union[GroqAsyncChatResponses, AsyncGenerator[GroqAsyncChatStreamChunks, None]]:
        formatted_messages = self._format_messages(messages)
        request_params: Dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "messages": formatted_messages,
        }

        if tools:
            request_params["tools"] = self._format_tools(tools)
        if temperature is not None:
            request_params["temperature"] = temperature
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if stream:
            request_params["stream"] = True

        request_params.update(kwargs)

        try:
            if stream:
                return self._stream(request_params)
            else:
                response = await self._client.chat.completions.create(**request_params)
                return GroqAsyncChatResponses(response)
        except Exception as e:
            raise self._handle_exception(e)

    async def _stream(
        self, request_params: Dict[str, Any]
    ) -> AsyncGenerator[GroqAsyncChatStreamChunks, None]:
        try:
            response = await self._client.chat.completions.create(**request_params)
            async for chunk in response:
                yield GroqAsyncChatStreamChunks(chunk)
        except Exception as e:
            raise self._handle_exception(e)

    def _format_messages(
        self, messages: Union[str, Message, List[Message]]
    ) -> List[Dict[str, Any]]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if not isinstance(messages, list):
            messages = [messages]
        return [self._format_single_message(msg) for msg in messages]

    def _format_single_message(self, msg: Message) -> Dict[str, Any]:
        if isinstance(msg, SystemMessage):
            content = " ".join(block.text for block in msg.content)
            return {"role": "system", "content": content}
        elif isinstance(msg, UserMessage):
            content = self._format_content_blocks(msg.content)
            if len(content) == 1 and content[0].get("type") == "text":
                return {"role": "user", "content": content[0]["text"]}
            return {"role": "user", "content": content}
        elif isinstance(msg, AssistantMessage):
            result: Dict[str, Any] = {"role": "assistant"}
            if msg.content:
                result["content"] = " ".join(block.text for block in msg.content)
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return result
        elif isinstance(msg, ToolMessage):
            content = " ".join(block.text for block in msg.content)
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": content,
            }
        else:
            raise InvalidRequestError(f"Unknown message type: {type(msg)}")

    def _format_content_blocks(
        self, content: List[Union[TextBlock, ImageBlock, DocumentBlock]]
    ) -> List[Dict[str, Any]]:
        formatted = []
        for block in content:
            if isinstance(block, TextBlock):
                formatted.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                formatted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": block.url, "detail": block.detail},
                    }
                )
            elif isinstance(block, DocumentBlock):
                formatted.append(
                    {"type": "text", "text": f"[Document: {json.dumps(block.data)}]"}
                )
        return formatted

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for tool in tools:
            if "type" not in tool:
                formatted.append({"type": "function", "function": tool})
            else:
                formatted.append(tool)
        return formatted

    def _handle_exception(self, e: Exception) -> Exception:
        try:
            import groq
        except ImportError:
            return APIError(str(e))

        if isinstance(e, groq.AuthenticationError):
            return AuthenticationError(str(e))
        elif isinstance(e, groq.RateLimitError):
            retry_after = None
            if hasattr(e, "response") and e.response:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    retry_after = float(retry_after)
            return RateLimitError(str(e), retry_after=retry_after)
        elif isinstance(e, groq.BadRequestError):
            error_msg = str(e).lower()
            if "context_length" in error_msg or "maximum context length" in error_msg:
                return ContextLengthExceededError(str(e))
            if (
                "content_filter" in error_msg
                or "content management policy" in error_msg
            ):
                return ContentFilterError(str(e))
            return InvalidRequestError(str(e))
        elif isinstance(e, groq.APIError):
            return APIError(
                str(e),
                status_code=getattr(e, "status_code", None),
                response=getattr(e, "response", None),
            )
        else:
            return APIError(str(e))
