from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ...exceptions import ProviderImportError, AuthenticationError
from ...base import (
    LLMConnector,
    ChatCompletion,
    AsyncChatCompletion,
    BatchProcess,
    AsyncBatchProcess,
    FileAPI,
    AsyncFileAPI,
)

try:
    import openai
    from openai import OpenAI, AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class OpenAIConnector(LLMConnector):
    """
    OpenAI LLM Connector.

    Provides both synchronous and asynchronous interfaces for OpenAI API.

    Config options:
        api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        base_url: Optional custom base URL
        organization: Optional organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries

    Usage:
        # Sync usage
        connector = ConnectorFactory.create("openai")
        response = connector.chat().invoke(messages="Hello!")

        # Async usage
        response = await connector.async_chat().invoke(messages="Hello!")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not OPENAI_AVAILABLE:
            raise ProviderImportError(
                "OpenAI package is not installed. " "Install with: pip install openai"
            )

        super().__init__(config)

        client_kwargs: Dict[str, Any] = {}

        api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]
        if self.config.get("organization"):
            client_kwargs["organization"] = self.config["organization"]
        if self.config.get("timeout"):
            client_kwargs["timeout"] = self.config["timeout"]
        if self.config.get("max_retries") is not None:
            client_kwargs["max_retries"] = self.config["max_retries"]

        self._client = OpenAI(**client_kwargs)
        self._async_client: Optional["AsyncOpenAI"] = None  # type: ignore
        self._client_kwargs = client_kwargs

        self._chat: Optional[ChatCompletion] = None
        self._batch: Optional[BatchProcess] = None
        self._file: Optional[FileAPI] = None

        self._async_chat_instance: Optional[AsyncChatCompletion] = None
        self._async_batch_instance: Optional[AsyncBatchProcess] = None
        self._async_file_instance: Optional[AsyncFileAPI] = None

    def _validate_config(self) -> None:
        """Validate configuration."""
        api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "OpenAI API key not found. "
                "Set it via config['api_key'] or OPENAI_API_KEY environment variable."
            )

    def _get_async_client(self) -> "AsyncOpenAI":  # type: ignore
        """Get or create the async client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(**self._client_kwargs)
        return self._async_client

    # ==================== Sync Methods ====================

    def chat(self) -> ChatCompletion:
        """Get the chat completion interface."""
        if self._chat is None:
            from .completion import OpenAIChatCompletion

            self._chat = OpenAIChatCompletion(self._client)
        return self._chat

    def batch(self) -> BatchProcess:
        """Get the batch processing interface."""
        if self._batch is None:
            from .batch import OpenAIBatchProcess

            self._batch = OpenAIBatchProcess(self._client)
        return self._batch

    def file(self) -> FileAPI:
        """Get the file API interface."""
        if self._file is None:
            from .fileapi import OpenAIFileAPI

            self._file = OpenAIFileAPI(self._client)
        return self._file

    # ==================== Async Methods ====================

    def async_chat(self) -> AsyncChatCompletion:
        """Get the async chat completion interface."""
        if self._async_chat_instance is None:
            from .completion import OpenAIAsyncChatCompletion

            self._async_chat_instance = OpenAIAsyncChatCompletion(
                self._get_async_client()
            )
        return self._async_chat_instance

    def async_batch(self) -> AsyncBatchProcess:
        """Get the async batch processing interface."""
        if self._async_batch_instance is None:
            from .batch import OpenAIAsyncBatchProcess

            self._async_batch_instance = OpenAIAsyncBatchProcess(
                self._get_async_client()
            )
        return self._async_batch_instance

    def async_file(self) -> AsyncFileAPI:
        """Get the async file API interface."""
        if self._async_file_instance is None:
            from .fileapi import OpenAIAsyncFileAPI

            self._async_file_instance = OpenAIAsyncFileAPI(self._get_async_client())
        return self._async_file_instance

    # ==================== Properties ====================

    @property
    def client(self) -> "OpenAI":  # type: ignore
        """Access the underlying sync OpenAI client."""
        return self._client

    @property
    def async_client(self) -> "AsyncOpenAI":  # type: ignore
        """Access the underlying async OpenAI client."""
        return self._get_async_client()
