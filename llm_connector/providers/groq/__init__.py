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
    import groq
    from groq import Groq, AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    groq = None  # type: ignore
    Groq = None  # type: ignore
    AsyncGroq = None  # type: ignore
    GROQ_AVAILABLE = False


class GroqConnector(LLMConnector):
    """
    Groq LLM Connector.

    Provides both synchronous and asynchronous interfaces for Groq API.

    Config options:
        api_key: Groq API key (or use GROQ_API_KEY env var)
        base_url: Optional custom base URL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries

    Usage:
        # Sync usage
        connector = ConnectorFactory.create("groq")
        response = connector.chat().invoke(messages="Hello!")

        # Async usage
        response = await connector.async_chat().invoke(messages="Hello!")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not GROQ_AVAILABLE:
            raise ProviderImportError(
                "Groq package is not installed. " "Install with: pip install groq"
            )

        super().__init__(config)

        client_kwargs: Dict[str, Any] = {}

        api_key = self.config.get("api_key") or os.environ.get("GROQ_API_KEY")
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]
        if self.config.get("timeout"):
            client_kwargs["timeout"] = self.config["timeout"]
        if self.config.get("max_retries") is not None:
            client_kwargs["max_retries"] = self.config["max_retries"]

        self._client = Groq(**client_kwargs)
        self._async_client: Optional["AsyncGroq"] = None  # type: ignore
        self._client_kwargs = client_kwargs

        self._chat: Optional[ChatCompletion] = None
        self._batch: Optional[BatchProcess] = None
        self._file: Optional[FileAPI] = None

        self._async_chat_instance: Optional[AsyncChatCompletion] = None
        self._async_batch_instance: Optional[AsyncBatchProcess] = None
        self._async_file_instance: Optional[AsyncFileAPI] = None

    def _validate_config(self) -> None:
        """Validate configuration."""
        api_key = self.config.get("api_key") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "Groq API key not found. "
                "Set it via config['api_key'] or GROQ_API_KEY environment variable."
            )

    def _get_async_client(self) -> "AsyncGroq":  # type: ignore
        """Get or create the async client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = AsyncGroq(**self._client_kwargs)
        return self._async_client

    # ==================== Sync Methods ====================

    def chat(self) -> ChatCompletion:
        """Get the chat completion interface."""
        if self._chat is None:
            from .completion import GroqChatCompletion

            self._chat = GroqChatCompletion(self._client)
        return self._chat

    def batch(self) -> BatchProcess:
        """Get the batch processing interface."""
        if self._batch is None:
            from .batch import GroqBatchProcess

            self._batch = GroqBatchProcess(self._client)
        return self._batch

    def file(self) -> FileAPI:
        """Get the file API interface."""
        if self._file is None:
            from .fileapi import GroqFileAPI

            self._file = GroqFileAPI(self._client)
        return self._file

    # ==================== Async Methods ====================

    def async_chat(self) -> AsyncChatCompletion:
        """Get the async chat completion interface."""
        if self._async_chat_instance is None:
            from .completion import GroqAsyncChatCompletion

            self._async_chat_instance = GroqAsyncChatCompletion(
                self._get_async_client()
            )
        return self._async_chat_instance

    def async_batch(self) -> AsyncBatchProcess:
        """Get the async batch processing interface."""
        if self._async_batch_instance is None:
            from .batch import GroqAsyncBatchProcess

            self._async_batch_instance = GroqAsyncBatchProcess(self._get_async_client())
        return self._async_batch_instance

    def async_file(self) -> AsyncFileAPI:
        """Get the async file API interface."""
        if self._async_file_instance is None:
            from .fileapi import GroqAsyncFileAPI

            self._async_file_instance = GroqAsyncFileAPI(self._get_async_client())
        return self._async_file_instance

    # ==================== Properties ====================

    @property
    def client(self) -> "Groq":  # type: ignore
        """Access the underlying sync Groq client."""
        return self._client

    @property
    def async_client(self) -> "AsyncGroq":  # type: ignore
        """Access the underlying async Groq client."""
        return self._get_async_client()
