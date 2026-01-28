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
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    Anthropic = None  # type: ignore
    AsyncAnthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False


class AnthropicConnector(LLMConnector):
    """
    Anthropic LLM Connector.

    Provides both synchronous and asynchronous interfaces for Anthropic API.

    Config options:
        api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        base_url: Optional custom base URL
        organization: Optional organization ID
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries

    Usage:
        # Sync usage
        connector = ConnectorFactory.create("anthropic")
        response = connector.chat().invoke(messages="Hello!")

        # Async usage
        response = await connector.async_chat().invoke(messages="Hello!")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not ANTHROPIC_AVAILABLE:
            raise ProviderImportError(
                "Anthropic package is not installed. "
                "Install with: pip install anthropic"
            )

        super().__init__(config)

        client_kwargs: Dict[str, Any] = {}

        api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
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

        self._client = Anthropic(**client_kwargs)
        self._async_client: Optional["AsyncAnthropic"] = None  # type: ignore
        self._client_kwargs = client_kwargs

        self._chat: Optional[ChatCompletion] = None
        self._batch: Optional[BatchProcess] = None
        self._file: Optional[FileAPI] = None

        self._async_chat_instance: Optional[AsyncChatCompletion] = None
        self._async_batch_instance: Optional[AsyncBatchProcess] = None
        self._async_file_instance: Optional[AsyncFileAPI] = None

    def _validate_config(self):
        """Validate configuration."""
        api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "Anthropic API key not found. "
                "Set it via config['api_key'] or ANTHROPIC_API_KEY environment variable."
            )

    def _get_async_client(self) -> "AsyncAnthropic": # type: ignore
        """Get or create the async client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = AsyncAnthropic(**self._client_kwargs)  # type: ignore
        return self._async_client

    # ==================== Sync Methods ====================

    def chat(self) -> ChatCompletion:
        """Get the chat completion interface."""
        if self._chat is None:
            from .completion import AnthropicChatCompletion

            self._chat = AnthropicChatCompletion(self._client)
        return self._chat

    def batch(self) -> BatchProcess:
        """Get the batch processing interface."""
        if self._batch is None:
            from .batch import AnthropicBatchProcess

            self._batch = AnthropicBatchProcess(self._client)
        return self._batch

    def file(self) -> FileAPI:
        """Get the file API interface."""
        if self._file is None:
            from .fileapi import AnthropicFileAPI

            self._file = AnthropicFileAPI(self._client)
        return self._file

    # ==================== Async Methods ====================

    def async_chat(self) -> AsyncChatCompletion:
        """Get the async chat completion interface."""
        if self._async_chat_instance is None:
            from .completion import AnthropicAsyncChatCompletion

            self._async_chat_instance = AnthropicAsyncChatCompletion(
                self._get_async_client()
            )
        return self._async_chat_instance

    def async_batch(self) -> AsyncBatchProcess:
        """Get the async batch processing interface."""
        if self._async_batch_instance is None:
            from .batch import AnthropicAsyncBatchProcess

            self._async_batch_instance = AnthropicAsyncBatchProcess(
                self._get_async_client()
            )
        return self._async_batch_instance

    def async_file(self) -> AsyncFileAPI:
        """Get the async file API interface."""
        if self._async_file_instance is None:
            from .fileapi import AnthropicAsyncFileAPI

            self._async_file_instance = AnthropicAsyncFileAPI(self._get_async_client())
        return self._async_file_instance

    # ==================== Properties ====================

    @property
    def client(self) -> "Anthropic":  # type: ignore
        """Access the underlying sync Anthropic client."""
        return self._client

    @property
    def async_client(self) -> "AsyncAnthropic":  # type: ignore
        """Access the underlying async Anthropic client."""
        return self._get_async_client()
