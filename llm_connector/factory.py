from __future__ import annotations

import importlib
from typing import Any, Dict, Type, Union

from .base import LLMConnector
from .exceptions import ProviderNotSupportedError


class ConnectorFactory:
    """Factory for creating LLM connectors."""

    _registry: Dict[str, Union[Type[LLMConnector], str]] = {
        "openai": "OpenAIConnector",
        "groq": "GroqConnector",
        # "anthropic": "AnthropicConnector",  # Coming soon
    }

    @classmethod
    def register(cls, provider: str, connector_cls: Type[LLMConnector]) -> None:
        """
        Register a custom connector class.

        Args:
            provider: Provider name (e.g., 'openai', 'groq')
            connector_cls: Connector class that inherits from LLMConnector
        """
        if not issubclass(connector_cls, LLMConnector):
            raise TypeError(f"{connector_cls.__name__} must inherit from LLMConnector")
        cls._registry[provider.lower()] = connector_cls

    @classmethod
    def _resolve_connector(cls, provider: str) -> Type[LLMConnector]:
        """Resolve provider string to connector class."""
        provider_key = provider.lower()

        entry = cls._registry.get(provider_key)
        if entry is None:
            raise ProviderNotSupportedError(
                f"Provider '{provider}' is not registered. "
                f"Available providers: {list(cls._registry.keys())}"
            )

        if isinstance(entry, type):
            return entry

        if isinstance(entry, str):
            try:
                module = importlib.import_module(".providers", package=__package__)
                connector_cls = getattr(module, entry)
            except AttributeError:
                raise ProviderNotSupportedError(
                    f"Provider '{provider}' connector class '{entry}' not found. "
                    f"This provider may not be implemented yet."
                )
            except Exception as e:
                raise ImportError(f"Failed to resolve provider '{provider}'") from e

            if not issubclass(connector_cls, LLMConnector):
                raise TypeError(f"{entry} must inherit from LLMConnector")

            cls._registry[provider_key] = connector_cls
            return connector_cls

        raise TypeError(f"Invalid registry entry for provider '{provider}': {entry!r}")

    @classmethod
    def create(
        cls, provider: str, *, config: Dict[str, Any] | None = None
    ) -> LLMConnector:
        """
        Create a connector instance for the specified provider.

        Args:
            provider: Provider name (e.g., 'openai', 'groq')
            config: Configuration dictionary for the connector

        Returns:
            Configured LLMConnector instance
        """
        connector_cls = cls._resolve_connector(provider)
        return connector_cls(config=config)

    @classmethod
    def supported_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._registry.keys())

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from the registry."""
        provider_key = provider.lower()
        if provider_key in cls._registry:
            del cls._registry[provider_key]
