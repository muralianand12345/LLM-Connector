import pytest
from unittest.mock import patch, MagicMock

from llm_connector import ConnectorFactory
from llm_connector.base import LLMConnector
from llm_connector.exceptions import ProviderNotSupportedError


class TestConnectorFactory:
    """Tests for ConnectorFactory."""

    def test_supported_providers(self):
        """Test listing supported providers."""
        providers = ConnectorFactory.supported_providers()
        assert isinstance(providers, list)
        assert "openai" in providers

    def test_create_unsupported_provider(self):
        """Test creating an unsupported provider raises error."""
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            ConnectorFactory.create("unsupported_provider")
        assert "unsupported_provider" in str(exc_info.value)

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_create_openai_connector(self, mock_openai):
        """Test creating OpenAI connector."""
        mock_openai.return_value = MagicMock()

        connector = ConnectorFactory.create("openai", config={"api_key": "test-key"})

        assert connector is not None
        assert hasattr(connector, "chat")
        assert hasattr(connector, "batch")
        assert hasattr(connector, "file")
        assert hasattr(connector, "async_chat")
        assert hasattr(connector, "async_batch")
        assert hasattr(connector, "async_file")

    @patch("llm_connector.providers.openai.OPENAI_AVAILABLE", True)
    @patch("llm_connector.providers.openai.OpenAI")
    def test_create_openai_case_insensitive(self, mock_openai):
        """Test provider name is case insensitive."""
        mock_openai.return_value = MagicMock()

        connector1 = ConnectorFactory.create("openai", config={"api_key": "test-key"})
        connector2 = ConnectorFactory.create("OpenAI", config={"api_key": "test-key"})
        connector3 = ConnectorFactory.create("OPENAI", config={"api_key": "test-key"})

        assert connector1 is not None
        assert connector2 is not None
        assert connector3 is not None

    def test_register_custom_connector(self):
        """Test registering a custom connector."""

        class CustomConnector(LLMConnector):
            def chat(self):
                return None

            def batch(self):
                return None

            def file(self):
                return None

            def async_chat(self):
                return None

            def async_batch(self):
                return None

            def async_file(self):
                return None

        ConnectorFactory.register("custom", CustomConnector)

        assert "custom" in ConnectorFactory.supported_providers()

        connector = ConnectorFactory.create("custom")
        assert isinstance(connector, CustomConnector)

        # Cleanup
        ConnectorFactory.unregister("custom")

    def test_register_invalid_connector(self):
        """Test registering an invalid connector raises error."""

        class NotAConnector:
            pass

        with pytest.raises(TypeError):
            ConnectorFactory.register("invalid", NotAConnector)

    def test_unregister_provider(self):
        """Test unregistering a provider."""

        class TempConnector(LLMConnector):
            def chat(self):
                return None

            def batch(self):
                return None

            def file(self):
                return None

            def async_chat(self):
                return None

            def async_batch(self):
                return None

            def async_file(self):
                return None

        ConnectorFactory.register("temp", TempConnector)
        assert "temp" in ConnectorFactory.supported_providers()

        ConnectorFactory.unregister("temp")
        assert "temp" not in ConnectorFactory.supported_providers()

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a non-existent provider doesn't raise."""
        # Should not raise
        ConnectorFactory.unregister("nonexistent")
