# LLM Connector

A unified Python interface for interacting with multiple LLM providers.

## Installation

```bash
# Base installation (no providers)
pip install llm-connector

# With OpenAI support
pip install llm-connector[openai]

# With Groq support
pip install llm-connector[groq]

# With all providers
pip install llm-connector[all]
```

## Quick Start

### OpenAI

```python
from llm_connector import ConnectorFactory

# Create a connector
connector = ConnectorFactory.create("openai", config={"api_key": "your-api-key"})

# Chat completion
response = connector.chat().invoke(messages="Hello, how are you?")
print(response.content)

# Streaming
for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
    print(chunk.delta_content, end="", flush=True)
```

### Groq

```python
from llm_connector import ConnectorFactory

# Create a connector
connector = ConnectorFactory.create("groq", config={"api_key": "your-groq-api-key"})

# Chat completion
response = connector.chat().invoke(messages="Hello, how are you?")
print(response.content)

# Streaming
for chunk in connector.chat().invoke(messages="Tell me a story", stream=True):
    print(chunk.delta_content, end="", flush=True)
```

## Features

- **Unified Interface**: Single API for multiple LLM providers
- **Streaming Support**: Real-time response streaming
- **Tool Calling**: Function/tool calling support
- **Batch Processing**: Batch API support for bulk operations
- **File API**: File upload and management
- **Type Safety**: Full type hints and Pydantic models
- **Async Support**: Full async/await support for all operations

## Supported Providers

| Provider | Status | Installation |
|----------|--------|--------------|
| OpenAI | ✅ Supported | `pip install llm-connector[openai]` |
| Groq | ✅ Supported | `pip install llm-connector[groq]` |
| Anthropic | 🚧 Coming Soon | - |

## Usage Examples

### Basic Chat

```python
from llm_connector import ConnectorFactory

# OpenAI
connector = ConnectorFactory.create("openai", config={
    "api_key": "your-api-key",
    # Or set OPENAI_API_KEY environment variable
})

# Groq
connector = ConnectorFactory.create("groq", config={
    "api_key": "your-groq-api-key",
    # Or set GROQ_API_KEY environment variable
})

response = connector.chat().invoke(
    messages="What is the capital of France?",
    model="gpt-4o-mini",  # or "llama-3.3-70b-versatile" for Groq
    temperature=0.7,
)

print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Async Chat

```python
import asyncio
from llm_connector import ConnectorFactory

async def main():
    connector = ConnectorFactory.create("groq")
    
    # Async chat
    response = await connector.async_chat().invoke(messages="Hello!")
    print(response.content)
    
    # Async streaming
    stream = await connector.async_chat().invoke(
        messages="Tell me a story",
        stream=True
    )
    async for chunk in stream:
        print(chunk.delta_content, end="", flush=True)

asyncio.run(main())
```

### Structured Messages

```python
from llm_connector import (
    ConnectorFactory,
    UserMessage,
    SystemMessage,
    TextBlock,
)

connector = ConnectorFactory.create("openai")

messages = [
    SystemMessage(
        role="system",
        content=[TextBlock(text="You are a helpful assistant.")]
    ),
    UserMessage(
        role="user",
        content=[TextBlock(text="Hello!")]
    ),
]

response = connector.chat().invoke(messages=messages)
```

### Streaming

```python
for chunk in connector.chat().invoke(
    messages="Write a poem about Python",
    stream=True
):
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
```

### Tool Calling

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

response = connector.chat().invoke(
    messages="What's the weather in Tokyo?",
    tools=tools,
)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Arguments: {tool_call.arguments}")
```

### Batch Processing

```python
# Create batch job from JSONL file
batch_request = connector.batch().create(
    file="batch_requests.jsonl",
    completion_window="24h"
)

# Check status
status = connector.batch().status(batch_request.id)
print(f"Status: {status.status}")

# Get results when completed
if status.status == "completed":
    results = connector.batch().result(batch_request.id)
    for record in results.records:
        print(record)
```

### File API

```python
# Upload file
file_id = connector.file().upload(
    file=b'{"test": "data"}',
    purpose="batch"
)

# Retrieve metadata
file_info = connector.file().retrieve(file_id=file_id)

# Download content
content = connector.file().download(file_id=file_id)

# Delete file
connector.file().delete(file_id=file_id)
```

## Custom Providers

You can register custom providers:

```python
from llm_connector import ConnectorFactory, LLMConnector

class MyCustomConnector(LLMConnector):
    # Implement required methods
    ...

ConnectorFactory.register("custom", MyCustomConnector)
connector = ConnectorFactory.create("custom", config={...})
```

## Error Handling

```python
from llm_connector import (
    ConnectorFactory,
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError,
)

try:
    response = connector.chat().invoke(messages="Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except ContextLengthExceededError:
    print("Input too long")
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub](https://github.com/muralianand12345/llm-connector)
- [Issues](https://github.com/muralianand12345/llm-connector/issues)
