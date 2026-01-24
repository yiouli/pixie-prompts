"""Tests for utils module - conversion between OpenAI and Pydantic AI message formats."""

import base64
import json

import pytest

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ThinkingPart,
    ImageUrl,
    AudioUrl,
    VideoUrl,
    DocumentUrl,
    BinaryContent,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from pixie.prompts.utils import (
    openai_messages_to_pydantic_ai_messages,
    pydantic_ai_messages_to_openai_messages,
    assemble_model_request_parameters,
)


class TestOpenAIToPydanticAI:
    """Tests for converting OpenAI message format to Pydantic AI messages."""

    def test_simple_system_message(self):
        """Test converting a simple system message."""
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], SystemPromptPart)
        assert result[0].parts[0].content == "You are a helpful assistant."

    def test_simple_user_message(self):
        """Test converting a simple user message."""
        openai_messages = [{"role": "user", "content": "Hello, how are you?"}]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == "Hello, how are you?"

    def test_simple_assistant_message(self):
        """Test converting a simple assistant message."""
        openai_messages = [
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "I'm doing well, thank you!"

    def test_developer_role_treated_as_system(self):
        """Test that 'developer' role is treated as system message."""
        openai_messages = [
            {"role": "developer", "content": "You are a code assistant."}
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], SystemPromptPart)
        assert result[0].parts[0].content == "You are a code assistant."

    def test_assistant_message_with_tool_calls(self):
        """Test converting an assistant message with tool calls."""
        openai_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris, France"}',
                        },
                    }
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], ToolCallPart)
        assert result[0].parts[0].tool_name == "get_weather"
        assert result[0].parts[0].tool_call_id == "call_abc123"
        assert result[0].parts[0].args == '{"location": "Paris, France"}'

    def test_assistant_message_with_multiple_tool_calls(self):
        """Test converting an assistant message with multiple tool calls."""
        openai_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "id": "call_def456",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 2
        assert all(isinstance(part, ToolCallPart) for part in result[0].parts)
        assert result[0].parts[0].tool_call_id == "call_abc123"  # type: ignore
        assert result[0].parts[1].tool_call_id == "call_def456"  # type: ignore

    def test_assistant_message_with_content_and_tool_calls(self):
        """Test assistant message that has both text content and tool calls."""
        openai_messages = [
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 2
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "I'll check the weather for you."
        assert isinstance(result[0].parts[1], ToolCallPart)

    def test_tool_message(self):
        """Test converting a tool result message."""
        openai_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": "15°C", "conditions": "sunny"}',
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], ToolReturnPart)
        assert result[0].parts[0].tool_call_id == "call_abc123"
        assert (
            result[0].parts[0].content
            == '{"temperature": "15°C", "conditions": "sunny"}'
        )

    def test_tool_message_with_name(self):
        """Test converting a tool result message that includes the function name."""
        openai_messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": "15°C and sunny",
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], ToolReturnPart)
        assert result[0].parts[0].tool_name == "get_weather"
        assert result[0].parts[0].tool_call_id == "call_abc123"

    def test_full_conversation(self):
        """Test converting a complete conversation with multiple message types."""
        openai_messages = [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "name": "get_weather",
                "content": "15°C and sunny",
            },
            {"role": "assistant", "content": "The weather in Paris is 15°C and sunny."},
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        # Should have 5 messages
        assert len(result) == 5

        # Message 1: System prompt
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], SystemPromptPart)

        # Message 2: User prompt
        assert isinstance(result[1], ModelRequest)
        assert isinstance(result[1].parts[0], UserPromptPart)

        # Message 3: Assistant with tool call
        assert isinstance(result[2], ModelResponse)
        assert isinstance(result[2].parts[0], ToolCallPart)

        # Message 4: Tool result
        assert isinstance(result[3], ModelRequest)
        assert isinstance(result[3].parts[0], ToolReturnPart)

        # Message 5: Final assistant response
        assert isinstance(result[4], ModelResponse)
        assert isinstance(result[4].parts[0], TextPart)

    def test_empty_messages_list(self):
        """Test converting an empty messages list."""
        result = openai_messages_to_pydantic_ai_messages([])
        assert result == []

    def test_assistant_message_with_empty_content(self):
        """Test assistant message with empty string content (no tool calls)."""
        openai_messages = [{"role": "assistant", "content": ""}]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == ""

    # ===== Multimedia content tests (OpenAI -> Pydantic AI) =====

    def test_user_message_with_image_url(self):
        """Test converting a user message with an image URL."""
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        # Content should be a sequence with text and ImageUrl
        assert isinstance(part.content, list)
        assert len(part.content) == 2
        assert part.content[0] == "What's in this image?"
        assert isinstance(part.content[1], ImageUrl)
        assert part.content[1].url == "https://example.com/image.png"

    def test_user_message_with_image_url_and_detail(self):
        """Test converting a user message with image URL that has detail level."""
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/highres.jpg",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert len(part.content) == 2
        image_content = part.content[1]
        assert isinstance(image_content, ImageUrl)
        assert image_content.url == "https://example.com/highres.jpg"
        # Detail should be in vendor_metadata
        assert image_content.vendor_metadata == {"detail": "high"}

    def test_user_message_with_base64_image(self):
        """Test converting a user message with a base64 encoded image."""
        # Small 1x1 red PNG
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this pixel?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                        },
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert len(part.content) == 2
        binary_content = part.content[1]
        assert isinstance(binary_content, BinaryContent)
        assert binary_content.media_type == "image/png"

    def test_user_message_with_multiple_images(self):
        """Test converting a user message with multiple images."""
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image1.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image2.jpg"},
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert len(part.content) == 3
        assert part.content[0] == "Compare these two images:"
        assert isinstance(part.content[1], ImageUrl)
        assert isinstance(part.content[2], ImageUrl)
        assert part.content[1].url == "https://example.com/image1.png"
        assert part.content[2].url == "https://example.com/image2.jpg"

    def test_user_message_with_audio_input(self):
        """Test converting a user message with audio input."""
        # Base64 audio data (minimal MP3 header for testing)
        audio_data = base64.b64encode(b"\xff\xfb\x90\x00" + b"\x00" * 100).decode()
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio:"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "mp3",
                        },
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert len(part.content) == 2
        assert part.content[0] == "Transcribe this audio:"
        audio_content = part.content[1]
        assert isinstance(audio_content, BinaryContent)
        assert audio_content.media_type == "audio/mpeg"

    def test_user_message_text_only_in_content_array(self):
        """Test user message with content array containing only text parts."""
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        # Should have text content (joined or as list)
        if isinstance(part.content, str):
            assert "First part" in part.content
            assert "Second part" in part.content
        else:
            assert "First part." in part.content
            assert "Second part." in part.content

    def test_user_message_image_without_text(self):
        """Test user message with only an image (no text)."""
        openai_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/standalone.png"},
                    },
                ],
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert isinstance(part.content, list)
        assert len(part.content) == 1
        assert isinstance(part.content[0], ImageUrl)

    def test_function_call_deprecated_format(self):
        """Test converting deprecated function_call format (pre-tool_calls)."""
        openai_messages = [
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], ToolCallPart)
        assert result[0].parts[0].tool_name == "get_weather"

    def test_function_role_message(self):
        """Test converting deprecated function role message."""
        openai_messages = [
            {"role": "function", "name": "get_weather", "content": "15°C and sunny"}
        ]

        result = openai_messages_to_pydantic_ai_messages(openai_messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], ToolReturnPart)
        assert result[0].parts[0].tool_name == "get_weather"


class TestPydanticAIToOpenAI:
    """Tests for converting Pydantic AI messages to OpenAI format."""

    def test_simple_system_prompt(self):
        """Test converting a simple system prompt."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[SystemPromptPart(content="You are a helpful assistant.")]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."

    def test_simple_user_prompt(self):
        """Test converting a simple user prompt."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello!")])
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_simple_assistant_text_response(self):
        """Test converting a simple assistant text response."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(parts=[TextPart(content="Hello! How can I help you?")])
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello! How can I help you?"

    def test_assistant_with_tool_call(self):
        """Test converting an assistant response with a tool call."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        args='{"location": "Paris"}',
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_abc123"
        assert result[0]["tool_calls"][0]["type"] == "function"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert (
            result[0]["tool_calls"][0]["function"]["arguments"]
            == '{"location": "Paris"}'
        )

    def test_assistant_with_multiple_tool_calls(self):
        """Test converting an assistant response with multiple tool calls."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_1",
                        args='{"location": "Paris"}',
                    ),
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_2",
                        args='{"location": "London"}',
                    ),
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 2
        assert result[0]["tool_calls"][0]["id"] == "call_1"
        assert result[0]["tool_calls"][1]["id"] == "call_2"

    def test_assistant_with_text_and_tool_calls(self):
        """Test converting an assistant response with both text and tool calls."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    TextPart(content="Let me check the weather."),
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        args='{"location": "Paris"}',
                    ),
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check the weather."
        assert len(result[0]["tool_calls"]) == 1

    def test_tool_return_part(self):
        """Test converting a tool return part."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        content="15°C and sunny",
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_abc123"
        assert result[0]["content"] == "15°C and sunny"

    def test_tool_return_with_dict_content(self):
        """Test converting a tool return with dict content (should be JSON serialized)."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        content={"temperature": "15°C", "conditions": "sunny"},
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        # Content should be JSON string
        content = result[0]["content"]
        assert isinstance(content, str)
        parsed = json.loads(content)
        assert parsed == {"temperature": "15°C", "conditions": "sunny"}

    def test_tool_call_with_dict_args(self):
        """Test converting a tool call where args is a dict instead of string."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        args={"location": "Paris"},  # Dict instead of string
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        # Arguments should be JSON string in OpenAI format
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"location": "Paris"}

    def test_full_conversation(self):
        """Test converting a complete conversation."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[SystemPromptPart(content="You are a weather assistant.")]
            ),
            ModelRequest(
                parts=[UserPromptPart(content="What's the weather in Paris?")]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        args='{"location": "Paris"}',
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                        content="15°C and sunny",
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content="The weather in Paris is 15°C and sunny.")]
            ),
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert "tool_calls" in result[2]
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"
        assert result[4]["content"] == "The weather in Paris is 15°C and sunny."

    def test_empty_messages_list(self):
        """Test converting an empty messages list."""
        result = pydantic_ai_messages_to_openai_messages([])
        assert result == []

    def test_retry_prompt_part(self):
        """Test converting a retry prompt part."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Invalid location format",
                        tool_name="get_weather",
                        tool_call_id="call_abc123",
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        # Retry prompt should be converted to tool message with error
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_abc123"

    def test_retry_prompt_without_tool_name(self):
        """Test converting a retry prompt part without tool name (user-style message)."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Please try again with valid input.", tool_name=None
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        # Without tool_name, should be converted to user message
        assert result[0]["role"] == "user"

    def test_mixed_request_parts(self):
        """Test a ModelRequest with multiple different part types."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="System instructions"),
                    UserPromptPart(content="User question"),
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        # Should produce separate messages for each part type
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_multiple_tool_returns(self):
        """Test a ModelRequest with multiple tool returns."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="func1", tool_call_id="call_1", content="result1"
                    ),
                    ToolReturnPart(
                        tool_name="func2", tool_call_id="call_2", content="result2"
                    ),
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        # Each tool return should be a separate message
        assert len(result) == 2
        assert all(msg["role"] == "tool" for msg in result)
        assert result[0]["tool_call_id"] == "call_1"
        assert result[1]["tool_call_id"] == "call_2"

    # ===== Multimedia content tests (Pydantic AI -> OpenAI) =====

    def test_user_prompt_with_image_url(self):
        """Test converting UserPromptPart with ImageUrl to OpenAI format."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What's in this image?",
                            ImageUrl(url="https://example.com/image.png"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "What's in this image?"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/image.png"

    def test_user_prompt_with_image_url_and_detail(self):
        """Test converting ImageUrl with vendor_metadata (detail) to OpenAI format."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Describe in detail.",
                            ImageUrl(
                                url="https://example.com/highres.jpg",
                                vendor_metadata={"detail": "high"},
                            ),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        image_part = content[1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"] == "https://example.com/highres.jpg"
        assert image_part["image_url"]["detail"] == "high"

    def test_user_prompt_with_multiple_images(self):
        """Test converting UserPromptPart with multiple images."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Compare these:",
                            ImageUrl(url="https://example.com/img1.png"),
                            ImageUrl(url="https://example.com/img2.jpg"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 3
        assert content[0] == {"type": "text", "text": "Compare these:"}
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    def test_user_prompt_with_binary_image(self):
        """Test converting UserPromptPart with BinaryContent image."""
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Minimal PNG-like data
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What is this?",
                            BinaryContent(data=image_bytes, media_type="image/png"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        image_part = content[1]
        assert image_part["type"] == "image_url"
        # Should be a data URI
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_user_prompt_with_audio_binary(self):
        """Test converting UserPromptPart with BinaryContent audio."""
        audio_bytes = b"\xff\xfb\x90\x00" + b"\x00" * 100  # Minimal MP3-like data
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Transcribe this:",
                            BinaryContent(data=audio_bytes, media_type="audio/mpeg"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        audio_part = content[1]
        assert audio_part["type"] == "input_audio"
        assert "data" in audio_part["input_audio"]
        assert audio_part["input_audio"]["format"] == "mp3"

    def test_user_prompt_with_audio_url(self):
        """Test converting UserPromptPart with AudioUrl."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What is being said?",
                            AudioUrl(url="https://example.com/audio.mp3"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        # Audio URL should be converted appropriately
        audio_part = content[1]
        # AudioUrl is typically converted to input_audio or a URL reference
        assert audio_part["type"] in ("input_audio", "audio_url")

    def test_user_prompt_with_document_url(self):
        """Test converting UserPromptPart with DocumentUrl."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Summarize this document:",
                            DocumentUrl(url="https://example.com/doc.pdf"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        # Document URL should be included
        doc_part = content[1]
        assert "url" in str(doc_part) or doc_part.get("type") == "file"

    def test_user_prompt_with_video_url(self):
        """Test converting UserPromptPart with VideoUrl."""
        pydantic_messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "Describe this video:",
                            VideoUrl(url="https://example.com/video.mp4"),
                        ]
                    )
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        # Video URL should be included in some format
        video_part = content[1]
        assert "url" in str(video_part) or video_part.get("type") == "video"

    def test_thinking_part_is_excluded(self):
        """Test that ThinkingPart is excluded from OpenAI output (it's internal)."""
        pydantic_messages: list[ModelMessage] = [
            ModelResponse(
                parts=[
                    ThinkingPart(content="Let me think about this..."),
                    TextPart(content="Here's my answer."),
                ]
            )
        ]

        result = pydantic_ai_messages_to_openai_messages(pydantic_messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Only the TextPart content should be included
        assert result[0]["content"] == "Here's my answer."


class TestRoundTrip:
    """Tests for round-trip conversions (OpenAI -> Pydantic AI -> OpenAI)."""

    def test_simple_conversation_roundtrip(self):
        """Test that a simple conversation survives round-trip conversion."""
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        pydantic_msgs = openai_messages_to_pydantic_ai_messages(original)
        result = pydantic_ai_messages_to_openai_messages(pydantic_msgs)

        assert len(result) == len(original)
        for orig, res in zip(original, result):
            assert orig["role"] == res["role"]
            assert orig["content"] == res["content"]

    def test_tool_call_roundtrip(self):
        """Test that tool calls survive round-trip conversion."""
        original = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "15°C"},
            {"role": "assistant", "content": "It's 15°C in Paris."},
        ]

        pydantic_msgs = openai_messages_to_pydantic_ai_messages(original)
        result = pydantic_ai_messages_to_openai_messages(pydantic_msgs)

        assert len(result) == len(original)

        # Check user message
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "What's the weather?"

        # Check tool call
        assert result[1]["role"] == "assistant"
        assert result[1]["tool_calls"][0]["id"] == "call_abc123"
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather"

        # Check tool response
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_abc123"

        # Check final response
        assert result[3]["role"] == "assistant"
        assert result[3]["content"] == "It's 15°C in Paris."

    def test_image_url_roundtrip(self):
        """Test that image URLs survive round-trip conversion."""
        original = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                ],
            }
        ]

        pydantic_msgs = openai_messages_to_pydantic_ai_messages(original)
        result = pydantic_ai_messages_to_openai_messages(pydantic_msgs)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://example.com/image.png"

    def test_image_with_detail_roundtrip(self):
        """Test that image detail level survives round-trip conversion."""
        original = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/highres.jpg",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]

        pydantic_msgs = openai_messages_to_pydantic_ai_messages(original)
        result = pydantic_ai_messages_to_openai_messages(pydantic_msgs)

        assert len(result) == 1
        content = result[0]["content"]
        assert content[1]["image_url"]["detail"] == "high"

    def test_multiple_images_roundtrip(self):
        """Test that multiple images survive round-trip conversion."""
        original = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/a.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/b.png"},
                    },
                ],
            }
        ]

        pydantic_msgs = openai_messages_to_pydantic_ai_messages(original)
        result = pydantic_ai_messages_to_openai_messages(pydantic_msgs)

        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 3
        assert content[1]["image_url"]["url"] == "https://example.com/a.png"
        assert content[2]["image_url"]["url"] == "https://example.com/b.png"


class TestAssembleModelRequestParameters:
    """Tests for assembling ModelRequestParameters from OpenAI format tools and output_schema."""

    # ===== Tests for output_schema conversion =====

    def test_output_schema_only_simple(self):
        """Test converting a simple output schema without tools."""
        output_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None
        )

        assert isinstance(result, ModelRequestParameters)
        assert result.output_object is not None
        assert result.output_object.json_schema == output_schema
        assert result.output_mode == "native"
        assert result.function_tools == []
        assert result.output_tools == []

    def test_output_schema_with_title_and_description(self):
        """Test output schema with title and description are extracted."""
        output_schema = {
            "type": "object",
            "title": "WeatherResponse",
            "description": "A weather response object",
            "properties": {
                "temperature": {"type": "number"},
                "conditions": {"type": "string"},
            },
            "required": ["temperature", "conditions"],
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None
        )

        assert result.output_object is not None
        assert result.output_object.name == "WeatherResponse"
        assert result.output_object.description == "A weather response object"
        assert result.output_object.json_schema == output_schema

    def test_output_schema_with_strict_true(self):
        """Test output schema with strict mode enabled."""
        output_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None, strict=True
        )

        assert result.output_object is not None
        assert result.output_object.strict is True

    def test_output_schema_with_strict_false(self):
        """Test output schema with strict mode disabled."""
        output_schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None, strict=False
        )

        assert result.output_object is not None
        assert result.output_object.strict is False

    def test_output_schema_none_returns_text_mode(self):
        """Test that when output_schema is None, no output_object is set."""
        result = assemble_model_request_parameters(output_schema=None, tools=None)

        assert result.output_object is None
        assert result.output_mode == "text"

    # ===== Tests for tools conversion =====

    def test_single_function_tool(self):
        """Test converting a single function tool."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 1
        tool = result.function_tools[0]
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "get_weather"
        assert tool.description == "Get the current weather for a location"
        assert tool.parameters_json_schema["properties"]["location"]["type"] == "string"
        assert tool.kind == "function"

    def test_multiple_function_tools(self):
        """Test converting multiple function tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Send an email",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"},
                        },
                        "required": ["to", "subject", "body"],
                    },
                },
            },
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 3
        assert result.function_tools[0].name == "get_weather"
        assert result.function_tools[1].name == "search_web"
        assert result.function_tools[2].name == "send_email"

    def test_tool_with_strict_true(self):
        """Test converting a tool with strict mode enabled."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 1
        assert result.function_tools[0].strict is True

    def test_tool_with_strict_false(self):
        """Test converting a tool with strict mode disabled."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert result.function_tools[0].strict is False

    def test_tool_without_description(self):
        """Test converting a tool without a description."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 1
        assert result.function_tools[0].name == "simple_tool"
        assert result.function_tools[0].description is None

    def test_tool_without_parameters(self):
        """Test converting a tool without parameters (empty schema)."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 1
        assert result.function_tools[0].name == "get_time"
        # Should have default empty schema
        assert result.function_tools[0].parameters_json_schema == {
            "type": "object",
            "properties": {},
        }

    def test_tools_empty_list(self):
        """Test that empty tools list results in no function_tools."""
        result = assemble_model_request_parameters(output_schema=None, tools=[])

        assert result.function_tools == []

    def test_tools_none(self):
        """Test that None tools results in no function_tools."""
        result = assemble_model_request_parameters(output_schema=None, tools=None)

        assert result.function_tools == []

    # ===== Tests for combined output_schema and tools =====

    def test_output_schema_and_tools_combined(self):
        """Test converting both output_schema and tools together."""
        output_schema = {
            "type": "object",
            "title": "SearchResult",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "count": {"type": "integer"},
            },
            "required": ["results", "count"],
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=tools
        )

        # Check output_object
        assert result.output_object is not None
        assert result.output_object.name == "SearchResult"
        assert result.output_mode == "native"

        # Check function_tools
        assert len(result.function_tools) == 1
        assert result.function_tools[0].name == "search"

    # ===== Tests for edge cases and validation =====

    def test_non_function_tool_type_raises_error(self):
        """Test that non-function tool types raise ValueError."""
        tools = [
            {
                "type": "code_interpreter",  # Not a function type
                "code_interpreter": {},
            }
        ]

        with pytest.raises(ValueError, match="Unsupported tool type"):
            assemble_model_request_parameters(output_schema=None, tools=tools)

    def test_tool_missing_function_key_raises_error(self):
        """Test that tools without 'function' key raise ValueError."""
        tools = [
            {
                "type": "function",
                # Missing 'function' key
            }
        ]

        with pytest.raises(ValueError, match="Missing 'function' key"):
            assemble_model_request_parameters(output_schema=None, tools=tools)

    def test_tool_missing_name_raises_error(self):
        """Test that tools without a name raise ValueError."""
        tools = [
            {
                "type": "function",
                "function": {
                    "description": "A tool without a name",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with pytest.raises(ValueError, match="Missing 'name'"):
            assemble_model_request_parameters(output_schema=None, tools=tools)

    def test_complex_nested_parameters_schema(self):
        """Test converting a tool with complex nested parameters."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_order",
                    "description": "Create a new order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string", "format": "email"},
                                    "address": {
                                        "type": "object",
                                        "properties": {
                                            "street": {"type": "string"},
                                            "city": {"type": "string"},
                                            "zip": {"type": "string"},
                                        },
                                        "required": ["street", "city"],
                                    },
                                },
                                "required": ["name", "email"],
                            },
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product_id": {"type": "string"},
                                        "quantity": {"type": "integer", "minimum": 1},
                                        "price": {"type": "number"},
                                    },
                                    "required": ["product_id", "quantity"],
                                },
                            },
                        },
                        "required": ["customer", "items"],
                    },
                },
            }
        ]

        result = assemble_model_request_parameters(output_schema=None, tools=tools)

        assert len(result.function_tools) == 1
        tool = result.function_tools[0]
        assert tool.name == "create_order"

        # Verify nested structure is preserved
        schema = tool.parameters_json_schema
        assert schema["properties"]["customer"]["type"] == "object"
        assert (
            schema["properties"]["customer"]["properties"]["address"]["type"]
            == "object"
        )
        assert schema["properties"]["items"]["type"] == "array"
        assert schema["properties"]["items"]["items"]["type"] == "object"

    def test_complex_output_schema_with_refs(self):
        """Test output schema with JSON schema references ($defs)."""
        output_schema = {
            "type": "object",
            "title": "Report",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Section"},
                },
            },
            "$defs": {
                "Section": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["title", "content"],
                },
            },
            "required": ["sections"],
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None
        )

        assert result.output_object is not None
        assert result.output_object.json_schema == output_schema
        assert "$defs" in result.output_object.json_schema

    def test_output_mode_default_with_output_schema(self):
        """Test that output_mode defaults to 'native' when output_schema is provided."""
        output_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None
        )

        assert result.output_mode == "native"

    def test_output_mode_override(self):
        """Test that output_mode can be overridden."""
        output_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None, output_mode="tool"
        )

        assert result.output_mode == "tool"

    def test_allow_text_output_default(self):
        """Test that allow_text_output defaults appropriately."""
        result = assemble_model_request_parameters(output_schema=None, tools=None)

        assert result.allow_text_output is True

    def test_allow_text_output_override(self):
        """Test that allow_text_output can be overridden."""
        output_schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
        }

        result = assemble_model_request_parameters(
            output_schema=output_schema, tools=None, allow_text_output=False
        )

        assert result.allow_text_output is False
