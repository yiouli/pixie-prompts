"""Utilities for converting between different message formats."""

from __future__ import annotations

import base64
import json
from typing import Any, Literal, Sequence

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
    UserContent,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.output import OutputObjectDefinition


# Mapping from OpenAI audio formats to media types
AUDIO_FORMAT_TO_MEDIA_TYPE = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "opus": "audio/opus",
}

# Mapping from media types to OpenAI audio formats
MEDIA_TYPE_TO_AUDIO_FORMAT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/pcm": "pcm",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
    "audio/opus": "opus",
}


def _convert_openai_content_array_to_pydantic(
    content_array: list[dict[str, Any]],
) -> list[UserContent]:
    """Convert OpenAI content array format to Pydantic AI UserContent list.

    Args:
        content_array: List of content parts in OpenAI format

    Returns:
        List of Pydantic AI UserContent items (str, ImageUrl, AudioUrl, BinaryContent, etc.)
    """
    user_content: list[UserContent] = []

    for part in content_array:
        part_type = part.get("type")

        if part_type == "text":
            user_content.append(part.get("text", ""))

        elif part_type == "image_url":
            image_data = part.get("image_url", {})
            url = image_data.get("url", "")
            detail = image_data.get("detail")

            # Check if it's a data URI (base64 encoded)
            if url.startswith("data:"):
                binary_content = BinaryContent.from_data_uri(url)
                if detail:
                    binary_content.vendor_metadata = {"detail": detail}
                user_content.append(binary_content)
            else:
                # Regular URL
                vendor_metadata = {"detail": detail} if detail else None
                user_content.append(ImageUrl(url=url, vendor_metadata=vendor_metadata))

        elif part_type == "input_audio":
            audio_data = part.get("input_audio", {})
            data_b64 = audio_data.get("data", "")
            audio_format = audio_data.get("format", "mp3")

            # Convert base64 to binary
            audio_bytes = base64.b64decode(data_b64)
            media_type = AUDIO_FORMAT_TO_MEDIA_TYPE.get(
                audio_format, f"audio/{audio_format}"
            )

            user_content.append(BinaryContent(data=audio_bytes, media_type=media_type))

        elif part_type == "file":
            # File input - convert to DocumentUrl or appropriate type
            file_data = part.get("file", {})
            file_id = file_data.get("file_id", "")
            # OpenAI file references are URLs to their file storage
            if file_id:
                user_content.append(DocumentUrl(url=f"openai://file/{file_id}"))

        else:
            # Unknown type - treat as text if possible
            if "text" in part:
                user_content.append(part["text"])

    return user_content


def _convert_pydantic_content_to_openai(
    content: Sequence[UserContent],
) -> list[dict[str, Any]]:
    """Convert Pydantic AI UserContent sequence to OpenAI content array format.

    Args:
        content: Sequence of Pydantic AI UserContent items

    Returns:
        List of OpenAI content parts
    """
    content_array: list[dict[str, Any]] = []

    for item in content:
        if isinstance(item, str):
            content_array.append({"type": "text", "text": item})

        elif isinstance(item, ImageUrl):
            image_url_data: dict[str, Any] = {"url": item.url}
            # Include detail if present in vendor_metadata
            if item.vendor_metadata and "detail" in item.vendor_metadata:
                image_url_data["detail"] = item.vendor_metadata["detail"]
            content_array.append({"type": "image_url", "image_url": image_url_data})

        elif isinstance(item, BinaryContent):
            if item.is_image:
                # Convert binary image to data URI
                image_url_data: dict[str, Any] = {"url": item.data_uri}
                if item.vendor_metadata and "detail" in item.vendor_metadata:
                    image_url_data["detail"] = item.vendor_metadata["detail"]
                content_array.append({"type": "image_url", "image_url": image_url_data})

            elif item.is_audio:
                # Convert to OpenAI input_audio format
                audio_format = MEDIA_TYPE_TO_AUDIO_FORMAT.get(
                    item.media_type, item.format
                )
                content_array.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": item.base64,
                            "format": audio_format,
                        },
                    }
                )

            elif item.is_video:
                # Video as data URI (limited support in OpenAI)
                content_array.append(
                    {
                        "type": "video",
                        "video": {"url": item.data_uri},
                    }
                )

            elif item.is_document:
                # Document as data URI
                content_array.append(
                    {
                        "type": "file",
                        "file": {"url": item.data_uri},
                    }
                )

            else:
                # Unknown binary type - try as file
                content_array.append(
                    {
                        "type": "file",
                        "file": {"url": item.data_uri},
                    }
                )

        elif isinstance(item, AudioUrl):
            # Audio URL - OpenAI prefers input_audio with base64 data,
            # but we can reference the URL
            content_array.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": item.url},
                }
            )

        elif isinstance(item, VideoUrl):
            # Video URL
            content_array.append(
                {
                    "type": "video",
                    "video": {"url": item.url},
                }
            )

        elif isinstance(item, DocumentUrl):
            # Document URL
            content_array.append(
                {
                    "type": "file",
                    "file": {"url": item.url},
                }
            )

        # Skip CachePoint and other non-content types

    return content_array


def openai_messages_to_pydantic_ai_messages(
    messages: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert OpenAI chat completion message format to Pydantic AI messages.

    This function converts the OpenAI message format (used in chat completions API)
    to the Pydantic AI message format.

    Supported message roles:
    - system/developer: Converted to ModelRequest with SystemPromptPart
    - user: Converted to ModelRequest with UserPromptPart
    - assistant: Converted to ModelResponse with TextPart and/or ToolCallPart
    - tool/function: Converted to ModelRequest with ToolReturnPart

    Args:
        messages: List of OpenAI format messages

    Returns:
        List of Pydantic AI ModelMessage objects

    Raises:
        NotImplementedError: If multimedia content (images, audio, etc.) is encountered
        ValueError: If an unknown message role is encountered
    """
    result: list[ModelMessage] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "developer"):
            # System/developer messages become ModelRequest with SystemPromptPart
            result.append(ModelRequest(parts=[SystemPromptPart(content=content or "")]))

        elif role == "user":
            # Check for multimodal content (content array)
            if isinstance(content, list):
                user_content = _convert_openai_content_array_to_pydantic(content)
                result.append(
                    ModelRequest(parts=[UserPromptPart(content=user_content)])
                )
            else:
                result.append(
                    ModelRequest(parts=[UserPromptPart(content=content or "")])
                )

        elif role == "assistant":
            parts: list[TextPart | ToolCallPart] = []

            # Handle text content if present
            if content:
                parts.append(TextPart(content=content))

            # Handle tool_calls (modern format)
            tool_calls = msg.get("tool_calls", [])
            for tool_call in tool_calls:
                if tool_call.get("type") == "function":
                    func = tool_call.get("function", {})
                    parts.append(
                        ToolCallPart(
                            tool_name=func.get("name", ""),
                            tool_call_id=tool_call.get("id", ""),
                            args=func.get("arguments", "{}"),
                        )
                    )

            # Handle deprecated function_call format
            function_call = msg.get("function_call")
            if function_call:
                parts.append(
                    ToolCallPart(
                        tool_name=function_call.get("name", ""),
                        args=function_call.get("arguments", "{}"),
                    )
                )

            # If no parts were created but we have an assistant message,
            # create an empty text part
            if not parts:
                parts.append(TextPart(content=content or ""))

            result.append(ModelResponse(parts=parts))

        elif role == "tool":
            # Tool response message
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")  # Optional in OpenAI format
            tool_content = msg.get("content", "")

            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            content=tool_content,
                        )
                    ]
                )
            )

        elif role == "function":
            # Deprecated function role message
            func_name = msg.get("name", "")
            func_content = msg.get("content", "")

            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=func_name,
                            content=func_content,
                        )
                    ]
                )
            )

        else:
            raise ValueError(f"Unknown message role: {role}")

    return result


def pydantic_ai_messages_to_openai_messages(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    """Convert Pydantic AI messages to OpenAI chat completion message format.

    This function converts Pydantic AI messages to the OpenAI message format
    that can be used with the chat completions API.

    Supported Pydantic AI parts:
    - SystemPromptPart: Converted to system role message
    - UserPromptPart: Converted to user role message
    - TextPart: Part of assistant role message
    - ToolCallPart: Part of assistant role message with tool_calls
    - ToolReturnPart: Converted to tool role message
    - RetryPromptPart: Converted to tool or user role message
    - ThinkingPart: Excluded from output (internal to model)

    Args:
        messages: List of Pydantic AI ModelMessage objects

    Returns:
        List of OpenAI format messages

    Raises:
        NotImplementedError: If multimedia content is encountered in UserPromptPart
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            # Process each part of the request
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})

                elif isinstance(part, UserPromptPart):
                    # Check for multimodal content
                    if not isinstance(part.content, str):
                        # Content is a sequence - convert to OpenAI content array
                        content_array = _convert_pydantic_content_to_openai(
                            part.content
                        )
                        result.append({"role": "user", "content": content_array})
                    else:
                        result.append({"role": "user", "content": part.content})

                elif isinstance(part, ToolReturnPart):
                    # Serialize content if it's not a string
                    content = part.content
                    if not isinstance(content, str):
                        content = json.dumps(content)

                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": content,
                        }
                    )

                elif isinstance(part, RetryPromptPart):
                    # Convert retry prompt based on whether it has a tool name
                    if part.tool_name:
                        # Retry for a tool call - send as tool message
                        content = (
                            part.content
                            if isinstance(part.content, str)
                            else json.dumps(part.content)
                        )
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": part.model_response(),
                            }
                        )
                    else:
                        # General retry - send as user message
                        result.append(
                            {"role": "user", "content": part.model_response()}
                        )

        elif isinstance(msg, ModelResponse):
            # Collect text parts and tool call parts
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_parts.append(part.content)
                elif isinstance(part, ToolCallPart):
                    # Convert args to string if it's a dict
                    args = part.args
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    elif args is None:
                        args = "{}"

                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.tool_name,
                                "arguments": args,
                            },
                        }
                    )
                elif isinstance(part, ThinkingPart):
                    # ThinkingPart is internal, skip it
                    pass
                # Other part types (BuiltinToolCallPart, BuiltinToolReturnPart, FilePart)
                # are not directly mappable to OpenAI format

            # Build the assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}

            if text_parts:
                assistant_msg["content"] = "\n\n".join(text_parts)
            else:
                assistant_msg["content"] = None

            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            result.append(assistant_msg)

    return result


def assemble_model_request_parameters(
    output_schema: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    *,
    output_mode: Literal["text", "tool", "native", "prompted"] | None = None,
    strict: bool | None = None,
    allow_text_output: bool = True,
) -> ModelRequestParameters:
    """Assemble Pydantic AI ModelRequestParameters from OpenAI format tools and output schema.

    This function converts OpenAI format tools definitions and JSON schema output specification
    to the Pydantic AI ModelRequestParameters format that can be used with model requests.

    Args:
        output_schema: A JSON schema defining the expected structured output format.
            If provided, creates an OutputObjectDefinition. The schema can include
            'title' and 'description' fields which will be extracted.
        tools: List of OpenAI format tool definitions. Each tool should have the format:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "Tool description",
                    "parameters": { ... JSON schema ... },
                    "strict": true/false  # optional
                }
            }
        output_mode: The output mode for structured output. Defaults to "native" when
            output_schema is provided, otherwise "text".
        strict: Whether to enforce strict JSON schema validation for output.
            Only applies when output_schema is provided.
        allow_text_output: Whether plain text output is allowed alongside structured output.
            Defaults to True.

    Returns:
        ModelRequestParameters configured with function_tools and/or output_object.

    Raises:
        ValueError: If a tool has an unsupported type (not "function"),
            if a tool is missing the 'function' key, or if a tool is missing a 'name'.

    Example:
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_weather",
        ...             "description": "Get weather for a location",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {"location": {"type": "string"}},
        ...                 "required": ["location"]
        ...             }
        ...         }
        ...     }
        ... ]
        >>> output_schema = {
        ...     "type": "object",
        ...     "properties": {"temperature": {"type": "number"}},
        ...     "required": ["temperature"]
        ... }
        >>> params = assemble_model_request_parameters(
        ...     output_schema=output_schema,
        ...     tools=tools
        ... )
    """
    function_tools: list[ToolDefinition] = []
    output_object: OutputObjectDefinition | None = None

    # Convert tools to ToolDefinition objects
    if tools:
        for tool in tools:
            tool_type = tool.get("type")
            if tool_type != "function":
                raise ValueError(
                    f"Unsupported tool type: {tool_type}. Only 'function' type is supported."
                )

            function_def = tool.get("function")
            if function_def is None:
                raise ValueError(
                    "Missing 'function' key in tool definition. "
                    "Expected format: {'type': 'function', 'function': {...}}"
                )

            name = function_def.get("name")
            if not name:
                raise ValueError(
                    "Missing 'name' in function definition. "
                    "Every tool must have a name."
                )

            description = function_def.get("description")
            parameters = function_def.get("parameters")
            tool_strict = function_def.get("strict")

            # Build parameters schema, defaulting to empty object if not provided
            parameters_json_schema: dict[str, Any] = (
                parameters
                if parameters is not None
                else {"type": "object", "properties": {}}
            )

            function_tools.append(
                ToolDefinition(
                    name=name,
                    description=description,
                    parameters_json_schema=parameters_json_schema,
                    strict=tool_strict,
                    kind="function",
                )
            )

    # Convert output_schema to OutputObjectDefinition
    if output_schema is not None:
        # Extract optional title and description from schema
        schema_name = output_schema.get("title")
        schema_description = output_schema.get("description")

        output_object = OutputObjectDefinition(
            json_schema=output_schema,
            name=schema_name,
            description=schema_description,
            strict=strict,
        )

    # Determine output_mode
    if output_mode is None:
        output_mode = "native" if output_schema is not None else "text"

    return ModelRequestParameters(
        function_tools=function_tools,
        output_mode=output_mode,
        output_object=output_object,
        allow_text_output=allow_text_output,
    )
