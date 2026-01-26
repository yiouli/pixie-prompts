"""GraphQL schema for SDK server."""

from datetime import datetime
import json
import logging
import os
from typing import Any, Optional, cast, get_args

from graphql import GraphQLError
import jinja2
from pydantic_ai import ModelSettings
from pydantic_ai.direct import model_request
from pydantic_ai.models import KnownModelName
import strawberry
from strawberry.scalars import JSON

from pixie.prompts.prompt import variables_definition_to_schema
from pixie.prompts.prompt_management import get_prompt, list_prompts
from pixie.prompts.utils import (
    assemble_model_request_parameters,
    openai_messages_to_pydantic_ai_messages,
    pydantic_ai_messages_to_openai_messages,
)

logger = logging.getLogger(__name__)


@strawberry.type
class PromptMetadata:
    """Metadata for a registered prompt via create_prompt."""

    id: strawberry.ID
    variables_schema: JSON
    version_count: int
    description: Optional[str] = None
    module: Optional[str] = None


@strawberry.input
class IKeyValue:
    """Key-value attribute."""

    key: str
    value: str


@strawberry.type
class TKeyValue:
    """Key-value attribute."""

    key: str
    value: str


@strawberry.type
class PromptVersion:
    """Prompt version information."""

    version_id: str
    content: str
    created_at: float


@strawberry.type
class Prompt:
    """Full prompt information including versions."""

    id: strawberry.ID
    variables_schema: JSON
    versions: list[PromptVersion]
    default_version_id: str | None
    """default version id can only be None if versions is empty"""
    description: Optional[str] = None
    module: Optional[str] = None


@strawberry.type
class ToolCall:
    """Tool call information."""

    name: str
    args: JSON
    tool_call_id: strawberry.ID


@strawberry.type
class LlmCallResult:

    input: JSON
    output: JSON | None
    tool_calls: list[ToolCall] | None
    usage: JSON
    cost: float
    timestamp: datetime
    reasoning: str | None


def is_demo_mode() -> bool:
    is_demo_mode = os.getenv("IS_DEMO_MODE", "0") in ("1", "true", "True")
    return is_demo_mode


@strawberry.type
class Query:
    """GraphQL queries."""

    @strawberry.field
    async def possible_models(self) -> list[str]:
        """List possible model names.

        Returns:
            A list of model names supported by the server.
        """
        return list(get_args(KnownModelName.__value__))

    @strawberry.field
    async def health_check(self) -> str:
        """Health check endpoint."""
        logger.debug("Health check endpoint called")
        return "0.0.0"

    @strawberry.field
    def list_prompts(self) -> list[PromptMetadata]:
        """List all registered prompt templates.

        Returns:
            A list of PromptMetadata objects containing id, variables_schema, version_count,
            description, and module for each registered prompt.
        """

        return [
            PromptMetadata(
                id=strawberry.ID(p.prompt.id),
                variables_schema=JSON(
                    # NOTE: avoid p.get_variables_schema() to prevent potential fetching from storage
                    # this in theory could be different from the stored schema but in practice should not be
                    variables_definition_to_schema(p.prompt.variables_definition)
                ),
                version_count=p.prompt.get_version_count(),
                description=p.description,
                module=p.module,
            )
            for p in list_prompts()
        ]

    @strawberry.field
    async def get_prompt(self, id: strawberry.ID) -> Prompt:
        """Get full prompt information including versions.

        Args:
            id: The unique identifier of the prompt.
        Returns:
            Prompt object containing id, variables_schema, versions,
            and default_version_id.
        Raises:
            GraphQLError: If prompt with given id is not found.
        """
        prompt_with_registration = get_prompt((str(id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{id}' not found.")
        prompt = prompt_with_registration.prompt
        if not prompt.exists_in_storage():
            return Prompt(
                id=id,
                variables_schema=JSON(
                    # NOTE: avoid prompt.get_variables_schema() to prevent potential fetching from storage
                    variables_definition_to_schema(prompt.variables_definition)
                ),
                versions=[],
                default_version_id=None,
                description=prompt_with_registration.description,
                module=prompt_with_registration.module,
            )
        versions_dict = prompt.get_versions()
        versions = [
            PromptVersion(
                version_id=k, content=v, created_at=prompt.get_version_creation_time(k)
            )
            for k, v in versions_dict.items()
        ]
        default_version_id: str = prompt.get_default_version_id()
        variables_schema = prompt.get_variables_schema()
        return Prompt(
            id=id,
            variables_schema=JSON(variables_schema),
            versions=versions,
            default_version_id=default_version_id,
            description=prompt_with_registration.description,
            module=prompt_with_registration.module,
        )


@strawberry.type
class Mutation:
    """GraphQL mutations."""

    @strawberry.mutation
    async def call_llm(
        self,
        model: str,
        prompt_template: str,
        variables: Optional[JSON],
        prompt_placeholder: str,
        input_messages: list[JSON],
        output_schema: Optional[JSON] = None,
        tools: Optional[list[JSON]] = None,
        model_parameters: Optional[JSON] = None,
    ) -> LlmCallResult:
        """Call LLM with the given inputs.

        Args:
            model: The model name to use (e.g., "openai:gpt-4").
            prompt_template: prompt template string.
            variables: variables for the prompt template.
            prompt_placeholder: placeholder string in the prompt template to be replaced.
            input_messages: List of messages as JSON objects in openai format, containing prompt_placeholder in content.
            output_schema: Optional output schema.
            tools: Optional tools configuration (not yet implemented).
            model_parameters: Optional model parameters.

        Returns:
            LLM call result

        Raises:
            GraphQLError: If the LLM call fails.
        """
        try:
            if is_demo_mode():
                model = "openai:gpt-4o-mini"
            template = jinja2.Template(prompt_template)
            prompt = template.render(**(cast(dict[str, Any], variables) or {}))
            print(prompt)
            print(type(prompt))
            pydantic_messages = openai_messages_to_pydantic_ai_messages(
                cast(list[dict[str, Any]], input_messages)
            )
            for msg in pydantic_messages:
                for part in msg.parts:
                    if part.part_kind == "user-prompt":
                        if isinstance(part.content, str):
                            part.content = part.content.replace(
                                prompt_placeholder,
                                prompt,
                            )
                        else:
                            part.content = [
                                p.replace(prompt_placeholder, prompt)
                                for p in part.content
                                if isinstance(p, str)
                            ]
                    elif part.part_kind == "system-prompt":
                        part.content = part.content.replace(prompt_placeholder, prompt)

            # Replace the placeholder in input messages
            response = await model_request(
                model=model,
                messages=pydantic_messages,
                model_settings=cast(ModelSettings | None, model_parameters),
                model_request_parameters=assemble_model_request_parameters(
                    cast(dict[str, Any] | None, output_schema),
                    cast(list[dict[str, Any]] | None, tools),
                    strict=True,
                    allow_text_output=False,
                ),
            )
            return LlmCallResult(
                input=JSON(pydantic_ai_messages_to_openai_messages(pydantic_messages)),
                output=(
                    JSON(json.loads(response.text) if output_schema else response.text)
                    if response.text
                    else None
                ),
                tool_calls=(
                    [
                        ToolCall(
                            name=tc.tool_name,
                            args=JSON(tc.args_as_dict()),
                            tool_call_id=strawberry.ID(tc.tool_call_id),
                        )
                        for tc in response.tool_calls
                    ]
                    if response.tool_calls
                    else None
                ),
                usage=JSON(
                    {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                ),
                cost=float(response.cost().total_price),
                timestamp=response.timestamp,
                reasoning=response.thinking,
            )
        except Exception as e:
            logger.error("Error running LLM: %s", str(e))
            raise GraphQLError(f"Failed to run LLM: {str(e)}") from e

    @strawberry.mutation
    async def add_prompt_version(
        self,
        prompt_id: strawberry.ID,
        version_id: str,
        content: str,
        set_as_default: bool = False,
    ) -> str:
        """Add a new version to an existing prompt.

        Args:
            prompt_id: The unique identifier of the prompt.
            version_id: The identifier for the new version.
            content: The content of the new prompt version.
            set_as_default: Whether to set this version as the default.

        Returns:
            The updated BasePrompt object.
        """
        if is_demo_mode():
            raise GraphQLError("Modifications are not allowed in demo mode.")
        prompt_with_registration = get_prompt((str(prompt_id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{prompt_id}' not found.")
        prompt = prompt_with_registration.prompt
        try:
            prompt.append_version(
                version_id=version_id,
                content=content,
                set_as_default=set_as_default,
            )
        except Exception as e:
            raise GraphQLError(f"Failed to add prompt version: {str(e)}") from e
        return "OK"

    @strawberry.mutation
    async def update_default_prompt_version(
        self,
        prompt_id: strawberry.ID,
        default_version_id: str,
    ) -> str:
        """Update the default version of an existing prompt.

        Args:
            prompt_id: The unique identifier of the prompt.
            default_version_id: The identifier of the version to set as default.

        Returns:
            True if the update was successful.
        """
        if is_demo_mode():
            raise GraphQLError("Modifications are not allowed in demo mode.")
        prompt_with_registration = get_prompt((str(prompt_id)))
        if prompt_with_registration is None:
            raise GraphQLError(f"Prompt with id '{prompt_id}' not found.")
        prompt = prompt_with_registration.prompt
        try:
            prompt.update_default_version_id(default_version_id)
        except Exception as e:
            raise GraphQLError(
                f"Failed to update default prompt version: {str(e)}"
            ) from e
        return "OK"


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
