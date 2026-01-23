"""GraphQL schema for SDK server."""

import logging
from typing import Optional

from graphql import GraphQLError
import strawberry
from strawberry.scalars import JSON

from pixie.prompts.prompt import variables_definition_to_schema
from pixie.prompts.prompt_management import get_prompt, list_prompts

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
class Prompt:
    """Full prompt information including versions."""

    id: strawberry.ID
    variables_schema: JSON
    versions: list[TKeyValue]
    created_at: float | None
    default_version_id: str | None
    """default version id can only be None if versions is empty"""
    description: Optional[str] = None
    module: Optional[str] = None


@strawberry.type
class Query:
    """GraphQL queries."""

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
                created_at=None,
            )
        versions_dict = prompt.get_versions()
        versions = [TKeyValue(key=k, value=v) for k, v in versions_dict.items()]
        default_version_id: str = prompt.get_default_version_id()
        variables_schema = prompt.get_variables_schema()
        return Prompt(
            id=id,
            variables_schema=JSON(variables_schema),
            versions=versions,
            default_version_id=default_version_id,
            description=prompt_with_registration.description,
            module=prompt_with_registration.module,
            created_at=prompt.created_at,
        )


@strawberry.type
class Mutation:
    """GraphQL mutations."""

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
