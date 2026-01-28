"""Unit tests for GraphQL endpoints."""

import os
import pytest
import tempfile
from typing import Any

from pixie.prompts.graphql import schema
from pixie.prompts.prompt_management import create_prompt, _registry
from pixie.prompts import storage as storage_module


class TestGraphQLQueries:
    """Test GraphQL query endpoints."""

    def setup_method(self) -> None:
        """Clear the registry before each test."""
        _registry.clear()
        # Reset storage instance
        storage_module._storage_instance = None
        # Set up temp directory for storage via environment variable
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = self.temp_dir

    def teardown_method(self) -> None:
        """Clean up after each test."""
        _registry.clear()
        storage_module._storage_instance = None
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """Test health_check query."""
        query = """
        query {
            healthCheck
        }
        """

        result = await schema.execute(query)

        assert result.errors is None
        assert result.data is not None
        assert result.data["healthCheck"] == "0.0.0"

    def test_list_prompts_empty(self) -> None:
        """Test list_prompts query with no prompts."""
        query = """
        query {
            listPrompts {
                id
                versionCount
            }
        }
        """

        result = schema.execute_sync(query)

        assert result.errors is None
        assert result.data is not None
        assert result.data["listPrompts"] == []

    def test_list_prompts_with_prompts(self) -> None:
        """Test list_prompts query with prompts."""
        # Create a test prompt
        create_prompt("test_prompt", description="Test description")

        query = """
        query {
            listPrompts {
                id
                versionCount
                description
            }
        }
        """

        result = schema.execute_sync(query)

        assert result.errors is None
        assert result.data is not None
        assert len(result.data["listPrompts"]) == 1
        prompt_data = result.data["listPrompts"][0]
        assert prompt_data["id"] == "test_prompt"
        assert prompt_data["versionCount"] == 0
        assert prompt_data["description"] == "Test description"

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """Test get_prompt query with non-existent prompt."""
        query = """
        query GetPrompt($id: ID!) {
            getPrompt(id: $id) {
                id
                versions {
                    versionId
                    content
                    createdAt
                }
                defaultVersionId
            }
        }
        """

        result = await schema.execute(query, variable_values={"id": "non_existent"})

        assert result.errors is not None
        assert result.errors
        assert "not found" in str(result.errors[0])

    @pytest.mark.asyncio
    async def test_get_prompt_exists_no_storage(self) -> None:
        """Test get_prompt query for prompt that exists but not in storage."""
        # Create a test prompt
        create_prompt("test_prompt", description="Test description")

        query = """
        query GetPrompt($id: ID!) {
            getPrompt(id: $id) {
                id
                versions {
                    versionId
                    content
                    createdAt
                }
                defaultVersionId
                description
            }
        }
        """

        result = await schema.execute(query, variable_values={"id": "test_prompt"})

        assert result.errors is None
        assert result.data is not None
        assert result.data["getPrompt"]["id"] == "test_prompt"
        assert result.data["getPrompt"]["versions"] == []
        assert result.data["getPrompt"]["defaultVersionId"] is None
        assert result.data["getPrompt"]["description"] == "Test description"

    @pytest.mark.asyncio
    async def test_possible_models(self) -> None:
        """Test possible_models query."""
        query = """
        query {
            possibleModels
        }
        """

        result = await schema.execute(query)

        assert result.errors is None
        assert result.data is not None
        assert "possibleModels" in result.data
        assert isinstance(result.data["possibleModels"], list)
        assert len(result.data["possibleModels"]) > 0


class TestGraphQLMutations:
    """Test GraphQL mutation endpoints."""

    def setup_method(self) -> None:
        """Clear the registry before each test."""
        _registry.clear()
        # Reset storage instance
        storage_module._storage_instance = None
        # Set up temp directory for storage via environment variable
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = self.temp_dir

    def teardown_method(self) -> None:
        """Clean up after each test."""
        _registry.clear()
        storage_module._storage_instance = None
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]

    @pytest.mark.asyncio
    async def test_add_prompt_version_success(self) -> None:
        """Test add_prompt_version mutation."""
        # Create a test prompt
        create_prompt("test_prompt")

        mutation = """
        mutation AddPromptVersion(
            $promptId: ID!
            $versionId: String!
            $content: String!
            $setAsDefault: Boolean
        ) {
            addPromptVersion(
                promptId: $promptId
                versionId: $versionId
                content: $content
                setAsDefault: $setAsDefault
            )
        }
        """

        variables = {
            "promptId": "test_prompt",
            "versionId": "v1",
            "content": "Test content",
            "setAsDefault": True,
        }

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is None
        assert result.data is not None
        assert result.data["addPromptVersion"] == "OK"

    @pytest.mark.asyncio
    async def test_add_prompt_version_not_found(self) -> None:
        """Test add_prompt_version mutation with non-existent prompt."""
        mutation = """
        mutation AddPromptVersion(
            $promptId: ID!
            $versionId: String!
            $content: String!
        ) {
            addPromptVersion(
                promptId: $promptId
                versionId: $versionId
                content: $content
            )
        }
        """

        variables = {
            "promptId": "non_existent",
            "versionId": "v1",
            "content": "Test content",
        }

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is not None
        assert result.errors
        assert "not found" in str(result.errors[0])

    @pytest.mark.asyncio
    async def test_update_default_prompt_version_success(self) -> None:
        """Test update_default_prompt_version mutation."""
        # Create a test prompt and add versions
        prompt: Any = create_prompt("test_prompt")
        prompt.append_version("v1", "Content v1")
        prompt.append_version("v2", "Content v2")

        mutation = """
        mutation UpdateDefaultPromptVersion(
            $promptId: ID!
            $defaultVersionId: String!
        ) {
            updateDefaultPromptVersion(
                promptId: $promptId
                defaultVersionId: $defaultVersionId
            )
        }
        """

        variables = {"promptId": "test_prompt", "defaultVersionId": "v2"}

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is None
        assert result.data is not None
        assert result.data["updateDefaultPromptVersion"] == "OK"

    @pytest.mark.asyncio
    async def test_update_default_prompt_version_not_found(self) -> None:
        """Test update_default_prompt_version mutation with non-existent prompt."""
        mutation = """
        mutation UpdateDefaultPromptVersion(
            $promptId: ID!
            $defaultVersionId: String!
        ) {
            updateDefaultPromptVersion(
                promptId: $promptId
                defaultVersionId: $defaultVersionId
            )
        }
        """

        variables = {"promptId": "non_existent", "defaultVersionId": "v1"}

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is not None
        assert result.errors
        assert "not found" in str(result.errors[0])
