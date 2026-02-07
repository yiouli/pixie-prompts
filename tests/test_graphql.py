"""Unit tests for GraphQL endpoints."""

import os
import pytest
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pixie.prompts.graphql import (
    LlmCallUsage,
    schema,
    is_demo_mode,
    execute_single_llm_call,
)
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

    @pytest.mark.asyncio
    async def test_possible_models_demo_mode(self) -> None:
        """Test possible_models query in demo mode."""
        # Set demo mode
        os.environ["IS_DEMO_MODE"] = "1"
        try:
            query = """
            query {
                possibleModels
            }
            """

            result = await schema.execute(query)

            assert result.errors is None
            assert result.data is not None
            assert result.data["possibleModels"] == ["openai:gpt-4o-mini"]
        finally:
            # Clean up
            if "IS_DEMO_MODE" in os.environ:
                del os.environ["IS_DEMO_MODE"]


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
    async def test_add_prompt_version_demo_mode(self) -> None:
        """Test add_prompt_version mutation in demo mode."""
        # Set demo mode
        os.environ["IS_DEMO_MODE"] = "1"
        try:
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
                "promptId": "test_prompt",
                "versionId": "v1",
                "content": "Test content",
            }

            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is not None
            assert result.errors
            assert "Modifications are not allowed in demo mode" in str(result.errors[0])
        finally:
            # Clean up
            if "IS_DEMO_MODE" in os.environ:
                del os.environ["IS_DEMO_MODE"]

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

    @pytest.mark.asyncio
    async def test_update_default_prompt_version_demo_mode(self) -> None:
        """Test update_default_prompt_version mutation in demo mode."""
        # Set demo mode
        os.environ["IS_DEMO_MODE"] = "1"
        try:
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

            variables = {"promptId": "test_prompt", "defaultVersionId": "v1"}

            result = await schema.execute(mutation, variable_values=variables)

            assert result.errors is not None
            assert result.errors
            assert "Modifications are not allowed in demo mode" in str(result.errors[0])
        finally:
            # Clean up
            if "IS_DEMO_MODE" in os.environ:
                del os.environ["IS_DEMO_MODE"]

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.execute_single_llm_call")
    async def test_call_llm_success(self, mock_execute: AsyncMock) -> None:
        """Test call_llm mutation success."""
        from pixie.prompts.graphql import LlmCallResult
        from strawberry.scalars import JSON

        mock_result = LlmCallResult(
            input=JSON({"messages": []}),
            output=JSON({"response": "test"}),
            tool_calls=None,
            usage=LlmCallUsage(input_tokens=10, output_tokens=0, total_tokens=10),
            cost=0.01,
            timestamp=MagicMock(),
            reasoning=None,
        )
        mock_execute.return_value = mock_result

        mutation = """
        mutation CallLLM(
            $model: String!
            $promptTemplate: String!
            $variables: JSON
            $promptPlaceholder: String!
            $inputMessages: [JSON!]!
            $outputSchema: JSON
        ) {
            callLlm(
                model: $model
                promptTemplate: $promptTemplate
                variables: $variables
                promptPlaceholder: $promptPlaceholder
                inputMessages: $inputMessages
                outputSchema: $outputSchema
            ) {
                output
                usage {
                    inputTokens
                    outputTokens
                    totalTokens
                }
                cost
            }
        }
        """

        variables = {
            "model": "openai:gpt-4",
            "promptTemplate": "Hello {{name}}",
            "variables": {"name": "World"},
            "promptPlaceholder": "{{prompt}}",
            "inputMessages": [{"role": "user", "content": "Hello"}],
            "outputSchema": {"type": "object"},
        }

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is None
        assert result.data is not None
        assert "callLlm" in result.data
        call_result = result.data["callLlm"]
        assert call_result["output"] == {"response": "test"}
        assert call_result["usage"] == {
            "inputTokens": 10,
            "outputTokens": 0,
            "totalTokens": 10,
        }
        assert call_result["cost"] == 0.01

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.execute_single_llm_call")
    async def test_call_llm_error(self, mock_execute: AsyncMock) -> None:
        """Test call_llm mutation error."""
        mock_execute.side_effect = Exception("Test error")

        mutation = """
        mutation CallLLM(
            $model: String!
            $promptTemplate: String!
            $variables: JSON
            $promptPlaceholder: String!
            $inputMessages: [JSON!]!
        ) {
            callLlm(
                model: $model
                promptTemplate: $promptTemplate
                variables: $variables
                promptPlaceholder: $promptPlaceholder
                inputMessages: $inputMessages
            ) {
                output
            }
        }
        """

        variables = {
            "model": "openai:gpt-4",
            "promptTemplate": "Hello",
            "variables": None,
            "promptPlaceholder": "{{prompt}}",
            "inputMessages": [{"role": "user", "content": "Hello"}],
        }

        result = await schema.execute(mutation, variable_values=variables)

        assert result.errors is not None
        assert result.errors
        assert "Test error" in str(result.errors[0])


class TestUtilityFunctions:
    """Test utility functions in graphql.py."""

    def test_is_demo_mode_false_by_default(self) -> None:
        """Test is_demo_mode returns False when IS_DEMO_MODE is not set."""
        if "IS_DEMO_MODE" in os.environ:
            del os.environ["IS_DEMO_MODE"]
        assert is_demo_mode() is False

    def test_is_demo_mode_false_for_invalid_values(self) -> None:
        """Test is_demo_mode returns False for invalid values."""
        os.environ["IS_DEMO_MODE"] = "false"
        assert is_demo_mode() is False
        os.environ["IS_DEMO_MODE"] = "0"
        assert is_demo_mode() is False

    def test_is_demo_mode_true_for_valid_values(self) -> None:
        """Test is_demo_mode returns True for valid true values."""
        for value in ["1", "true", "True"]:
            os.environ["IS_DEMO_MODE"] = value
            assert is_demo_mode() is True
        if "IS_DEMO_MODE" in os.environ:
            del os.environ["IS_DEMO_MODE"]


class TestExecuteSingleLlmCall:
    """Test execute_single_llm_call function."""

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.model_request")
    async def test_execute_single_llm_call_success(
        self, mock_model_request: MagicMock
    ) -> None:
        """Test successful execution of single LLM call."""
        from pixie.prompts.graphql import LlmCallInput, LlmCallResult
        from strawberry.scalars import JSON
        import strawberry

        # Mock the response
        mock_response = MagicMock()
        mock_response.text = '{"result": "test"}'
        mock_response.tool_calls = []
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.cost.return_value.total_price = 0.01
        mock_response.timestamp = MagicMock()
        mock_response.thinking = "Test reasoning"
        mock_model_request.return_value = mock_response

        call_input = LlmCallInput(
            id=strawberry.ID("test-id"),
            model="openai:gpt-4",
            prompt_template="Hello {{name}}",
            variables=JSON({"name": "World"}),
            prompt_placeholder="{{prompt}}",
            input_messages=[JSON({"role": "user", "content": "Hello"})],
            output_schema=JSON({"type": "object"}),
        )

        result = await execute_single_llm_call(call_input)

        assert isinstance(result, LlmCallResult)
        assert result.output == {"result": "test"}
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.cost == 0.01
        assert result.reasoning == "Test reasoning"
        assert result.tool_calls is None

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.model_request")
    async def test_execute_single_llm_call_no_output_schema(
        self, mock_model_request: MagicMock
    ) -> None:
        """Test LLM call without output schema."""
        from pixie.prompts.graphql import LlmCallInput, LlmCallResult
        from strawberry.scalars import JSON
        import strawberry

        mock_response = MagicMock()
        mock_response.text = "Plain text response"
        mock_response.tool_calls = []
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.cost.return_value.total_price = 0.005
        mock_response.timestamp = MagicMock()
        mock_response.thinking = None
        mock_model_request.return_value = mock_response

        call_input = LlmCallInput(
            id=strawberry.ID("test-id"),
            model="openai:gpt-4",
            prompt_template="Hello",
            variables=None,
            prompt_placeholder="{{prompt}}",
            input_messages=[JSON({"role": "user", "content": "Hello"})],
            output_schema=None,
        )

        result = await execute_single_llm_call(call_input)

        assert isinstance(result, LlmCallResult)
        assert result.output == "Plain text response"

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.model_request")
    async def test_execute_single_llm_call_with_tool_calls(
        self, mock_model_request: MagicMock
    ) -> None:
        """Test LLM call with tool calls."""
        from pixie.prompts.graphql import LlmCallInput, LlmCallResult
        from strawberry.scalars import JSON
        import strawberry

        mock_tool_call = MagicMock()
        mock_tool_call.tool_name = "test_tool"
        mock_tool_call.args_as_dict.return_value = {"arg": "value"}
        mock_tool_call.tool_call_id = "tool-1"

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_response.cost.return_value.total_price = 0.005
        mock_response.timestamp = MagicMock()
        mock_response.thinking = None
        mock_model_request.return_value = mock_response

        call_input = LlmCallInput(
            id=strawberry.ID("test-id"),
            model="openai:gpt-4",
            prompt_template="Hello",
            variables=None,
            prompt_placeholder="{{prompt}}",
            input_messages=[JSON({"role": "user", "content": "Hello"})],
            output_schema=None,
        )

        result = await execute_single_llm_call(call_input)

        assert isinstance(result, LlmCallResult)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"
        assert result.tool_calls[0].args == {"arg": "value"}

    @pytest.mark.asyncio
    @patch("pixie.prompts.graphql.model_request")
    async def test_execute_single_llm_call_error(
        self, mock_model_request: MagicMock
    ) -> None:
        """Test LLM call that raises an exception."""
        from pixie.prompts.graphql import LlmCallInput
        from strawberry.scalars import JSON
        import strawberry

        mock_model_request.side_effect = Exception("Test error")

        call_input = LlmCallInput(
            id=strawberry.ID("test-id"),
            model="openai:gpt-4",
            prompt_template="Hello",
            variables=None,
            prompt_placeholder="{{prompt}}",
            input_messages=[JSON({"role": "user", "content": "Hello"})],
            output_schema=None,
        )

        with pytest.raises(Exception, match="Test error"):
            await execute_single_llm_call(call_input)
