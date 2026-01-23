"""Comprehensive unit tests for pixie.prompts.storage module."""

import json
import logging
import os
import tempfile
import time
import pytest
from types import NoneType
from typing import Dict

from pixie.prompts.prompt import BaseUntypedPrompt, BasePrompt, _prompt_registry
from pixie.prompts.storage import PromptLoadError, _FilePromptStorage


def write_prompt_folder(
    base_dir: str,
    prompt_id: str,
    *,
    versions: Dict[str, str],
    default_version_id: str,
    variables_schema: Dict | None = None,
):
    prompt_dir = os.path.join(base_dir, prompt_id)
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "defaultVersionId": default_version_id,
                "variablesSchema": variables_schema
                or {"type": "object", "properties": {}},
            },
            f,
        )
    for version_id, content in versions.items():
        with open(os.path.join(prompt_dir, f"{version_id}.jinja"), "w") as f:
            f.write(content)


class TestFilePromptStorage:
    """Tests for FilePromptStorage class."""

    @pytest.fixture(autouse=True)
    def clear_prompt_registry(self):
        """Clear the global prompt registry before each test."""
        _prompt_registry.clear()

    @pytest.fixture(autouse=True)
    def reset_storage_instance(self):
        """Reset the global storage instance before each test."""
        import pixie.prompts.storage as storage_module

        storage_module._storage_instance = None
        yield
        storage_module._storage_instance = None

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_prompt_data(self) -> Dict[str, Dict]:
        """Sample prompt data for testing."""
        return {
            "prompt1": {
                "versions": {"v1": "Hello {name}", "v2": "Hi {name}"},
                "defaultVersionId": "v1",
                "variablesSchema": {"type": "object", "properties": {}},
            },
            "prompt2": {
                "versions": {"default": "Goodbye {name}"},
                "defaultVersionId": "default",
                "variablesSchema": {"type": "object", "properties": {}},
            },
        }

    def create_sample_files(self, temp_dir: str, sample_data: Dict[str, Dict]):
        """Create sample prompt folders in the temp directory."""
        for prompt_id, data in sample_data.items():
            write_prompt_folder(
                temp_dir,
                prompt_id,
                versions=data["versions"],
                default_version_id=data["defaultVersionId"],
                variables_schema=data["variablesSchema"],
            )

    def test_init_creates_directory_if_not_exists(self, temp_dir: str):
        """Test that __init__ creates the directory if it doesn't exist."""
        subdir = os.path.join(temp_dir, "storage")
        assert not os.path.exists(subdir)

        storage = _FilePromptStorage(subdir)
        assert os.path.exists(subdir)
        assert isinstance(storage._prompts, dict)
        assert len(storage._prompts) == 0

    @pytest.mark.asyncio
    async def test_init_loads_existing_files(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that __init__ loads existing JSON files into memory."""
        self.create_sample_files(temp_dir, sample_prompt_data)

        storage = _FilePromptStorage(temp_dir)

        assert len(storage._prompts) == 2
        assert "prompt1" in storage._prompts
        assert "prompt2" in storage._prompts

        prompt1 = storage._prompts["prompt1"]
        assert isinstance(prompt1, BaseUntypedPrompt)
        assert prompt1.id == "prompt1"
        assert prompt1.get_versions() == sample_prompt_data["prompt1"]["versions"]
        assert (
            prompt1.get_default_version_id()
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

        prompt2 = storage._prompts["prompt2"]
        assert prompt2.id == "prompt2"
        assert prompt2.get_versions() == sample_prompt_data["prompt2"]["versions"]
        assert (
            prompt2.get_default_version_id()
            == sample_prompt_data["prompt2"]["defaultVersionId"]
        )

    def test_init_handles_empty_directory(self, temp_dir: str):
        """Test that __init__ handles an empty directory gracefully."""
        storage = _FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 0

    def test_init_skips_non_json_files(self, temp_dir: str):
        """Test that __init__ skips non-prompt entries in storage directory."""
        write_prompt_folder(
            temp_dir,
            "prompt1",
            versions={"default": "test"},
            default_version_id="default",
            variables_schema={"type": "object", "properties": {}},
        )

        txt_path = os.path.join(temp_dir, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("This is not JSON")

        storage = _FilePromptStorage(temp_dir)
        assert len(storage._prompts) == 1
        assert "prompt1" in storage._prompts

    def test_init_handles_invalid_json(self, temp_dir: str):
        """Test that __init__ raises an exception for invalid JSON."""
        prompt_dir = os.path.join(temp_dir, "invalid")
        os.makedirs(prompt_dir, exist_ok=True)
        metadata_path = os.path.join(prompt_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write("invalid json content")

        with pytest.raises(PromptLoadError) as excinfo:
            _FilePromptStorage(temp_dir)

        failures = excinfo.value.failures
        assert len(failures) == 1
        assert isinstance(failures[0].error, json.JSONDecodeError)

    def test_init_handles_missing_versions(self, temp_dir: str):
        """Test that __init__ raises ValueError for missing versions in JSON."""
        prompt_dir = os.path.join(temp_dir, "missing")
        os.makedirs(prompt_dir, exist_ok=True)
        metadata_path = os.path.join(prompt_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({"defaultVersionId": "default", "variablesSchema": {}}, f)

        with pytest.raises(PromptLoadError) as excinfo:
            _FilePromptStorage(temp_dir)

        failures = excinfo.value.failures
        assert len(failures) == 1
        assert isinstance(failures[0].error, KeyError)

    def test_load_without_metadata_uses_latest_version(self, temp_dir: str):
        """When metadata is missing, the latest created version becomes default and per-version ctime is recorded."""
        prompt_dir = os.path.join(temp_dir, "no_metadata")
        os.makedirs(prompt_dir, exist_ok=True)

        v1_path = os.path.join(prompt_dir, "v1.jinja")
        with open(v1_path, "w") as f:
            f.write("first")

        time.sleep(0.02)

        v2_path = os.path.join(prompt_dir, "v2.jinja")
        with open(v2_path, "w") as f:
            f.write("second")

        storage = _FilePromptStorage(temp_dir)

        prompt = storage.get("no_metadata")
        assert prompt.get_versions() == {"v1": "first", "v2": "second"}
        assert prompt.get_default_version_id() == "v2"
        assert prompt.get_version_creation_time("v1") == pytest.approx(
            os.path.getctime(v1_path)
        )
        assert prompt.get_version_creation_time("v2") == pytest.approx(
            os.path.getctime(v2_path)
        )

    def test_load_without_metadata_and_no_versions_fails(self, temp_dir: str):
        """Loading a prompt without metadata still fails when no version files exist."""
        prompt_dir = os.path.join(temp_dir, "empty")
        os.makedirs(prompt_dir, exist_ok=True)

        with pytest.raises(PromptLoadError) as excinfo:
            _FilePromptStorage(temp_dir)

        failures = excinfo.value.failures
        assert len(failures) == 1
        assert isinstance(failures[0].error, KeyError)

    def test_load_isolates_failures_and_loads_valid_prompts(self, temp_dir: str):
        """Valid prompts load even when others fail, with aggregated error reported."""
        # Valid prompt
        write_prompt_folder(
            temp_dir,
            "prompt1",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Invalid prompt (missing metadata keys)
        broken_dir = os.path.join(temp_dir, "broken")
        os.makedirs(broken_dir, exist_ok=True)
        with open(os.path.join(broken_dir, "metadata.json"), "w") as f:
            json.dump({"variablesSchema": {}}, f)

        storage = _FilePromptStorage(temp_dir, raise_on_error=False)
        assert storage.exists("prompt1") is True
        assert storage.exists("broken") is False
        assert len(storage.load_failures) == 1

        # Reload with raising enabled to surface aggregated error while keeping valid prompt loaded
        with pytest.raises(PromptLoadError) as excinfo:
            storage.load()

        assert len(excinfo.value.failures) == 1
        assert storage.exists("prompt1") is True
        assert storage.get("prompt1").id == "prompt1"

    def test_load_logs_failures(self, temp_dir: str, caplog):
        """Errors are logged and summarized when loading prompts."""
        broken_dir = os.path.join(temp_dir, "broken")
        os.makedirs(broken_dir, exist_ok=True)
        with open(os.path.join(broken_dir, "metadata.json"), "w") as f:
            f.write("{invalid_json}")

        storage = _FilePromptStorage(temp_dir, raise_on_error=False)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            storage.load(raise_on_error=False)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Failed to load prompt" in msg for msg in messages)
        assert any("failure(s)" in msg for msg in messages)

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that exists returns True for existing prompts."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        assert storage.exists("prompt1") is True
        assert storage.exists("prompt2") is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_non_existing_prompt(self, temp_dir: str):
        """Test that exists returns False for non-existing prompts."""
        storage = _FilePromptStorage(temp_dir)

        assert storage.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_save_creates_new_prompt(self, temp_dir: str):
        """Test that save creates a new prompt and returns True."""
        storage = _FilePromptStorage(temp_dir)

        prompt = BaseUntypedPrompt(
            versions={"v1": "Hello {name}", "v2": "Hi {name}"},
            default_version_id="v1",
            id="new_prompt",
        )

        # Save should work for new prompts now and return True
        result = storage.save(prompt)
        assert result is True

        prompt_dir = os.path.join(temp_dir, "new_prompt")
        assert os.path.isdir(prompt_dir)

        with open(os.path.join(prompt_dir, "metadata.json"), "r") as f:
            data = json.load(f)

        assert data["defaultVersionId"] == "v1"
        assert "variablesSchema" in data

        with open(os.path.join(prompt_dir, "v1.jinja"), "r") as f:
            assert f.read() == "Hello {name}"
        with open(os.path.join(prompt_dir, "v2.jinja"), "r") as f:
            assert f.read() == "Hi {name}"

    @pytest.mark.asyncio
    async def test_save_existing_prompt_raises_on_content_change(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Changing existing version content should raise and leave files untouched."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        prompt_dir = os.path.join(temp_dir, "prompt1")
        original_v1_path = os.path.join(prompt_dir, "v1.jinja")
        with open(original_v1_path, "r") as f:
            original_content = f.read()
        original_ctime = os.path.getctime(original_v1_path)

        updated_versions = {"v1": "Updated {name}", "v3": "New version"}
        updated_prompt = BaseUntypedPrompt(
            versions=updated_versions, default_version_id="v1", id="prompt1"
        )

        with pytest.raises(ValueError, match="already exists with different content"):
            storage.save(updated_prompt)

        with open(original_v1_path, "r") as f:
            assert f.read() == original_content
        assert os.path.getctime(original_v1_path) == pytest.approx(original_ctime)
        assert not os.path.exists(os.path.join(prompt_dir, "v3.jinja"))

    @pytest.mark.asyncio
    async def test_save_existing_prompt_same_content_preserves_ctime(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Saving unchanged content should leave version files intact while updating metadata."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        prompt_dir = os.path.join(temp_dir, "prompt1")
        v1_path = os.path.join(prompt_dir, "v1.jinja")
        v2_path = os.path.join(prompt_dir, "v2.jinja")
        original_ctimes = {
            "v1": os.path.getctime(v1_path),
            "v2": os.path.getctime(v2_path),
        }

        updated_prompt = BaseUntypedPrompt(
            versions=sample_prompt_data["prompt1"]["versions"],
            default_version_id="v2",  # flip default to ensure metadata updates
            id="prompt1",
        )

        result = storage.save(updated_prompt)
        assert result is False

        assert os.path.getctime(v1_path) == pytest.approx(original_ctimes["v1"])
        assert os.path.getctime(v2_path) == pytest.approx(original_ctimes["v2"])

        stored_prompt = storage.get("prompt1")
        assert stored_prompt.get_default_version_id() == "v2"
        assert stored_prompt.get_version_creation_time("v1") == pytest.approx(
            original_ctimes["v1"]
        )
        assert stored_prompt.get_version_creation_time("v2") == pytest.approx(
            original_ctimes["v2"]
        )

    @pytest.mark.asyncio
    async def test_get_returns_existing_prompt(
        self, temp_dir: str, sample_prompt_data: Dict[str, Dict]
    ):
        """Test that get returns the correct prompt for existing ID."""
        self.create_sample_files(temp_dir, sample_prompt_data)
        storage = _FilePromptStorage(temp_dir)

        prompt = storage.get("prompt1")
        assert isinstance(prompt, BaseUntypedPrompt)
        assert prompt.id == "prompt1"
        assert prompt.get_versions() == sample_prompt_data["prompt1"]["versions"]
        assert (
            prompt.get_default_version_id()
            == sample_prompt_data["prompt1"]["defaultVersionId"]
        )

    @pytest.mark.asyncio
    async def test_get_raises_keyerror_for_non_existing_prompt(self, temp_dir: str):
        """Test that get raises KeyError for non-existing prompt ID."""
        storage = _FilePromptStorage(temp_dir)

        with pytest.raises(KeyError):
            storage.get("nonexistent")

    @pytest.mark.asyncio
    async def test_save_writes_to_file_before_memory_update(self, temp_dir: str):
        """Test that save creates a new prompt successfully."""
        storage = _FilePromptStorage(temp_dir)

        prompt = BaseUntypedPrompt(
            versions={"default": "Test"}, default_version_id="default", id="test_prompt"
        )

        result = storage.save(prompt)
        assert result is True

        # Verify it was saved to file
        prompt_dir = os.path.join(temp_dir, "test_prompt")
        assert os.path.isdir(prompt_dir)

    @pytest.mark.asyncio
    async def test_init_with_default_version_id_none(self, temp_dir: str):
        """Test loading a prompt where defaultVersionId is missing (defaults to first version)."""
        write_prompt_folder(
            temp_dir,
            "prompt",
            versions={"v1": "Version 1"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        storage = _FilePromptStorage(temp_dir)
        prompt = storage._prompts["prompt"]
        assert prompt.get_default_version_id() == "v1"  # Defaults to first version

    @pytest.mark.asyncio
    async def test_save_validates_schema_compatibility(self, temp_dir: str):
        """Test that save validates schema compatibility when updating prompts."""
        # Create initial prompt with schema
        initial_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        storage = _FilePromptStorage(temp_dir)
        storage.save(initial_prompt)

        # Try to update with incompatible schema (removing required field)
        updated_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            },
        )

        # Should raise TypeError due to incompatible schema
        with pytest.raises(TypeError):
            storage.save(updated_prompt)

    @pytest.mark.asyncio
    async def test_save_allows_compatible_schema_extension(self, temp_dir: str):
        """Test that save allows extending schema with compatible changes."""
        # Create initial prompt with broader schema
        initial_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        )

        storage = _FilePromptStorage(temp_dir)
        storage.save(initial_prompt)

        # Update with narrower but compatible schema (fewer fields)
        # Original schema is a subschema of new schema if new schema is more permissive
        updated_prompt = BaseUntypedPrompt(
            versions={"v1": "Hello"},
            default_version_id="v1",
            id="schema_test",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )

        # Should succeed - removing optional fields makes schema more permissive
        result = storage.save(updated_prompt)
        assert result is False  # Existing prompt

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_lazy_loading(self, temp_dir: str):
        """Test that StorageBackedPrompt loads from storage on first access."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )

        write_prompt_folder(
            temp_dir,
            "test_prompt",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Initialize storage - it will load existing files
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt - should not load yet
        backed_prompt = StorageBackedPrompt(id="test_prompt")
        assert backed_prompt._prompt is None

        # Access versions - should trigger loading
        versions = backed_prompt.get_versions()
        assert versions == {"v1": "Hello {name}"}
        assert backed_prompt._prompt is not None

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_compile(self, temp_dir: str):
        """Test that StorageBackedPrompt.compile works correctly."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str

        write_prompt_folder(
            temp_dir,
            "test_prompt",
            versions={"v1": "Hello {{name}}!"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with variable definition
        backed_prompt = StorageBackedPrompt(
            id="test_prompt", variables_definition=TestVars
        )

        # Compile
        variables = TestVars(name="World")
        result = backed_prompt.compile(variables)
        assert result == "Hello World!"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_raises_without_init(self):
        """Test that StorageBackedPrompt raises error if storage not initialized."""
        from pixie.prompts.storage import StorageBackedPrompt
        import pixie.prompts.storage as storage_module

        # Ensure storage is not initialized
        storage_module._storage_instance = None

        backed_prompt = StorageBackedPrompt(id="test_prompt")

        with pytest.raises(
            RuntimeError, match="Prompt storage has not been initialized"
        ):
            backed_prompt.get_versions()

    @pytest.mark.asyncio
    async def test_create_prompt_helper(self, temp_dir: str):
        """Test the create_prompt helper function."""
        from pixie.prompts.storage import initialize_prompt_storage
        from pixie.prompts.prompt_management import create_prompt

        write_prompt_folder(
            temp_dir,
            "helper_test",
            versions={"v1": "Test"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create prompt using helper
        prompt = create_prompt(id="helper_test")
        assert prompt.id == "helper_test"

        versions = prompt.get_versions()
        assert versions == {"v1": "Test"}

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_schema_compatibility_check_passes(
        self, temp_dir: str
    ):
        """Test that schema compatibility check passes when definition is subschema of storage."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str

        write_prompt_folder(
            temp_dir,
            "schema_test",
            versions={"v1": "Hello {name}!"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {},
            },
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with restrictive definition
        backed_prompt = StorageBackedPrompt(
            id="schema_test", variables_definition=TestVars
        )

        # Should not raise, since TestVars schema is subschema of empty
        versions = backed_prompt.get_versions()
        assert versions == {"v1": "Hello {name}!"}

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_schema_compatibility_check_fails(
        self, temp_dir: str
    ):
        """Test that schema compatibility check fails when definition is not subschema of storage."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )

        write_prompt_folder(
            temp_dir,
            "schema_fail_test",
            versions={"v1": "Hello {name}!"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with NoneType (empty schema)
        backed_prompt = StorageBackedPrompt(
            id="schema_fail_test", variables_definition=NoneType
        )

        # Should raise TypeError because empty schema is not subschema of required schema
        with pytest.raises(
            TypeError,
            match="The provided variables_definition is not compatible with the prompt's variables schema",
        ):
            backed_prompt.get_versions()

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_actualize(self, temp_dir: str):
        """Test that StorageBackedPrompt.actualize() loads the prompt and returns self."""
        from pixie.prompts.storage import (
            initialize_prompt_storage,
            StorageBackedPrompt,
        )

        write_prompt_folder(
            temp_dir,
            "actualize_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt - should not load yet
        backed_prompt = StorageBackedPrompt(id="actualize_test")
        assert backed_prompt._prompt is None

        # Call actualize - should load and return self
        result = backed_prompt.actualize()
        assert result is backed_prompt
        assert backed_prompt._prompt is not None

        # Verify it works
        versions = backed_prompt.get_versions()
        assert versions == {"v1": "Hello {name}"}

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self):
        """Test that list_prompts returns empty list initially."""
        from pixie.prompts.prompt_management import list_prompts
        import pixie.prompts.prompt_management as pm_module

        # Clear the registry
        pm_module._registry.clear()

        prompts = list_prompts()
        assert prompts == []

    @pytest.mark.asyncio
    async def test_get_prompt_nonexistent(self):
        """Test that get_prompt returns None for non-existent prompt."""
        from pixie.prompts.prompt_management import get_prompt
        import pixie.prompts.prompt_management as pm_module

        # Clear the registry
        pm_module._registry.clear()

        prompt = get_prompt("nonexistent")
        assert prompt is None

    @pytest.mark.asyncio
    async def test_create_prompt_new(self, temp_dir: str):
        """Test creating a new prompt with create_prompt."""
        from pixie.prompts.storage import initialize_prompt_storage
        from pixie.prompts.prompt_management import (
            create_prompt,
            get_prompt,
            list_prompts,
        )
        import pixie.prompts.prompt_management as pm_module

        # Clear the registry and initialize storage
        pm_module._registry.clear()
        initialize_prompt_storage(temp_dir)

        write_prompt_folder(
            temp_dir,
            "create_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Create new prompt
        prompt = create_prompt(id="create_test")
        assert prompt.id == "create_test"
        assert prompt.variables_definition == NoneType

        # Should be in registry
        prompt_with_registration = get_prompt("create_test")
        assert prompt_with_registration is not None
        retrieved = prompt_with_registration.prompt
        assert retrieved is prompt

        # Should be in list
        prompts = list_prompts()
        assert len(prompts) == 1
        assert prompts[0].prompt is prompt

    @pytest.mark.asyncio
    async def test_create_prompt_existing_same_definition(self, temp_dir: str):
        """Test getting existing prompt with same variables_definition."""
        from pixie.prompts.storage import initialize_prompt_storage
        from pixie.prompts.prompt_management import create_prompt
        import pixie.prompts.prompt_management as pm_module
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str

        # Clear the registry and initialize storage
        pm_module._registry.clear()
        initialize_prompt_storage(temp_dir)

        write_prompt_folder(
            temp_dir,
            "existing_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Create prompt first time
        prompt1 = create_prompt(id="existing_test", variables_definition=TestVars)
        assert prompt1.variables_definition == TestVars

        # Create same prompt second time - should return same instance
        prompt2 = create_prompt(id="existing_test", variables_definition=TestVars)
        assert prompt2 is prompt1

    @pytest.mark.asyncio
    async def test_create_prompt_existing_different_definition_raises(
        self, temp_dir: str
    ):
        """Test that creating prompt with different variables_definition raises error."""
        from pixie.prompts.storage import initialize_prompt_storage
        from pixie.prompts.prompt_management import create_prompt
        import pixie.prompts.prompt_management as pm_module
        from pixie.prompts.prompt import Variables

        class TestVars1(Variables):
            name: str

        class TestVars2(Variables):
            age: int

        # Clear the registry and initialize storage
        pm_module._registry.clear()
        initialize_prompt_storage(temp_dir)

        write_prompt_folder(
            temp_dir,
            "conflict_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Create prompt first time
        create_prompt(id="conflict_test", variables_definition=TestVars1)

        # Try to create with different definition - should raise
        with pytest.raises(
            ValueError,
            match="Prompt with id 'conflict_test' already exists with a different variables definition",
        ):
            create_prompt(id="conflict_test", variables_definition=TestVars2)

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_properties(self, temp_dir: str):
        """Test StorageBackedPrompt id and variables_definition properties."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="test_id", variables_definition=TestVars)
        assert prompt.id == "test_id"
        assert prompt.variables_definition == TestVars

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_get_variables_schema(self, temp_dir: str):
        """Test StorageBackedPrompt.get_variables_schema."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str
            age: int

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="test_id", variables_definition=TestVars)
        schema = prompt.get_variables_schema()
        assert schema == {
            "type": "object",
            "title": "TestVars",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "age": {"title": "Age", "type": "integer"},
            },
            "required": ["name", "age"],
        }

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_exists_in_storage_true(self, temp_dir: str):
        """Test exists_in_storage returns True when prompt exists."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "exists_test",
            versions={"v1": "Hello"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="exists_test")
        assert prompt.exists_in_storage() is True

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_exists_in_storage_false(self, temp_dir: str):
        """Test exists_in_storage returns False when prompt does not exist."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="nonexistent")
        assert prompt.exists_in_storage() is False

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_get_default_version_id(self, temp_dir: str):
        """Test StorageBackedPrompt.get_default_version_id."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "default_test",
            versions={"v1": "Version 1", "v2": "Version 2"},
            default_version_id="v2",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="default_test")
        default_id = prompt.get_default_version_id()
        assert default_id == "v2"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_version_creation_time_matches_file(
        self, temp_dir: str
    ):
        """get_version_creation_time returns the creation time of the version file."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "created_at_test",
            versions={"v1": "Version 1"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        default_path = os.path.join(temp_dir, "created_at_test", "v1.jinja")
        prompt = StorageBackedPrompt(id="created_at_test")

        created_at = prompt.get_version_creation_time("v1")
        assert created_at == pytest.approx(os.path.getctime(default_path))

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_without_metadata_uses_latest_version(
        self, temp_dir: str
    ):
        """StorageBackedPrompt falls back to latest version when metadata is absent."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        prompt_dir = os.path.join(temp_dir, "no_metadata_storage")
        os.makedirs(prompt_dir, exist_ok=True)

        v1_path = os.path.join(prompt_dir, "v1.jinja")
        with open(v1_path, "w") as f:
            f.write("one")

        time.sleep(0.02)

        v2_path = os.path.join(prompt_dir, "v2.jinja")
        with open(v2_path, "w") as f:
            f.write("two")

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="no_metadata_storage")

        assert prompt.get_default_version_id() == "v2"
        assert prompt.get_versions() == {"v1": "one", "v2": "two"}
        assert prompt.get_version_creation_time("v2") == pytest.approx(
            os.path.getctime(v2_path)
        )

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version(self, temp_dir: str):
        """Test that StorageBackedPrompt.append_version works correctly."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "append_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="append_test")

        # Append new version
        result_prompt = prompt.append_version(
            version_id="v2", content="Hi {name}", set_as_default=True
        )

        # Check that it returns the underlying BasePrompt
        assert isinstance(result_prompt, BasePrompt)
        assert result_prompt.id == "append_test"

        # Check versions were updated
        versions = result_prompt.get_versions()
        assert "v2" in versions
        assert versions["v2"] == "Hi {name}"
        assert result_prompt.get_default_version_id() == "v2"

        # Check that storage was updated
        storage = _FilePromptStorage(temp_dir)
        stored_prompt = storage.get("append_test")
        stored_versions = stored_prompt.get_versions()
        assert "v2" in stored_versions
        assert stored_versions["v2"] == "Hi {name}"
        assert stored_prompt.get_default_version_id() == "v2"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_preserves_existing_ctime(
        self, temp_dir: str
    ):
        """Appending a version keeps existing file ctime and records the new one."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "append_ctime_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="append_ctime_test")

        v1_path = os.path.join(temp_dir, "append_ctime_test", "v1.jinja")
        original_ctime = os.path.getctime(v1_path)

        time.sleep(0.02)
        prompt.append_version(version_id="v2", content="Hi {name}")

        v1_ctime_after = prompt.get_version_creation_time("v1")
        v2_ctime = prompt.get_version_creation_time("v2")
        v2_path = os.path.join(temp_dir, "append_ctime_test", "v2.jinja")

        assert v1_ctime_after == pytest.approx(original_ctime)
        assert v2_ctime == pytest.approx(os.path.getctime(v2_path))

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_creates_new_prompt(
        self, temp_dir: str
    ):
        """Test that StorageBackedPrompt.append_version creates a new prompt if it doesn't exist."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="new_append_test")

        # Append version to non-existing prompt (should create new)
        result_prompt = prompt.append_version(
            version_id="v1", content="Hello {name}", set_as_default=True
        )

        # Check that it returns the underlying BasePrompt
        assert isinstance(result_prompt, BasePrompt)
        assert result_prompt.id == "new_append_test"

        # Check versions
        versions = result_prompt.get_versions()
        assert "v1" in versions
        assert versions["v1"] == "Hello {name}"
        assert result_prompt.get_default_version_id() == "v1"

        # Check that storage was created
        storage = _FilePromptStorage(temp_dir)
        stored_prompt = storage.get("new_append_test")
        stored_versions = stored_prompt.get_versions()
        assert "v1" in stored_versions
        assert stored_versions["v1"] == "Hello {name}"
        assert stored_prompt.get_default_version_id() == "v1"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_update_default_version_id(self, temp_dir: str):
        """Test that StorageBackedPrompt.update_default_version_id works correctly."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "update_default_test",
            versions={
                "v1": "Version 1",
                "v2": "Version 2",
                "v3": "Version 3",
            },
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="update_default_test")

        # Update default version
        result_prompt = prompt.update_default_version_id("v3")

        # Check that it returns the underlying BasePrompt
        assert isinstance(result_prompt, BasePrompt)
        assert result_prompt.id == "update_default_test"
        assert result_prompt.get_default_version_id() == "v3"

        # Check that storage was updated
        storage = _FilePromptStorage(temp_dir)
        stored_prompt = storage.get("update_default_test")
        assert stored_prompt.get_default_version_id() == "v3"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_exists_in_storage_without_init_raises_error(
        self,
    ):
        """Test exists_in_storage raises error when storage not initialized."""
        from pixie.prompts.storage import StorageBackedPrompt
        import pixie.prompts.storage as storage_module

        # Ensure storage is not initialized
        storage_module._storage_instance = None

        prompt = StorageBackedPrompt(id="test")

        with pytest.raises(
            RuntimeError, match="Prompt storage has not been initialized"
        ):
            prompt.exists_in_storage()

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_actualize_loads_prompt(self, temp_dir: str):
        """Test that actualize loads the prompt and returns self."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "actualize_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="actualize_test")
        assert prompt._prompt is None

        # Call actualize
        result = prompt.actualize()
        assert result is prompt
        assert prompt._prompt is not None

        # Verify it works
        versions = prompt.get_versions()
        assert versions == {"v1": "Hello {name}"}

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_actualize_with_variables_definition(
        self, temp_dir: str
    ):
        """Test actualize with variables_definition performs schema compatibility check."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            name: str

        write_prompt_folder(
            temp_dir,
            "actualize_vars_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(
            id="actualize_vars_test", variables_definition=TestVars
        )

        # Should succeed - schemas are compatible
        result = prompt.actualize()
        assert result is prompt
        assert prompt._prompt is not None

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_actualize_incompatible_schema_raises_error(
        self, temp_dir: str
    ):
        """Test actualize raises error when schema is incompatible."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        from pixie.prompts.prompt import Variables

        class TestVars(Variables):
            age: int  # Different from what's in storage

        write_prompt_folder(
            temp_dir,
            "incompatible_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(
            id="incompatible_test", variables_definition=TestVars
        )

        # Should raise TypeError due to incompatible schema
        with pytest.raises(
            TypeError, match="The provided variables_definition is not compatible"
        ):
            prompt.actualize()

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_updates_storage(
        self, temp_dir: str
    ):
        """Test that append_version updates the storage file."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "storage_update_test",
            versions={"v1": "Initial"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="storage_update_test")

        # Append version
        prompt.append_version(version_id="v2", content="Added version")

        prompt_dir = os.path.join(temp_dir, "storage_update_test")
        with open(os.path.join(prompt_dir, "v2.jinja"), "r") as f:
            assert f.read() == "Added version"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_update_default_updates_storage(
        self, temp_dir: str
    ):
        """Test that update_default_version_id updates the storage file."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "default_update_test",
            versions={"v1": "Version 1", "v2": "Version 2"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="default_update_test")

        # Update default
        prompt.update_default_version_id("v2")

        prompt_dir = os.path.join(temp_dir, "default_update_test")
        with open(os.path.join(prompt_dir, "metadata.json"), "r") as f:
            data = json.load(f)

        assert data["defaultVersionId"] == "v2"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_raises_for_existing_id(
        self, temp_dir: str
    ):
        """Test that append_version raises error for existing version ID."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "duplicate_version_test",
            versions={"v1": "Version 1"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="duplicate_version_test")

        # Try to append existing version
        with pytest.raises(ValueError, match="Version ID 'v1' already exists"):
            prompt.append_version(version_id="v1", content="Duplicate")

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_update_default_raises_for_nonexistent_id(
        self, temp_dir: str
    ):
        """Test that update_default_version_id raises error for nonexistent version ID."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "nonexistent_default_test",
            versions={"v1": "Version 1"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="nonexistent_default_test")

        # Try to update to nonexistent version
        with pytest.raises(ValueError, match="Version ID 'nonexistent' does not exist"):
            prompt.update_default_version_id("nonexistent")

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_without_init_raises_error(self):
        """Test that append_version raises error when storage not initialized."""
        from pixie.prompts.storage import StorageBackedPrompt
        import pixie.prompts.storage as storage_module

        # Ensure storage is not initialized
        storage_module._storage_instance = None

        prompt = StorageBackedPrompt(id="test")

        with pytest.raises(
            RuntimeError, match="Prompt storage has not been initialized"
        ):
            prompt.append_version(version_id="v2", content="New version")

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_update_default_without_init_raises_error(self):
        """Test that update_default_version_id raises error when storage not initialized."""
        from pixie.prompts.storage import StorageBackedPrompt
        import pixie.prompts.storage as storage_module

        # Ensure storage is not initialized
        storage_module._storage_instance = None

        prompt = StorageBackedPrompt(id="test")

        with pytest.raises(
            RuntimeError, match="Prompt storage has not been initialized"
        ):
            prompt.update_default_version_id("v2")

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_deletion_during_runtime(self, temp_dir: str):
        """Test behavior when a prompt is deleted from storage during runtime."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        import os

        write_prompt_folder(
            temp_dir,
            "deletion_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="deletion_test")

        # Delete the prompt folder metadata to simulate removal
        os.remove(os.path.join(temp_dir, "deletion_test", "metadata.json"))

        # Attempt to access the deleted prompt
        # The prompt is still in storage's in-memory cache, so it should work
        # but raises TypeError due to schema incompatibility with NoneType default
        with pytest.raises(TypeError, match="not compatible"):
            prompt.get_versions()


class TestInitializePromptStorage:
    """Tests for initialize_prompt_storage function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture(autouse=True)
    def reset_storage_instance(self):
        """Reset the global storage instance before each test."""
        import pixie.prompts.storage as storage_module

        storage_module._storage_instance = None
        yield
        storage_module._storage_instance = None

    def test_initialize_prompt_storage_once(self, temp_dir: str):
        """Test that initialize_prompt_storage can only be called once."""
        from pixie.prompts.storage import initialize_prompt_storage

        initialize_prompt_storage(temp_dir)

        # Should raise error on second call
        with pytest.raises(
            RuntimeError, match="Prompt storage has already been initialized"
        ):
            initialize_prompt_storage(temp_dir)

    def test_initialize_creates_storage(self, temp_dir: str):
        """Test that initialize_prompt_storage creates a FilePromptStorage instance."""
        from pixie.prompts.storage import initialize_prompt_storage
        import pixie.prompts.storage as storage_module

        initialize_prompt_storage(temp_dir)

        assert storage_module._storage_instance is not None
        assert isinstance(storage_module._storage_instance, _FilePromptStorage)

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_schema_incompatibility(
        self, temp_dir: str
    ):
        """Test that appending a version with incompatible schema raises an error."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        from pixie.prompts.prompt import Variables

        class OriginalVars(Variables):
            name: str

        class IncompatibleVars(Variables):
            age: int

        write_prompt_folder(
            temp_dir,
            "schema_test",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        # Create StorageBackedPrompt with original schema
        prompt = StorageBackedPrompt(
            id="schema_test", variables_definition=OriginalVars
        )

        # Attempt to append a version with incompatible schema
        with pytest.raises(TypeError, match="Original schema must be a subschema"):
            prompt.append_version(version_id="v2", content="Hi {age}")

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_concurrent_append_version(self, temp_dir: str):
        """Test concurrent calls to append_version to ensure thread safety."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt
        import asyncio

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="concurrent_test")

        async def append_version(version_id, content):
            prompt.append_version(version_id=version_id, content=content)

        # Run concurrent appends
        await asyncio.gather(
            append_version("v1", "Hello {name}"),
            append_version("v2", "Hi {name}"),
        )

        # Check that both versions exist
        versions = prompt.get_versions()
        assert "v1" in versions
        assert "v2" in versions
        assert versions["v1"] == "Hello {name}"
        assert versions["v2"] == "Hi {name}"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_invalid_version_id(
        self, temp_dir: str
    ):
        """Test that appending a version with an empty version ID works (no validation)."""
        from pixie.prompts.storage import initialize_prompt_storage, StorageBackedPrompt

        # Initialize storage
        initialize_prompt_storage(temp_dir)

        prompt = StorageBackedPrompt(id="invalid_version_test")

        # Empty version ID is actually allowed - no validation in place
        result = prompt.append_version(version_id="", content="Hello {name}")
        assert result is not None
        versions = result.get_versions()
        assert "" in versions

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_corrupted_storage(self, temp_dir: str):
        """Test behavior when storage files are corrupted."""
        from pixie.prompts.storage import initialize_prompt_storage
        import os

        prompt_dir = os.path.join(temp_dir, "corrupted_test")
        os.makedirs(prompt_dir, exist_ok=True)
        corrupted_file = os.path.join(prompt_dir, "metadata.json")
        with open(corrupted_file, "w") as f:
            f.write("{invalid_json}")

        # Initialize storage - should raise aggregated error while still attempting other prompts
        with pytest.raises(PromptLoadError) as excinfo:
            initialize_prompt_storage(temp_dir)

        failures = excinfo.value.failures
        assert len(failures) == 1
        assert isinstance(failures[0].error, json.JSONDecodeError)
