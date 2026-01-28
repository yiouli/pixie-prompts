"""Comprehensive unit tests for pixie.prompts.storage module."""

import json
import logging
import os
import tempfile
import time
import pytest
from typing import Dict

from pixie.prompts.prompt import BaseUntypedPrompt, _prompt_registry
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
        # Clean up any env var from previous tests
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]
        yield
        storage_module._storage_instance = None
        # Clean up env var after test
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]

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
        from pixie.prompts.storage import StorageBackedPrompt

        write_prompt_folder(
            temp_dir,
            "test_prompt",
            versions={"v1": "Hello {name}"},
            default_version_id="v1",
            variables_schema={"type": "object", "properties": {}},
        )

        # Initialize storage - it will load existing files
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir

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
        from pixie.prompts.storage import StorageBackedPrompt
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
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir

        # Create StorageBackedPrompt with variable definition
        backed_prompt = StorageBackedPrompt(
            id="test_prompt", variables_definition=TestVars
        )

        # Compile
        variables = TestVars(name="World")
        result = backed_prompt.compile(variables)
        assert result == "Hello World!"


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
        # Clean up any env var from previous tests
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]
        yield
        storage_module._storage_instance = None
        # Clean up env var after test
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]

    def test_initialize_prompt_storage_once(self, temp_dir: str):
        """Test that initialize_prompt_storage can only be called once."""
        from pixie.prompts.storage import initialize_prompt_storage

        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir
        initialize_prompt_storage()

        # Should raise error on second call
        with pytest.raises(
            RuntimeError, match="Prompt storage has already been initialized"
        ):
            initialize_prompt_storage()

    def test_initialize_creates_storage(self, temp_dir: str):
        """Test that initialize_prompt_storage creates a FilePromptStorage instance."""
        from pixie.prompts.storage import initialize_prompt_storage
        import pixie.prompts.storage as storage_module

        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir
        initialize_prompt_storage()

        assert storage_module._storage_instance is not None
        assert isinstance(
            storage_module._storage_instance, storage_module._FilePromptStorage
        )

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_append_version_schema_incompatibility(
        self, temp_dir: str
    ):
        """Test that appending a version with incompatible schema raises an error."""
        from pixie.prompts.storage import StorageBackedPrompt
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
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir

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
        from pixie.prompts.storage import StorageBackedPrompt
        import asyncio

        # Initialize storage
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir

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
        from pixie.prompts.storage import StorageBackedPrompt

        # Initialize storage
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir

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
        os.environ["PIXIE_PROMPT_STORAGE_DIR"] = temp_dir
        with pytest.raises(PromptLoadError) as excinfo:
            initialize_prompt_storage()

        failures = excinfo.value.failures
        assert len(failures) == 1
        assert isinstance(failures[0].error, json.JSONDecodeError)
