import json
import logging
import os
from dataclasses import dataclass
from types import NoneType
from typing import Any, Dict, NotRequired, Protocol, Self, TypedDict

from jsonsubschema import isSubschema

from .prompt import (
    BasePrompt,
    BaseUntypedPrompt,
    Prompt,
    TPromptVar,
    variables_definition_to_schema,
)


logger = logging.getLogger(__name__)


VERSION_FILE_EXTENSION = ".jinja"


@dataclass(frozen=True)
class _PromptLoadFailure:
    prompt_id: str | None
    path: str
    error: Exception


class PromptLoadError(Exception):

    def __init__(self, failures: list[_PromptLoadFailure]):
        self.failures = failures
        message_lines = [
            f"- {failure.prompt_id or '<unknown>'} ({failure.path}): {failure.error}"
            for failure in failures
        ]
        message = "Failed to load prompts:\n" + "\n".join(message_lines)
        super().__init__(message)


class BaseUntypedPromptWithCreationTime(BaseUntypedPrompt):

    def __init__(
        self,
        *,
        id: str,
        versions: dict[str, str],
        default_version_id: str,
        variables_schema: dict[str, Any] | None = None,
        version_creation_times: dict[str, float],
    ) -> None:
        super().__init__(
            id=id,
            versions=versions,
            default_version_id=default_version_id,
            variables_schema=variables_schema,
        )
        self._version_creation_times = version_creation_times

    def get_version_creation_time(self, version_id: str) -> float:
        return self._version_creation_times[version_id]


class PromptStorage(Protocol):

    def load(self, *, raise_on_error: bool = True) -> list[_PromptLoadFailure]: ...

    def exists(self, prompt_id: str) -> bool: ...

    def save(self, prompt: BaseUntypedPrompt) -> bool: ...

    def get(self, prompt_id: str) -> BaseUntypedPromptWithCreationTime: ...


class _BasePromptMetadata(TypedDict):
    defaultVersionId: str
    variablesSchema: NotRequired[Dict[str, Any]]


class _FilePromptStorage(PromptStorage):

    def __init__(self, directory: str, *, raise_on_error: bool = True) -> None:
        self._directory = directory
        self._prompts: Dict[str, BaseUntypedPromptWithCreationTime] = {}
        self._load_failures: list[_PromptLoadFailure] = []
        self.load(raise_on_error=raise_on_error)

    def load(self, *, raise_on_error: bool = True) -> list[_PromptLoadFailure]:
        """Load prompts from storage with error isolation.

        Continues loading valid prompts even if some fail and aggregates failures.
        """
        logger.info("Loading prompts from directory %s", self._directory)
        self._prompts.clear()
        self._load_failures = []
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        for entry in os.listdir(self._directory):
            prompt_path = os.path.join(self._directory, entry)
            if not os.path.isdir(prompt_path):
                logger.debug("Skipping non-directory entry at %s", prompt_path)
                continue
            try:
                metadata_path = os.path.join(prompt_path, "metadata.json")
                metadata: _BasePromptMetadata | None = None
                if os.path.isfile(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                versions: dict[str, str] = {}
                version_creation_times: dict[str, float] = {}
                for filename in os.listdir(prompt_path):
                    if not filename.endswith(VERSION_FILE_EXTENSION):
                        continue
                    version_id, _ = os.path.splitext(filename)
                    version_path = os.path.join(prompt_path, filename)
                    with open(version_path, "r") as vf:
                        versions[version_id] = vf.read()
                    version_creation_times[version_id] = os.path.getctime(version_path)

                if not versions:
                    raise KeyError("No versions provided for the prompt.")

                if metadata is not None:
                    default_version_id = metadata["defaultVersionId"]
                    variables_schema = metadata.get("variablesSchema", None)
                else:
                    default_version_id, _ = max(
                        version_creation_times.items(),
                        key=lambda item: (item[1], item[0]),
                    )
                    variables_schema = None

                if default_version_id not in versions:
                    raise KeyError(
                        f"Default version '{default_version_id}' not found for prompt '{entry}'."
                    )

                prompt = BaseUntypedPromptWithCreationTime(
                    id=entry,
                    versions=versions,
                    default_version_id=default_version_id,
                    variables_schema=variables_schema,
                    version_creation_times=version_creation_times,
                )
                self._prompts[entry] = prompt
                logger.debug(
                    "Loaded prompt '%s' with %d version(s)", entry, len(versions)
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load prompt '%s' at %s", entry, prompt_path)
                self._load_failures.append(
                    _PromptLoadFailure(prompt_id=entry, path=prompt_path, error=exc)
                )
        if self._load_failures:
            logger.warning(
                "Completed loading prompts with %d failure(s)", len(self._load_failures)
            )
            if raise_on_error:
                raise PromptLoadError(self._load_failures)
        else:
            logger.info("Loaded %d prompt(s) successfully", len(self._prompts))
        return list(self._load_failures)

    @property
    def load_failures(self) -> list[_PromptLoadFailure]:
        return list(self._load_failures)

    def exists(self, prompt_id: str) -> bool:
        return prompt_id in self._prompts

    def save(self, prompt: BaseUntypedPrompt) -> bool:
        prompt_id = prompt.id
        original = self._prompts.get(prompt_id)
        new_schema = prompt.get_variables_schema()
        if original:
            original_schema = original.get_variables_schema()
            if not isSubschema(original_schema, new_schema):
                raise TypeError(
                    "Original schema must be a subschema of the new schema."
                )
        prompt_dir = os.path.join(self._directory, prompt_id)
        os.makedirs(prompt_dir, exist_ok=True)

        versions = prompt.get_versions()
        version_ids = set(versions.keys())
        existing_versions = {
            os.path.splitext(filename)[0]
            for filename in os.listdir(prompt_dir)
            if filename.endswith(VERSION_FILE_EXTENSION)
        }

        # Validate that we are not overwriting existing content with new data
        for version_id in version_ids & existing_versions:
            version_path = os.path.join(
                prompt_dir, f"{version_id}{VERSION_FILE_EXTENSION}"
            )
            with open(version_path, "r") as vf:
                existing_content = vf.read()
            if existing_content != versions[version_id]:
                raise ValueError(
                    f"Version '{version_id}' already exists with different content."
                )

        metadata: _BasePromptMetadata = {
            "defaultVersionId": prompt.get_default_version_id(),
            "variablesSchema": prompt.get_variables_schema(),
        }
        metadata_path = os.path.join(prompt_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Write only new versions; existing identical versions are left untouched
        for version_id, content in versions.items():
            version_path = os.path.join(
                prompt_dir, f"{version_id}{VERSION_FILE_EXTENSION}"
            )
            if os.path.exists(version_path):
                continue
            with open(version_path, "w") as vf:
                vf.write(content)

        for stale_version in existing_versions - set(versions.keys()):
            stale_path = os.path.join(
                prompt_dir, f"{stale_version}{VERSION_FILE_EXTENSION}"
            )
            os.remove(stale_path)

        version_creation_times = {
            version_id: os.path.getctime(
                os.path.join(prompt_dir, f"{version_id}{VERSION_FILE_EXTENSION}")
            )
            for version_id in versions.keys()
        }

        stored_prompt = BaseUntypedPromptWithCreationTime(
            id=prompt_id,
            versions=versions,
            default_version_id=prompt.get_default_version_id(),
            variables_schema=prompt.get_variables_schema(),
            version_creation_times=version_creation_times,
        )

        try:
            BasePrompt.update_prompt_registry(stored_prompt)
        except KeyError:
            # Prompt not in type prompt registry yet, meaning there's no usage in code
            # thus this untyped prompt would just be stored but not used in code
            pass
        self._prompts[prompt_id] = stored_prompt
        return original is None

    def get(self, prompt_id: str) -> BaseUntypedPromptWithCreationTime:
        return self._prompts[prompt_id]


_storage_instance: PromptStorage | None = None


def get_storage_root_directory() -> str:
    return os.getenv("PIXIE_PROMPT_STORAGE_DIR", ".pixie/prompts")


def _ensure_storage_initialized() -> PromptStorage:
    """Ensure storage is initialized, initializing it if necessary.

    Returns:
        The initialized storage instance.
    """
    global _storage_instance
    if _storage_instance is None:
        storage_directory = get_storage_root_directory()
        _storage_instance = _FilePromptStorage(storage_directory, raise_on_error=False)
        logger.info(
            "Auto-initialized prompt storage at directory: %s", storage_directory
        )
        if _storage_instance.load_failures:
            raise PromptLoadError(_storage_instance.load_failures)
    return _storage_instance


# TODO allow other storage types later
def initialize_prompt_storage() -> None:
    """Initialize prompt storage.

    The storage directory is read from the PIXIE_PROMPT_STORAGE_DIR environment
    variable, defaulting to '.pixie/prompts'.

    Raises:
        RuntimeError: If storage has already been initialized.
        PromptLoadError: If there are failures loading prompts.
    """
    global _storage_instance
    if _storage_instance is not None:
        raise RuntimeError("Prompt storage has already been initialized.")
    _ensure_storage_initialized()


class StorageBackedPrompt(Prompt[TPromptVar]):

    def __init__(
        self,
        id: str,
        *,
        variables_definition: type[TPromptVar] = NoneType,
    ) -> None:
        self._id = id
        self._variables_definition = variables_definition
        self._prompt: BasePrompt[TPromptVar] | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def variables_definition(self) -> type[TPromptVar]:
        return self._variables_definition

    def get_variables_schema(self) -> dict[str, Any]:
        return variables_definition_to_schema(self._variables_definition)

    def _get_prompt(self) -> BasePrompt[TPromptVar]:
        storage = _ensure_storage_initialized()
        if self._prompt is None:
            untyped_prompt = storage.get(self.id)
            self._prompt = BasePrompt.from_untyped(
                untyped_prompt,
                variables_definition=self.variables_definition,
            )
            schema_from_storage = untyped_prompt.get_variables_schema()
            schema_from_definition = self.get_variables_schema()
            if not isSubschema(schema_from_definition, schema_from_storage):
                raise TypeError(
                    "Schema from definition is not a subschema of the schema from storage."
                )
        return self._prompt

    def actualize(self) -> Self:
        self._get_prompt()
        return self

    def exists_in_storage(self) -> bool:
        _ensure_storage_initialized()
        try:
            self.actualize()
            return True
        except KeyError:
            return False

    def get_versions(self) -> dict[str, str]:
        prompt = self._get_prompt()
        return prompt.get_versions()

    def get_version_creation_time(self, version_id: str) -> float:
        storage = _ensure_storage_initialized()
        prompt_with_ctime = storage.get(self.id)
        if not prompt_with_ctime:
            raise KeyError(f"Prompt with id '{self.id}' not found in storage.")
        return prompt_with_ctime.get_version_creation_time(version_id)

    def get_version_count(self) -> int:
        try:
            prompt = self._get_prompt()
            versions_dict = prompt.get_versions()
            return len(versions_dict)
        except KeyError:
            return 0

    def get_default_version_id(self) -> str:
        prompt = self._get_prompt()
        return prompt.get_default_version_id()

    def compile(
        self,
        variables: TPromptVar = None,
        *,
        version_id: str | None = None,
    ) -> str:
        prompt = self._get_prompt()
        return prompt.compile(variables=variables, version_id=version_id)

    def append_version(
        self,
        version_id: str,
        content: str,
        set_as_default: bool = False,
    ) -> BasePrompt[TPromptVar]:
        storage = _ensure_storage_initialized()
        if self.exists_in_storage():
            prompt = self._get_prompt()
            prompt.append_version(
                version_id=version_id,
                content=content,
                set_as_default=set_as_default,
            )
            storage.save(prompt)
            return prompt
        else:
            # it should be safe to assume there's no actualized prompt for this id
            # thus it should be same to create a new instance of BasePrompt
            new_prompt = BasePrompt(
                id=self.id,
                versions={version_id: content},
                variables_definition=self.variables_definition,
                default_version_id=version_id,
            )
            storage.save(new_prompt)
            return new_prompt

    def update_default_version_id(
        self,
        version_id: str,
    ) -> BasePrompt[TPromptVar]:
        storage = _ensure_storage_initialized()
        prompt = self._get_prompt()
        prompt.update_default_version_id(version_id)
        storage.save(prompt)
        return prompt
