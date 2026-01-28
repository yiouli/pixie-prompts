from contextlib import asynccontextmanager
import os
import re
import sys
import importlib.util
from pathlib import Path
from fastapi import FastAPI
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
import asyncio
import logging

from pixie.prompts.storage import (
    PromptLoadError,
    get_storage_root_directory,
    initialize_prompt_storage,
)

logger = logging.getLogger(__name__)

_storage_observer: Observer | None = None  # type: ignore
_storage_reload_task: asyncio.Task | None = None


def _preload_pixie_modules(cwd: Path) -> None:
    """Pre-load pixie modules from the local source directory.

    This ensures that when dynamically loaded modules import pixie.registry,
    they get the same module instance as the rest of the application.

    Without this, namespace package collisions can occur when pixie-prompts
    is installed as a package while pixie-sdk-py is run from source. The
    dynamic module loading via importlib.util.spec_from_file_location() can
    create separate module instances, leading to separate _registry dicts.

    Args:
        cwd: The current working directory to search for pixie modules.
    """
    pixie_dir = cwd / "pixie"
    if not pixie_dir.is_dir():
        return

    # Find all pixie submodules that exist locally and pre-import them
    # This ensures they're in sys.modules before any dynamic loading
    pixie_modules_to_preload = []

    for py_file in pixie_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        # Skip if in __pycache__ or similar
        if "__pycache__" in py_file.parts:
            continue

        relative_path = py_file.relative_to(cwd)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")
        pixie_modules_to_preload.append((module_name, py_file))

    # Sort to ensure parent modules are loaded before children
    pixie_modules_to_preload.sort(key=lambda x: x[0])

    for module_name, py_file in pixie_modules_to_preload:
        if module_name in sys.modules:
            continue

        # Ensure parent packages are in sys.modules
        parts = module_name.split(".")
        for i in range(1, len(parts)):
            parent_name = ".".join(parts[:i])
            if parent_name not in sys.modules:
                parent_path = cwd / Path(*parts[:i])
                init_file = parent_path / "__init__.py"
                if init_file.exists():
                    spec = importlib.util.spec_from_file_location(
                        parent_name, init_file
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[parent_name] = module
                        try:
                            spec.loader.exec_module(module)
                        except Exception:
                            pass
                else:
                    # Create a namespace package
                    import types

                    ns_module = types.ModuleType(parent_name)
                    ns_module.__path__ = [str(parent_path)]
                    sys.modules[parent_name] = ns_module

        # Now load the actual module
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                # Don't fail if a module can't be loaded - it might have
                # dependencies that aren't available yet
                pass


def discover_and_load_modules():
    """Discover and load all Python files that use pixie.prompts.create_prompt, or pixie.create_prompt.

    This function recursively searches the current working directory for Python files
    """
    cwd = Path.cwd()
    # Recursively find all Python files
    python_files = list(cwd.rglob("*.py"))

    if not python_files:
        return

    # Add current directory to Python path if not already there
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    # Pre-load pixie modules from local source to avoid namespace package collisions
    # This ensures dynamically loaded modules use the same pixie.registry instance
    _preload_pixie_modules(cwd)

    loaded_count = 0
    for py_file in python_files:
        # Skip __init__.py, private files, and anything in site-packages/venv
        if py_file.name.startswith("_") or any(
            part in py_file.parts
            for part in ["site-packages", ".venv", "venv", "__pycache__"]
        ):
            continue

        # Quick check if file imports register_application
        content = py_file.read_text()
        if not re.search(r"(?:^|\s)pixie(?=[\s.])", content):
            continue

        # Check for ignore directive
        if any(
            re.fullmatch(r"#\s*pixie\s*:\s*ignore", line.strip())
            for line in content.splitlines()
        ):
            continue

        # Load the module with a unique name based on path
        relative_path = py_file.relative_to(cwd)
        module_name = str(relative_path.with_suffix("")).replace("/", ".")

        # Skip if module was already loaded (e.g., during preload)
        if module_name in sys.modules:
            loaded_count += 1
            continue

        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                loaded_count += 1
            except Exception as e:
                logger.error("Failed to load module %s: %s", module_name, e)


def _reload_prompt_storage() -> None:
    """Reload prompts from disk and refresh any actualized prompts."""
    from pixie.prompts import storage as storage_module

    storage_instance = storage_module._storage_instance
    if storage_instance is None:
        logger.warning("Prompt storage is not initialized; skip reload.")
        return

    storage_instance.load()


async def _debounced_reload(delay_seconds: float) -> None:
    """Reload storage after a debounce window to collapse bursty events."""
    try:
        await asyncio.sleep(delay_seconds)
        logger.info("Detected prompt storage change; reloading prompts...")
        _reload_prompt_storage()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Failed to reload prompts after storage change.")


class _StorageChangeHandler(FileSystemEventHandler):
    """Watchdog handler that schedules prompt reloads on any change."""

    # Patterns to ignore
    IGNORE_PATTERNS = {
        ".vscode",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        "~",  # Vim swap files
        ".swp",
        ".swo",
        ".swx",
        ".tmp",
        ".temp",
        ".DS_Store",
        "thumbs.db",
        ".lock",
    }

    ALLOWED_EVENT_TYPES = {"created", "modified", "deleted", "moved"}

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float,
        watch_extensions: set[str] | None = None,
    ) -> None:
        self._loop = loop
        self._debounce_seconds = debounce_seconds
        # Default to common prompt file formats if not specified
        self._watch_extensions = watch_extensions or {
            ".json",
            ".jinja",
        }
        self._last_seen: dict[Path, tuple[float | None, int | None]] = {}

    def _should_ignore_event(self, event: FileSystemEvent) -> bool:
        """Check if event should be ignored."""
        src = event.src_path
        if not isinstance(src, str):
            return True

        path = Path(src)

        # Check primary path
        if self._is_ignored_path(path):
            return True

        # For move events, also check destination
        if hasattr(event, "dest_path") and event.dest_path:
            dest = event.dest_path
            if not isinstance(dest, str):
                return True
            dest_path = Path(dest)
            # Ignore if either source or dest should be ignored
            if self._is_ignored_path(dest_path):
                return True

        return False

    def _is_ignored_path(self, path: Path) -> bool:
        """Check if a path should be ignored."""
        path_str = str(path)

        # Ignore if path contains any ignore pattern
        if any(pattern in path_str for pattern in self.IGNORE_PATTERNS):
            return True

        # Ignore temporary files (various editor patterns)
        filename = path.name

        # VSCode creates files like: .file.txt.12345~
        if filename.startswith(".") and "~" in filename:
            return True

        # Hidden files starting with .
        if filename.startswith("."):
            return True

        # Backup files
        if filename.startswith("~") or filename.endswith("~"):
            return True

        # Temporary suffixes
        if any(
            filename.endswith(suffix)
            for suffix in (".tmp", ".temp", ".swp", ".swo", ".swx", ".bak", ".lock")
        ):
            return True

        # For file events (not directories), check extension
        if path.is_file() or (not path.exists() and path.suffix):
            if path.suffix not in self._watch_extensions:
                return True

        return False

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Filter out unwanted events
        if event.event_type not in self.ALLOWED_EVENT_TYPES:
            return
        if self._should_ignore_event(event):
            return

        # Skip duplicate events where the file's mtime/size did not change
        if not event.is_directory and event.event_type in {"created", "modified"}:
            src = event.src_path
            if isinstance(src, str):
                path = Path(src)
                if self._is_ignored_path(path):
                    return
                if self._is_duplicate_event(path):
                    return

        # Ignore pure directory events (unless it's a delete/move that could affect contents)
        # We still want to know if a directory containing prompt files was deleted/moved
        if event.is_directory and event.event_type not in ("deleted", "moved"):
            return

        logger.debug(
            "Storage change detected: %s (%s)%s",
            event.src_path,
            event.event_type,
            f" -> {event.dest_path}" if hasattr(event, "dest_path") else "",
        )

        # Avoid blocking the watchdog thread; schedule reload on the event loop.
        self._loop.call_soon_threadsafe(self._trigger_reload)

    def _trigger_reload(self) -> None:
        global _storage_reload_task

        # Cancel existing reload task to reset the debounce timer
        if _storage_reload_task is not None and not _storage_reload_task.done():
            _storage_reload_task.cancel()

        _storage_reload_task = self._loop.create_task(
            _debounced_reload(self._debounce_seconds)
        )

    def _is_duplicate_event(self, path: Path) -> bool:
        """Return True if the event is a repeat with unchanged mtime/size."""
        try:
            stat = path.stat()
            fingerprint = (stat.st_mtime, stat.st_size)
        except FileNotFoundError:
            self._last_seen.pop(path, None)
            return False

        last = self._last_seen.get(path)
        if last == fingerprint:
            return True

        self._last_seen[path] = fingerprint
        return False


async def start_storage_watcher(
    storage_directory: Path,
    debounce_seconds: float,
    watch_extensions: set[str] | None = None,
) -> None:
    """Start a watchdog observer on the prompt storage directory.

    Args:
        storage_directory: Directory to watch for changes
        debounce_seconds: Delay before triggering reload (collapses rapid events)
        watch_extensions: Set of file extensions to watch (e.g., {'.yaml', '.json'})
    """
    global _storage_observer

    if _storage_observer is not None:
        return

    loop = asyncio.get_running_loop()
    storage_directory.mkdir(parents=True, exist_ok=True)

    handler = _StorageChangeHandler(loop, debounce_seconds, watch_extensions)
    observer = Observer()
    observer.schedule(handler, str(storage_directory), recursive=True)
    observer.start()

    _storage_observer = observer

    extensions_str = ", ".join(sorted(handler._watch_extensions))
    logger.info(
        "Watching prompt storage at %s for %s files (debounce %.2fs)",
        storage_directory,
        extensions_str,
        debounce_seconds,
    )


async def stop_storage_watcher() -> None:
    """Stop the watchdog observer if running."""
    global _storage_observer, _storage_reload_task

    if _storage_reload_task is not None:
        _storage_reload_task.cancel()
        try:
            await _storage_reload_task
        except asyncio.CancelledError:
            pass
        _storage_reload_task = None

    if _storage_observer is not None:
        _storage_observer.stop()
        await asyncio.to_thread(_storage_observer.join, 5.0)
        _storage_observer = None


def init_prompt_storage():
    """Initialize prompt storage and return lifespan context manager."""
    storage_directory = get_storage_root_directory()
    try:
        initialize_prompt_storage()
    except PromptLoadError as e:
        for err in e.failures:
            logger.error("Prompt load error: %s", err)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            storage_path = Path(storage_directory)
            watch_interval = float(os.getenv("PIXIE_PROMPT_WATCH_INTERVAL", "1.0"))
            await start_storage_watcher(storage_path, watch_interval)
            yield
        finally:
            await stop_storage_watcher()

    return lifespan
