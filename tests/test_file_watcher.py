# type: ignore
"""Unit tests for pixie.prompts.file_watcher public functions."""

import asyncio
import builtins
import sys

import pytest

from pixie.prompts import file_watcher


@pytest.fixture(autouse=True)
def reset_globals():
    file_watcher._storage_observer = None
    file_watcher._storage_reload_task = None
    yield
    file_watcher._storage_observer = None
    file_watcher._storage_reload_task = None


@pytest.fixture
def fresh_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_discover_and_load_prompts_loads_non_ignored_files(
    monkeypatch, tmp_path, fresh_sys_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "_pixie_test_markers", [], raising=False)

    (tmp_path / "main.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('root')\n",
        encoding="utf-8",
    )
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "module.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('nested')\n",
        encoding="utf-8",
    )
    (tmp_path / "__init__.py").write_text("ignored = True\n", encoding="utf-8")
    (tmp_path / "_private.py").write_text(
        "import builtins\nbuiltins._pixie_test_markers.append('private')\n",
        encoding="utf-8",
    )
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "shadow.py").write_text(
        "import builtins\nbuiltins._pixie_test_markers.append('venv')\n",
        encoding="utf-8",
    )

    file_watcher.discover_and_load_modules()

    markers = builtins._pixie_test_markers  # type: ignore
    assert set(markers) == {"root", "nested"}
    assert str(tmp_path) in sys.path

    for module_name in ["main", "nested.module"]:
        sys.modules.pop(module_name, None)


def test_discover_and_load_modules_filters_by_pixie_content(
    monkeypatch, tmp_path, fresh_sys_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "_pixie_test_markers", [], raising=False)

    # File with pixie import - should be loaded
    (tmp_path / "with_pixie.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('loaded')\n",
        encoding="utf-8",
    )
    # File without pixie - should not be loaded
    (tmp_path / "without_pixie.py").write_text(
        "import builtins\nbuiltins._pixie_test_markers.append('not_loaded')\n",
        encoding="utf-8",
    )
    # File with pixie in comment - should be loaded
    (tmp_path / "pixie_comment.py").write_text(
        "# import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('comment_loaded')\n",
        encoding="utf-8",
    )

    file_watcher.discover_and_load_modules()

    markers = builtins._pixie_test_markers  # type: ignore
    assert set(markers) == {"loaded", "comment_loaded"}
    assert str(tmp_path) in sys.path

    for module_name in ["with_pixie", "pixie_comment"]:
        sys.modules.pop(module_name, None)


def test_discover_and_load_modules_ignores_pixie_files_in_ignored_dirs(
    monkeypatch, tmp_path, fresh_sys_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "_pixie_test_markers", [], raising=False)

    # File with pixie in __pycache__ - should be ignored
    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "cached.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('cached')\n",
        encoding="utf-8",
    )
    # File with pixie in .venv - should be ignored
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "venv_file.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('venv')\n",
        encoding="utf-8",
    )
    # Normal file with pixie - should be loaded
    (tmp_path / "normal.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('normal')\n",
        encoding="utf-8",
    )

    file_watcher.discover_and_load_modules()

    markers = builtins._pixie_test_markers  # type: ignore
    assert set(markers) == {"normal"}
    assert str(tmp_path) in sys.path

    sys.modules.pop("normal", None)


def test_discover_and_load_modules_handles_load_errors_gracefully(
    monkeypatch, tmp_path, fresh_sys_path, caplog
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "_pixie_test_markers", [], raising=False)

    # Valid file with pixie
    (tmp_path / "valid.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('valid')\n",
        encoding="utf-8",
    )
    # File with syntax error
    (tmp_path / "syntax_error.py").write_text(
        "import pixie\nimport builtins\nbuiltins._pixie_test_markers.append('error'\n",  # Missing closing paren
        encoding="utf-8",
    )
    # File that will fail on import (non-existent module)
    (tmp_path / "import_error.py").write_text(
        "import pixie\nimport nonexistent_module\nbuiltins._pixie_test_markers.append('import_fail')\n",
        encoding="utf-8",
    )

    with caplog.at_level("ERROR"):
        file_watcher.discover_and_load_modules()

    markers = builtins._pixie_test_markers  # type: ignore
    assert set(markers) == {"valid"}  # Only valid one loaded

    # Check that errors were logged
    error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
    assert len(error_logs) >= 2  # At least syntax and import errors

    assert str(tmp_path) in sys.path
    sys.modules.pop("valid", None)


@pytest.mark.asyncio
async def test_init_prompt_storage_lifespan_starts_and_stops_watcher(
    monkeypatch, tmp_path
):
    # Mock the storage initialization
    initialized = []
    monkeypatch.setattr(
        file_watcher, "initialize_prompt_storage", lambda path: initialized.append(path)
    )

    # Mock start and stop
    start_calls = []
    stop_calls = []

    async def mock_start(path, interval):
        start_calls.append((path, interval))

    async def mock_stop():
        stop_calls.append(True)

    monkeypatch.setattr(file_watcher, "start_storage_watcher", mock_start)
    monkeypatch.setattr(file_watcher, "stop_storage_watcher", mock_stop)

    # Mock FastAPI app
    class MockApp:
        pass

    app = MockApp()

    lifespan = file_watcher.init_prompt_storage()

    # Test that it's a callable
    assert callable(lifespan)

    # Test the context manager
    async with lifespan(app):
        pass

    # Check that storage was initialized with default path
    assert initialized == [".pixie/prompts"]

    # Check that watcher was started and stopped
    assert len(start_calls) == 1
    assert len(stop_calls) == 1

    # Check start call arguments
    path, interval = start_calls[0]
    assert str(path) == ".pixie/prompts"
    assert interval == 1.0  # Default watch interval


@pytest.mark.asyncio
async def test_init_prompt_storage_lifespan_handles_env_vars(monkeypatch, tmp_path):
    # Set environment variables
    monkeypatch.setenv("PIXIE_PROMPT_STORAGE_DIR", str(tmp_path / "custom_prompts"))
    monkeypatch.setenv("PIXIE_PROMPT_WATCH_INTERVAL", "2.5")

    initialized = []
    monkeypatch.setattr(
        file_watcher, "initialize_prompt_storage", lambda path: initialized.append(path)
    )

    start_calls = []
    stop_calls = []

    async def mock_start(path, interval):
        start_calls.append((path, interval))

    async def mock_stop():
        stop_calls.append(True)

    monkeypatch.setattr(file_watcher, "start_storage_watcher", mock_start)
    monkeypatch.setattr(file_watcher, "stop_storage_watcher", mock_stop)

    class MockApp:
        pass

    app = MockApp()
    lifespan = file_watcher.init_prompt_storage()

    async with lifespan(app):
        pass

    # Check custom path and interval
    assert initialized == [str(tmp_path / "custom_prompts")]
    path, interval = start_calls[0]
    assert str(path) == str(tmp_path / "custom_prompts")
    assert interval == 2.5


def test_storage_change_handler_watches_mustache(monkeypatch):
    calls: list[str] = []

    class FakeLoop:
        def call_soon_threadsafe(self, fn):
            calls.append("scheduled")
            fn()

    handler = file_watcher._StorageChangeHandler(FakeLoop(), 0.01)
    handler._trigger_reload = lambda: calls.append("triggered")  # type: ignore[attr-defined]

    class FakeEvent:
        src_path = "/tmp/prompt/sample.jinja"
        dest_path = None
        is_directory = False
        event_type = "created"

    handler.on_any_event(FakeEvent())

    assert ".jinja" in handler._watch_extensions
    assert calls == ["scheduled", "triggered"]

    class IgnoredEvent(FakeEvent):
        src_path = "/tmp/prompt/notes.log"

    calls.clear()
    handler.on_any_event(IgnoredEvent())
    assert calls == []


def test_storage_change_handler_dedupes_repeated_events(monkeypatch, tmp_path):
    prompt_file = tmp_path / "sample.jinja"
    prompt_file.write_text("hello", encoding="utf-8")

    calls: list[str] = []

    class FakeLoop:
        def call_soon_threadsafe(self, fn):
            calls.append("scheduled")
            fn()

    handler = file_watcher._StorageChangeHandler(FakeLoop(), 0.01)
    handler._trigger_reload = lambda: calls.append("triggered")  # type: ignore[attr-defined]

    class Event:
        src_path = str(prompt_file)
        dest_path = None
        is_directory = False
        event_type = "modified"

    handler.on_any_event(Event())
    assert calls == ["scheduled", "triggered"]

    calls.clear()
    handler.on_any_event(Event())
    assert calls == []

    prompt_file.write_text("updated", encoding="utf-8")
    handler.on_any_event(Event())
    assert calls == ["scheduled", "triggered"]


@pytest.mark.asyncio
async def test_start_storage_watcher_initializes_observer(monkeypatch, tmp_path):
    class DummyObserver:
        def __init__(self):
            self.schedule_calls: list[tuple[object, str, bool]] = []
            self.started = False

        def schedule(self, handler, path: str, recursive: bool):
            self.schedule_calls.append((handler, path, recursive))

        def start(self):
            self.started = True

    dummy_observer = DummyObserver
    monkeypatch.setattr(file_watcher, "Observer", dummy_observer)

    await file_watcher.start_storage_watcher(
        tmp_path, debounce_seconds=0.05, watch_extensions={".yaml"}
    )

    assert isinstance(file_watcher._storage_observer, DummyObserver)
    observer = file_watcher._storage_observer
    assert observer.started is True
    assert len(observer.schedule_calls) == 1

    handler, path, recursive = observer.schedule_calls[0]
    assert isinstance(handler, file_watcher._StorageChangeHandler)
    assert path == str(tmp_path)
    assert recursive is True
    assert handler._watch_extensions == {".yaml"}


@pytest.mark.asyncio
async def test_start_storage_watcher_is_noop_when_already_running(
    monkeypatch, tmp_path
):
    sentinel = object()
    file_watcher._storage_observer = sentinel

    monkeypatch.setattr(
        file_watcher,
        "Observer",
        lambda: (_ for _ in ()).throw(RuntimeError("should not instantiate")),
    )

    await file_watcher.start_storage_watcher(tmp_path, debounce_seconds=0.1)

    assert file_watcher._storage_observer is sentinel


@pytest.mark.asyncio
async def test_stop_storage_watcher_cancels_task_and_stops_observer(monkeypatch):
    class DummyObserver:
        def __init__(self):
            self.stop_called = False
            self.join_called_with: tuple | None = None

        def stop(self):
            self.stop_called = True

        def join(self, timeout):
            self.join_called_with = (timeout,)

    observer = DummyObserver()
    file_watcher._storage_observer = observer

    loop = asyncio.get_running_loop()
    reload_task = loop.create_task(asyncio.sleep(10))
    file_watcher._storage_reload_task = reload_task

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(file_watcher.asyncio, "to_thread", fake_to_thread)

    await file_watcher.stop_storage_watcher()

    assert reload_task.cancelled() is True
    assert observer.stop_called is True
    assert observer.join_called_with == (5.0,)
    assert file_watcher._storage_observer is None
    assert file_watcher._storage_reload_task is None
