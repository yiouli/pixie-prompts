"""Pytest configuration and fixtures for pixie-prompts tests."""

import sys

import pytest


@pytest.fixture(autouse=True)
def ensure_fresh_prompt_module():
    """Ensure the prompt module has fresh registries for each test.

    This is needed because file_watcher tests may remove pixie modules from
    sys.modules, causing subsequent imports to get new instances of global
    variables like _compiled_prompt_registry and _prompt_registry.
    """
    # Store original module references if they exist
    original_modules = {}
    for name in list(sys.modules.keys()):
        if name.startswith("pixie.prompts"):
            original_modules[name] = sys.modules[name]

    yield

    # After test, check if modules were removed and re-add the originals
    # This ensures subsequent tests use the same module instances
    for name, module in original_modules.items():
        if name not in sys.modules:
            sys.modules[name] = module
