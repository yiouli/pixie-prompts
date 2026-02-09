"""Unit tests for prompt override functionality.

Tests ensure that prompt_overrides context manager works correctly for:
1. BasePrompt - direct prompt overriding
2. StorageBackedPrompt - storage-backed prompt overriding
3. ContextVar propagation, separation, and nesting
"""

import asyncio
import os
import pytest
import tempfile
import threading

from pixie.prompts.prompt import (
    BasePrompt,
    Variables,
    prompt_overrides,
    _prompt_overrides,
)
from pixie.prompts.storage import (
    StorageBackedPrompt,
    initialize_prompt_storage,
)


class SampleVariables(Variables):
    """Sample variables for testing."""

    name: str
    message: str = "default message"


class TestPromptOverridesBasePrompt:
    """Tests for prompt_overrides with BasePrompt."""

    def test_basic_override_single_prompt(self):
        """Test basic override of a single prompt."""
        prompt = BasePrompt(
            id="test_prompt_1",
            versions="Original template",
        )

        # Without override
        result = prompt.compile()
        assert result == "Original template"

        # With override
        with prompt_overrides({"test_prompt_1": "Overridden template"}):
            result = prompt.compile()
            assert result == "Overridden template"

        # After override context exits
        result = prompt.compile()
        assert result == "Original template"

    def test_override_multiple_prompts(self):
        """Test overriding multiple prompts at once."""
        prompt1 = BasePrompt(id="prompt_1", versions="Template 1")
        prompt2 = BasePrompt(id="prompt_2", versions="Template 2")

        with prompt_overrides(
            {
                "prompt_1": "Override 1",
                "prompt_2": "Override 2",
            }
        ):
            assert prompt1.compile() == "Override 1"
            assert prompt2.compile() == "Override 2"

        assert prompt1.compile() == "Template 1"
        assert prompt2.compile() == "Template 2"

    def test_override_with_variables(self):
        """Test that overridden templates work with variables."""
        prompt = BasePrompt(
            id="var_prompt",
            versions="Hello {{ name }}",
            variables_definition=SampleVariables,
        )

        variables = SampleVariables(name="Alice")

        # Without override
        result = prompt.compile(variables)
        assert result == "Hello Alice"

        # With override including variables
        with prompt_overrides({"var_prompt": "Hi {{ name }}, {{ message }}"}):
            result = prompt.compile(variables)
            assert result == "Hi Alice, default message"

        # After context exits
        result = prompt.compile(variables)
        assert result == "Hello Alice"

    def test_override_version_id_is_special(self):
        """Test that override uses special version_id."""
        prompt = BasePrompt(
            id="version_test",
            versions={"v1": "Version 1", "v2": "Version 2"},
            default_version_id="v1",
        )

        # Normal compilation uses specified version
        assert prompt.compile(version_id="v2") == "Version 2"

        # Override should take precedence even with explicit version_id
        with prompt_overrides({"version_test": "Override"}):
            result = prompt.compile(version_id="v2")
            assert result == "Override"

    def test_non_matching_prompt_id_not_affected(self):
        """Test that prompts with non-matching IDs are not affected."""
        prompt1 = BasePrompt(id="prompt_to_override", versions="Original 1")
        prompt2 = BasePrompt(id="prompt_not_overridden", versions="Original 2")

        with prompt_overrides({"prompt_to_override": "Overridden"}):
            assert prompt1.compile() == "Overridden"
            assert prompt2.compile() == "Original 2"


class TestPromptOverridesNesting:
    """Tests for nested prompt_overrides contexts."""

    def test_nested_overrides_different_prompts(self):
        """Test nested overrides affecting different prompts."""
        prompt1 = BasePrompt(id="p1", versions="Original 1")
        prompt2 = BasePrompt(id="p2", versions="Original 2")

        with prompt_overrides({"p1": "Override 1"}):
            assert prompt1.compile() == "Override 1"
            assert prompt2.compile() == "Original 2"

            with prompt_overrides({"p2": "Override 2"}):
                assert prompt1.compile() == "Override 1"
                assert prompt2.compile() == "Override 2"

            # After inner context exits
            assert prompt1.compile() == "Override 1"
            assert prompt2.compile() == "Original 2"

        # After outer context exits
        assert prompt1.compile() == "Original 1"
        assert prompt2.compile() == "Original 2"

    def test_nested_overrides_same_prompt(self):
        """Test that inner override takes precedence for same prompt."""
        prompt = BasePrompt(id="nested_test", versions="Original")

        with prompt_overrides({"nested_test": "Outer override"}):
            assert prompt.compile() == "Outer override"

            with prompt_overrides({"nested_test": "Inner override"}):
                assert prompt.compile() == "Inner override"

            # After inner context, back to outer
            assert prompt.compile() == "Outer override"

        # After all contexts exit
        assert prompt.compile() == "Original"

    def test_triple_nesting(self):
        """Test deeply nested overrides."""
        prompt = BasePrompt(id="deep", versions="Level 0")

        with prompt_overrides({"deep": "Level 1"}):
            assert prompt.compile() == "Level 1"

            with prompt_overrides({"deep": "Level 2"}):
                assert prompt.compile() == "Level 2"

                with prompt_overrides({"deep": "Level 3"}):
                    assert prompt.compile() == "Level 3"

                assert prompt.compile() == "Level 2"

            assert prompt.compile() == "Level 1"

        assert prompt.compile() == "Level 0"

    def test_nested_with_partial_overlap(self):
        """Test nested contexts with partial overlap in prompt IDs."""
        prompt1 = BasePrompt(id="p1", versions="Original 1")
        prompt2 = BasePrompt(id="p2", versions="Original 2")
        prompt3 = BasePrompt(id="p3", versions="Original 3")

        with prompt_overrides({"p1": "Outer 1", "p2": "Outer 2"}):
            assert prompt1.compile() == "Outer 1"
            assert prompt2.compile() == "Outer 2"
            assert prompt3.compile() == "Original 3"

            with prompt_overrides({"p2": "Inner 2", "p3": "Inner 3"}):
                assert prompt1.compile() == "Outer 1"  # From outer context
                assert prompt2.compile() == "Inner 2"  # Overridden in inner
                assert prompt3.compile() == "Inner 3"  # New in inner

            # After inner context exits
            assert prompt1.compile() == "Outer 1"
            assert prompt2.compile() == "Outer 2"
            assert prompt3.compile() == "Original 3"


class TestPromptOverridesContextVarSeparation:
    """Tests for contextvar separation across threads and async contexts."""

    def test_thread_isolation(self):
        """Test that overrides in one thread don't affect another thread."""
        prompt = BasePrompt(id="thread_test", versions="Original")
        results = {}
        errors = []

        def thread1_func():
            try:
                # Thread 1 has override
                with prompt_overrides({"thread_test": "Thread 1 override"}):
                    import time

                    time.sleep(0.1)  # Give thread 2 time to run
                    results["thread1"] = prompt.compile()
            except Exception as e:
                errors.append(("thread1", e))

        def thread2_func():
            try:
                # Thread 2 has no override
                import time

                time.sleep(0.05)  # Ensure thread1 is in override context
                results["thread2"] = prompt.compile()
            except Exception as e:
                errors.append(("thread2", e))

        t1 = threading.Thread(target=thread1_func)
        t2 = threading.Thread(target=thread2_func)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Errors in threads: {errors}"
        assert results["thread1"] == "Thread 1 override"
        assert results["thread2"] == "Original"

    def test_thread_different_overrides(self):
        """Test that different threads can have different overrides."""
        prompt = BasePrompt(id="thread_diff", versions="Original")
        results = {}
        errors = []

        def thread_func(thread_id, override_value):
            try:
                with prompt_overrides({"thread_diff": override_value}):
                    import time

                    time.sleep(0.05)
                    results[thread_id] = prompt.compile()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=thread_func, args=(1, "Override 1")),
            threading.Thread(target=thread_func, args=(2, "Override 2")),
            threading.Thread(target=thread_func, args=(3, "Override 3")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert results[1] == "Override 1"
        assert results[2] == "Override 2"
        assert results[3] == "Override 3"

    @pytest.mark.asyncio
    async def test_async_context_isolation(self):
        """Test that overrides work correctly in async contexts."""
        prompt = BasePrompt(id="async_test", versions="Original")

        async def task_with_override(override_value):
            with prompt_overrides({"async_test": override_value}):
                await asyncio.sleep(0.05)
                return prompt.compile()

        async def task_without_override():
            await asyncio.sleep(0.05)
            return prompt.compile()

        # Run tasks concurrently
        results = await asyncio.gather(
            task_with_override("Async override 1"),
            task_without_override(),
            task_with_override("Async override 2"),
        )

        assert results[0] == "Async override 1"
        assert results[1] == "Original"
        assert results[2] == "Async override 2"

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test that prompt_overrides works as async context manager."""
        prompt = BasePrompt(id="async_cm", versions="Original")

        async with prompt_overrides({"async_cm": "Async override"}):
            result = prompt.compile()
            assert result == "Async override"

        result = prompt.compile()
        assert result == "Original"

    @pytest.mark.asyncio
    async def test_async_nested_overrides(self):
        """Test nested async overrides."""
        prompt = BasePrompt(id="async_nested", versions="Original")

        async with prompt_overrides({"async_nested": "Outer"}):
            assert prompt.compile() == "Outer"

            async with prompt_overrides({"async_nested": "Inner"}):
                await asyncio.sleep(0.01)
                assert prompt.compile() == "Inner"

            assert prompt.compile() == "Outer"

        assert prompt.compile() == "Original"


class TestPromptOverridesStorageBackedPrompt:
    """Tests for prompt_overrides with StorageBackedPrompt."""

    @pytest.fixture(autouse=True)
    def reset_storage_instance(self):
        """Reset the global storage instance before each test."""
        import pixie.prompts.storage as storage_module

        storage_module._storage_instance = None
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]
        yield
        storage_module._storage_instance = None
        if "PIXIE_PROMPT_STORAGE_DIR" in os.environ:
            del os.environ["PIXIE_PROMPT_STORAGE_DIR"]

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set environment variable and initialize storage
            os.environ["PIXIE_PROMPT_STORAGE_DIR"] = tmpdir
            initialize_prompt_storage()
            yield tmpdir

    def test_storage_backed_prompt_override(self, temp_storage):
        """Test basic override of StorageBackedPrompt."""
        # Create a storage-backed prompt with default
        prompt = StorageBackedPrompt(
            id="stored_prompt",
            default="Default from code",
        )

        # Without override, uses default
        result = prompt.compile()
        assert result == "Default from code"

        # With override
        with prompt_overrides({"stored_prompt": "Overridden"}):
            result = prompt.compile()
            assert result == "Overridden"

        # After context exits
        result = prompt.compile()
        assert result == "Default from code"

    def test_storage_backed_prompt_with_variables(self, temp_storage):
        """Test StorageBackedPrompt override with variables."""
        prompt = StorageBackedPrompt(
            id="stored_var_prompt",
            default="Hello {{ name }}",
            variables_definition=SampleVariables,
        )

        variables = SampleVariables(name="Bob")

        result = prompt.compile(variables)
        assert result == "Hello Bob"

        with prompt_overrides({"stored_var_prompt": "Hi {{ name }}, {{ message }}"}):
            result = prompt.compile(variables)
            assert result == "Hi Bob, default message"

    def test_storage_backed_prompt_override_nested(self, temp_storage):
        """Test nested overrides with StorageBackedPrompt."""
        prompt = StorageBackedPrompt(
            id="nested_stored",
            default="Original",
        )

        with prompt_overrides({"nested_stored": "Outer"}):
            assert prompt.compile() == "Outer"

            with prompt_overrides({"nested_stored": "Inner"}):
                assert prompt.compile() == "Inner"

            assert prompt.compile() == "Outer"

        assert prompt.compile() == "Original"

    def test_storage_backed_prompt_thread_isolation(self, temp_storage):
        """Test thread isolation with StorageBackedPrompt."""
        prompt = StorageBackedPrompt(
            id="thread_stored",
            default="Original",
        )
        results = {}
        errors = []

        def thread_func(thread_id, override_value):
            try:
                if override_value:
                    with prompt_overrides({"thread_stored": override_value}):
                        import time

                        time.sleep(0.05)
                        results[thread_id] = prompt.compile()
                else:
                    import time

                    time.sleep(0.05)
                    results[thread_id] = prompt.compile()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=thread_func, args=(1, "Thread 1")),
            threading.Thread(target=thread_func, args=(2, None)),
            threading.Thread(target=thread_func, args=(3, "Thread 3")),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert results[1] == "Thread 1"
        assert results[2] == "Original"
        assert results[3] == "Thread 3"

    @pytest.mark.asyncio
    async def test_storage_backed_prompt_async_isolation(self, temp_storage):
        """Test async isolation with StorageBackedPrompt."""
        prompt = StorageBackedPrompt(
            id="async_stored",
            default="Original",
        )

        async def task(override_value):
            if override_value:
                async with prompt_overrides({"async_stored": override_value}):
                    await asyncio.sleep(0.02)
                    return prompt.compile()
            else:
                await asyncio.sleep(0.02)
                return prompt.compile()

        results = await asyncio.gather(
            task("Async 1"),
            task(None),
            task("Async 2"),
        )

        assert results[0] == "Async 1"
        assert results[1] == "Original"
        assert results[2] == "Async 2"


class TestPromptOverridesEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_override_dict(self):
        """Test that empty override dict doesn't affect anything."""
        prompt = BasePrompt(id="empty_test", versions="Original")

        with prompt_overrides({}):
            result = prompt.compile()
            assert result == "Original"

    def test_override_with_none_value_not_applied(self):
        """Test that None values in override dict don't override."""
        prompt = BasePrompt(id="none_test", versions="Original")

        # This should not cause an override since the ID isn't in the dict
        with prompt_overrides({"other_prompt": "Override"}):
            result = prompt.compile()
            assert result == "Original"

    def test_override_survives_exception_in_context(self):
        """Test that override context is properly cleaned up on exception."""
        prompt = BasePrompt(id="exception_test", versions="Original")

        try:
            with prompt_overrides({"exception_test": "Override"}):
                assert prompt.compile() == "Override"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # After exception, override should be cleaned up
        result = prompt.compile()
        assert result == "Original"

    def test_override_with_invalid_jinja_template(self):
        """Test that invalid Jinja template in override raises appropriate error."""
        prompt = BasePrompt(
            id="invalid_jinja",
            versions="Hello {{ name }}",
            variables_definition=SampleVariables,
        )
        variables = SampleVariables(name="Test")

        with prompt_overrides({"invalid_jinja": "Hello {{ undefined_var }}"}):
            with pytest.raises(Exception):  # UndefinedError or similar
                prompt.compile(variables)

    def test_contextvar_default_is_empty_dict(self):
        """Test that the default value of contextvar is an empty dict."""
        # This is important for proper isolation
        default_value = _prompt_overrides.get()
        assert default_value == {}
        assert isinstance(default_value, dict)

    def test_multiple_sequential_overrides(self):
        """Test multiple sequential (non-nested) override contexts."""
        prompt = BasePrompt(id="sequential", versions="Original")

        with prompt_overrides({"sequential": "First"}):
            assert prompt.compile() == "First"

        with prompt_overrides({"sequential": "Second"}):
            assert prompt.compile() == "Second"

        with prompt_overrides({"sequential": "Third"}):
            assert prompt.compile() == "Third"

        assert prompt.compile() == "Original"

    def test_override_does_not_mutate_original_versions(self):
        """Test that override doesn't mutate the prompt's original versions."""
        prompt = BasePrompt(
            id="immutable_test",
            versions={"v1": "Version 1", "v2": "Version 2"},
            default_version_id="v1",
        )

        with prompt_overrides({"immutable_test": "Override"}):
            prompt.compile()

        # Original versions should be unchanged
        versions = prompt.get_versions()
        assert versions == {"v1": "Version 1", "v2": "Version 2"}
        assert "v1" in versions
        assert "v2" in versions
        assert "__override__" not in versions  # Special version ID shouldn't persist


class TestPromptOverridesPropagation:
    """Tests for contextvar propagation in various scenarios."""

    def test_propagation_to_nested_function_calls(self):
        """Test that overrides propagate to nested function calls."""
        prompt = BasePrompt(id="nested_func", versions="Original")

        def inner_function():
            return prompt.compile()

        def outer_function():
            return inner_function()

        with prompt_overrides({"nested_func": "Override"}):
            result = outer_function()
            assert result == "Override"

        result = outer_function()
        assert result == "Original"

    @pytest.mark.asyncio
    async def test_propagation_to_async_nested_calls(self):
        """Test that overrides propagate through async call chains."""
        prompt = BasePrompt(id="async_chain", versions="Original")

        async def inner():
            await asyncio.sleep(0.01)
            return prompt.compile()

        async def middle():
            return await inner()

        async def outer():
            return await middle()

        async with prompt_overrides({"async_chain": "Override"}):
            result = await outer()
            assert result == "Override"

        result = await outer()
        assert result == "Original"

    def test_override_state_independent_per_context(self):
        """Test that override state is independent for each context entry."""
        prompt = BasePrompt(id="independent", versions="Original")

        # First context
        with prompt_overrides({"independent": "First context"}):
            result1 = prompt.compile()

        # Second context (should not be affected by first)
        with prompt_overrides({"independent": "Second context"}):
            result2 = prompt.compile()

        assert result1 == "First context"
        assert result2 == "Second context"

    @pytest.mark.asyncio
    async def test_override_in_task_created_inside_context(self):
        """Test that tasks created inside override context inherit the override."""
        prompt = BasePrompt(id="task_inherit", versions="Original")

        async with prompt_overrides({"task_inherit": "Override"}):
            # Create task inside override context
            async def delayed_compile():
                await asyncio.sleep(0.01)
                return prompt.compile()

            task = asyncio.create_task(delayed_compile())
            result = await task

        assert result == "Override"

    @pytest.mark.asyncio
    async def test_override_not_inherited_by_task_created_outside(self):
        """Test that tasks created outside don't inherit overrides from caller."""
        prompt = BasePrompt(id="task_no_inherit", versions="Original")

        async def independent_task():
            await asyncio.sleep(0.02)
            return prompt.compile()

        # Create task outside override context
        task = asyncio.create_task(independent_task())

        async with prompt_overrides({"task_no_inherit": "Override"}):
            # Wait for task while in override context
            # But task was created outside, so it shouldn't see the override
            await asyncio.sleep(0.01)

        result = await task
        assert result == "Original"
