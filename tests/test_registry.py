"""Unit tests for the MetricRegistry."""

import pytest
from rt_games.utils.registry import MetricRegistry


class TestMetricRegistry:
    """Tests for MetricRegistry class."""

    def test_register_and_get(self):
        """Test registering and retrieving a metric."""
        registry = MetricRegistry()

        @registry.register("test_metric")
        def my_metric():
            return 42

        assert registry.has("test_metric")
        assert registry.get("test_metric")() == 42

    def test_register_case_insensitive(self):
        """Registry should be case-insensitive."""
        registry = MetricRegistry()

        @registry.register("MyMetric")
        def my_metric():
            return 1

        assert registry.has("mymetric")
        assert registry.has("MYMETRIC")
        assert registry.has("MyMetric")

    def test_get_case_insensitive(self):
        """Get should work with any case."""
        registry = MetricRegistry()

        @registry.register("TestFunc")
        def test_func():
            return "ok"

        assert registry.get("testfunc")() == "ok"
        assert registry.get("TESTFUNC")() == "ok"

    def test_has_returns_false_for_unknown(self):
        """has() should return False for unregistered metrics."""
        registry = MetricRegistry()
        assert not registry.has("nonexistent")

    def test_get_raises_for_unknown(self):
        """get() should raise KeyError for unregistered metrics."""
        registry = MetricRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_names_returns_sorted_list(self):
        """names() should return sorted list of registered metrics."""
        registry = MetricRegistry()

        @registry.register("zebra")
        def z():
            pass

        @registry.register("alpha")
        def a():
            pass

        @registry.register("beta")
        def b():
            pass

        assert registry.names() == ["alpha", "beta", "zebra"]

    def test_names_empty_registry(self):
        """names() should return empty list for empty registry."""
        registry = MetricRegistry()
        assert registry.names() == []

    def test_decorator_returns_original_function(self):
        """The register decorator should return the original function."""
        registry = MetricRegistry()

        def original():
            return "original"

        decorated = registry.register("test")(original)
        assert decorated is original
