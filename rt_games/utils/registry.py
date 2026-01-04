from typing import Callable, Dict


class MetricRegistry:
    """
    Simple registry to map metric names to callables.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._registry[name.lower()] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        return self._registry[name.lower()]

    def has(self, name: str) -> bool:
        return name.lower() in self._registry

    def names(self):
        return sorted(self._registry.keys())


METRICS_REGISTRY = MetricRegistry()
