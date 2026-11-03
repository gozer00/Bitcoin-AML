# registration and lookup for concrete ModelAdapter implementations

from __future__ import annotations

from typing import Any, Callable, Mapping

from bitcoin_aml.models.base import ModelAdapter

MODEL_REGISTRY: dict[str, type[ModelAdapter]] = {}


def register_model(
    name: str,
) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    """
    class decorator registering a ModelAdapter implementation under `name`
    """

    def decorator(adapter_cls: type[ModelAdapter]) -> type[ModelAdapter]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")

        MODEL_REGISTRY[name] = adapter_cls
        return adapter_cls

    return decorator


def create_model(name: str, parameters: Mapping[str, Any]) -> ModelAdapter:
    """
    instantiates a registered ModelAdapter by name
    """
    adapter_cls = MODEL_REGISTRY.get(name)

    if adapter_cls is None:
        raise KeyError(
            f"unknown model {name!r}; registered models: {sorted(MODEL_REGISTRY)}"
        )

    return adapter_cls(**parameters)