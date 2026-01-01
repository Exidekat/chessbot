"""
VLA Model Registry

Provides registration and lookup of VLA model implementations.
"""

from typing import Dict, Type, List

# Forward reference to avoid circular import
VLAModelBase = "VLAModelBase"

# Registry of available models
MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """
    Decorator to register a VLA model class.

    Usage:
        @register_model("pi0")
        class PI0Model(VLAModelBase):
            ...

    Args:
        name: Short identifier for the model (e.g., "pi0", "smolvla")

    Returns:
        Decorator function that registers the class
    """
    def decorator(cls: Type) -> Type:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model_class(name: str) -> Type:
    """
    Get model class by name.

    Args:
        name: Model identifier (e.g., "pi0", "smolvla")

    Returns:
        The model class

    Raises:
        ValueError: If model name is not registered
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: '{name}'. Available models: {available or '(none registered)'}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """
    List all registered model names.

    Returns:
        Sorted list of model identifiers
    """
    return sorted(MODEL_REGISTRY.keys())
