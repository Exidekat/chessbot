"""
VLA Model Abstractions

This module provides a unified interface for loading and using different VLA models
(PI0, SmolVLA) with a factory pattern.

Usage:
    from vla.models import load_vla_model, list_models

    # Load PI0 model (default)
    model, tokenizer = load_vla_model("pi0", device="cuda")

    # Load SmolVLA model
    model, tokenizer = load_vla_model("smolvla", device="cuda")

    # List available models
    print(list_models())  # ["pi0", "smolvla"]
"""

from .registry import MODEL_REGISTRY, register_model, get_model_class, list_models
from .factory import load_vla_model
from .base import VLAModelBase

# Import model implementations to register them
from . import pi0
from . import smolvla

__all__ = [
    "VLAModelBase",
    "load_vla_model",
    "list_models",
    "get_model_class",
    "register_model",
    "MODEL_REGISTRY",
]
