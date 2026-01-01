"""
VLA Training Configuration Module

Provides model-specific training configurations with a unified interface.

Usage:
    from vla.configs import get_training_config

    # Get PI0 config (default)
    config = get_training_config("pi0")

    # Get SmolVLA config
    config = get_training_config("smolvla")

    # Access model-specific values
    print(config.learning_rate)      # Model-specific default
    print(config.image_size)         # (224, 224) for PI0, (512, 512) for SmolVLA
    print(config.checkpoint_dir)     # "checkpoints/chess_pi0" or "checkpoints/chess_smolvla"
"""

from typing import Type

from .base_config import BaseTrainingConfig
from .pi0_config import PI0TrainingConfig
from .smolvla_config import SmolVLATrainingConfig


# Registry of model-specific configs
CONFIG_REGISTRY = {
    "pi0": PI0TrainingConfig,
    "smolvla": SmolVLATrainingConfig,
}


def get_training_config(model_name: str = "pi0") -> BaseTrainingConfig:
    """
    Get training configuration for specified model.

    Args:
        model_name: Model identifier ("pi0" or "smolvla")

    Returns:
        Model-specific training configuration instance

    Raises:
        ValueError: If model_name is not registered
    """
    if model_name not in CONFIG_REGISTRY:
        available = ", ".join(sorted(CONFIG_REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{model_name}'. Available: {available}")

    return CONFIG_REGISTRY[model_name]()


def get_config_class(model_name: str) -> Type[BaseTrainingConfig]:
    """
    Get configuration class for specified model.

    Args:
        model_name: Model identifier

    Returns:
        Configuration class (not instance)
    """
    if model_name not in CONFIG_REGISTRY:
        available = ", ".join(sorted(CONFIG_REGISTRY.keys()))
        raise ValueError(f"Unknown model: '{model_name}'. Available: {available}")

    return CONFIG_REGISTRY[model_name]


def list_config_models() -> list:
    """List all models with registered configs."""
    return sorted(CONFIG_REGISTRY.keys())


__all__ = [
    "BaseTrainingConfig",
    "PI0TrainingConfig",
    "SmolVLATrainingConfig",
    "get_training_config",
    "get_config_class",
    "list_config_models",
    "CONFIG_REGISTRY",
]
