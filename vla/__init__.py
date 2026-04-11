"""
VLA (Vision-Language-Action) Module

End-to-end robot control using VLA models (PI0, SmolVLA).
Includes episode collection, finetuning, and deployment.

Components:
- models/: Model abstractions and factory pattern for PI0/SmolVLA
- configs/: Per-model training configuration classes
- chess_dataloader: Load LeRobot datasets for training
- losses: VLA loss functions
- evaluate: Model evaluation utilities

Scripts (in scripts/):
- scripts/vla_finetune.py: Fine-tune VLA on collected chess episodes
- scripts/vla_deploy.py: Deploy model for real-time inference
- scripts/collect_vla_episodes.py: Episode collection
- scripts/validate_vla_episodes.py: Episode validation

Usage:
    # New API (preferred)
    from vla.models import load_vla_model, list_models
    from vla.configs import get_training_config

    model, tokenizer = load_vla_model("pi0")  # or "smolvla"
    config = get_training_config("pi0")       # or "smolvla"
"""

# New API (preferred) - multi-model support
from .models import load_vla_model, list_models, get_model_class
from .configs import get_training_config, get_config_class, list_config_models

# Common utilities (unchanged)
from .chess_dataloader import ChessEpisodeDataset, create_dataloaders, collate_fn
from .losses import VLALoss, compute_action_accuracy, create_loss_fn

__all__ = [
    # New API (multi-model)
    "load_vla_model",
    "list_models",
    "get_model_class",
    "get_training_config",
    "get_config_class",
    "list_config_models",
    # Data loading
    "ChessEpisodeDataset",
    "create_dataloaders",
    "collate_fn",
    # Losses
    "VLALoss",
    "compute_action_accuracy",
    "create_loss_fn",
]
