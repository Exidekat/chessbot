"""
VLA (Vision-Language-Action) Module

End-to-end robot control using Physical Intelligence's pi0.5 model.
Includes episode collection, finetuning, and deployment.

Components:
- vla_load_model: Load pi0.5 base or fine-tuned checkpoints
- vla_deploy: Deploy model for real-time inference
- vla_finetune: Fine-tune on collected chess episodes
- chess_dataloader: Load LeRobot datasets for training
- training_config: Training configuration management
- losses: VLA loss functions
- evaluate: Model evaluation utilities

Episode collection is in scripts/collect_vla_episodes.py
Episode validation is in scripts/validate_vla_episodes.py
"""

from .vla_load_model import load_pi0_model, get_model_info
from .training_config import ChessTrainingConfig, load_config, get_default_config
from .chess_dataloader import ChessEpisodeDataset, create_dataloaders, collate_fn
from .losses import VLALoss, compute_action_accuracy, create_loss_fn

__all__ = [
    # Model loading
    "load_pi0_model",
    "get_model_info",
    # Training config
    "ChessTrainingConfig",
    "load_config",
    "get_default_config",
    # Data loading
    "ChessEpisodeDataset",
    "create_dataloaders",
    "collate_fn",
    # Losses
    "VLALoss",
    "compute_action_accuracy",
    "create_loss_fn",
]
