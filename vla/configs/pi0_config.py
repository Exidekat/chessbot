"""
PI0 Training Configuration

Model-specific training configuration for PI0/PI0.5 VLA model.
"""

from dataclasses import dataclass
from typing import Tuple

from .base_config import BaseTrainingConfig


@dataclass
class PI0TrainingConfig(BaseTrainingConfig):
    """
    PI0/PI0.5 specific training configuration.

    Inherits common settings from BaseTrainingConfig and provides
    PI0-specific defaults for:
        - Learning rate: 2.5e-5 (lower than SmolVLA)
        - Image size: 224x224 (PaliGemma resolution)
        - Checkpoint directory: checkpoints/chess_pi0
        - Base model: lerobot/pi05_base

    These defaults are tuned for LoRA-style finetuning on chess episodes
    with 22GB VRAM (RTX 3090 / A5000 class GPU).
    """

    # PI0-specific defaults (can be overridden)
    _learning_rate: float = 2.5e-5  # PI0 optimal LR (lower than SmolVLA)

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return "pi0"

    @property
    def learning_rate(self) -> float:
        """PI0 default learning rate (2.5e-5)."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """Allow overriding learning rate."""
        self._learning_rate = value

    @property
    def image_size(self) -> Tuple[int, int]:
        """PI0 uses 224x224 images (PaliGemma resolution)."""
        return (224, 224)

    @property
    def checkpoint_dir(self) -> str:
        """PI0 checkpoint directory."""
        return "checkpoints/chess_pi0"

    @property
    def base_model(self) -> str:
        """PI0 base model on HuggingFace."""
        return "lerobot/pi05_base"

    @property
    def camera_keys(self) -> dict:
        """PI0 camera key mapping."""
        return {
            'global': 'observation.images.base_0_rgb',
            'gripper': 'observation.images.left_wrist_0_rgb',
            'unused': 'observation.images.right_wrist_0_rgb',
        }

    @property
    def state_dim(self) -> int:
        """PI0 expects 32D state (padded)."""
        return 32
