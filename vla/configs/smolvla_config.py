"""
SmolVLA Training Configuration

Model-specific training configuration for SmolVLA VLA model.
"""

from dataclasses import dataclass
from typing import Tuple

from .base_config import BaseTrainingConfig


@dataclass
class SmolVLATrainingConfig(BaseTrainingConfig):
    """
    SmolVLA specific training configuration.

    Inherits common settings from BaseTrainingConfig and provides
    SmolVLA-specific defaults for:
        - Learning rate: 1e-4 (higher than PI0)
        - Image size: 256x256 (SmolVLA native resolution)
        - Checkpoint directory: checkpoints/chess_smolvla
        - Base model: lerobot/smolvla_base
        - Camera keys: camera1, camera2, camera3

    SmolVLA is more efficient than PI0 and can handle higher learning rates.
    """

    # SmolVLA-specific defaults (can be overridden)
    _learning_rate: float = 1e-4  # SmolVLA optimal LR (higher than PI0)

    # SmolVLA can handle larger batches (500M params vs PI0's 3B)
    batch_size: int = 16

    # SmolVLA has higher gradient clip norm by default
    max_grad_norm: float = 10.0  # SmolVLA uses 10.0 vs PI0's 1.0

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return "smolvla"

    @property
    def learning_rate(self) -> float:
        """SmolVLA default learning rate (1e-4)."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        """Allow overriding learning rate."""
        self._learning_rate = value

    @property
    def image_size(self) -> Tuple[int, int]:
        """SmolVLA uses 256x256 images."""
        return (256, 256)

    @property
    def checkpoint_dir(self) -> str:
        """SmolVLA checkpoint directory."""
        return "checkpoints/chess_smolvla"

    @property
    def base_model(self) -> str:
        """SmolVLA base model on HuggingFace."""
        return "lerobot/smolvla_base"

    @property
    def camera_keys(self) -> dict:
        """SmolVLA camera key mapping."""
        return {
            'global': 'observation.images.camera1',
            'gripper': 'observation.images.camera2',
            'unused': 'observation.images.camera3',
        }

    @property
    def state_dim(self) -> int:
        """SmolVLA expects 6D state (raw joint positions)."""
        return 6
