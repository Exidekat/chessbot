"""
VLA Model Base Protocol

Defines the interface that all VLA model implementations must follow.
"""

from typing import Protocol, Dict, Any, Tuple, Optional, runtime_checkable
import numpy as np

# Type alias for tokenizer (varies by model)
Tokenizer = Any


@runtime_checkable
class VLAModelBase(Protocol):
    """
    Protocol defining the VLA model interface.

    All VLA model implementations (PI0, SmolVLA, etc.) must implement this interface
    to be compatible with the unified training and deployment scripts.

    Class Attributes:
        MODEL_NAME: Short identifier for the model (e.g., "pi0", "smolvla")
        DEFAULT_IMAGE_SIZE: Default input image resolution as (height, width)
        DEFAULT_PRETRAINED_PATH: HuggingFace path to pretrained weights
        TOKENIZER_PATH: Path to tokenizer (if separate from model)
    """

    MODEL_NAME: str
    DEFAULT_IMAGE_SIZE: Tuple[int, int]
    DEFAULT_PRETRAINED_PATH: str
    TOKENIZER_PATH: Optional[str]

    # Instance attributes set after loading
    policy: Any  # The underlying LeRobot policy
    tokenizer: Tokenizer
    device: str

    def preprocess_observation(
        self,
        global_frame: np.ndarray,
        gripper_frame: np.ndarray,
        robot_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Preprocess raw observations to model input format.

        Args:
            global_frame: Global camera frame (BGR, 720x1280)
            gripper_frame: Gripper camera frame (BGR, varies by source)
            robot_state: Robot joint positions (6D for SO-100)

        Returns:
            Dictionary with preprocessed observations ready for model input.
            Keys vary by model but typically include:
                - observation.images.* (preprocessed image tensors)
                - observation.state (padded state tensor)
        """
        ...

    def tokenize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize a language prompt for the model.

        Args:
            prompt: Natural language instruction (e.g., "Pick up the pawn from e2")
            max_length: Maximum token length (uses model default if None)

        Returns:
            Dictionary with tokenized prompt:
                - tokens or input_ids: Token tensor
                - attention_mask: Attention mask tensor
        """
        ...

    def predict_action(
        self,
        observation: Dict[str, Any],
        language_prompt: str,
        current_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Predict action from observation and language prompt.

        Args:
            observation: Preprocessed observation dict from preprocess_observation()
            language_prompt: Natural language instruction
            current_state: Current robot state for delta action computation

        Returns:
            Dictionary with:
                - joint_positions: Predicted joint positions (6D for SO-100)
                - confidence: Optional confidence score
                - raw_action: Raw model output (for debugging)
        """
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.

        Returns:
            Dictionary with model info:
                - model_name: Model identifier
                - image_size: Input image resolution
                - device: Current device
                - total_params: Total parameters
                - trainable_params: Trainable parameters
                - chunk_size: Action chunk size
                - n_action_steps: Number of action steps
        """
        ...

    @property
    def image_size(self) -> Tuple[int, int]:
        """Get model's expected input image size."""
        ...


class VLAModelMixin:
    """
    Mixin class providing common functionality for VLA models.

    Subclasses should inherit from this to get common utility methods.
    """

    policy: Any
    device: str
    DEFAULT_IMAGE_SIZE: Tuple[int, int]

    @property
    def image_size(self) -> Tuple[int, int]:
        """Get model's expected input image size."""
        return self.DEFAULT_IMAGE_SIZE

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        config = self.policy.config

        return {
            "model_name": getattr(self, "MODEL_NAME", "unknown"),
            "image_size": self.image_size,
            "device": self.device,
            "n_obs_steps": getattr(config, "n_obs_steps", 1),
            "n_action_steps": getattr(config, "n_action_steps", 50),
            "chunk_size": getattr(config, "chunk_size", 50),
            "max_action_dim": getattr(config, "max_action_dim", 32),
            "max_state_dim": getattr(config, "max_state_dim", 32),
            "total_params": sum(p.numel() for p in self.policy.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.policy.parameters() if p.requires_grad
            ),
        }
