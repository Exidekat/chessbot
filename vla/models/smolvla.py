"""
SmolVLA VLA Model Implementation

This module provides the SmolVLA model wrapper implementing the VLAModelBase interface.
SmolVLA is an efficient vision-language-action model trained on the LeRobot community data.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import sys

import numpy as np

try:
    import torch
    import cv2
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType
except ImportError as e:
    print(f"[X] Failed to import SmolVLA dependencies: {e}")
    print("    Run: conda activate ltx && pip install lerobot")
    sys.exit(1)

from .registry import register_model
from .base import VLAModelMixin


# Chess robot camera configuration for SmolVLA
# SmolVLA uses camera1/camera2/camera3 naming convention (256x256)
CHESS_INPUT_FEATURES_SMOLVLA = {
    'observation.images.camera1': PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 256, 256)
    ),
    'observation.images.camera2': PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 256, 256)
    ),
    'observation.images.camera3': PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 256, 256)
    ),
}

# Camera key mapping from our naming to SmolVLA's expected names
SMOLVLA_CAMERA_KEYS = {
    'global_camera': 'observation.images.camera1',
    'gripper_camera': 'observation.images.camera2',
    'unused_camera': 'observation.images.camera3',
}

CHESS_OUTPUT_FEATURES_SMOLVLA = {
    'action': PolicyFeature(
        type=FeatureType.ACTION, shape=(6,)  # SO-100 has 6 joints
    ),
}


@register_model("smolvla")
class SmolVLAModel(VLAModelMixin):
    """
    SmolVLA VLA model implementation.

    SmolVLA is an efficient VLA model using SmolVLM2 as the vision-language backbone.
    It's optimized for lower compute requirements while maintaining good performance.

    Key differences from PI0:
        - Uses 512x512 images (vs 224x224 for PI0)
        - SmolVLM2 backbone (vs PaliGemma)
        - Tokenizer comes from the model processor
        - Higher default learning rate (1e-4 vs 2.5e-5)

    Attributes:
        MODEL_NAME: "smolvla"
        DEFAULT_IMAGE_SIZE: (512, 512)
        DEFAULT_PRETRAINED_PATH: "lerobot/smolvla_base"
        TOKENIZER_PATH: None (tokenizer from model processor)
    """

    MODEL_NAME = "smolvla"
    DEFAULT_IMAGE_SIZE = (256, 256)  # SmolVLA uses 256x256
    DEFAULT_PRETRAINED_PATH = "lerobot/smolvla_base"
    TOKENIZER_PATH = None  # Tokenizer comes from model processor

    # Camera key names expected by the model
    CAMERA_KEYS = {
        'global': 'observation.images.camera1',
        'gripper': 'observation.images.camera2',
        'unused': 'observation.images.camera3',
    }

    # SmolVLA normalization (ImageNet)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        """Initialize empty model (populated by from_pretrained)."""
        self.policy: Optional[SmolVLAPolicy] = None
        self.tokenizer = None
        self.device: str = "cpu"
        self._mean_tensor: Optional[torch.Tensor] = None
        self._std_tensor: Optional[torch.Tensor] = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        for_training: bool = False,
    ) -> "SmolVLAModel":
        """
        Load SmolVLA model from pretrained weights or checkpoint.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint. If None or doesn't exist,
                           loads base weights from HuggingFace.
            device: Device to run model on ("cuda" or "cpu").
            for_training: If True, set model to training mode.

        Returns:
            SmolVLAModel instance ready for inference or training.
        """
        instance = cls()
        instance.device = device

        # Determine which weights to load
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[SmolVLA] Loading fine-tuned checkpoint: {checkpoint_path}")
            pretrained_path = checkpoint_path
        else:
            print(f"[SmolVLA] Loading base weights from HuggingFace ({cls.DEFAULT_PRETRAINED_PATH})")
            pretrained_path = cls.DEFAULT_PRETRAINED_PATH

        # Load SmolVLA policy
        print(f"[SmolVLA] Loading model from: {pretrained_path}")
        instance.policy = SmolVLAPolicy.from_pretrained(pretrained_path)
        instance.policy = instance.policy.to(device)

        # Set training/eval mode
        if for_training:
            instance.policy.train()
            print(f"[SmolVLA] Model loaded in training mode on {device}")
        else:
            instance.policy.eval()
            print(f"[SmolVLA] Model loaded in eval mode on {device}")

        # Get tokenizer from model processor
        # SmolVLA uses SmolVLM2's processor which includes the tokenizer
        print(f"[SmolVLA] Getting tokenizer from model processor...")
        instance.tokenizer = instance.policy.model.vlm_with_expert.processor.tokenizer
        print(f"[SmolVLA] Tokenizer loaded")

        # Pre-compute normalization tensors
        instance._mean_tensor = torch.tensor(cls.IMAGENET_MEAN).view(3, 1, 1).to(device)
        instance._std_tensor = torch.tensor(cls.IMAGENET_STD).view(3, 1, 1).to(device)

        return instance

    def preprocess_observation(
        self,
        global_frame: np.ndarray,
        gripper_frame: np.ndarray,
        robot_state: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw observations for SmolVLA model input.

        Args:
            global_frame: Global camera frame (BGR, any resolution)
            gripper_frame: Gripper camera frame (BGR, any resolution)
            robot_state: Robot joint positions (6D for SO-100), or None

        Returns:
            Dictionary with preprocessed tensors:
                - observation.images.camera1: Global camera (1, 3, 256, 256)
                - observation.images.camera2: Gripper camera (1, 3, 256, 256)
                - observation.images.camera3: Zeros (1, 3, 256, 256)
                - observation.state: State vector (1, 6)
        """
        img_size = self.DEFAULT_IMAGE_SIZE[0]  # 256

        # Ensure frames are in (H, W, C) format
        if len(global_frame.shape) == 3 and global_frame.shape[0] == 3:
            global_frame = np.transpose(global_frame, (1, 2, 0))
        if len(gripper_frame.shape) == 3 and gripper_frame.shape[0] == 3:
            gripper_frame = np.transpose(gripper_frame, (1, 2, 0))

        # Resize to 256x256 (SmolVLA image size)
        global_resized = cv2.resize(global_frame, (img_size, img_size))
        gripper_resized = cv2.resize(gripper_frame, (img_size, img_size))

        # Convert BGR to RGB
        global_rgb = cv2.cvtColor(global_resized, cv2.COLOR_BGR2RGB)
        gripper_rgb = cv2.cvtColor(gripper_resized, cv2.COLOR_BGR2RGB)

        # Prepare state vector (6D for SmolVLA)
        if robot_state is not None:
            state_6d = robot_state.astype(np.float32)
        else:
            state_6d = np.zeros(6, dtype=np.float32)
            state_6d[5] = 0.5  # Gripper at 50% open

        # Convert to tensors with normalization
        global_tensor = self._preprocess_image(global_rgb)
        gripper_tensor = self._preprocess_image(gripper_rgb)

        # Camera3 is unused (zeros)
        camera3_tensor = torch.zeros(1, 3, img_size, img_size, device=self.device)

        # State tensor (6D for SmolVLA, not 32D)
        state_tensor = torch.from_numpy(state_6d).float().unsqueeze(0).to(self.device)

        return {
            self.CAMERA_KEYS['global']: global_tensor,     # observation.images.camera1
            self.CAMERA_KEYS['gripper']: gripper_tensor,   # observation.images.camera2
            self.CAMERA_KEYS['unused']: camera3_tensor,    # observation.images.camera3
            "observation.state": state_tensor,
        }

    def _preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Apply SmolVLA image preprocessing."""
        tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        tensor = tensor / 255.0  # Normalize to [0, 1]
        tensor = tensor.to(self.device)
        tensor = (tensor - self._mean_tensor) / self._std_tensor  # Normalization
        tensor = tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
        return tensor

    def tokenize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a language prompt for SmolVLA.

        Args:
            prompt: Natural language instruction
            max_length: Maximum token length (uses model config if None)

        Returns:
            Dictionary with tokenized prompt tensors
        """
        if max_length is None:
            max_length = self.policy.config.tokenizer_max_length

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

        return {
            "tokens": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device).bool(),
        }

    def predict_action(
        self,
        observation: Dict[str, torch.Tensor],
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
                - joint_positions: Predicted joint positions (6D)
                - confidence: Confidence score (always 1.0)
                - raw_action: Raw model output for debugging
        """
        # Tokenize prompt and add to observation
        token_dict = self.tokenize_prompt(language_prompt)
        observation["observation.language.tokens"] = token_dict["tokens"]
        observation["observation.language.attention_mask"] = token_dict["attention_mask"]

        # Run inference
        with torch.inference_mode():
            action_tensor = self.policy.select_action(observation)

        # Extract action as numpy
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.array(action_tensor)

        # Handle batch dimension
        if len(action.shape) > 1:
            action = action[0]

        # Get current state for delta computation
        if current_state is not None:
            state_6d = current_state.astype(np.float32)
        else:
            # Extract from observation if available
            state_tensor = observation.get("observation.state")
            if state_tensor is not None:
                state_6d = state_tensor.cpu().numpy()[0, :6]
            else:
                state_6d = np.zeros(6, dtype=np.float32)

        # SmolVLA also outputs deltas (flow-matching), apply to current state
        if len(action) >= 6:
            action_deltas = action[:6].astype(np.float32)
            predicted_joints = state_6d + action_deltas
            predicted_joints = np.clip(predicted_joints, 0.0, 2 * np.pi)
        else:
            predicted_joints = state_6d

        return {
            "joint_positions": predicted_joints,
            "confidence": 1.0,
            "raw_action": action[:6] if len(action) >= 6 else action,
        }
