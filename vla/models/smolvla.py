"""
SmolVLA VLA Model Implementation

This module provides the SmolVLA model wrapper implementing the VLAModelBase interface.
SmolVLA is an efficient vision-language-action model trained on the LeRobot community data.

Key changes from naive implementation:
- Uses quantile denormalization for action output (industry practice)
- Treats model output as ABSOLUTE positions, not deltas
- Supports action chunking
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
    print("    Run: conda activate cb && pip install lerobot")
    sys.exit(1)

from .registry import register_model
from .base import VLAModelMixin
from vla.normalization import ActionNormalizer


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
        - Uses 256x256 images (vs 224x224 for PI0)
        - SmolVLM2 backbone (vs PaliGemma)
        - Tokenizer comes from the model processor
        - Higher default learning rate (1e-4 vs 2.5e-5)

    Attributes:
        MODEL_NAME: "smolvla"
        DEFAULT_IMAGE_SIZE: (256, 256)
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

    # Valid joint range for clipping (assumes SO-100 robot)
    JOINT_RANGE = (0.0, 2 * np.pi)

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
        self.normalizer: Optional[ActionNormalizer] = None
        self._warned_no_normalizer: bool = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        for_training: bool = False,
        normalizer: Optional[ActionNormalizer] = None,
        norm_stats_path: Optional[str] = None,
    ) -> "SmolVLAModel":
        """
        Load SmolVLA model from pretrained weights or checkpoint.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (.pt file with model_state_dict).
                           If None or doesn't exist, loads base weights from HuggingFace.
            device: Device to run model on ("cuda" or "cpu").
            for_training: If True, set model to training mode.
            normalizer: ActionNormalizer for denormalizing outputs.
            norm_stats_path: Path to norm_stats.json (used if normalizer is None).

        Returns:
            SmolVLAModel instance ready for inference or training.
        """
        instance = cls()
        instance.device = device

        # Check if we have a custom checkpoint (.pt file with model_state_dict)
        has_custom_checkpoint = False
        if checkpoint_path and Path(checkpoint_path).exists():
            # Check if it's our custom format (has model_state_dict key)
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                if "model_state_dict" in ckpt:
                    has_custom_checkpoint = True
                    print(f"[SmolVLA] Found fine-tuned checkpoint: {checkpoint_path}")
                del ckpt
            except Exception:
                pass

        # Always load base model architecture from HuggingFace first
        print(f"[SmolVLA] Loading base model from: {cls.DEFAULT_PRETRAINED_PATH}")
        instance.policy = SmolVLAPolicy.from_pretrained(cls.DEFAULT_PRETRAINED_PATH)
        instance.policy = instance.policy.to(device)

        # Load custom checkpoint weights on top of base model
        if has_custom_checkpoint:
            print(f"[SmolVLA] Loading fine-tuned weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU to avoid OOM
            instance.policy.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            print(f"[SmolVLA] Loaded checkpoint from epoch {epoch}")
            del ckpt

        # Set training/eval mode
        if for_training:
            instance.policy.train()
            print(f"[SmolVLA] Model loaded in training mode on {device}")
        else:
            instance.policy.eval()
            print(f"[SmolVLA] Model loaded in eval mode on {device}")

        # Get tokenizer from model processor
        print(f"[SmolVLA] Getting tokenizer from model processor...")
        try:
            instance.tokenizer = instance.policy.model.vlm_with_expert.processor.tokenizer
        except AttributeError as e:
            print(f"[WARN] Could not access tokenizer via policy.model.vlm_with_expert.processor.tokenizer: {e}")
            print(f"[WARN] SmolVLA tokenizer not available - tokenize_prompt() will fail")
            instance.tokenizer = None
        else:
            print(f"[SmolVLA] Tokenizer loaded")

        # Pre-compute normalization tensors
        instance._mean_tensor = torch.tensor(cls.IMAGENET_MEAN).view(3, 1, 1).to(device)
        instance._std_tensor = torch.tensor(cls.IMAGENET_STD).view(3, 1, 1).to(device)

        # Setup action normalizer for denormalization during inference
        if normalizer is not None:
            instance.normalizer = normalizer
            print(f"[SmolVLA] Using provided normalizer")
        elif norm_stats_path and Path(norm_stats_path).exists():
            instance.normalizer = ActionNormalizer(norm_stats_path)
            print(f"[SmolVLA] Loaded normalizer from {norm_stats_path}")
        else:
            instance.normalizer = None
            print(f"[SmolVLA] No normalizer loaded - model outputs will be raw (not denormalized)")

        return instance

    def preprocess_observation(
        self,
        global_frame: np.ndarray,
        gripper_frame: np.ndarray,
        robot_state: Optional[np.ndarray] = None,
        tile_mode: str = "multi_tile",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw observations for SmolVLA model input.

        Args:
            global_frame: Global camera frame (BGR, any resolution)
            gripper_frame: Gripper camera frame (BGR, any resolution)
            robot_state: Robot joint positions (6D for SO-100), or None
            tile_mode: How to handle global camera:
                - "multi_tile" (default): Split into left/right tiles (3.5x resolution)
                - "letterbox": Single frame with black bar padding

        Returns:
            Dictionary with preprocessed tensors:
                - observation.images.camera1: Global left tile or letterboxed (1, 3, 256, 256)
                - observation.images.camera2: Gripper camera (1, 3, 256, 256)
                - observation.images.camera3: Global right tile or zeros (1, 3, 256, 256)
                - observation.state: State vector (1, 6)
        """
        img_size = self.DEFAULT_IMAGE_SIZE[0]  # 256

        # Ensure frames are in (H, W, C) format
        if len(global_frame.shape) == 3 and global_frame.shape[0] == 3:
            global_frame = np.transpose(global_frame, (1, 2, 0))
        if len(gripper_frame.shape) == 3 and gripper_frame.shape[0] == 3:
            gripper_frame = np.transpose(gripper_frame, (1, 2, 0))

        # Process global camera based on tile_mode
        if tile_mode == "multi_tile":
            # MULTI-TILE: Split global into left/right with 5% overlap (3.5x resolution)
            h, w = global_frame.shape[:2]
            mid = w // 2
            overlap = int(w * 0.05)  # 5% overlap = 64 pixels for 1280 width

            left_tile = global_frame[:, :mid + overlap]
            right_tile = global_frame[:, mid - overlap:]

            # Resize each tile to img_size x img_size
            left_resized = cv2.resize(left_tile, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            right_resized = cv2.resize(right_tile, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            left_rgb = cv2.cvtColor(left_resized, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right_resized, cv2.COLOR_BGR2RGB)

            # Convert to tensors
            left_tensor = self._preprocess_image(left_rgb)
            right_tensor = self._preprocess_image(right_rgb)
        else:
            # LETTERBOX: Single frame with black bar padding
            gh, gw = global_frame.shape[:2]
            scale = min(img_size / gh, img_size / gw)
            new_h, new_w = int(gh * scale), int(gw * scale)

            global_resized = cv2.resize(global_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create padded output (black padding)
            global_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            pad_top = (img_size - new_h) // 2
            pad_left = (img_size - new_w) // 2
            global_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = global_resized

            # Convert BGR to RGB
            global_rgb = cv2.cvtColor(global_padded, cv2.COLOR_BGR2RGB)
            left_tensor = self._preprocess_image(global_rgb)
            right_tensor = torch.zeros(1, 3, img_size, img_size, device=self.device)

        # GRIPPER: Resize to Xx256 (height=256), then center crop width
        # Handles any input resolution (camera-agnostic)
        gh, gw = gripper_frame.shape[:2]
        scale = img_size / gh
        new_w = int(gw * scale)
        gripper_resized = cv2.resize(gripper_frame, (new_w, img_size), interpolation=cv2.INTER_LINEAR)

        # Center crop width to img_size
        if new_w > img_size:
            left = (new_w - img_size) // 2
            gripper_resized = gripper_resized[:, left:left+img_size]
        elif new_w < img_size:
            # Pad if width is smaller (rare)
            gripper_padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            left = (img_size - new_w) // 2
            gripper_padded[:, left:left+new_w] = gripper_resized
            gripper_resized = gripper_padded

        # Convert BGR to RGB
        gripper_rgb = cv2.cvtColor(gripper_resized, cv2.COLOR_BGR2RGB)

        # Prepare state vector (6D for SmolVLA)
        if robot_state is not None:
            state_6d = robot_state.astype(np.float32)
        else:
            state_6d = np.zeros(6, dtype=np.float32)
            state_6d[5] = 0.5  # Gripper at 50% open

        # Convert to tensors with normalization
        gripper_tensor = self._preprocess_image(gripper_rgb)

        # State tensor (6D for SmolVLA, not 32D)
        state_tensor = torch.from_numpy(state_6d).float().unsqueeze(0).to(self.device)

        return {
            self.CAMERA_KEYS['global']: left_tensor,      # observation.images.camera1 (left tile or letterbox)
            self.CAMERA_KEYS['gripper']: gripper_tensor,  # observation.images.camera2
            self.CAMERA_KEYS['unused']: right_tensor,     # observation.images.camera3 (right tile or zeros)
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

        SmolVLA outputs normalized action values in [-1, 1] range.
        We denormalize to get absolute joint positions.

        NOTE: Per industry practice (SmolVLA paper), the model predicts
        ABSOLUTE joint positions, not deltas. The current_state parameter
        is kept for compatibility but is not used for delta computation.

        Args:
            observation: Preprocessed observation dict from preprocess_observation()
            language_prompt: Natural language instruction
            current_state: Current robot state (for reference, not used for deltas)

        Returns:
            Dictionary with:
                - joint_positions: Predicted absolute joint positions (6D)
                - action_chunk: Full action chunk if available
                - confidence: Confidence score (always 1.0)
                - raw_action: Raw normalized model output for debugging
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
            raw_action = action_tensor.cpu().numpy()
        else:
            raw_action = np.array(action_tensor)

        # Handle batch and chunk dimensions
        if len(raw_action.shape) == 3:
            # (batch, chunk, dim) -> take first batch
            raw_action = raw_action[0]
        if len(raw_action.shape) == 2:
            # (chunk, dim) -> this is an action chunk
            action_chunk = raw_action[:, :6].astype(np.float32)
            # Use first timestep for immediate execution
            normalized_action = raw_action[0, :6].astype(np.float32)
        else:
            # (dim,) -> single action
            normalized_action = raw_action[:6].astype(np.float32)
            action_chunk = None

        # Denormalize action (from [-1, 1] to absolute joint positions)
        # This is the industry-standard approach used by SmolVLA
        if self.normalizer is not None and self.normalizer.has_stats("action"):
            # Denormalize to get absolute joint positions
            predicted_joints = self.normalizer.denormalize(
                normalized_action, key="action"
            )
            if action_chunk is not None:
                action_chunk = self.normalizer.denormalize(
                    action_chunk, key="action"
                )
        else:
            # No normalizer - output is raw (may not be in valid joint range)
            predicted_joints = normalized_action
            if not self._warned_no_normalizer:
                print("[WARN] No normalizer - using raw action output (warning shown once)")
                self._warned_no_normalizer = True

        # Clip to valid joint range (assumes SO-100 robot)
        predicted_joints = np.clip(predicted_joints, *self.JOINT_RANGE)
        if action_chunk is not None:
            action_chunk = np.clip(action_chunk, *self.JOINT_RANGE)

        result = {
            "joint_positions": predicted_joints,
            "confidence": 1.0,
            "raw_action": normalized_action,
        }

        # Include full action chunk if available (for chunk-based execution)
        if action_chunk is not None:
            result["action_chunk"] = action_chunk

        return result
