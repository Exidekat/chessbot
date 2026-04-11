"""
PI0 (Pi Zero Point Five) VLA Model Implementation

This module provides the PI0 model wrapper implementing the VLAModelBase interface.
PI0 is Physical Intelligence's vision-language-action model from LeRobot.

Key changes from naive implementation:
- Uses quantile denormalization for action output (industry practice)
- Treats model output as ABSOLUTE positions, not deltas
- Supports action chunking (50-step trajectories)
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import contextlib
import sys

import numpy as np

try:
    import torch
    import cv2
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.configs.types import PolicyFeature, FeatureType
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"[X] Failed to import PI0 dependencies: {e}")
    print("    Run: conda activate cb && pip install lerobot transformers")
    sys.exit(1)

from .registry import register_model
from .base import VLAModelMixin
from vla.normalization import ActionNormalizer


CHESS_OUTPUT_FEATURES = {
    'action': PolicyFeature(
        type=FeatureType.ACTION, shape=(6,)  # SO-100 has 6 joints
    ),
}


@register_model("pi0")
class PI0Model(VLAModelMixin):
    """
    PI0 (Pi Zero Point Five) VLA model implementation.

    This model uses PaliGemma as the vision-language backbone with a
    flow-matching action expert for predicting robot actions.

    Attributes:
        MODEL_NAME: "pi0"
        DEFAULT_IMAGE_SIZE: (224, 224) - Fixed resolution for PaliGemma
        DEFAULT_PRETRAINED_PATH: "lerobot/pi05_base"
        TOKENIZER_PATH: "google/paligemma-3b-pt-224"
    """

    MODEL_NAME = "pi0"
    DEFAULT_IMAGE_SIZE = (224, 224)
    DEFAULT_PRETRAINED_PATH = "lerobot/pi05_base"
    TOKENIZER_PATH = "google/paligemma-3b-pt-224"

    # Camera key names expected by PI0
    CAMERA_KEYS = {
        'global': 'observation.images.base_0_rgb',
        'gripper': 'observation.images.left_wrist_0_rgb',
        'unused': 'observation.images.right_wrist_0_rgb',
    }

    # Valid joint range for clipping (assumes SO-100 robot)
    JOINT_RANGE = (0.0, 2 * np.pi)

    # ImageNet normalization (matches training in chess_dataloader.py)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self):
        """Initialize empty model (populated by from_pretrained)."""
        self.policy: Optional[PI05Policy] = None
        self.tokenizer: Optional[AutoTokenizer] = None
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
    ) -> "PI0Model":
        """
        Load PI0 model from pretrained weights or checkpoint.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (.pt file with model_state_dict).
                           If None or doesn't exist, loads base weights from HuggingFace.
            device: Device to run model on ("cuda" or "cpu").
            for_training: If True, set model to training mode.
            normalizer: ActionNormalizer for denormalizing outputs.
            norm_stats_path: Path to norm_stats.json (used if normalizer is None).

        Returns:
            PI0Model instance ready for inference or training.
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
                    print(f"[PI0] Found fine-tuned checkpoint: {checkpoint_path}")
                del ckpt
            except Exception:
                pass

        # Always load base model architecture from HuggingFace first
        print(f"[PI0] Loading base model from: {cls.DEFAULT_PRETRAINED_PATH}")
        instance.policy = PI05Policy.from_pretrained(cls.DEFAULT_PRETRAINED_PATH)

        # Convert to bfloat16 for memory efficiency (13GB -> 7GB)
        # bfloat16 is preferred over float16 for training stability
        instance.policy = instance.policy.to(device, dtype=torch.bfloat16)
        print(f"[PI0] Model converted to bfloat16 for memory efficiency")

        # Load custom checkpoint weights on top of base model
        if has_custom_checkpoint:
            print(f"[PI0] Loading fine-tuned weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU to avoid OOM
            instance.policy.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            print(f"[PI0] Loaded checkpoint from epoch {epoch}")
            del ckpt

        # Set training/eval mode
        if for_training:
            instance.policy.train()
            print(f"[PI0] Model loaded in training mode on {device}")
        else:
            instance.policy.eval()
            print(f"[PI0] Model loaded in eval mode on {device}")

        # Load tokenizer
        print(f"[PI0] Loading PaliGemma tokenizer...")
        instance.tokenizer = AutoTokenizer.from_pretrained(cls.TOKENIZER_PATH)
        print(f"[PI0] Tokenizer loaded")

        # Pre-compute normalization tensors
        instance._mean_tensor = torch.tensor(cls.IMAGENET_MEAN).view(3, 1, 1).to(device)
        instance._std_tensor = torch.tensor(cls.IMAGENET_STD).view(3, 1, 1).to(device)

        # Setup action normalizer for denormalization during inference
        if normalizer is not None:
            instance.normalizer = normalizer
            print(f"[PI0] Using provided normalizer")
        elif norm_stats_path and Path(norm_stats_path).exists():
            instance.normalizer = ActionNormalizer(norm_stats_path)
            print(f"[PI0] Loaded normalizer from {norm_stats_path}")
        else:
            instance.normalizer = None
            print(f"[PI0] No normalizer loaded - model outputs will be raw (not denormalized)")

        return instance

    def preprocess_observation(
        self,
        global_frame: np.ndarray,
        gripper_frame: np.ndarray,
        robot_state: Optional[np.ndarray] = None,
        tile_mode: str = "multi_tile",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw observations for PI0 model input.

        Args:
            global_frame: Global camera frame (BGR, any resolution)
            gripper_frame: Gripper camera frame (BGR, any resolution)
            robot_state: Robot joint positions (6D for SO-100), or None
            tile_mode: How to handle global camera:
                - "multi_tile" (default): Split into left/right tiles (3.5x resolution)
                - "letterbox": Single frame with black bar padding

        Returns:
            Dictionary with preprocessed tensors:
                - observation.images.base_0_rgb: Global left tile or letterboxed (1, 3, 224, 224)
                - observation.images.left_wrist_0_rgb: Gripper camera (1, 3, 224, 224)
                - observation.images.right_wrist_0_rgb: Global right tile or zeros (1, 3, 224, 224)
                - observation.state: Padded state (1, 32)
        """
        img_size = 224  # PI0 uses 224x224

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

        # GRIPPER: Resize to Xx224 (height=224), then center crop width
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

        # Prepare state vector (pad to 32D for PI0)
        if robot_state is not None:
            state_6d = robot_state.astype(np.float32)
        else:
            state_6d = np.zeros(6, dtype=np.float32)
            state_6d[5] = 0.5  # Gripper at 50% open

        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:6] = state_6d

        # Convert to tensors with ImageNet normalization
        gripper_tensor = self._preprocess_image(gripper_rgb)

        # State tensor
        state_tensor = torch.from_numpy(state_32d).float().unsqueeze(0).to(self.device)

        return {
            "observation.images.base_0_rgb": left_tensor,           # Left tile or letterbox
            "observation.images.left_wrist_0_rgb": gripper_tensor,  # Gripper
            "observation.images.right_wrist_0_rgb": right_tensor,   # Right tile or zeros
            "observation.state": state_tensor,
        }

    def _preprocess_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Apply PI0 image preprocessing (ImageNet normalization)."""
        tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        tensor = tensor / 255.0  # Normalize to [0, 1]
        tensor = tensor.to(self.device)
        tensor = (tensor - self._mean_tensor) / self._std_tensor  # ImageNet normalization
        tensor = tensor.unsqueeze(0)  # Add batch dim: (1, C, H, W)
        return tensor

    def tokenize_prompt(
        self,
        prompt: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a language prompt for PI0.

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

        PI0 outputs normalized action values in [-1, 1] range.
        We denormalize to get absolute joint positions.

        NOTE: Per industry practice (OpenPI, SmolVLA), the model predicts
        ABSOLUTE joint positions, not deltas. The current_state parameter
        is kept for compatibility but is not used for delta computation.

        Args:
            observation: Preprocessed observation dict from preprocess_observation()
            language_prompt: Natural language instruction
            current_state: Current robot state (for reference, not used for deltas)

        Returns:
            Dictionary with:
                - joint_positions: Predicted absolute joint positions (6D)
                - action_chunk: Full action chunk if available (50, 6)
                - confidence: Confidence score (always 1.0 for PI0)
                - raw_action: Raw normalized model output for debugging
        """
        # Tokenize prompt and add to observation
        token_dict = self.tokenize_prompt(language_prompt)
        observation["observation.language.tokens"] = token_dict["tokens"]
        observation["observation.language.attention_mask"] = token_dict["attention_mask"]

        # Run inference (autocast needed: model is bfloat16 but lerobot's
        # sample_noise() creates float32 tensors internally)
        if 'cuda' in str(self.device):
            ctx = torch.autocast('cuda', dtype=torch.bfloat16)
        else:
            ctx = contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            action_tensor = self.policy.select_action(observation)

        # Extract action as numpy
        if isinstance(action_tensor, torch.Tensor):
            raw_action = action_tensor.cpu().numpy()
        else:
            raw_action = np.array(action_tensor)

        # Handle batch and chunk dimensions
        # PI0 can output (batch, chunk, dim) or (chunk, dim) or (dim,)
        if len(raw_action.shape) == 3:
            # (batch, chunk, dim) -> take first batch
            raw_action = raw_action[0]
        if len(raw_action.shape) == 2:
            # (chunk, dim) -> this is an action chunk (50 timesteps)
            action_chunk = raw_action[:, :6].astype(np.float32)
            # Use first timestep for immediate execution
            normalized_action = raw_action[0, :6].astype(np.float32)
        else:
            # (dim,) -> single action
            normalized_action = raw_action[:6].astype(np.float32)
            action_chunk = None

        # Denormalize action (from [-1, 1] to absolute joint positions)
        # This is the industry-standard approach used by PI0 and SmolVLA
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
