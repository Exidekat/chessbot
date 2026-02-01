"""
Chess Episode Data Loader

Loads collected episodes from LeRobot v3.0 dataset format for VLA training.
Transforms data to model input format (PI0, SmolVLA).

Key features:
- LeRobot v3.0 format with video storage
- Quantile normalization of actions and states (industry practice)
- Action chunking support (50 timesteps)
- Multi-model support (PI0, SmolVLA)
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

# Configure HuggingFace for offline/local usage
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except ImportError as e:
    print(f"[X] Failed to import torch: {e}")
    sys.exit(1)

# Import normalization utilities
from vla.normalization import ActionNormalizer

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARN] LeRobot not available. Install with: pip install lerobot")


# Camera keys for PI0 (maps our naming to PI0's expected names)
PI0_CAMERA_KEYS = {
    'global': 'observation.images.base_0_rgb',
    'gripper': 'observation.images.left_wrist_0_rgb',
    'unused': 'observation.images.right_wrist_0_rgb',
}

# Camera keys for SmolVLA
SMOLVLA_CAMERA_KEYS = {
    'global': 'observation.images.camera1',
    'gripper': 'observation.images.camera2',
    'unused': 'observation.images.camera3',
}

# Our dataset's camera keys (from collect_vla_episodes.py)
DATASET_CAMERA_KEYS = {
    'global': 'observation.images.global',
    'gripper': 'observation.images.gripper',
}


class ChessEpisodeDataset(Dataset):
    """
    PyTorch Dataset wrapping LeRobot chess episodes.

    Loads observation frames (global + gripper cameras), joint positions,
    actions, and language instructions for VLA training.

    The dataset returns batches compatible with both PI0 and SmolVLA:
    - observation.images.*: Camera view tensors (keys from model_camera_keys)
    - observation.state: Joint position tensor
    - action: Target action tensor
    - language_instruction: VLM prompt string
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),  # PI0: 224x224, SmolVLA: 256x256
        model_camera_keys: Optional[Dict[str, str]] = None,  # Target model's camera keys
        state_dim: int = 32,  # PI0: 32, SmolVLA: 6
        use_global_camera: bool = True,
        use_gripper_camera: bool = True,
        normalizer: Optional[ActionNormalizer] = None,  # For quantile normalization
        normalize_actions: bool = True,  # Apply normalization to actions
        global_tile_mode: str = "multi_tile",  # "multi_tile" (default) or "letterbox"
        video_backend: str = "torchcodec",  # "torchcodec" (GPU-accelerated) or "pyav" (CPU fallback)
    ):
        """
        Initialize chess episode dataset.

        Automatically detects dataset format (video vs PNG) from meta/info.json:
        - If features have dtype="video": Uses video_backend for decoding (slower)
        - If features have dtype="image": Loads PNG frames directly (10-20x faster)

        Args:
            dataset_path: Path to LeRobot dataset directory
            split: Dataset split ("train", "val", or "all")
            transform: Optional torchvision transforms for images
            image_size: Target image size for model input (default 224x224)
            model_camera_keys: Target model's camera key mapping, e.g.:
                         PI0: PI0_CAMERA_KEYS
                         SmolVLA: SMOLVLA_CAMERA_KEYS
                         If None, uses PI0_CAMERA_KEYS as default.
            state_dim: State vector dimension (32 for PI0, 6 for SmolVLA)
            use_global_camera: Include global camera observations
            use_gripper_camera: Include gripper camera observations
            normalizer: ActionNormalizer instance for quantile normalization.
                       If None, loads from dataset_path/norm_stats.json if exists.
            normalize_actions: Whether to apply normalization to actions
            global_tile_mode: How to handle global camera:
                - "multi_tile" (default): Split into left/right tiles with 5% overlap
                - "letterbox": Single frame with black bar padding
            video_backend: Video decoding backend (only used for video-based datasets):
                - "torchcodec" (default): GPU-accelerated decoding (requires FFmpeg 5+)
                - "pyav": CPU decoding (fallback, more compatible)
                Note: Ignored for PNG image-based datasets (auto-detected from info.json)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.model_camera_keys = model_camera_keys or PI0_CAMERA_KEYS
        self.state_dim = state_dim
        self.use_global_camera = use_global_camera
        self.use_gripper_camera = use_gripper_camera
        self.normalize_actions = normalize_actions
        self.global_tile_mode = global_tile_mode
        self.video_backend = video_backend

        # Pre-compute ImageNet normalization tensors (avoid creating in __getitem__)
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Cache for per-episode home positions (avoid repeated file reads)
        self._episode_homes: Dict[int, Optional[np.ndarray]] = {}

        # Setup normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            # Try to load from dataset directory
            stats_path = self.dataset_path / "norm_stats.json"
            if stats_path.exists():
                self.normalizer = ActionNormalizer(str(stats_path))
                print(f"[OK] Loaded normalization stats from {stats_path}")
            else:
                self.normalizer = None
                if normalize_actions:
                    print(f"[WARN] No norm_stats.json found at {stats_path}")
                    print(f"       Run: python -m vla.normalization --dataset {dataset_path}")

        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is required. Install with: pip install lerobot")

        # Detect dataset format from info.json (video vs image-based)
        # This determines whether we need a video backend or can use direct image loading
        info_path = self.dataset_path / "meta" / "info.json"
        self.uses_video = True  # Default to video for backward compatibility

        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                # Check if any camera feature uses video or image dtype
                self.uses_video = any(
                    feat.get("dtype") == "video"
                    for key, feat in info.get("features", {}).items()
                    if "images" in key
                )
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARN] Could not read info.json: {e}")
                print(f"       Assuming video-based dataset")

        # Load LeRobot dataset with appropriate backend
        print(f"Loading LeRobot dataset from: {dataset_path}")

        if self.uses_video:
            # Video-based dataset: use GPU-accelerated video decoding
            # torchcodec uses NVIDIA NVDEC for faster decoding, falls back to pyav if unavailable
            try:
                self.lerobot_dataset = LeRobotDataset(
                    repo_id="local/chess_episodes",
                    root=self.dataset_path,
                    video_backend=video_backend,
                    revision="main",  # Bypass HF Hub version check for local datasets
                )
                print(f"[OK] Using {video_backend} video backend")
            except Exception as e:
                if video_backend == "torchcodec":
                    print(f"[WARN] torchcodec failed: {e}")
                    print(f"[INFO] Falling back to pyav backend")
                    self.lerobot_dataset = LeRobotDataset(
                        repo_id="local/chess_episodes",
                        root=self.dataset_path,
                        video_backend="pyav",
                        revision="main",  # Bypass HF Hub version check for local datasets
                    )
                    self.video_backend = "pyav"
                else:
                    raise
        else:
            # Image-based dataset (PNG frames): no video backend needed
            # This is ~10-20x faster for random access as it avoids video decoding
            self.lerobot_dataset = LeRobotDataset(
                repo_id="local/chess_episodes",
                root=self.dataset_path,
                revision="main",  # Bypass HF Hub version check for local datasets
            )
            self.video_backend = None
            print(f"[OK] Using PNG image backend (no video decoding)")

        # Get dataset info
        self.total_frames = len(self.lerobot_dataset)
        self.fps = self.lerobot_dataset.fps
        self.features = self.lerobot_dataset.features

        print(f"[OK] Loaded {self.total_frames} frames at {self.fps} FPS")

        # Build index mapping for train/val split
        self._build_split_indices(split)

        # Setup image transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

    def _build_split_indices(self, split: str, val_ratio: float = 0.1):
        """
        Build indices for train/val split based on episodes.

        Uses ceiling for validation count to ensure at least 1 episode is
        used for validation (Leave-One-Episode-Out minimum).
        """
        # Get episode boundaries from metadata (fast - no video decoding needed)
        # LeRobot stores dataset_from_index and dataset_to_index per episode
        episode_indices = []

        try:
            # Use episode metadata for fast boundary lookup
            n_episodes = self.lerobot_dataset.meta.total_episodes
            for ep_idx in range(n_episodes):
                ep_meta = self.lerobot_dataset.meta.episodes[ep_idx]
                from_idx = ep_meta.get("dataset_from_index", 0)
                episode_indices.append(from_idx)
            episode_indices.append(len(self.lerobot_dataset))  # End marker
        except (AttributeError, KeyError):
            # Fallback: scan parquet data (still fast, no video decoding)
            import pyarrow.parquet as pq
            data_dir = self.dataset_path / "data"
            all_indices = []
            for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
                df = pq.read_table(pq_file, columns=["episode_index", "index"]).to_pandas()
                for ep_idx in df["episode_index"].unique():
                    ep_df = df[df["episode_index"] == ep_idx]
                    all_indices.append((ep_idx, ep_df["index"].min()))
            all_indices.sort(key=lambda x: x[0])
            episode_indices = [idx for _, idx in all_indices]
            episode_indices.append(len(self.lerobot_dataset))  # End marker
            n_episodes = len(episode_indices) - 1

        # Use ceiling to ensure at least 1 episode for validation
        # This implements Leave-One-Episode-Out as minimum validation
        n_val = max(1, math.ceil(n_episodes * val_ratio))
        n_train = n_episodes - n_val

        # Ensure we have at least 1 training episode
        if n_train < 1 and n_episodes > 1:
            n_train = 1
            n_val = n_episodes - 1

        # Split by episodes (not frames) to avoid data leakage
        if split == "train":
            # First n_train episodes
            start_idx = episode_indices[0]
            end_idx = episode_indices[n_train]
            self.indices = list(range(start_idx, end_idx))
        elif split == "val":
            # Last n_val episodes
            start_idx = episode_indices[n_train]
            end_idx = episode_indices[-1]
            self.indices = list(range(start_idx, end_idx))
        else:
            # Use all data
            self.indices = list(range(len(self.lerobot_dataset)))

        n_split = n_train if split == "train" else (n_val if split == "val" else n_episodes)
        print(f"[{split}] Using {len(self.indices)} frames from {n_split} episodes")

    def _load_episode_home(self, episode_index: int) -> Optional[np.ndarray]:
        """Load recording home positions for an episode from robot_configs/."""
        config_path = self.dataset_path / "robot_configs" / f"episode_{episode_index:06d}.csv"
        if not config_path.exists():
            return None

        home_positions = []
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                home_positions.append(float(row['home_rad']))

        return np.array(home_positions, dtype=np.float32) if home_positions else None

    def _get_episode_home(self, episode_index: int) -> Optional[np.ndarray]:
        """Get cached home positions for an episode, loading if needed."""
        if episode_index not in self._episode_homes:
            self._episode_homes[episode_index] = self._load_episode_home(episode_index)
        return self._episode_homes[episode_index]

    def _default_transform(self) -> T.Compose:
        """
        Default image preprocessing (fallback for PIL images only).

        Note: LeRobot returns torch tensors, so this path is rarely used.
        The main path in _process_image() handles tensors directly for better performance.
        """
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _process_global_multi_tile(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split global frame into left and right tiles with 5% overlap.

        This provides 3.5x more effective resolution than letterboxing:
        - Letterbox: 1280x720 -> 256x144 content = 36,864 pixels
        - Multi-tile: 2 x (640x720 -> 256x256) = 131,072 pixels

        Args:
            img: Input tensor (C, H, W) normalized to [0, 1]

        Returns:
            Tuple of (left_tile, right_tile), each (C, target_h, target_w) with ImageNet normalization
        """
        h, w = img.shape[1], img.shape[2]
        mid = w // 2
        overlap = int(w * 0.05)  # 5% overlap = 64 pixels for 1280 width

        # Split with overlap (ensures board center is in both tiles)
        left_tile = img[:, :, :mid + overlap]
        right_tile = img[:, :, mid - overlap:]

        target_h, target_w = self.image_size

        # Resize each tile to target size
        left_resized = torch.nn.functional.interpolate(
            left_tile.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        right_resized = torch.nn.functional.interpolate(
            right_tile.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)

        # ImageNet normalization (use pre-computed tensors)
        left_norm = (left_resized - self._norm_mean) / self._norm_std
        right_norm = (right_resized - self._norm_mean) / self._norm_std

        return left_norm, right_norm

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample.

        Returns:
            Dictionary with model-specific keys:
                - observation.images.{model_key}: (C, H, W) tensor (cameras)
                - observation.state: (state_dim,) joint positions tensor
                - action: (50, 32) action chunk tensor
                - language_instruction: str VLM prompt
        """
        # Map to dataset index
        dataset_idx = self.indices[idx]
        sample = self.lerobot_dataset[dataset_idx]

        # Get episode index for home-relative normalization
        episode_index = sample.get("episode_index", 0)
        if isinstance(episode_index, torch.Tensor):
            episode_index = episode_index.item()
        recording_home = self._get_episode_home(episode_index)

        result = {}

        # Dataset camera key -> Model camera key
        # Dataset uses: observation.images.global, observation.images.gripper
        # Model uses: PI0_CAMERA_KEYS or SMOLVLA_CAMERA_KEYS
        dataset_global_key = DATASET_CAMERA_KEYS['global']  # observation.images.global
        dataset_gripper_key = DATASET_CAMERA_KEYS['gripper']  # observation.images.gripper
        model_global_key = self.model_camera_keys['global']
        model_gripper_key = self.model_camera_keys['gripper']

        # Process global camera -> maps to model-specific key
        if self.use_global_camera and dataset_global_key in sample:
            global_img = sample[dataset_global_key]

            # Skip if image is None (can happen with some dataset formats)
            if global_img is None:
                print(f"[WARN] Global image is None for index {idx}, skipping")
            else:
                # Convert to tensor format for processing
                if isinstance(global_img, torch.Tensor):
                    if global_img.dim() == 3 and global_img.shape[0] != 3:
                        global_img = global_img.permute(2, 0, 1)
                    global_img = global_img.float() / 255.0 if global_img.max() > 1.0 else global_img.float()
                elif isinstance(global_img, np.ndarray):
                    global_img = torch.from_numpy(global_img).permute(2, 0, 1).float() / 255.0

                if self.global_tile_mode == "multi_tile":
                    # Multi-tile: Split into left/right tiles (3.5x more resolution)
                    # Left tile -> model's 'global' slot, Right tile -> model's 'unused' slot
                    left_tile, right_tile = self._process_global_multi_tile(global_img)
                    result[model_global_key] = left_tile
                    if 'unused' in self.model_camera_keys:
                        result[self.model_camera_keys['unused']] = right_tile
                else:
                    # Letterbox: Single frame with black bar padding
                    global_processed = self._process_image(global_img, is_global=True)
                    result[model_global_key] = global_processed
                    # Add zeros for unused third camera slot
                    if 'unused' in self.model_camera_keys:
                        result[self.model_camera_keys['unused']] = torch.zeros(3, self.image_size[0], self.image_size[1])

        # Process gripper camera -> maps to model-specific key
        # Gripper: height-first resize + center crop (camera-agnostic)
        if self.use_gripper_camera and dataset_gripper_key in sample:
            gripper_img = sample[dataset_gripper_key]
            # Skip if image is None (can happen with some dataset formats)
            if gripper_img is not None:
                gripper_img = self._process_image(gripper_img, is_global=False)
                result[model_gripper_key] = gripper_img

        # Process joint positions (observation.state in our dataset)
        # PI0 expects 32-dim state (padded), SmolVLA expects 6-dim state
        if "observation.state" in sample:
            joint_pos = sample["observation.state"]
            if isinstance(joint_pos, np.ndarray):
                joint_pos = torch.from_numpy(joint_pos).float()
            elif not isinstance(joint_pos, torch.Tensor):
                joint_pos = torch.tensor(joint_pos, dtype=torch.float32)

            # Subtract recording home for home-relative normalization
            if recording_home is not None:
                home_tensor = torch.from_numpy(recording_home[:len(joint_pos)]).float()
                joint_pos = joint_pos - home_tensor

            # Pad/truncate to model's expected state dimension
            state_vector = torch.zeros(self.state_dim)
            n_joints = min(len(joint_pos), self.state_dim)
            state_vector[:n_joints] = joint_pos[:n_joints]
            result["observation.state"] = state_vector

        # Process action (target)
        # Pretrained PI05 expects action chunks: (chunk_size, action_dim) = (50, 32)
        # We have a single 6-joint action, so we:
        # 1. Subtract recording home for home-relative values
        # 2. Pad the action dimension to 32
        # 3. Apply quantile normalization to [-1, 1] range
        # 4. Repeat the action 50 times (for action chunking during training)
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)

            # Subtract recording home for home-relative normalization
            if recording_home is not None:
                home_tensor = torch.from_numpy(recording_home[:len(action)]).float()
                action = action - home_tensor

            # Pad to 32 dimensions first
            padded_action = torch.zeros(32)
            padded_action[:len(action)] = action

            # Apply quantile normalization (industry practice)
            # This maps actions to [-1, 1] range for stable training
            # Note: normalization applied AFTER padding since stats were computed on padded data
            if self.normalize_actions and self.normalizer is not None:
                padded_action = self.normalizer.normalize(padded_action, key="action")

            # Repeat for action chunking (50 timesteps)
            chunk_size = 50
            action_chunk = padded_action.unsqueeze(0).repeat(chunk_size, 1)  # (50, 32)
            result["action"] = action_chunk

        # Process language instruction (from task field in LeRobot)
        # LeRobot stores task descriptions which we use as language instructions
        if "task" in sample:
            lang = sample["task"]
            if isinstance(lang, (list, tuple)):
                lang = lang[0] if lang else ""
            result["language_instruction"] = str(lang)
        else:
            result["language_instruction"] = ""

        return result

    def _process_image(self, img, is_global: bool = False) -> torch.Tensor:
        """
        Process image with aspect-ratio-aware resizing.

        Args:
            img: Input image (tensor, numpy, or PIL)
            is_global: If True, use letterbox padding to preserve full frame
                      If False (gripper), resize height first then center crop width
        """
        # Convert to tensor if needed
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] != 3:
                # (H, W, C) -> (C, H, W)
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0 if img.max() > 1.0 else img
        elif isinstance(img, np.ndarray):
            # Numpy array (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            # PIL Image - use default transform (already handles resize)
            return self.transform(img)

        h, w = img.shape[1], img.shape[2]
        target_h, target_w = self.image_size

        if is_global:
            # GLOBAL CAMERA: Letterbox resize (preserve full frame, add padding)
            # e.g., 1280x720 -> 256x256 with black bars (top/bottom)
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize maintaining aspect ratio
            img_resized = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)

            # Create padded output (black bars)
            padded = torch.zeros(3, target_h, target_w, device=img.device)

            # Center the image
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
            padded[:, pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized

            img = padded
        else:
            # GRIPPER CAMERA: Resize to Xx256 (height=target_h), then center crop width
            # Handles any input resolution (camera-agnostic)
            # Important visuals are centered, so center crop is safe

            # Step 1: Resize so height = target_h, width scales proportionally
            scale = target_h / h
            new_w = int(w * scale)
            img_resized = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(target_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)

            # Step 2: Center crop width to target_w
            if new_w > target_w:
                left = (new_w - target_w) // 2
                img = img_resized[:, :, left:left+target_w]
            elif new_w < target_w:
                # Width is smaller than target - pad with black (rare case)
                padded = torch.zeros(3, target_h, target_w, device=img.device)
                left = (target_w - new_w) // 2
                padded[:, :, left:left+new_w] = img_resized
                img = padded
            else:
                img = img_resized

        # ImageNet normalization (use pre-computed tensors)
        return (img - self._norm_mean) / self._norm_std


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for chess episode batches.

    Handles variable-length language instructions and stacks tensors.
    Dynamically handles camera keys from both PI0 and SmolVLA.
    """
    result = {}

    # Stack all image tensors dynamically (handles both PI0 and SmolVLA keys)
    for key in batch[0].keys():
        if key.startswith("observation.images."):
            result[key] = torch.stack([b[key] for b in batch])

    # Stack state tensors
    if "observation.state" in batch[0]:
        result["observation.state"] = torch.stack(
            [b["observation.state"] for b in batch]
        )

    # Stack action tensors
    if "action" in batch[0]:
        result["action"] = torch.stack(
            [b["action"] for b in batch]
        )

    # Keep language instructions as list of strings
    if "language_instruction" in batch[0]:
        result["language_instruction"] = [
            b["language_instruction"] for b in batch
        ]

    return result


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.1,
    image_size: Tuple[int, int] = (224, 224),
    model_camera_keys: Optional[Dict[str, str]] = None,
    state_dim: int = 32,
    normalizer: Optional[ActionNormalizer] = None,
    normalize_actions: bool = True,
    global_tile_mode: str = "multi_tile",
    video_backend: str = "torchcodec",
) -> Tuple[DataLoader, DataLoader, Optional[ActionNormalizer]]:
    """
    Create train and validation dataloaders.

    Automatically detects dataset format (video vs PNG) from meta/info.json:
    - Video-based datasets: Use video_backend for decoding
    - PNG image datasets: Direct image loading (10-20x faster, no video decoding)

    Args:
        dataset_path: Path to LeRobot dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Fraction of data for validation
        image_size: Target image size
        model_camera_keys: Model's camera key mapping (PI0_CAMERA_KEYS or SMOLVLA_CAMERA_KEYS)
        state_dim: State vector dimension (32 for PI0, 6 for SmolVLA)
        normalizer: ActionNormalizer for quantile normalization (auto-loads if None)
        normalize_actions: Whether to apply normalization to actions
        global_tile_mode: How to handle global camera:
            - "multi_tile" (default): Split into left/right tiles (3.5x resolution)
            - "letterbox": Single frame with black bar padding
        video_backend: Video decoding backend (only used for video-based datasets):
            - "torchcodec" (default): GPU-accelerated decoding
            - "pyav": CPU decoding (more compatible)
            Note: Ignored for PNG image-based datasets (auto-detected)

    Returns:
        Tuple of (train_loader, val_loader, normalizer)
    """
    # Load or create normalizer (shared between train/val)
    if normalizer is None and normalize_actions:
        stats_path = Path(dataset_path) / "norm_stats.json"
        if stats_path.exists():
            normalizer = ActionNormalizer(str(stats_path))
        else:
            print(f"[WARN] norm_stats.json not found. Computing statistics...")
            # Compute stats from training data
            temp_dataset = ChessEpisodeDataset(
                dataset_path=dataset_path,
                split="all",
                image_size=image_size,
                model_camera_keys=model_camera_keys,
                state_dim=state_dim,
                normalize_actions=False,  # Don't normalize for stats computation
                global_tile_mode=global_tile_mode,
                video_backend=video_backend,
            )
            normalizer = ActionNormalizer()
            normalizer.compute_stats(temp_dataset)
            normalizer.save(str(stats_path))

    train_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="train",
        image_size=image_size,
        model_camera_keys=model_camera_keys,
        state_dim=state_dim,
        normalizer=normalizer,
        normalize_actions=normalize_actions,
        global_tile_mode=global_tile_mode,
        video_backend=video_backend,
    )

    val_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="val",
        image_size=image_size,
        model_camera_keys=model_camera_keys,
        state_dim=state_dim,
        normalizer=normalizer,
        normalize_actions=normalize_actions,
        global_tile_mode=global_tile_mode,
        video_backend=video_backend,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=8 if num_workers > 0 else None,  # Prefetch 8 batches per worker for video decode
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=8 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, normalizer


if __name__ == "__main__":
    """Test data loading."""
    import argparse

    parser = argparse.ArgumentParser(description="Test chess episode data loader")
    parser.add_argument("--dataset", type=str, default="data/episodes",
                        help="Path to LeRobot dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing ChessEpisodeDataset")
    print("=" * 60)
    print()

    try:
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            dataset_path=args.dataset,
            batch_size=2,
            num_workers=0,  # For debugging
        )

        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test one batch
        print("\nTesting train batch:")
        for batch in train_loader:
            print(f"  observation.image shape: {batch.get('observation.image', torch.zeros(1)).shape}")
            print(f"  observation.state shape: {batch.get('observation.state', torch.zeros(1)).shape}")
            print(f"  action shape: {batch.get('action', torch.zeros(1)).shape}")
            print(f"  language_instruction: {batch.get('language_instruction', [''])[0][:50]}...")
            break

        print("\n[OK] Data loading test successful!")

    except Exception as e:
        print(f"\n[X] Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
