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
    ):
        """
        Initialize chess episode dataset.

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
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.model_camera_keys = model_camera_keys or PI0_CAMERA_KEYS
        self.state_dim = state_dim
        self.use_global_camera = use_global_camera
        self.use_gripper_camera = use_gripper_camera
        self.normalize_actions = normalize_actions

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

        # Load LeRobot dataset
        # Note: repo_id is a logical identifier, root is the actual local path
        # When root is provided, LeRobot loads from local disk without HuggingFace Hub access
        # Use pyav backend for video decoding (torchcodec has FFmpeg compatibility issues)
        print(f"Loading LeRobot dataset from: {dataset_path}")
        self.lerobot_dataset = LeRobotDataset(
            repo_id="local/chess_episodes",  # Logical ID (not used when root is provided)
            root=self.dataset_path,  # Actual local path
            video_backend="pyav",  # Use pyav instead of torchcodec for better FFmpeg compatibility
        )

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

    def _default_transform(self) -> T.Compose:
        """Default image preprocessing for π₀.₅."""
        return T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

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

        result = {}

        # Dataset camera key -> Model camera key
        # Dataset uses: observation.images.global, observation.images.gripper
        # Model uses: PI0_CAMERA_KEYS or SMOLVLA_CAMERA_KEYS
        dataset_global_key = DATASET_CAMERA_KEYS['global']  # observation.images.global
        dataset_gripper_key = DATASET_CAMERA_KEYS['gripper']  # observation.images.gripper
        model_global_key = self.model_camera_keys['global']
        model_gripper_key = self.model_camera_keys['gripper']

        # Process global camera -> maps to model-specific key
        # All images resized to self.image_size (PI0: 224x224, SmolVLA: 256x256)
        if self.use_global_camera and dataset_global_key in sample:
            global_img = sample[dataset_global_key]
            global_img = self._process_image(global_img)
            result[model_global_key] = global_img

        # Process gripper camera -> maps to model-specific key
        if self.use_gripper_camera and dataset_gripper_key in sample:
            gripper_img = sample[dataset_gripper_key]
            gripper_img = self._process_image(gripper_img)
            result[model_gripper_key] = gripper_img

        # Add dummy third camera - zeros for unused slot
        # This maintains compatibility with the 3-camera pretrained architecture
        if 'unused' in self.model_camera_keys:
            unused_key = self.model_camera_keys['unused']
            result[unused_key] = torch.zeros(3, self.image_size[0], self.image_size[1])

        # Process joint positions (observation.state in our dataset)
        # PI0 expects 32-dim state (padded), SmolVLA expects 6-dim state
        if "observation.state" in sample:
            joint_pos = sample["observation.state"]
            if isinstance(joint_pos, np.ndarray):
                joint_pos = torch.from_numpy(joint_pos).float()
            elif not isinstance(joint_pos, torch.Tensor):
                joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
            # Pad/truncate to model's expected state dimension
            state_vector = torch.zeros(self.state_dim)
            n_joints = min(len(joint_pos), self.state_dim)
            state_vector[:n_joints] = joint_pos[:n_joints]
            result["observation.state"] = state_vector

        # Process action (target)
        # Pretrained PI05 expects action chunks: (chunk_size, action_dim) = (50, 32)
        # We have a single 6-joint action, so we:
        # 1. Pad the action dimension to 32
        # 2. Apply quantile normalization to [-1, 1] range
        # 3. Repeat the action 50 times (for action chunking during training)
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)

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

    def _process_image(self, img) -> torch.Tensor:
        """Process image to normalized tensor."""
        if isinstance(img, torch.Tensor):
            # Already tensor (C, H, W) or (H, W, C)
            if img.dim() == 3 and img.shape[0] != 3:
                # (H, W, C) -> (C, H, W)
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0 if img.max() > 1.0 else img
            # Resize to model input size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False
            ).squeeze(0)
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
        else:
            # Numpy array or PIL Image
            img = self.transform(img)
        return img


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
) -> Tuple[DataLoader, DataLoader, Optional[ActionNormalizer]]:
    """
    Create train and validation dataloaders.

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
    )

    val_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="val",
        image_size=image_size,
        model_camera_keys=model_camera_keys,
        state_dim=state_dim,
        normalizer=normalizer,
        normalize_actions=normalize_actions,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
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
