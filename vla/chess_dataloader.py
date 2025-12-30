"""
Chess Episode Data Loader

Loads collected episodes from LeRobot dataset format for VLA training.
Transforms data to π₀.₅ model input format.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except ImportError as e:
    print(f"[X] Failed to import torch: {e}")
    sys.exit(1)

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LEROBOT_AVAILABLE = True
    except ImportError:
        LEROBOT_AVAILABLE = False
        print("[WARN] LeRobot not available. Install with: pip install lerobot")


class ChessEpisodeDataset(Dataset):
    """
    PyTorch Dataset wrapping LeRobot chess episodes.

    Loads observation frames (global + gripper cameras), joint positions,
    actions, and language instructions for VLA training.

    The dataset returns batches compatible with π₀.₅ model input format:
    - observation.image: Combined camera views tensor
    - observation.state: Joint position tensor
    - action: Target action tensor
    - language_instruction: VLM prompt string
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
        image_size: Tuple[int, int] = (224, 224),  # π₀.₅ expects 224x224
        use_global_camera: bool = True,
        use_gripper_camera: bool = True,
    ):
        """
        Initialize chess episode dataset.

        Args:
            dataset_path: Path to LeRobot dataset directory
            split: Dataset split ("train" or "val")
            transform: Optional torchvision transforms for images
            image_size: Target image size for model input (default 224x224)
            use_global_camera: Include global camera observations
            use_gripper_camera: Include gripper camera observations
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.use_global_camera = use_global_camera
        self.use_gripper_camera = use_gripper_camera

        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot is required. Install with: pip install lerobot")

        # Load LeRobot dataset
        print(f"Loading LeRobot dataset from: {dataset_path}")
        self.lerobot_dataset = LeRobotDataset(str(dataset_path))

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
        """Build indices for train/val split based on episodes."""
        # Get episode boundaries
        episode_indices = []
        current_episode = -1

        for idx in range(len(self.lerobot_dataset)):
            sample = self.lerobot_dataset[idx]
            ep_idx = sample.get("episode_index", 0)
            if ep_idx != current_episode:
                episode_indices.append(idx)
                current_episode = ep_idx

        episode_indices.append(len(self.lerobot_dataset))  # End marker

        n_episodes = len(episode_indices) - 1
        n_val = max(1, int(n_episodes * val_ratio))
        n_train = n_episodes - n_val

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

        print(f"[{split}] Using {len(self.indices)} frames from {n_train if split == 'train' else n_val} episodes")

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
            Dictionary with:
                - observation.image: (C, H, W) tensor (global camera)
                - observation.gripper_image: (C, H, W) tensor (gripper camera)
                - observation.state: (6,) joint positions tensor
                - action: (6,) action tensor
                - language_instruction: str VLM prompt
        """
        # Map to dataset index
        dataset_idx = self.indices[idx]
        sample = self.lerobot_dataset[dataset_idx]

        result = {}

        # Process global camera
        if self.use_global_camera and "observation.global_camera" in sample:
            global_img = sample["observation.global_camera"]
            if isinstance(global_img, torch.Tensor):
                # Already tensor (C, H, W) or (H, W, C)
                if global_img.dim() == 3 and global_img.shape[0] != 3:
                    # (H, W, C) -> (C, H, W)
                    global_img = global_img.permute(2, 0, 1)
                global_img = global_img.float() / 255.0 if global_img.max() > 1.0 else global_img
            else:
                # Numpy array
                global_img = self.transform(global_img)
            result["observation.image"] = global_img

        # Process gripper camera
        if self.use_gripper_camera and "observation.gripper_camera" in sample:
            gripper_img = sample["observation.gripper_camera"]
            if isinstance(gripper_img, torch.Tensor):
                if gripper_img.dim() == 3 and gripper_img.shape[0] != 3:
                    gripper_img = gripper_img.permute(2, 0, 1)
                gripper_img = gripper_img.float() / 255.0 if gripper_img.max() > 1.0 else gripper_img
            else:
                gripper_img = self.transform(gripper_img)
            result["observation.gripper_image"] = gripper_img

        # Process joint positions (observation)
        if "observation.joint_positions" in sample:
            joint_pos = sample["observation.joint_positions"]
            if isinstance(joint_pos, np.ndarray):
                joint_pos = torch.from_numpy(joint_pos).float()
            elif not isinstance(joint_pos, torch.Tensor):
                joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
            result["observation.state"] = joint_pos

        # Process action (target)
        if "action" in sample:
            action = sample["action"]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            result["action"] = action

        # Process language instruction
        if "language_instruction" in sample:
            lang = sample["language_instruction"]
            if isinstance(lang, (list, tuple)):
                lang = lang[0] if lang else ""
            result["language_instruction"] = str(lang)
        else:
            result["language_instruction"] = ""

        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for chess episode batches.

    Handles variable-length language instructions and stacks tensors.
    """
    result = {}

    # Stack image tensors
    if "observation.image" in batch[0]:
        result["observation.image"] = torch.stack(
            [b["observation.image"] for b in batch]
        )

    if "observation.gripper_image" in batch[0]:
        result["observation.gripper_image"] = torch.stack(
            [b["observation.gripper_image"] for b in batch]
        )

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
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        dataset_path: Path to LeRobot dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Fraction of data for validation
        image_size: Target image size

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="train",
        image_size=image_size,
    )

    val_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="val",
        image_size=image_size,
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

    return train_loader, val_loader


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
