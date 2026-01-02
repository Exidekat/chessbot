"""
Action and State Normalization for VLA Training

Implements quantile-based normalization as used by PI0 and SmolVLA:
1. Compute 1st and 99th percentiles per dimension from training data
2. Map values to [-1, 1] range during training
3. Denormalize (inverse transform) during inference

This matches industry practice from OpenPI and LeRobot.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass, asdict

try:
    import torch
except ImportError:
    torch = None


@dataclass
class NormStats:
    """Normalization statistics for a single feature."""
    q01: List[float]  # 1st percentile per dimension
    q99: List[float]  # 99th percentile per dimension
    mean: List[float]
    std: List[float]
    min: List[float]
    max: List[float]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "NormStats":
        return cls(**d)


class ActionNormalizer:
    """
    Quantile-based action normalizer for VLA models.

    Maps action values to [-1, 1] range using 1st/99th percentiles.
    This matches the normalization used by PI0 and SmolVLA.
    """

    def __init__(self, stats_path: Optional[str] = None):
        """
        Initialize normalizer.

        Args:
            stats_path: Path to load existing statistics from.
                       If None, statistics must be computed or loaded later.
        """
        self.stats: Dict[str, NormStats] = {}
        self.stats_path = stats_path

        if stats_path and Path(stats_path).exists():
            self.load(stats_path)

    def compute_stats(
        self,
        dataset,
        feature_keys: List[str] = ["action", "observation.joint_positions"],
        num_samples: Optional[int] = None,
    ) -> Dict[str, NormStats]:
        """
        Compute normalization statistics from a dataset.

        Args:
            dataset: PyTorch dataset or iterable yielding dicts with feature_keys
            feature_keys: Features to compute statistics for
            num_samples: Number of samples to use (None = all)

        Returns:
            Dictionary mapping feature keys to NormStats
        """
        # Collect samples
        samples = {key: [] for key in feature_keys}

        n = 0
        for item in dataset:
            for key in feature_keys:
                if key in item:
                    val = item[key]
                    if torch is not None and isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    if isinstance(val, np.ndarray):
                        # Handle action chunks (50, 32) - take first timestep
                        if val.ndim == 2:
                            val = val[0]
                        samples[key].append(val)

            n += 1
            if num_samples and n >= num_samples:
                break

        # Compute statistics per feature
        for key in feature_keys:
            if not samples[key]:
                continue

            data = np.stack(samples[key])  # (N, dim)

            self.stats[key] = NormStats(
                q01=np.percentile(data, 1, axis=0).tolist(),
                q99=np.percentile(data, 99, axis=0).tolist(),
                mean=np.mean(data, axis=0).tolist(),
                std=np.std(data, axis=0).tolist(),
                min=np.min(data, axis=0).tolist(),
                max=np.max(data, axis=0).tolist(),
            )

            print(f"[NormStats] Computed stats for '{key}':")
            print(f"  Shape: {data.shape}")
            print(f"  q01: {self.stats[key].q01[:6]}...")
            print(f"  q99: {self.stats[key].q99[:6]}...")

        return self.stats

    def normalize(
        self,
        values: Union[np.ndarray, "torch.Tensor"],
        key: str = "action",
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Normalize values to [-1, 1] range using quantile scaling.

        Formula: normalized = (value - q01) / (q99 - q01) * 2 - 1

        Args:
            values: Values to normalize, shape (..., dim)
            key: Feature key to use statistics for

        Returns:
            Normalized values in [-1, 1] range
        """
        if key not in self.stats:
            print(f"[WARN] No stats for '{key}', returning unnormalized")
            return values

        stats = self.stats[key]
        is_torch = torch is not None and isinstance(values, torch.Tensor)

        if is_torch:
            q01 = torch.tensor(stats.q01, dtype=values.dtype, device=values.device)
            q99 = torch.tensor(stats.q99, dtype=values.dtype, device=values.device)
        else:
            q01 = np.array(stats.q01, dtype=np.float32)
            q99 = np.array(stats.q99, dtype=np.float32)

        # Handle dimension mismatch: truncate stats to match input dimension
        # This happens when stats were saved with padding (e.g., 32D for PI0)
        # but model uses fewer dimensions (e.g., 6D for SmolVLA)
        input_dim = values.shape[-1]
        stats_dim = len(stats.q01)
        if input_dim < stats_dim:
            q01 = q01[:input_dim]
            q99 = q99[:input_dim]

        # Handle shape broadcasting
        # Values can be (dim,), (batch, dim), or (batch, chunk, dim)
        while q01.ndim < values.ndim:
            if is_torch:
                q01 = q01.unsqueeze(0)
                q99 = q99.unsqueeze(0)
            else:
                q01 = np.expand_dims(q01, 0)
                q99 = np.expand_dims(q99, 0)

        # Avoid division by zero
        range_val = q99 - q01
        if is_torch:
            range_val = torch.clamp(range_val, min=1e-6)
        else:
            range_val = np.clip(range_val, 1e-6, None)

        # Normalize to [-1, 1]
        normalized = (values - q01) / range_val * 2 - 1

        # Clip to [-1, 1] for safety
        if is_torch:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        else:
            normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def denormalize(
        self,
        values: Union[np.ndarray, "torch.Tensor"],
        key: str = "action",
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Denormalize values from [-1, 1] range to original scale.

        Formula: original = (normalized + 1) / 2 * (q99 - q01) + q01

        Args:
            values: Normalized values in [-1, 1] range
            key: Feature key to use statistics for

        Returns:
            Denormalized values in original scale
        """
        if key not in self.stats:
            print(f"[WARN] No stats for '{key}', returning as-is")
            return values

        stats = self.stats[key]
        is_torch = torch is not None and isinstance(values, torch.Tensor)

        if is_torch:
            q01 = torch.tensor(stats.q01, dtype=values.dtype, device=values.device)
            q99 = torch.tensor(stats.q99, dtype=values.dtype, device=values.device)
        else:
            q01 = np.array(stats.q01, dtype=np.float32)
            q99 = np.array(stats.q99, dtype=np.float32)

        # Handle dimension mismatch: truncate stats to match input dimension
        # This happens when stats were saved with padding (e.g., 32D for PI0)
        # but model uses fewer dimensions (e.g., 6D for SmolVLA)
        input_dim = values.shape[-1]
        stats_dim = len(stats.q01)
        if input_dim < stats_dim:
            q01 = q01[:input_dim]
            q99 = q99[:input_dim]

        # Handle shape broadcasting
        while q01.ndim < values.ndim:
            if is_torch:
                q01 = q01.unsqueeze(0)
                q99 = q99.unsqueeze(0)
            else:
                q01 = np.expand_dims(q01, 0)
                q99 = np.expand_dims(q99, 0)

        # Denormalize from [-1, 1]
        denormalized = (values + 1) / 2 * (q99 - q01) + q01

        return denormalized

    def save(self, path: str):
        """Save normalization statistics to JSON file."""
        save_dict = {key: stats.to_dict() for key, stats in self.stats.items()}

        with open(path, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"[NormStats] Saved to {path}")

    def load(self, path: str):
        """Load normalization statistics from JSON file."""
        with open(path, "r") as f:
            load_dict = json.load(f)

        self.stats = {key: NormStats.from_dict(d) for key, d in load_dict.items()}
        self.stats_path = path

        print(f"[NormStats] Loaded from {path}")
        for key in self.stats:
            print(f"  - {key}: {len(self.stats[key].q01)} dimensions")

    def has_stats(self, key: str = "action") -> bool:
        """Check if statistics exist for a feature."""
        return key in self.stats


def load_episodes_from_custom_format(dataset_path: str) -> List[Dict[str, np.ndarray]]:
    """
    Load episodes from custom ChessBot format (PNG files + metadata.json).

    This format stores episodes as:
        dataset_path/
            episode_000000/
                metadata.json  (contains joint_positions, vlm_prompt, etc.)
                global_000000.png
                gripper_000000.png
                ...
            episode_000001/
                ...

    Args:
        dataset_path: Path to episodes directory

    Returns:
        List of dictionaries with 'action' and 'observation.joint_positions' keys
    """
    dataset_dir = Path(dataset_path)
    samples = []

    # Find all episode directories
    episode_dirs = sorted(dataset_dir.glob("episode_*"))

    if not episode_dirs:
        raise ValueError(f"No episode directories found in {dataset_path}")

    print(f"[NormStats] Found {len(episode_dirs)} episodes")

    for ep_dir in episode_dirs:
        metadata_path = ep_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"[WARN] Skipping {ep_dir.name}: no metadata.json")
            continue

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract joint positions (array of arrays, one per frame)
        joint_positions = metadata.get("joint_positions", [])
        if not joint_positions:
            print(f"[WARN] Skipping {ep_dir.name}: no joint_positions")
            continue

        # Each frame becomes a sample
        # For VLA training, action at frame t is joint position at frame t+1
        # (or we can use the current position as the target for simplicity)
        for i, jp in enumerate(joint_positions):
            jp_array = np.array(jp, dtype=np.float32)

            # Action = joint position (target position for the robot)
            # For training, we treat current joint position as the action label
            samples.append({
                "action": jp_array,
                "observation.joint_positions": jp_array,
            })

    print(f"[NormStats] Loaded {len(samples)} frames from {len(episode_dirs)} episodes")
    return samples


def compute_dataset_stats(
    dataset_path: str,
    output_path: Optional[str] = None,
    feature_keys: List[str] = ["action", "observation.state"],
) -> ActionNormalizer:
    """
    Convenience function to compute and save normalization statistics.

    Supports both:
    1. Custom ChessBot format (PNG files + metadata.json per episode)
    2. LeRobot format (parquet files + HuggingFace metadata)

    Args:
        dataset_path: Path to dataset directory
        output_path: Path to save stats JSON (default: dataset_path/norm_stats.json)
        feature_keys: Features to compute statistics for

    Returns:
        ActionNormalizer with computed statistics
    """
    if output_path is None:
        output_path = str(Path(dataset_path) / "norm_stats.json")

    # Check if this is custom format (has episode_* directories with metadata.json)
    dataset_dir = Path(dataset_path)
    episode_dirs = list(dataset_dir.glob("episode_*/metadata.json"))

    if episode_dirs:
        # Custom ChessBot format
        print(f"[NormStats] Detected custom ChessBot format")
        samples = load_episodes_from_custom_format(dataset_path)
    else:
        # Try LeRobot format
        print(f"[NormStats] Trying LeRobot format...")
        try:
            from vla.chess_dataloader import ChessEpisodeDataset
            dataset = ChessEpisodeDataset(
                dataset_path=dataset_path,
                split="all",
                normalize_actions=False,  # Don't normalize during stats computation
            )
            # Convert to list of dicts
            samples = [dataset[i] for i in range(len(dataset))]
        except Exception as e:
            raise ValueError(f"Could not load dataset: {e}")

    # Compute stats
    normalizer = ActionNormalizer()
    normalizer.compute_stats(samples, feature_keys=feature_keys)

    # Save
    normalizer.save(output_path)

    return normalizer


if __name__ == "__main__":
    """Compute normalization statistics for a dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--dataset", type=str, default="data/episodes",
                        help="Path to LeRobot dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for stats JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("Computing Normalization Statistics")
    print("=" * 60)

    normalizer = compute_dataset_stats(
        dataset_path=args.dataset,
        output_path=args.output,
    )

    print("\n[OK] Statistics computed and saved!")
