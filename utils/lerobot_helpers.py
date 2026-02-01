"""
LeRobot Dataset Manipulation Helpers

Shared utilities for working with LeRobot v3.0 datasets, used by:
- clean_lerobot_dataset.py
- reindex_dataset.py
- validate_vla_episodes.py
- salvage_vla_episodes.py

These helpers reduce code duplication and provide a consistent API for:
- Parquet file operations (load, write, validate)
- Episode index management (contiguity checks, remapping)
- Metadata operations (info.json, stats.json)
- Quality annotations (episode_qualities.json)
- File operations (safe renaming, video/config file management)
- Robot config loading (per-episode calibration files)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Parquet Operations
# =============================================================================

def load_all_data_parquets(dataset_path: Path) -> pd.DataFrame:
    """
    Load all data parquet files from a LeRobot dataset.

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Combined DataFrame with all frame data
    """
    data_dir = dataset_path / "data"
    all_dfs = []

    for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pq.read_table(pq_file).to_pandas()
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {pq_file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def load_all_episode_metadata(dataset_path: Path) -> pd.DataFrame:
    """
    Load all episode metadata parquet files from a LeRobot dataset.

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Combined DataFrame with all episode metadata
    """
    episodes_dir = dataset_path / "meta" / "episodes"
    all_dfs = []

    for pq_file in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pq.read_table(pq_file).to_pandas()
            all_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read episode metadata {pq_file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def write_parquet_chunked(
    df: pd.DataFrame,
    output_dir: Path,
    chunk_size: int = 5,
    group_by: str = "episode_index"
) -> List[Path]:
    """
    Write a DataFrame to parquet files, grouped by episode index.

    Args:
        df: DataFrame to write
        output_dir: Directory to write files to (e.g., data/chunk-000)
        chunk_size: Number of episodes per file
        group_by: Column to group by for chunking

    Returns:
        List of paths to created files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    unique_groups = df[group_by].unique()
    file_idx = 0

    for i in range(0, len(unique_groups), chunk_size):
        group_batch = unique_groups[i:i + chunk_size]
        batch_df = df[df[group_by].isin(group_batch)]

        out_file = output_dir / f"file-{file_idx:03d}.parquet"
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        pq.write_table(table, out_file)

        created_files.append(out_file)
        file_idx += 1

    return created_files


def validate_parquet_file(path: Path) -> Tuple[bool, int]:
    """
    Check if a parquet file is valid and readable.

    Args:
        path: Path to the parquet file

    Returns:
        (is_valid, row_count) - row_count is 0 if invalid
    """
    try:
        table = pq.read_table(str(path))
        return True, len(table)
    except Exception:
        return False, 0


# =============================================================================
# Episode Index Management
# =============================================================================

def get_episode_indices(dataset_path: Path) -> List[int]:
    """
    Get all episode indices from a LeRobot dataset's data files.

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Sorted list of episode indices
    """
    data_dir = dataset_path / "data"
    episode_indices = set()

    for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(pq_file)
            episode_indices.update(df["episode_index"].unique())
        except Exception as e:
            print(f"[WARN] Failed to read {pq_file}: {e}")

    return sorted(episode_indices)


def is_contiguous(indices: List[int]) -> bool:
    """
    Check if a list of indices is contiguous starting from 0.

    Args:
        indices: List of integer indices

    Returns:
        True if indices are [0, 1, 2, ..., n-1]
    """
    if not indices:
        return True
    expected = list(range(len(indices)))
    return sorted(indices) == expected


def build_index_mapping(old_indices: List[int]) -> Dict[int, int]:
    """
    Build a mapping from old indices to new contiguous indices.

    Args:
        old_indices: List of old episode indices (may have gaps)

    Returns:
        Dict mapping old_index -> new_index (0, 1, 2, ...)
    """
    sorted_indices = sorted(old_indices)
    return {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}


def remap_episode_indices(df: pd.DataFrame, mapping: Dict[int, int]) -> pd.DataFrame:
    """
    Remap episode indices in a DataFrame using a mapping.

    Args:
        df: DataFrame with 'episode_index' column
        mapping: Dict mapping old_index -> new_index

    Returns:
        DataFrame with remapped episode indices
    """
    df = df.copy()
    df["episode_index"] = df["episode_index"].map(mapping)
    return df


# =============================================================================
# Metadata Operations
# =============================================================================

def load_info_json(dataset_path: Path) -> Dict:
    """
    Load info.json from a LeRobot dataset.

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Dict with dataset info, or empty dict if not found
    """
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {}

    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load info.json: {e}")
        return {}


def save_info_json(dataset_path: Path, info: Dict):
    """
    Save info.json to a LeRobot dataset.

    Args:
        dataset_path: Root path to the LeRobot dataset
        info: Dict with dataset info
    """
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def update_episode_counts(
    dataset_path: Path,
    total_episodes: int,
    total_frames: int
):
    """
    Update episode and frame counts in info.json.

    Also updates the splits field to reflect the new episode count.

    Args:
        dataset_path: Root path to the LeRobot dataset
        total_episodes: New total episode count
        total_frames: New total frame count
    """
    info = load_info_json(dataset_path)

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}

    save_info_json(dataset_path, info)


def calculate_total_frames(dataset_path: Path) -> int:
    """
    Calculate total frame count from data parquet files.

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Total number of frames across all valid parquet files
    """
    data_dir = dataset_path / "data"
    total_frames = 0

    for pq_file in data_dir.glob("chunk-*/file-*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            total_frames += len(df)
        except Exception:
            pass

    return total_frames


# =============================================================================
# Quality Annotations
# =============================================================================

def load_quality_annotations(dataset_path: Path) -> Dict[int, str]:
    """
    Load episode quality annotations from a LeRobot dataset.

    Quality annotations are stored in episode_qualities.json with format:
    {"0": "good", "1": "bad", "2": "unreviewed", ...}

    Args:
        dataset_path: Root path to the LeRobot dataset

    Returns:
        Dict mapping episode_index (int) -> quality ("good", "bad", or "unreviewed")
    """
    quality_file = dataset_path / "episode_qualities.json"
    if not quality_file.exists():
        return {}

    try:
        with open(quality_file, 'r') as f:
            loaded = json.load(f)
            return {int(k): v for k, v in loaded.items()}
    except Exception as e:
        print(f"[WARN] Failed to load quality annotations: {e}")
        return {}


def save_quality_annotations(dataset_path: Path, annotations: Dict[int, str]):
    """
    Save episode quality annotations to a LeRobot dataset.

    Args:
        dataset_path: Root path to the LeRobot dataset
        annotations: Dict mapping episode_index (int) -> quality
    """
    quality_file = dataset_path / "episode_qualities.json"

    # Convert int keys to strings for JSON
    str_annotations = {str(k): v for k, v in annotations.items()}

    with open(quality_file, 'w') as f:
        json.dump(str_annotations, f, indent=2)


def get_bad_episodes(annotations: Dict[int, str]) -> Set[int]:
    """
    Get set of episode indices marked as "bad".

    Args:
        annotations: Quality annotations dict

    Returns:
        Set of episode indices with quality == "bad"
    """
    return {idx for idx, quality in annotations.items() if quality == "bad"}


def remap_quality_annotations(
    annotations: Dict[int, str],
    index_mapping: Dict[int, int]
) -> Dict[int, str]:
    """
    Remap quality annotations to new episode indices.

    Args:
        annotations: Original quality annotations
        index_mapping: Dict mapping old_index -> new_index

    Returns:
        Dict with remapped quality annotations
    """
    remapped = {}
    for old_idx, quality in annotations.items():
        if old_idx in index_mapping:
            new_idx = index_mapping[old_idx]
            remapped[new_idx] = quality
    return remapped


# =============================================================================
# File Operations
# =============================================================================

def safe_rename_files(renames: List[Tuple[Path, Path]]):
    """
    Safely rename files using temp files to avoid collisions.

    This handles cases where file-001.mp4 needs to become file-000.mp4
    but file-000.mp4 already exists - we first rename to temp, then
    to final destination.

    Args:
        renames: List of (source, destination) path tuples
    """
    if not renames:
        return

    # First pass: rename to temp names
    temp_mapping = []
    for src, dst in renames:
        temp_path = src.parent / f"{src.stem}.tmp{src.suffix}"
        src.rename(temp_path)
        temp_mapping.append((temp_path, dst))

    # Second pass: rename to final names
    for temp, dst in temp_mapping:
        temp.rename(dst)


def rename_video_files(dataset_path: Path, index_mapping: Dict[int, int]):
    """
    Rename video files to match new episode indices.

    Args:
        dataset_path: Root path to the LeRobot dataset
        index_mapping: Dict mapping old_index -> new_index
    """
    for video_key in ["observation.images.global", "observation.images.gripper"]:
        video_dir = dataset_path / "videos" / video_key
        if not video_dir.exists():
            continue

        renames = []
        for chunk_dir in sorted(video_dir.glob("chunk-*")):
            for video_file in sorted(chunk_dir.glob("file-*.mp4")):
                try:
                    old_idx = int(video_file.stem.replace("file-", ""))
                except ValueError:
                    continue

                if old_idx in index_mapping:
                    new_idx = index_mapping[old_idx]
                    if old_idx != new_idx:
                        new_path = chunk_dir / f"file-{new_idx:06d}.mp4"
                        renames.append((video_file, new_path))

        safe_rename_files(renames)


def rename_robot_configs(dataset_path: Path, index_mapping: Dict[int, int]):
    """
    Rename robot config files to match new episode indices.

    Args:
        dataset_path: Root path to the LeRobot dataset
        index_mapping: Dict mapping old_index -> new_index
    """
    robot_configs_dir = dataset_path / "robot_configs"
    if not robot_configs_dir.exists():
        return

    renames = []
    for config_file in sorted(robot_configs_dir.glob("episode_*.csv")):
        try:
            old_idx = int(config_file.stem.replace("episode_", ""))
        except ValueError:
            continue

        if old_idx in index_mapping:
            new_idx = index_mapping[old_idx]
            if old_idx != new_idx:
                new_path = robot_configs_dir / f"episode_{new_idx:06d}.csv"
                renames.append((config_file, new_path))

    safe_rename_files(renames)


def delete_episode_files(dataset_path: Path, episode_indices: List[int]):
    """
    Delete all files associated with specific episodes.

    Deletes:
    - Video files (observation.images.global, observation.images.gripper)
    - Robot config files

    Args:
        dataset_path: Root path to the LeRobot dataset
        episode_indices: List of episode indices to delete
    """
    deleted_set = set(episode_indices)

    # Delete video files
    for video_key in ["observation.images.global", "observation.images.gripper"]:
        video_dir = dataset_path / "videos" / video_key
        if not video_dir.exists():
            continue

        for chunk_dir in sorted(video_dir.glob("chunk-*")):
            for video_file in sorted(chunk_dir.glob("file-*.mp4")):
                try:
                    idx = int(video_file.stem.replace("file-", ""))
                except ValueError:
                    continue

                if idx in deleted_set:
                    video_file.unlink()

    # Delete robot config files
    robot_configs_dir = dataset_path / "robot_configs"
    if robot_configs_dir.exists():
        for config_file in sorted(robot_configs_dir.glob("episode_*.csv")):
            try:
                idx = int(config_file.stem.replace("episode_", ""))
            except ValueError:
                continue

            if idx in deleted_set:
                config_file.unlink()


def renumber_parquet_files(directory: Path) -> List[Tuple[str, str]]:
    """
    Renumber parquet files to be contiguous (file-000, file-001, ...).

    Args:
        directory: Directory containing parquet files (e.g., data/chunk-000)

    Returns:
        List of (old_name, new_name) tuples for files that were renamed
    """
    files = sorted(directory.glob("file-*.parquet"))
    renames = []

    for new_idx, f in enumerate(files):
        old_name = f.name
        new_name = f"file-{new_idx:03d}.parquet"
        if old_name != new_name:
            renames.append((old_name, new_name))

    # Execute renames using safe method
    rename_tuples = [(directory / old, directory / new) for old, new in renames]
    safe_rename_files(rename_tuples)

    return renames


# =============================================================================
# Robot Config Helpers
# =============================================================================

def load_robot_config(config_path: Path) -> Optional[np.ndarray]:
    """
    Load robot home positions from a per-episode config file.

    Config file format is CSV with columns: joint_name, home_rad, ...

    Args:
        config_path: Path to the episode config CSV file

    Returns:
        np.ndarray: Home positions (6 joints, radians), or None if not available
    """
    if not config_path.exists():
        return None

    try:
        import csv
        home_positions = []
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                home_positions.append(float(row['home_rad']))

        if len(home_positions) != 6:
            print(f"[WARN] Invalid robot config (expected 6 joints): {config_path}")
            return None

        return np.array(home_positions, dtype=np.float32)

    except Exception as e:
        print(f"[WARN] Failed to load robot config {config_path}: {e}")
        return None


def copy_robot_config(src: Path, dst: Path):
    """
    Copy a robot config file.

    Args:
        src: Source config file path
        dst: Destination config file path
    """
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


# =============================================================================
# Cleanup Helpers
# =============================================================================

def remove_empty_directories(dataset_path: Path):
    """
    Remove empty chunk directories from the dataset.

    Cleans up:
    - data/chunk-*
    - videos/*/chunk-*
    - meta/episodes/chunk-*

    Args:
        dataset_path: Root path to the LeRobot dataset
    """
    subdirs = [
        "data",
        "videos/observation.images.global",
        "videos/observation.images.gripper",
        "meta/episodes"
    ]

    for subdir in subdirs:
        target_dir = dataset_path / subdir
        if target_dir.exists():
            for chunk_dir in target_dir.glob("chunk-*"):
                if chunk_dir.is_dir() and not any(chunk_dir.iterdir()):
                    chunk_dir.rmdir()
