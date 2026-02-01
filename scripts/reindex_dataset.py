#!/usr/bin/env python3
"""
Re-index LeRobot dataset to have contiguous episode indices.

This script fixes datasets with gaps in episode indices (e.g., after episode deletion)
by renumbering all episodes to be contiguous (0, 1, 2, ...).

Usage:
    python scripts/reindex_dataset.py --dataset data/lerobot_episodes/
    python scripts/reindex_dataset.py --dataset data/lerobot_episodes/ --dry-run
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lerobot_helpers import (
    get_episode_indices,
    is_contiguous,
    build_index_mapping,
    remap_episode_indices,
    load_info_json,
    save_info_json,
    calculate_total_frames,
    safe_rename_files,
    rename_video_files,
    rename_robot_configs,
    renumber_parquet_files,
)


def check_file_contiguity(directory: Path, pattern: str) -> bool:
    """Check if file-NNN.parquet files are contiguously numbered."""
    files = sorted(directory.glob(pattern))
    if not files:
        return True
    indices = []
    for f in files:
        try:
            idx = int(f.stem.replace("file-", ""))
            indices.append(idx)
        except ValueError:
            continue
    return indices == list(range(len(indices)))


def reindex_dataset(dataset_path: Path, dry_run: bool = False):
    """
    Re-index a LeRobot dataset to have contiguous episode indices.

    Args:
        dataset_path: Path to the LeRobot dataset
        dry_run: If True, only show what would be done
    """
    data_dir = dataset_path / "data"
    episodes_dir = dataset_path / "meta" / "episodes"

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False

    # Step 1: Get all current episode indices using helper
    print("[INFO] Scanning dataset for episode indices...")
    current_episodes = get_episode_indices(dataset_path)

    if not current_episodes:
        print("[ERROR] No episodes found in dataset!")
        return False

    # Check if episode indices are contiguous
    indices_contiguous = is_contiguous(current_episodes)

    # Check if parquet file names are contiguous (LeRobot requires this)
    data_files_contiguous = all(
        check_file_contiguity(chunk_dir, "file-*.parquet")
        for chunk_dir in data_dir.glob("chunk-*")
    )
    meta_files_contiguous = all(
        check_file_contiguity(chunk_dir, "file-*.parquet")
        for chunk_dir in episodes_dir.glob("chunk-*")
    ) if episodes_dir.exists() else True

    if indices_contiguous and data_files_contiguous and meta_files_contiguous:
        print(f"[OK] Dataset already has contiguous indices (0-{len(current_episodes)-1})")
        return True

    # Report what needs fixing
    if not indices_contiguous:
        print(f"[INFO] Episode indices have gaps: {min(current_episodes)}-{max(current_episodes)} ({len(current_episodes)} episodes)")
    if not data_files_contiguous:
        print(f"[INFO] Data parquet files have gaps in numbering")
    if not meta_files_contiguous:
        print(f"[INFO] Meta/episodes parquet files have gaps in numbering")

    # Create old_index -> new_index mapping using helper
    index_mapping = build_index_mapping(current_episodes)

    print(f"\n{'='*60}")
    print("RE-INDEXING PLAN")
    print(f"{'='*60}")
    print(f"Current episodes: {len(current_episodes)} (indices: {min(current_episodes)}-{max(current_episodes)} with gaps)")
    print(f"New indices: 0-{len(current_episodes)-1} (contiguous)")
    print(f"\nMapping (showing changed indices):")
    changes = [(old, new) for old, new in index_mapping.items() if old != new]
    if len(changes) <= 20:
        for old, new in changes:
            print(f"  {old} -> {new}")
    else:
        for old, new in changes[:10]:
            print(f"  {old} -> {new}")
        print(f"  ... ({len(changes) - 20} more)")
        for old, new in changes[-10:]:
            print(f"  {old} -> {new}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] No changes made. Remove --dry-run to apply changes.")
        return True

    # Confirm
    response = input("Proceed with re-indexing? (yes/no): ")
    if response.lower() != "yes":
        print("[INFO] Re-indexing cancelled")
        return False

    # Step 2: Update data parquet files
    print("\n[INFO] Updating data parquet files...")
    for data_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(data_file)
            df = remap_episode_indices(df, index_mapping)
            df.to_parquet(data_file, index=False)
            print(f"  Updated: {data_file.relative_to(dataset_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to update {data_file}: {e}")
            return False

    # Step 3: Update meta/episodes parquet files
    print("\n[INFO] Updating episode metadata...")
    for ep_file in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(ep_file)
            df = remap_episode_indices(df, index_mapping)
            df.to_parquet(ep_file, index=False)
            print(f"  Updated: {ep_file.relative_to(dataset_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to update {ep_file}: {e}")
            return False

    # Step 3b: Renumber data parquet file names to be contiguous
    print("\n[INFO] Renumbering data parquet files...")
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        renames = renumber_parquet_files(chunk_dir)
        for old_name, new_name in renames:
            print(f"  {chunk_dir.name}/{old_name} -> {new_name}")

    # Step 3c: Renumber meta/episodes parquet file names to be contiguous
    print("\n[INFO] Renumbering episode metadata files...")
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        renames = renumber_parquet_files(chunk_dir)
        for old_name, new_name in renames:
            print(f"  {chunk_dir.name}/{old_name} -> {new_name}")

    # Step 4: Rename video files using helper
    print("\n[INFO] Renaming video files...")
    rename_video_files(dataset_path, index_mapping)

    # Step 5: Rename robot_configs per-episode files using helper
    robot_configs_dir = dataset_path / "robot_configs"
    if robot_configs_dir.exists():
        print("\n[INFO] Renaming robot config files...")
        rename_robot_configs(dataset_path, index_mapping)

    # Step 6: Update meta/info.json using helpers
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        print("\n[INFO] Updating info.json...")
        try:
            info = load_info_json(dataset_path)

            new_total = len(current_episodes)
            info["total_episodes"] = new_total

            # Recalculate total frames using helper
            total_frames = calculate_total_frames(dataset_path)
            info["total_frames"] = total_frames

            # Set splits to contiguous range
            info["splits"] = {"train": f"0:{new_total}"}

            save_info_json(dataset_path, info)

            print(f"  Episodes: {new_total}")
            print(f"  Frames: {total_frames}")
            print(f"  Splits: 0:{new_total}")

        except Exception as e:
            print(f"[ERROR] Failed to update info.json: {e}")
            return False

    print(f"\n{'='*60}")
    print(f"[OK] Re-indexing complete!")
    print(f"    {len(current_episodes)} episodes now indexed 0-{len(current_episodes)-1}")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Re-index LeRobot dataset to have contiguous episode indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview changes (dry run)
    python scripts/reindex_dataset.py --dataset data/lerobot_episodes/ --dry-run

    # Apply re-indexing
    python scripts/reindex_dataset.py --dataset data/lerobot_episodes/
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/lerobot_episodes",
        help="Path to LeRobot dataset directory"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return 1

    if not (dataset_path / "meta" / "info.json").exists():
        print(f"[ERROR] Not a LeRobot dataset: {dataset_path}")
        return 1

    success = reindex_dataset(dataset_path, dry_run=args.dry_run)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
