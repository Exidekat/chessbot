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
import json
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def reindex_dataset(dataset_path: Path, dry_run: bool = False):
    """
    Re-index a LeRobot dataset to have contiguous episode indices.

    Args:
        dataset_path: Path to the LeRobot dataset
        dry_run: If True, only show what would be done
    """
    import pandas as pd

    data_dir = dataset_path / "data"
    episodes_dir = dataset_path / "meta" / "episodes"

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False

    # Step 1: Get all current episode indices
    print("[INFO] Scanning dataset for episode indices...")
    current_episodes = set()
    for data_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(data_file)
            current_episodes.update(df["episode_index"].unique())
        except Exception as e:
            print(f"[WARNING] Failed to read {data_file}: {e}")

    current_episodes = sorted(current_episodes)

    if not current_episodes:
        print("[ERROR] No episodes found in dataset!")
        return False

    # Check if episode indices are contiguous
    expected_contiguous = list(range(len(current_episodes)))
    indices_contiguous = (current_episodes == expected_contiguous)

    # Check if parquet file names are contiguous (LeRobot requires this)
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

    # Create old_index -> new_index mapping
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(current_episodes)}

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
            df["episode_index"] = df["episode_index"].map(index_mapping)
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
            df["episode_index"] = df["episode_index"].map(index_mapping)
            df.to_parquet(ep_file, index=False)
            print(f"  Updated: {ep_file.relative_to(dataset_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to update {ep_file}: {e}")
            return False

    # Step 3b: Renumber data parquet file names to be contiguous
    print("\n[INFO] Renumbering data parquet files...")
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        files = sorted(chunk_dir.glob("file-*.parquet"))
        # First pass: rename to temp to avoid collisions
        temp_renames = []
        for new_idx, f in enumerate(files):
            old_name = f.name
            new_name = f"file-{new_idx:03d}.parquet"
            if old_name != new_name:
                temp_path = chunk_dir / f"{f.stem}.tmp"
                final_path = chunk_dir / new_name
                temp_renames.append((f, temp_path, final_path, old_name, new_name))
        # Rename to temp
        for src, temp, _, _, _ in temp_renames:
            src.rename(temp)
        # Rename to final
        for _, temp, final, old_name, new_name in temp_renames:
            temp.rename(final)
            print(f"  {chunk_dir.name}/{old_name} -> {new_name}")

    # Step 3c: Renumber meta/episodes parquet file names to be contiguous
    print("\n[INFO] Renumbering episode metadata files...")
    for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
        files = sorted(chunk_dir.glob("file-*.parquet"))
        # First pass: rename to temp to avoid collisions
        temp_renames = []
        for new_idx, f in enumerate(files):
            old_name = f.name
            new_name = f"file-{new_idx:03d}.parquet"
            if old_name != new_name:
                temp_path = chunk_dir / f"{f.stem}.tmp"
                final_path = chunk_dir / new_name
                temp_renames.append((f, temp_path, final_path, old_name, new_name))
        # Rename to temp
        for src, temp, _, _, _ in temp_renames:
            src.rename(temp)
        # Rename to final
        for _, temp, final, old_name, new_name in temp_renames:
            temp.rename(final)
            print(f"  {chunk_dir.name}/{old_name} -> {new_name}")

    # Step 4: Rename video files
    print("\n[INFO] Renaming video files...")
    for video_key in ["observation.images.global", "observation.images.gripper"]:
        video_dir = dataset_path / "videos" / video_key
        if not video_dir.exists():
            continue

        # First pass: rename to temp names to avoid collisions
        temp_renames = []
        for chunk_dir in sorted(video_dir.glob("chunk-*")):
            for video_file in sorted(chunk_dir.glob("file-*.mp4")):
                try:
                    old_idx = int(video_file.stem.replace("file-", ""))
                except ValueError:
                    continue

                if old_idx in index_mapping:
                    new_idx = index_mapping[old_idx]
                    if old_idx != new_idx:
                        temp_path = chunk_dir / f"file-{old_idx:06d}.mp4.tmp"
                        final_path = chunk_dir / f"file-{new_idx:06d}.mp4"
                        temp_renames.append((video_file, temp_path, final_path))

        # Rename to temp
        for src, temp, _ in temp_renames:
            src.rename(temp)

        # Rename to final
        for _, temp, final in temp_renames:
            temp.rename(final)
            print(f"  Renamed: {video_key}/.../{temp.stem.replace('.tmp', '')} -> {final.name}")

    # Step 5: Rename robot_configs per-episode files
    robot_configs_dir = dataset_path / "robot_configs"
    if robot_configs_dir.exists():
        print("\n[INFO] Renaming robot config files...")

        # First pass: rename to temp names
        temp_renames = []
        for config_file in sorted(robot_configs_dir.glob("episode_*.csv")):
            try:
                old_idx = int(config_file.stem.replace("episode_", ""))
            except ValueError:
                continue

            if old_idx in index_mapping:
                new_idx = index_mapping[old_idx]
                if old_idx != new_idx:
                    temp_path = robot_configs_dir / f"episode_{old_idx:06d}.csv.tmp"
                    final_path = robot_configs_dir / f"episode_{new_idx:06d}.csv"
                    temp_renames.append((config_file, temp_path, final_path))

        # Rename to temp
        for src, temp, _ in temp_renames:
            src.rename(temp)

        # Rename to final
        for _, temp, final in temp_renames:
            temp.rename(final)
            print(f"  Renamed: {temp.stem.replace('.tmp', '')} -> {final.name}")

    # Step 6: Update meta/info.json
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        print("\n[INFO] Updating info.json...")
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)

            new_total = len(current_episodes)
            info["total_episodes"] = new_total

            # Recalculate total frames
            total_frames = 0
            for data_file in data_dir.glob("chunk-*/file-*.parquet"):
                try:
                    df = pd.read_parquet(data_file)
                    total_frames += len(df)
                except:
                    pass
            info["total_frames"] = total_frames

            # Set splits to contiguous range
            info["splits"] = {"train": f"0:{new_total}"}

            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)

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
