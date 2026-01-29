#!/usr/bin/env python3
"""
Migrate existing dataset to include per-episode robot configs for home-relative normalization.

This script:
1. Creates robot_configs/ directory in dataset
2. Copies robot configs as per-episode files (episode_000000.csv, episode_000001.csv, etc.)

Usage:
    # Migrate with known episode ranges
    python scripts/migrate_episode_configs.py --dataset data/lerobot_episodes/ \
        --config data/so100_config_ttyACM0.csv --episodes 0-16 \
        --config data/so100_config_ttyACM3.csv --episodes 39-105

    # List existing episodes in dataset
    python scripts/migrate_episode_configs.py --dataset data/lerobot_episodes/ --list
"""

import argparse
import json
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_episode_range(range_str: str) -> list:
    """
    Parse episode range string into list of episode indices.

    Supports:
    - Single: "5" -> [5]
    - Range: "0-16" -> [0, 1, 2, ..., 16]
    - Multiple: "0-16,39-105" -> [0, 1, ..., 16, 39, 40, ..., 105]
    """
    episodes = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return episodes


def list_episodes(dataset_path: Path):
    """List all episodes in the dataset."""
    # Check for LeRobot format
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)

        total_episodes = info.get("total_episodes", 0)
        total_frames = info.get("total_frames", 0)

        print(f"\nDataset: {dataset_path}")
        print(f"Format: LeRobot v3.0")
        print(f"Total episodes: {total_episodes}")
        print(f"Total frames: {total_frames}")

        # Check for robot_configs directory
        robot_configs_dir = dataset_path / "robot_configs"
        if robot_configs_dir.exists():
            configs = sorted(robot_configs_dir.glob("episode_*.csv"))
            print(f"\nPer-episode robot configs: {len(configs)}")
            if configs:
                # Show range
                indices = [int(c.stem.replace("episode_", "")) for c in configs]
                print(f"  Episodes with configs: {min(indices)}-{max(indices)} ({len(indices)} files)")
        else:
            print("\nNo robot_configs/ directory found.")

        # Check for legacy mapping file
        legacy_mapping = dataset_path / "episode_robot_configs.json"
        if legacy_mapping.exists():
            print(f"\n[WARN] Legacy episode_robot_configs.json found - consider migrating to per-episode format")

        return True
    else:
        print(f"[ERROR] Not a LeRobot dataset: {dataset_path}")
        return False


def migrate_configs(
    dataset_path: Path,
    config_episode_pairs: list,
    dry_run: bool = False
):
    """
    Migrate robot configs to per-episode files in dataset.

    Args:
        dataset_path: Path to LeRobot dataset
        config_episode_pairs: List of (config_path, episodes) tuples
        dry_run: If True, only show what would be done
    """
    robot_configs_dir = dataset_path / "robot_configs"

    # Create robot_configs directory
    if not dry_run:
        robot_configs_dir.mkdir(parents=True, exist_ok=True)

    # Track what we're doing
    total_created = 0
    total_skipped = 0

    for config_path, episodes in config_episode_pairs:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            continue

        print(f"\n[INFO] Processing {config_path.name} for {len(episodes)} episodes...")

        for ep_idx in episodes:
            dest_path = robot_configs_dir / f"episode_{ep_idx:06d}.csv"

            if dest_path.exists():
                total_skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would create {dest_path.name}")
            else:
                shutil.copy2(config_path, dest_path)

            total_created += 1

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)

    if dry_run:
        print(f"[DRY RUN] Would create {total_created} per-episode config files")
        print(f"[DRY RUN] Would skip {total_skipped} existing files")
    else:
        print(f"Created: {total_created} per-episode config files")
        print(f"Skipped: {total_skipped} existing files")
        print(f"Location: {robot_configs_dir}")

    # Show breakdown by config
    print("\nBreakdown by source config:")
    for config_path, episodes in config_episode_pairs:
        config_path = Path(config_path)
        # Compress to ranges for display
        episodes = sorted(episodes)
        if episodes:
            ranges = []
            start = episodes[0]
            end = episodes[0]
            for i in range(1, len(episodes)):
                if episodes[i] == end + 1:
                    end = episodes[i]
                else:
                    ranges.append(f"{start}-{end}" if start != end else str(start))
                    start = episodes[i]
                    end = episodes[i]
            ranges.append(f"{start}-{end}" if start != end else str(start))
            print(f"  {config_path.name}: episodes {', '.join(ranges)} ({len(episodes)} total)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing dataset to include per-episode robot configs for home-relative normalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List episodes in dataset
    python scripts/migrate_episode_configs.py --dataset data/lerobot_episodes/ --list

    # Migrate with known episode ranges (dry run)
    python scripts/migrate_episode_configs.py --dataset data/lerobot_episodes/ \\
        --config data/so100_config_ttyACM0.csv --episodes 0-16 \\
        --config data/so100_config_ttyACM3.csv --episodes 39-105 \\
        --dry-run

    # Actually perform migration
    python scripts/migrate_episode_configs.py --dataset data/lerobot_episodes/ \\
        --config data/so100_config_ttyACM0.csv --episodes 0-16 \\
        --config data/so100_config_ttyACM3.csv --episodes 39-105
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/lerobot_episodes",
        help="Path to LeRobot dataset directory"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List episodes and existing config files, then exit"
    )

    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=[],
        help="Path to robot config CSV (can be specified multiple times)"
    )

    parser.add_argument(
        "--episodes",
        type=str,
        action="append",
        default=[],
        help="Episode range for preceding --config (e.g., '0-16' or '39,40,41')"
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

    if args.list:
        list_episodes(dataset_path)
        return 0

    # Validate config/episode pairs
    if len(args.config) != len(args.episodes):
        print("[ERROR] Each --config must be followed by --episodes")
        print(f"       Got {len(args.config)} configs and {len(args.episodes)} episode ranges")
        return 1

    if len(args.config) == 0:
        print("[ERROR] No configs specified. Use --config and --episodes to specify mappings.")
        print("       Use --list to see existing episodes.")
        return 1

    # Parse episode ranges
    config_episode_pairs = []
    for config_path, episode_range in zip(args.config, args.episodes):
        episodes = parse_episode_range(episode_range)
        config_episode_pairs.append((config_path, episodes))
        print(f"[INFO] {config_path} -> episodes {episode_range} ({len(episodes)} total)")

    # Perform migration
    migrate_configs(
        dataset_path=dataset_path,
        config_episode_pairs=config_episode_pairs,
        dry_run=args.dry_run
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
