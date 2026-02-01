#!/usr/bin/env python3
"""
Salvage VLA Episodes Script

Repairs a corrupted LeRobot dataset by:
1. Identifying and removing corrupted parquet files
2. Recalculating frame counts from valid data files
3. Updating metadata to match valid episode count

Usage:
    # Dry run (show what would be done)
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes

    # Actually repair the dataset
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes --fix

    # Create backup before repair
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes --fix --backup
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.lerobot_helpers import (
    validate_parquet_file,
    load_info_json,
    save_info_json,
)


def scan_dataset(dataset_path: Path) -> dict:
    """
    Scan dataset and identify valid/corrupted files.

    Returns:
        Dictionary with scan results
    """
    results = {
        "data_files": [],
        "episode_meta_files": [],
        "valid_frames": 0,
        "valid_episodes": 0,
        "corrupted_files": [],
        "info_json": None,
    }

    # Check info.json
    results["info_json"] = load_info_json(dataset_path)
    if not results["info_json"]:
        print(f"[X] No info.json found at {dataset_path / 'meta' / 'info.json'}")
        return results

    # Scan data files
    data_dir = dataset_path / "data"
    if data_dir.exists():
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                is_valid, row_count = validate_parquet_file(parquet_file)
                file_info = {
                    "path": parquet_file,
                    "valid": is_valid,
                    "rows": row_count,
                }
                results["data_files"].append(file_info)

                if is_valid:
                    results["valid_frames"] += row_count
                else:
                    results["corrupted_files"].append(parquet_file)

    # Scan episode metadata files
    episodes_dir = dataset_path / "meta" / "episodes"
    if episodes_dir.exists():
        for chunk_dir in sorted(episodes_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                is_valid, row_count = validate_parquet_file(parquet_file)
                file_info = {
                    "path": parquet_file,
                    "valid": is_valid,
                    "rows": row_count,
                }
                results["episode_meta_files"].append(file_info)

                if is_valid:
                    results["valid_episodes"] += row_count
                else:
                    results["corrupted_files"].append(parquet_file)

    return results


def print_scan_results(results: dict):
    """Print scan results in a readable format."""
    print("\n" + "=" * 60)
    print("Dataset Scan Results")
    print("=" * 60)

    info = results["info_json"]
    if info:
        print(f"\nMetadata (info.json):")
        print(f"  Total episodes (claimed): {info.get('total_episodes', 'N/A')}")
        print(f"  Total frames (claimed):   {info.get('total_frames', 'N/A')}")

    print(f"\nData Files:")
    for f in results["data_files"]:
        status = "[OK]" if f["valid"] else "[CORRUPTED]"
        print(f"  {status} {f['path'].name}: {f['rows']} rows")

    print(f"\nEpisode Metadata Files:")
    for f in results["episode_meta_files"]:
        status = "[OK]" if f["valid"] else "[CORRUPTED]"
        print(f"  {status} {f['path'].name}: {f['rows']} episodes")

    print(f"\nSummary:")
    print(f"  Valid frames:   {results['valid_frames']}")
    print(f"  Valid episodes: {results['valid_episodes']}")
    print(f"  Corrupted files: {len(results['corrupted_files'])}")

    if results["corrupted_files"]:
        print(f"\nCorrupted files to remove:")
        for f in results["corrupted_files"]:
            print(f"  - {f}")

    # Check for mismatches
    if info:
        claimed_episodes = info.get("total_episodes", 0)
        claimed_frames = info.get("total_frames", 0)

        if claimed_episodes != results["valid_episodes"]:
            print(f"\n[!] Episode count mismatch: metadata says {claimed_episodes}, found {results['valid_episodes']}")

        if claimed_frames != results["valid_frames"]:
            print(f"[!] Frame count mismatch: metadata says {claimed_frames}, found {results['valid_frames']}")

    print("=" * 60)


def repair_dataset(dataset_path: Path, results: dict, create_backup: bool = False):
    """
    Repair the dataset by removing corrupted files and updating metadata.
    """
    print("\n" + "=" * 60)
    print("Repairing Dataset")
    print("=" * 60)

    # Create backup if requested
    if create_backup:
        backup_path = dataset_path.parent / f"{dataset_path.name}_backup"
        if backup_path.exists():
            # Add timestamp to make unique
            import time
            timestamp = int(time.time())
            backup_path = dataset_path.parent / f"{dataset_path.name}_backup_{timestamp}"

        print(f"\n[INFO] Creating backup at {backup_path}")
        shutil.copytree(dataset_path, backup_path)
        print(f"[OK] Backup created")

    # Remove corrupted files
    for corrupted_file in results["corrupted_files"]:
        print(f"\n[INFO] Removing corrupted file: {corrupted_file}")
        corrupted_file.unlink()
        print(f"[OK] Removed {corrupted_file.name}")

    # Update info.json using helper
    info = results["info_json"]
    old_episodes = info.get("total_episodes", 0)
    old_frames = info.get("total_frames", 0)

    info["total_episodes"] = results["valid_episodes"]
    info["total_frames"] = results["valid_frames"]

    # Update splits to reflect new episode count
    if results["valid_episodes"] > 0:
        info["splits"] = {"train": f"0:{results['valid_episodes']}"}
    else:
        info["splits"] = {}

    print(f"\n[INFO] Updating info.json:")
    print(f"  total_episodes: {old_episodes} -> {results['valid_episodes']}")
    print(f"  total_frames: {old_frames} -> {results['valid_frames']}")

    save_info_json(dataset_path, info)
    print(f"[OK] Updated {dataset_path / 'meta' / 'info.json'}")

    print("\n" + "=" * 60)
    print("[OK] Dataset repair complete!")
    print(f"     Valid episodes: {results['valid_episodes']}")
    print(f"     Valid frames: {results['valid_frames']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Salvage corrupted LeRobot VLA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (show what would be done)
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes

    # Actually repair the dataset
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes --fix

    # Create backup before repair
    python scripts/salvage_vla_episodes.py --dataset data/lerobot_episodes --fix --backup
"""
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/lerobot_episodes",
        help="Path to LeRobot dataset directory (default: data/lerobot_episodes)"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Actually repair the dataset (without this flag, only shows what would be done)"
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before repairing (only used with --fix)"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"[X] Dataset not found: {dataset_path}")
        return 1

    print(f"[INFO] Scanning dataset: {dataset_path}")

    # Scan the dataset
    results = scan_dataset(dataset_path)

    # Print results
    print_scan_results(results)

    # Check if repair is needed
    if not results["corrupted_files"]:
        info = results["info_json"]
        if (info and
            info.get("total_episodes") == results["valid_episodes"] and
            info.get("total_frames") == results["valid_frames"]):
            print("\n[OK] Dataset appears healthy, no repair needed.")
            return 0

    # Repair if --fix flag is set
    if args.fix:
        if not results["corrupted_files"] and results["info_json"]:
            # No corrupted files but metadata mismatch - still need to fix metadata
            info = results["info_json"]
            if (info.get("total_episodes") != results["valid_episodes"] or
                info.get("total_frames") != results["valid_frames"]):
                print("\n[INFO] No corrupted files, but metadata needs updating...")
                repair_dataset(dataset_path, results, args.backup)
        else:
            repair_dataset(dataset_path, results, args.backup)
    else:
        print("\n[INFO] Dry run complete. Use --fix to actually repair the dataset.")
        if args.backup:
            print("       Use --backup with --fix to create a backup first.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
