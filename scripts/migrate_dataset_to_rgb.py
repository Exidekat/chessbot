#!/usr/bin/env python3
"""
Migrate Labeled Dataset to RGB Mode

Replaces grayscale+CLAHE preprocessed images in labeled datasets with original RGB
captures while preserving labels (coordinates are unchanged since no resizing occurred).

This enables A/B testing between grayscale and RGB models using existing labeled data.

Expected directory structure:
    data/training/
      board_photos/           <- Original RGB captures (source)
      corner_dataset/
        images/               <- Currently grayscale, will be replaced with RGB
        labels/               <- Unchanged (coordinates still valid)
      piece_dataset/
        images/               <- Currently grayscale, will be replaced with RGB
        labels/               <- Unchanged

Usage:
    # Migrate corner dataset
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/corner_dataset

    # Migrate piece dataset with backup
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/piece_dataset \\
        --backup

    # Dry run (show what would be done without making changes)
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/corner_dataset \\
        --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2


def find_matching_source(image_stem: str, source_dir: Path) -> Path | None:
    """
    Find matching source image by filename stem.

    Args:
        image_stem: Filename stem (without extension) to match
        source_dir: Directory containing original RGB images

    Returns:
        Path to matching source image, or None if not found
    """
    # Common image extensions
    extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    for ext in extensions:
        candidate = source_dir / (image_stem + ext)
        if candidate.exists():
            return candidate

    return None


def verify_dimensions(source_path: Path, dataset_path: Path) -> tuple[bool, str]:
    """
    Verify that source and dataset images have matching dimensions.

    Args:
        source_path: Path to original RGB image
        dataset_path: Path to preprocessed image in dataset

    Returns:
        Tuple of (dimensions_match, message)
    """
    source_img = cv2.imread(str(source_path))
    dataset_img = cv2.imread(str(dataset_path))

    if source_img is None:
        return False, f"Failed to read source: {source_path}"
    if dataset_img is None:
        return False, f"Failed to read dataset: {dataset_path}"

    source_shape = source_img.shape[:2]  # (height, width)
    dataset_shape = dataset_img.shape[:2]

    if source_shape == dataset_shape:
        return True, f"{source_shape[1]}x{source_shape[0]}"
    else:
        return False, f"Dimension mismatch: source={source_shape}, dataset={dataset_shape}"


def migrate_dataset(
    source_dir: Path,
    dataset_dir: Path,
    backup: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Migrate dataset images from grayscale+CLAHE to original RGB.

    Args:
        source_dir: Directory containing original RGB images
        dataset_dir: Directory containing labeled dataset (with images/ subdirectory)
        backup: If True, backup grayscale images before replacing
        dry_run: If True, don't make changes, just report what would be done

    Returns:
        Dictionary with migration statistics
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists():
        raise ValueError(f"Dataset images directory not found: {images_dir}")

    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")

    # Create backup directory if needed
    backup_dir = None
    if backup and not dry_run:
        backup_dir = dataset_dir / "images_grayscale_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Backup directory: {backup_dir}")

    # Get all images in dataset
    image_extensions = [".png", ".jpg", ".jpeg"]
    dataset_images = []
    for ext in image_extensions:
        dataset_images.extend(images_dir.glob(f"*{ext}"))
        dataset_images.extend(images_dir.glob(f"*{ext.upper()}"))

    # Statistics
    stats = {
        "total": len(dataset_images),
        "replaced": 0,
        "missing_source": 0,
        "dimension_mismatch": 0,
        "errors": [],
    }

    print(f"\n[INFO] Found {len(dataset_images)} images in dataset")
    print(f"[INFO] Source directory: {source_dir}")
    print()

    for dataset_image in sorted(dataset_images):
        stem = dataset_image.stem

        # Find matching source
        source_image = find_matching_source(stem, source_dir)

        if source_image is None:
            print(f"  [MISSING] {dataset_image.name} - no matching source found")
            stats["missing_source"] += 1
            stats["errors"].append(f"Missing source: {stem}")
            continue

        # Verify dimensions match
        dims_ok, dims_msg = verify_dimensions(source_image, dataset_image)

        if not dims_ok:
            print(f"  [MISMATCH] {dataset_image.name} - {dims_msg}")
            stats["dimension_mismatch"] += 1
            stats["errors"].append(f"Dimension mismatch: {stem} - {dims_msg}")
            continue

        # Perform migration
        if dry_run:
            print(f"  [DRY-RUN] {dataset_image.name} <- {source_image.name} ({dims_msg})")
        else:
            # Backup if requested
            if backup_dir:
                backup_path = backup_dir / dataset_image.name
                shutil.copy2(dataset_image, backup_path)

            # Replace with RGB source
            shutil.copy2(source_image, dataset_image)
            print(f"  [OK] {dataset_image.name} <- {source_image.name} ({dims_msg})")

        stats["replaced"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate labeled dataset from grayscale+CLAHE to original RGB images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate corner dataset
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/corner_dataset

    # Migrate with backup
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/piece_dataset \\
        --backup

    # Dry run (show what would happen)
    python scripts/migrate_dataset_to_rgb.py \\
        --source data/training/board_photos \\
        --dataset data/training/corner_dataset \\
        --dry-run
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory containing original RGB board photos"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Directory containing labeled dataset (with images/ and labels/ subdirectories)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup grayscale images before replacing (saves to images_grayscale_backup/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    dataset_dir = Path(args.dataset)

    print("=" * 60)
    print("DATASET MIGRATION: Grayscale+CLAHE -> RGB")
    print("=" * 60)
    print(f"Source (RGB originals): {source_dir}")
    print(f"Dataset to migrate: {dataset_dir}")
    print(f"Backup: {'Yes' if args.backup else 'No'}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")
    print("=" * 60)

    try:
        stats = migrate_dataset(
            source_dir=source_dir,
            dataset_dir=dataset_dir,
            backup=args.backup,
            dry_run=args.dry_run,
        )

        print()
        print("=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Total images in dataset: {stats['total']}")
        print(f"Successfully {'would replace' if args.dry_run else 'replaced'}: {stats['replaced']}")
        print(f"Missing source images: {stats['missing_source']}")
        print(f"Dimension mismatches: {stats['dimension_mismatch']}")

        if stats["errors"]:
            print()
            print("Errors:")
            for error in stats["errors"]:
                print(f"  - {error}")

        print("=" * 60)

        if args.dry_run:
            print("\n[INFO] This was a dry run. No changes were made.")
            print("       Remove --dry-run to perform actual migration.")

        # Return non-zero if there were issues
        if stats["missing_source"] > 0 or stats["dimension_mismatch"] > 0:
            print("\n[WARN] Some images could not be migrated. Check errors above.")
            return 1

        return 0

    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
