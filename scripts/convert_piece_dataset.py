"""
Convert Existing Piece Dataset to Preprocessed Format

Applies grayscale + CLAHE preprocessing to existing piece detection training images.
This ensures the training data matches the preprocessing applied during inference.

IMPORTANT: Labels (bounding boxes) remain valid because preprocessing does not
change image dimensions - only the pixel values are modified.

Usage:
    python scripts/convert_piece_dataset.py --input data/training/piece_dataset --output data/training/piece_dataset_preprocessed
"""

import argparse
import sys
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_preprocessing import preprocess_dataset_for_pieces


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing piece dataset to preprocessed format (grayscale + CLAHE)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing original piece dataset (with images/ subdirectory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for preprocessed dataset"
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=3.0,
        help="CLAHE clip limit (default: 3.0)"
    )
    parser.add_argument(
        "--clahe-tile",
        type=int,
        default=8,
        help="CLAHE tile size (default: 8)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input
    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    data_yaml = input_path / "data.yaml"

    if not images_dir.exists():
        print(f"[X] Images directory not found: {images_dir}")
        return 1

    if not labels_dir.exists():
        print(f"[X] Labels directory not found: {labels_dir}")
        return 1

    print("=" * 60)
    print("PIECE DATASET CONVERSION")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"CLAHE clip limit: {args.clahe_clip}")
    print(f"CLAHE tile size:  {args.clahe_tile}")
    print("=" * 60)

    # Create output directory structure
    output_images_dir = output_path / "images"
    output_labels_dir = output_path / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess images
    print("\n[1/3] Preprocessing images with grayscale + CLAHE...")
    count = preprocess_dataset_for_pieces(
        images_dir,
        output_images_dir,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_size=args.clahe_tile
    )
    print(f"       [OK] Preprocessed {count} images")

    # Copy labels (unchanged - preprocessing doesn't affect bounding boxes)
    print("\n[2/3] Copying labels (unchanged)...")
    label_count = 0
    for label_file in labels_dir.glob("*.txt"):
        shutil.copy(label_file, output_labels_dir / label_file.name)
        label_count += 1
    print(f"       [OK] Copied {label_count} label files")

    # Copy data.yaml (update path)
    print("\n[3/3] Creating data.yaml...")
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update path to new location
        data_config['path'] = str(output_path.absolute())

        output_yaml = output_path / "data.yaml"
        with open(output_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print(f"       [OK] Created {output_yaml}")
    else:
        print(f"       [WARN] No data.yaml found in input, skipping")

    print("\n" + "=" * 60)
    print("[OK] Dataset conversion complete!")
    print("=" * 60)
    print(f"Images: {output_images_dir}")
    print(f"Labels: {output_labels_dir}")
    print(f"Config: {output_path / 'data.yaml'}")
    print("\nNext steps:")
    print(f"  1. Verify converted images look correct")
    print(f"  2. Fine-tune the model:")
    print(f"     python scripts/finetune_pieces.py --data {output_path}/data.yaml")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
