"""
Fine-tune Corner Detection Model

Fine-tunes the existing CORNER DETECTION model (data/best_corners.pt) on your
specific chessboard setup.

NOTE: This is for CORNER detection only, not piece detection.
      For piece detection fine-tuning, use a separate script.

By default, applies grayscale + CLAHE preprocessing to match inference pipeline.
Use --rgb flag to train on original RGB images (no preprocessing).

Usage:
    # Default: grayscale + CLAHE preprocessing (matches existing pipeline)
    python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml

    # RGB mode: no preprocessing (for A/B testing)
    python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml --rgb
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import random
import shutil
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_preprocessing import preprocess_dataset_for_corners


def create_train_val_split(dataset_dir, val_fraction=0.0, use_rgb=False):
    """
    Create temporary train/val split directories.

    If use_rgb=False (default), preprocesses images with grayscale + CLAHE to match
    the inference pipeline. If use_rgb=True, uses original RGB images.

    Args:
        dataset_dir: Path to dataset directory containing images/ and labels/
        val_fraction: Fraction of data to use for validation (default: 0.0 = use full set for both)
        use_rgb: If True, use original RGB images; if False, apply grayscale + CLAHE preprocessing

    Returns:
        Tuple of (path to temporary data.yaml, num_train, num_val, temp_preprocessed_dir or None)
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    # Determine split strategy
    if val_fraction <= 0:
        # Use full dataset for both train and val (no holdout)
        train_files = image_files
        val_files = image_files
        num_train = len(image_files)
        num_val = len(image_files)

        print(f"\n[Dataset Split]")
        print(f"  Total images: {len(image_files)}")
        print(f"  Training: {num_train} images (100%)")
        print(f"  Validation: {num_val} images (100% - same as training)")
        print(f"  Mode: Full dataset for both train/val (no holdout)")
    else:
        # Random split with holdout
        num_val = max(1, int(len(image_files) * val_fraction))
        num_train = len(image_files) - num_val

        random.seed(42)  # For reproducibility
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)

        val_files = shuffled_files[:num_val]
        train_files = shuffled_files[num_val:]

        print(f"\n[Dataset Split]")
        print(f"  Total images: {len(image_files)}")
        print(f"  Training: {num_train} images ({100*(1-val_fraction):.0f}%)")
        print(f"  Validation: {num_val} images ({100*val_fraction:.0f}%)")
        print(f"  Random seed: 42")

    # Preprocessing step (unless RGB mode)
    temp_preprocessed_dir = None
    source_images_dir = images_dir  # Default: use original images

    if not use_rgb:
        # Apply grayscale + CLAHE preprocessing to temp directory
        temp_preprocessed_dir = dataset_path / "temp_preprocessed" / "images"

        # Clean up old preprocessed directory if exists
        if temp_preprocessed_dir.parent.exists():
            shutil.rmtree(temp_preprocessed_dir.parent)

        print(f"\n[Preprocessing] Applying grayscale + CLAHE to images...")
        num_preprocessed = preprocess_dataset_for_corners(images_dir, temp_preprocessed_dir)
        print(f"[Preprocessing] Preprocessed {num_preprocessed} images")

        # Use preprocessed images as source
        source_images_dir = temp_preprocessed_dir

        # Update image_files list to point to preprocessed images
        train_files = [temp_preprocessed_dir / f.name for f in train_files]
        val_files = [temp_preprocessed_dir / f.name for f in val_files]
    else:
        print(f"\n[RGB Mode] Using original RGB images (no preprocessing)")

    # Create temporary split directories
    temp_dir = dataset_path / "temp_split"
    train_img_dir = temp_dir / "train" / "images"
    train_lbl_dir = temp_dir / "train" / "labels"
    val_img_dir = temp_dir / "val" / "images"
    val_lbl_dir = temp_dir / "val" / "labels"

    # Clean up old temp directory if exists
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Create directories
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Symlink training files
    for img_file in train_files:
        # Labels are still in original labels_dir
        lbl_file = labels_dir / (img_file.stem + ".txt")

        (train_img_dir / img_file.name).symlink_to(img_file.absolute())
        if lbl_file.exists():
            (train_lbl_dir / lbl_file.name).symlink_to(lbl_file.absolute())

    # Symlink validation files
    for img_file in val_files:
        lbl_file = labels_dir / (img_file.stem + ".txt")

        # Skip if already exists (when using full dataset for both)
        val_img_link = val_img_dir / img_file.name
        val_lbl_link = val_lbl_dir / lbl_file.name
        if not val_img_link.exists():
            val_img_link.symlink_to(img_file.absolute())
        if lbl_file.exists() and not val_lbl_link.exists():
            val_lbl_link.symlink_to(lbl_file.absolute())

    # Create temporary data.yaml
    temp_data_yaml = {
        'path': str(temp_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['corner']
    }

    temp_yaml_path = temp_dir / "data.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(temp_data_yaml, f, default_flow_style=False)

    print(f"  Created temp split: {temp_dir}")
    if temp_preprocessed_dir:
        print(f"  Preprocessed images: {temp_preprocessed_dir}")
    print()

    return str(temp_yaml_path), num_train, num_val, temp_preprocessed_dir


def finetune_corner_model(data_yaml, base_model, output_dir, epochs=50, imgsz=1280,
                          val_fraction=0.0, lr=0.0001, patience=20, freeze=10, use_rgb=False):
    """
    Fine-tune corner detection model.

    Args:
        data_yaml: Path to dataset YAML config
        base_model: Path to base model to fine-tune from
        output_dir: Directory for training outputs
        epochs: Number of training epochs
        imgsz: Image size for training (default: 1280 to match 1280x720 camera resolution)
        val_fraction: Fraction of data for validation (default: 0.0 = use full set for both)
        lr: Learning rate (default: 0.0001 - low for fine-tuning)
        patience: Early stopping patience (default: 20)
        freeze: Number of layers to freeze (default: 10 - freeze backbone)
        use_rgb: If True, train on RGB images without grayscale + CLAHE preprocessing
    """
    preprocessing_mode = "RGB (no preprocessing)" if use_rgb else "Grayscale + CLAHE"

    print("=" * 60)
    print("CORNER DETECTION MODEL FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Dataset: {data_yaml}")
    print(f"Preprocessing: {preprocessing_mode}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Learning rate: {lr}")
    print(f"Validation fraction: {val_fraction*100:.0f}%")
    print(f"Early stopping patience: {patience}")
    print(f"Frozen layers: {freeze}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    # Create train/val split (with optional preprocessing)
    dataset_dir = Path(data_yaml).parent
    temp_yaml, num_train, num_val, temp_preprocessed_dir = create_train_val_split(
        dataset_dir, val_fraction=val_fraction, use_rgb=use_rgb
    )

    # Track temp directories for cleanup
    temp_dir = Path(temp_yaml).parent

    # Check image dimensions to ensure no unintentional resizing
    images_dir = dataset_dir / "images"
    sample_images = list(images_dir.glob("*.png"))[:3]
    if sample_images:
        from PIL import Image as PILImage
        sample_img = PILImage.open(sample_images[0])
        img_width, img_height = sample_img.size
        print(f"[Image Check] Training data resolution: {img_width}x{img_height}")
        print(f"[Image Check] Training imgsz: {imgsz}")

        # Warn if there's a mismatch
        if img_width != imgsz and img_height != imgsz:
            print(f"[WARNING] Training resolution ({img_width}x{img_height}) != imgsz ({imgsz})")
            print(f"[WARNING] YOLO will resize/pad images - may cause aspect ratio issues")
            print(f"[WARNING] Recommended: Use imgsz matching your image dimensions")
        print()

    try:
        # Load base model
        print("Loading base model...")
        model = YOLO(base_model)
        print("[OK] Model loaded")
        print()

        # Fine-tune
        print("Starting fine-tuning...")
        print("This may take 10-30 minutes depending on dataset size and hardware")
        print()
        print("Training configuration:")
        print(f"  - Train/val split: {num_train}/{num_val} images ({100*(1-val_fraction):.0f}%/{100*val_fraction:.0f}%)")
        print(f"  - Frozen backbone layers: {freeze}")
        print(f"  - Learning rate: {lr} (low for fine-tuning)")
        print(f"  - Early stopping patience: {patience} epochs")
        print()
        # Set augmentation parameters based on preprocessing mode
        if use_rgb:
            # RGB mode: enable color augmentation
            hsv_h = 0.015   # Hue variation (color augmentation)
            hsv_s = 0.4     # Saturation variation
            hsv_v = 0.35    # Brightness variation
            print("Augmentations enabled (RGB mode):")
            print("  - Color augmentation: hue 1.5%, saturation 40%, brightness 35%")
        else:
            # Grayscale mode: disable hue/saturation (no color in grayscale)
            hsv_h = 0.0     # Disabled - no hue in grayscale images
            hsv_s = 0.0     # Disabled - no saturation in grayscale images
            hsv_v = 0.35    # Brightness variation only
            print("Augmentations enabled (Grayscale mode):")
            print("  - Brightness variation: 35% (no color augmentation)")

        print("  - Geometric: rotation 5deg, translation 5%, scale 10%")
        print(f"  - Horizontal flip: 50% (effectively ~{num_train * 2} augmented samples)")
        print()

        results = model.train(
            data=temp_yaml,  # Use temp split YAML
            epochs=epochs,
            imgsz=imgsz,
            project=output_dir,
            name="corner_finetune",
            exist_ok=True,
            patience=patience,  # Early stopping patience (configurable)
            save=True,
            plots=True,
            device='cuda',  # Use CUDA GPU (auto-selects available device)
            batch=16,  # Larger batch for GPU
            optimizer='AdamW',  # AdamW has better weight decay handling
            lr0=lr,  # Learning rate (configurable, default low for fine-tuning)
            lrf=0.01,
            weight_decay=0.001,  # Slightly higher for regularization
            warmup_epochs=5,  # Longer warmup for stability
            amp=True,  # Enable AMP for GPU acceleration
            seed=42,  # Reproducibility per project conventions
            freeze=freeze,  # Freeze backbone layers for fine-tuning

            # Lighting/color augmentation (mode-dependent)
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,

            # Additional augmentations
            degrees=5.0,      # Small rotation (board may not be perfectly aligned)
            translate=0.05,   # Small translation
            scale=0.1,        # Small scale variation
            fliplr=0.5,       # 50% horizontal flip (doubles effective dataset size)
            flipud=0.0,       # No vertical flip (board is always upright)
            mosaic=0.0,       # No mosaic (not useful for corner detection)
        )

        print("\n" + "=" * 60)
        print("[OK] Fine-tuning complete!")
        print("=" * 60)

        # Find best model
        best_model_path = Path(output_dir) / "corner_finetune" / "weights" / "best.pt"

        if best_model_path.exists():
            print(f"Best model: {best_model_path}")
            print()
            print("To use the fine-tuned model:")
            print(f"  1. Backup original: mv data/best_corners.pt data/best_corners_original.pt")
            print(f"  2. Copy fine-tuned: cp {best_model_path} data/best_corners.pt")
            print(f"  3. Test: python scripts/best_move_demo.py --debug")
        else:
            print("[X] Warning: Could not find best model weights")

        print("=" * 60)

        return results

    finally:
        # Clean up temporary directories
        if temp_dir.exists():
            print(f"\n[Cleanup] Removing temporary split directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        if temp_preprocessed_dir and temp_preprocessed_dir.parent.exists():
            print(f"[Cleanup] Removing preprocessed images: {temp_preprocessed_dir.parent}")
            shutil.rmtree(temp_preprocessed_dir.parent)

        print("[OK] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune corner detection model on your chessboard"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML config (e.g., data/training/corner_dataset/data.yaml)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="data/best_corners.pt",
        help="Base model to fine-tune from (default: data/best_corners.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/runs",
        help="Output directory for training results (default: data/training/runs)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for training (default: 1280 to match 1280x720 camera resolution)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="Fraction of data for validation (default: 0.0 = use full set for both train/val)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001 - low for fine-tuning to prevent overfitting)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Number of backbone layers to freeze (default: 10 - freeze backbone, train detection head)"
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Train on RGB images without grayscale + CLAHE preprocessing (for A/B testing)"
    )

    args = parser.parse_args()

    # Validate inputs
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[X] Dataset YAML not found: {data_yaml}")
        print("\nPlease create dataset first:")
        print("  python scripts/label_corners.py --input data/training/board_photos")
        return 1

    base_model = Path(args.base_model)
    if not base_model.exists():
        print(f"[X] Base model not found: {base_model}")
        print("\nPlease download models first:")
        print("  python scripts/download.py")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fine-tune
    try:
        results = finetune_corner_model(
            data_yaml=str(data_yaml),
            base_model=str(base_model),
            output_dir=str(output_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            val_fraction=args.val_fraction,
            lr=args.lr,
            patience=args.patience,
            freeze=args.freeze,
            use_rgb=args.rgb
        )
        return 0
    except Exception as e:
        print(f"\n[X] Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
