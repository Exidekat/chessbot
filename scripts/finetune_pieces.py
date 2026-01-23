"""
Fine-tune Piece Detection Model

Fine-tunes the existing PIECE DETECTION model (data/best_transformed_detection.pt)
on your specific chess piece set.

NOTE: This is for PIECE detection only, not corner detection.
      For corner detection fine-tuning, use finetune_corners.py.

CLASSIFICATION PRIORITY: This script saves models based on LOWEST cls_loss,
prioritizing correct piece classification over bbox accuracy.

By default, applies grayscale + CLAHE preprocessing to match inference pipeline.
Use --rgb flag to train on original RGB images (no preprocessing).

Usage:
    # Default: grayscale + CLAHE preprocessing (matches existing pipeline)
    python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml

    # RGB mode: no preprocessing (for A/B testing)
    python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml --rgb
"""

import argparse
import sys
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_preprocessing import preprocess_dataset_for_pieces


class BestClsLossTracker:
    """
    Custom callback tracker to save model with lowest classification loss.

    YOLO's default 'best.pt' is based on mAP (detection quality).
    For chess piece classification, we prioritize lowest cls_loss
    to ensure correct piece identification even at cost of bbox accuracy.
    """

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.best_cls_loss = float('inf')
        self.best_epoch = -1
        self.best_model_path = self.output_dir / "piece_finetune" / "weights" / "best_cls.pt"
        self.history = []  # Track (epoch, cls_loss, mAP50-95) for analysis

    def on_fit_epoch_end(self, trainer):
        """Called at end of each epoch - check if this is best cls_loss."""
        epoch = trainer.epoch

        # Get classification loss from trainer metrics
        # trainer.loss_items contains [box_loss, cls_loss, dfl_loss]
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            cls_loss = float(trainer.loss_items[1])  # Index 1 is cls_loss
        else:
            return

        # Get mAP for logging
        metrics = trainer.metrics
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)

        self.history.append((epoch, cls_loss, map50_95))

        if cls_loss < self.best_cls_loss:
            self.best_cls_loss = cls_loss
            self.best_epoch = epoch

            # Save current model as best_cls.pt
            last_model = self.output_dir / "piece_finetune" / "weights" / "last.pt"
            if last_model.exists():
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(last_model, self.best_model_path)
                print(f"\n[BEST CLS] Epoch {epoch}: cls_loss={cls_loss:.4f} (mAP50-95={map50_95:.3f}) -> Saved best_cls.pt")

    def on_train_end(self, trainer):
        """Called when training ends - print summary."""
        print("\n" + "=" * 60)
        print("CLASSIFICATION LOSS TRACKING SUMMARY")
        print("=" * 60)

        if self.best_epoch >= 0:
            print(f"Best cls_loss: {self.best_cls_loss:.4f} at epoch {self.best_epoch}")
            print(f"Saved to: {self.best_model_path}")

            # Find the epoch with best mAP for comparison
            if self.history:
                best_map_entry = max(self.history, key=lambda x: x[2])
                print(f"\nComparison:")
                print(f"  Best cls_loss: epoch {self.best_epoch}, cls={self.best_cls_loss:.4f}")
                print(f"  Best mAP50-95: epoch {best_map_entry[0]}, mAP={best_map_entry[2]:.3f}, cls={best_map_entry[1]:.4f}")
        else:
            print("[Warning] No cls_loss tracking data available")

        print("=" * 60)


def create_preprocessed_dataset(dataset_dir, use_rgb=False):
    """
    Create preprocessed dataset if needed.

    If use_rgb=False (default), preprocesses images with grayscale + CLAHE to match
    the inference pipeline. If use_rgb=True, uses original RGB images.

    Args:
        dataset_dir: Path to dataset directory containing images/ and labels/
        use_rgb: If True, use original RGB images; if False, apply grayscale + CLAHE preprocessing

    Returns:
        Tuple of (path to images dir to use, temp_preprocessed_dir or None)
    """
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"

    temp_preprocessed_dir = None

    if not use_rgb:
        # Apply grayscale + CLAHE preprocessing to temp directory
        temp_preprocessed_dir = dataset_path / "temp_preprocessed" / "images"

        # Clean up old preprocessed directory if exists
        if temp_preprocessed_dir.parent.exists():
            shutil.rmtree(temp_preprocessed_dir.parent)

        print(f"\n[Preprocessing] Applying grayscale + CLAHE to images...")
        num_preprocessed = preprocess_dataset_for_pieces(images_dir, temp_preprocessed_dir)
        print(f"[Preprocessing] Preprocessed {num_preprocessed} images")

        # Create a temporary data.yaml pointing to preprocessed images
        temp_data_yaml = temp_preprocessed_dir.parent / "data.yaml"

        # Copy labels to temp directory
        temp_labels_dir = temp_preprocessed_dir.parent / "labels"
        temp_labels_dir.mkdir(parents=True, exist_ok=True)

        labels_dir = dataset_path / "labels"
        for label_file in labels_dir.glob("*.txt"):
            shutil.copy(label_file, temp_labels_dir / label_file.name)

        # Read original data.yaml to get class names
        original_yaml = dataset_path / "data.yaml"
        with open(original_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Create temp data.yaml
        temp_config = {
            'path': str(temp_preprocessed_dir.parent.absolute()),
            'train': 'images',
            'val': 'images',
            'nc': data_config.get('nc', 12),
            'names': data_config.get('names', [])
        }

        with open(temp_data_yaml, 'w') as f:
            yaml.dump(temp_config, f, default_flow_style=False)

        return str(temp_data_yaml), temp_preprocessed_dir.parent
    else:
        print(f"\n[RGB Mode] Using original RGB images (no preprocessing)")
        return str(dataset_path / "data.yaml"), None


def finetune_piece_model(data_yaml, base_model, output_dir, epochs=100, imgsz=1280, use_rgb=False):
    """
    Fine-tune piece detection model.

    Args:
        data_yaml: Path to dataset YAML config
        base_model: Path to base model to fine-tune from
        output_dir: Directory for training outputs
        epochs: Number of training epochs
        imgsz: Image size for training (default: 1280 to match 1280x720 camera resolution)
        use_rgb: If True, train on RGB images without grayscale + CLAHE preprocessing
    """
    preprocessing_mode = "RGB (no preprocessing)" if use_rgb else "Grayscale + CLAHE"

    print("=" * 60)
    print("PIECE DETECTION MODEL FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Dataset: {data_yaml}")
    print(f"Preprocessing: {preprocessing_mode}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    # Apply preprocessing if needed
    dataset_dir = Path(data_yaml).parent
    train_data_yaml, temp_preprocessed_dir = create_preprocessed_dataset(dataset_dir, use_rgb=use_rgb)

    try:
        # Load base model
        print("Loading base model...")
        model = YOLO(base_model)
        print("[OK] Model loaded")
        print()

        # Register custom callback for cls_loss tracking
        cls_tracker = BestClsLossTracker(output_dir)
        model.add_callback("on_fit_epoch_end", cls_tracker.on_fit_epoch_end)
        model.add_callback("on_train_end", cls_tracker.on_train_end)
        print("[OK] Registered cls_loss tracker (saves best_cls.pt)")
        print()

        # Fine-tune
        print("Starting fine-tuning...")
        print("This may take 30-60 minutes depending on dataset size and hardware")
        print("(Piece detection requires more epochs than corner detection)")
        print()

        # Set augmentation parameters based on preprocessing mode
        if use_rgb:
            # RGB mode: enable color augmentation
            hsv_h = 0.015   # Hue variation (color augmentation)
            hsv_s = 0.4     # Saturation variation
            hsv_v = 0.7     # Strong brightness variation
            print("CONFIGURATION (RGB mode):")
            print("  - Color augmentation: hue 1.5%, saturation 40%, brightness 70%")
        else:
            # Grayscale mode: disable hue/saturation (no color in grayscale)
            hsv_h = 0.0     # Disabled - no hue in grayscale images
            hsv_s = 0.0     # Disabled - no saturation in grayscale images
            hsv_v = 0.7     # Strong brightness variation
            print("CONFIGURATION (Grayscale mode):")
            print("  - Brightness augmentation: 70% (no color augmentation)")

        print("  - Classification priority (cls=1.0, box=0.5)")
        print("  - Maximized augmentation (mosaic=1.0, mixup=0.2)")
        print("  - More rotation variation (degrees=15.0)")
        print("  - Lower final LR (lrf=0.001)")
        print("  - Image size 1280 (matches 1280x720 training images)")
        print("  - GPU training batch=8 (1280 uses 4x memory vs 640)")
        print()

        results = model.train(
            data=train_data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="piece_finetune",
        exist_ok=True,
        patience=30,  # Increased from 15 - allow more convergence
        save=True,
        plots=True,
        device=0,  # RTX A5500 GPU
        batch=8,  # REDUCED from 32 - 1280x1280 uses 4x memory vs 640x640
        optimizer='Adam',
        lr0=0.001,  # Lower learning rate for fine-tuning
        lrf=0.001,  # FIXED: was 0.01 - better convergence
        weight_decay=0.0005,
        warmup_epochs=5,
        amp=True,  # Enable AMP for GPU acceleration

        # CRITICAL: Loss weights - CONSERVATIVE for small dataset
        box=0.5,    # Moderate (was 0.05 - too low caused instability)
        cls=1.0,    # Moderate (was 2.5 - too high caused cls_loss explosion)
        dfl=1.0,    # Standard

        # Data augmentation - MAXIMIZED for small dataset
        # Augmentation settings depend on preprocessing mode (RGB vs grayscale)
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=15.0,  # INCREASED from 5.0 - more rotation variation
        translate=0.15,  # Increased from 0.1
        scale=0.3,    # Increased from 0.2
        shear=0.0,
        perspective=0.0,
        flipud=0.0,   # Don't flip vertically (would swap board up/down)
        fliplr=0.5,   # Horizontal flip 50% (doubles dataset, maintains left/right symmetry)
        mosaic=1.0,   # FIXED: was 0.5 - ALWAYS use mosaic for small dataset
        mixup=0.2,    # NEW: Enable mixup augmentation
        copy_paste=0.1,  # NEW: Enable copy-paste augmentation
        close_mosaic=10,  # Disable mosaic in last 10 epochs for stability
    )

        print("\n" + "=" * 60)
        print("[OK] Fine-tuning complete!")
        print("=" * 60)

        # Find best models
        weights_dir = Path(output_dir) / "piece_finetune" / "weights"
        best_map_path = weights_dir / "best.pt"      # YOLO's default (best mAP)
        best_cls_path = weights_dir / "best_cls.pt"  # Our custom (best cls_loss)

        print("\nSaved models:")
        if best_map_path.exists():
            print(f"  best.pt     - Best mAP50-95 (detection quality)")
        if best_cls_path.exists():
            print(f"  best_cls.pt - Best cls_loss (classification accuracy) [RECOMMENDED]")

        print()
        print("To use the fine-tuned model (choose ONE):")
        print()
        print("  Option A - Best CLASSIFICATION (recommended for chess):")
        print(f"    cp {best_cls_path} data/best_transformed_detection.pt")
        print()
        print("  Option B - Best DETECTION (if missing pieces is the issue):")
        print(f"    cp {best_map_path} data/best_transformed_detection.pt")
        print()
        print("  Then test: python scripts/best_move_demo.py --debug")
        print()
        print("  To analyze misclassifications:")
        print("    python scripts/validate_piece_labels_with_yolo.py --model best_cls.pt")

        print("=" * 60)

        return results

    finally:
        # Clean up temporary preprocessed directory
        if temp_preprocessed_dir and temp_preprocessed_dir.exists():
            print(f"\n[Cleanup] Removing preprocessed images: {temp_preprocessed_dir}")
            shutil.rmtree(temp_preprocessed_dir)
            print("[OK] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune piece detection model on your chess pieces"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML config (e.g., data/training/piece_dataset/data.yaml)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="data/best_transformed_detection.pt",
        help="Base model to fine-tune from (default: data/best_transformed_detection.pt)"
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
        default=150,
        help="Number of training epochs (default: 150, increased from 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Image size for training (default: 1280 to match 1280x720 images)"
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
        print("  1. Collect photos: python scripts/collect_piece_training_photos.py --device /dev/video0 --count 30")
        print("  2. Label pieces: python scripts/label_pieces.py --input data/training/piece_photos")
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
        results = finetune_piece_model(
            data_yaml=str(data_yaml),
            base_model=str(base_model),
            output_dir=str(output_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
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
