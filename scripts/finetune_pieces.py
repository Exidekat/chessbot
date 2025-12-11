"""
Fine-tune Piece Detection Model

Fine-tunes the existing PIECE DETECTION model (data/best_transformed_detection.pt)
on your specific chess piece set.

NOTE: This is for PIECE detection only, not corner detection.
      For corner detection fine-tuning, use finetune_corners.py.

Usage:
    python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def finetune_piece_model(data_yaml, base_model, output_dir, epochs=100, imgsz=640):
    """
    Fine-tune piece detection model.

    Args:
        data_yaml: Path to dataset YAML config
        base_model: Path to base model to fine-tune from
        output_dir: Directory for training outputs
        epochs: Number of training epochs
        imgsz: Image size for training
    """
    print("=" * 60)
    print("PIECE DETECTION MODEL FINE-TUNING")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()

    # Load base model
    print("Loading base model...")
    model = YOLO(base_model)
    print("[OK] Model loaded")
    print()

    # Fine-tune
    print("Starting fine-tuning...")
    print("This may take 30-60 minutes depending on dataset size and hardware")
    print("(Piece detection requires more epochs than corner detection)")
    print()
    print("Data augmentation enabled:")
    print("  - Horizontal flip: 50% (doubles effective dataset)")
    print("  - Color variation: HSV adjustments")
    print("  - Geometric: Small rotation, translation, scale")
    print()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="piece_finetune",
        exist_ok=True,
        patience=15,  # Early stopping patience (more than corner detection)
        save=True,
        plots=True,
        device='cpu',  # Use CPU (change to 0 for GPU if available)
        batch=8,  # Small batch for CPU
        optimizer='Adam',
        lr0=0.001,  # Lower learning rate for fine-tuning
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        amp=False,  # Disable AMP for CPU
        # Data augmentation (important for piece detection)
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value augmentation
        degrees=5.0,  # Small rotation (boards are mostly aligned)
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,   # Don't flip vertically (would swap board up/down)
        fliplr=0.5,   # Horizontal flip 50% (doubles dataset, maintains left/right symmetry)
        mosaic=0.5,   # Mosaic augmentation probability
    )

    print("\n" + "=" * 60)
    print("[OK] Fine-tuning complete!")
    print("=" * 60)

    # Find best model
    best_model_path = Path(output_dir) / "piece_finetune" / "weights" / "best.pt"

    if best_model_path.exists():
        print(f"Best model: {best_model_path}")
        print()
        print("To use the fine-tuned model:")
        print(f"  1. Backup original: mv data/best_transformed_detection.pt data/best_transformed_detection_original.pt")
        print(f"  2. Copy fine-tuned: cp {best_model_path} data/best_transformed_detection.pt")
        print(f"  3. Test: python scripts/best_move_demo.py --debug")
    else:
        print("[X] Warning: Could not find best model weights")

    print("=" * 60)

    return results


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
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)"
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
            imgsz=args.imgsz
        )
        return 0
    except Exception as e:
        print(f"\n[X] Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
