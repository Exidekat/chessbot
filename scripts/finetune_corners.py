"""
Fine-tune Corner Detection Model

Fine-tunes the existing CORNER DETECTION model (data/best_corners.pt) on your
specific chessboard setup.

NOTE: This is for CORNER detection only, not piece detection.
      For piece detection fine-tuning, use a separate script.

Usage:
    python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def finetune_corner_model(data_yaml, base_model, output_dir, epochs=50, imgsz=640):
    """
    Fine-tune corner detection model.

    Args:
        data_yaml: Path to dataset YAML config
        base_model: Path to base model to fine-tune from
        output_dir: Directory for training outputs
        epochs: Number of training epochs
        imgsz: Image size for training
    """
    print("=" * 60)
    print("CORNER DETECTION MODEL FINE-TUNING")
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
    print("This may take 10-30 minutes depending on dataset size and hardware")
    print()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="corner_finetune",
        exist_ok=True,
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
        device='cpu',  # Use CPU (change to 0 for GPU if available)
        batch=8,  # Small batch for CPU
        optimizer='Adam',
        lr0=0.001,  # Lower learning rate for fine-tuning
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        amp=False,  # Disable AMP for CPU
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
        default=640,
        help="Image size for training (default: 640)"
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
