"""
YOLO Training Script for Chess Piece Detection

This script trains a YOLOv8 model for chess piece detection with
custom augmentation and hyperparameters optimized for piece classification.

Usage:
    python -m guidance.training.train_yolo --data dataset/data.yaml --epochs 150
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml


def create_data_yaml(dataset_dir: Path) -> Path:
    """
    Create YOLO data configuration file.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Path to created data.yaml file
    """
    data_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'images',
        'val': 'images',  # Will be split automatically by YOLO

        'names': {
            0: 'black_bishop',
            1: 'black_king',
            2: 'black_knight',
            3: 'black_pawn',
            4: 'black_queen',
            5: 'black_rook',
            6: 'white_bishop',
            7: 'white_king',
            8: 'white_knight',
            9: 'white_pawn',
            10: 'white_queen',
            11: 'white_rook'
        }
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"Created data config: {yaml_path}")
    return yaml_path


def train_model(data_yaml: Path, epochs: int = 150, batch_size: int = 16,
                img_size: int = 1280, model_size: str = 'm'):
    """
    Train YOLO model with optimized hyperparameters.

    Args:
        data_yaml: Path to data.yaml configuration
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size (default: 1280 to match 1280x720 camera resolution)
        model_size: YOLO model size (n, s, m, l, x)
    """
    print("\n" + "=" * 60)
    print("YOLO Chess Piece Detection Training")
    print("=" * 60)
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print("=" * 60 + "\n")

    # Load pretrained model
    model = YOLO(f'yolov8{model_size}.pt')

    # Training hyperparameters optimized for chess pieces
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,

        # Learning rate
        lr0=0.001,  # Lower initial learning rate for fine details
        lrf=0.001,  # Final learning rate

        # Augmentation parameters
        mosaic=1.0,  # Mosaic augmentation probability
        mixup=0.15,  # Mixup augmentation probability
        copy_paste=0.1,  # Copy-paste augmentation

        # Geometric augmentations
        degrees=15.0,  # Rotation range (±15 degrees)
        translate=0.1,  # Translation range
        scale=0.3,  # Scale variation (0.7-1.3x)
        shear=0.0,  # Shear (disabled for pieces)
        perspective=0.0,  # Perspective transform (disabled - already done)
        flipud=0.0,  # Vertical flip (disabled for chess pieces)
        fliplr=0.5,  # Horizontal flip (board can be flipped)

        # Color augmentations for handling different chess sets
        hsv_h=0.015,  # Hue variation
        hsv_s=0.7,  # Saturation variation (important for wood tints)
        hsv_v=0.4,  # Brightness variation

        # Loss weights (higher cls weight for classification accuracy)
        box=0.05,  # Box loss weight
        cls=0.5,  # Class loss weight (HIGHER for better classification)
        dfl=1.5,  # Distribution focal loss weight

        # Other parameters
        patience=50,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save every N epochs
        cache=False,  # Don't cache images (may use too much RAM)
        device=0,  # Use CUDA GPU (device 0)
        workers=4,  # Number of dataloader workers
        project='runs/train',  # Project directory
        name='chess_pieces',  # Experiment name
        exist_ok=True,  # Overwrite existing project
        pretrained=True,  # Use pretrained weights
        optimizer='AdamW',  # Optimizer (AdamW often better than SGD)
        verbose=True,  # Verbose output
        seed=42,  # Random seed for reproducibility

        # Validation
        val=True,  # Validate during training
        split=0.2,  # Train/val split (80/20)

        # Advanced options for piece detection
        close_mosaic=10,  # Disable mosaic in last N epochs for stability
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model saved to: {model.trainer.best}")
    print(f"Results saved to: {model.trainer.save_dir}")
    print("=" * 60)

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for chess piece detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to data.yaml (will be created if not exists)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/",
        help="Dataset directory (default: dataset/)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1280,
        help="Input image size (default: 1280 to match 1280x720 camera resolution)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='m',
        help="Model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: m)"
    )

    args = parser.parse_args()

    # Create or use existing data.yaml
    if args.data:
        data_yaml = Path(args.data)
    else:
        dataset_dir = Path(args.dataset)
        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}")
            print("Please run data_collector.py first to create a dataset")
            return 1
        data_yaml = create_data_yaml(dataset_dir)

    if not data_yaml.exists():
        print(f"Error: Data config not found: {data_yaml}")
        return 1

    # Train model
    train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_size=args.model
    )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
