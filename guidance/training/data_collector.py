"""
Data Collection Tool for Chess Piece Detection Training

This tool helps collect and label chess piece images for YOLO training.
It can extract pieces from board images and prepare them in YOLO format.

Usage:
    python -m guidance.training.data_collector --source images/ --output dataset/
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
import json
from typing import List, Tuple, Dict
import sys

from guidance.board_detector import BoardDetector


class ChessPieceDataCollector:
    """Collects and labels chess piece images for training."""

    def __init__(self, output_dir: Path):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Piece class mapping (same as BoardDetector)
        self.piece_classes = {
            'b': 0,  'k': 1,  'n': 2,  'p': 3,  'q': 4,  'r': 5,   # black
            'B': 6,  'K': 7,  'N': 8,  'P': 9,  'Q': 10, 'R': 11   # white
        }

        # Statistics
        self.stats = {cls: 0 for cls in self.piece_classes.keys()}
        self.total_images = 0

    def extract_pieces_from_board(self, board_image_path: str,
                                 detector: BoardDetector = None) -> List[Dict]:
        """
        Extract individual piece crops from a chess board image.

        Args:
            board_image_path: Path to board image
            detector: BoardDetector instance (optional)

        Returns:
            List of piece dictionaries with crop info
        """
        if detector is None:
            detector = BoardDetector()

        try:
            # Detect corners and transform
            corners = detector.detect_corners(board_image_path, conf_threshold=0.1)
            transformed = detector.four_point_transform(board_image_path, corners)

            # Detect pieces
            detections, boxes = detector.detect_pieces(transformed, use_preprocessing=False)

            if len(detections) == 0:
                print(f"  No pieces detected in {board_image_path}")
                return []

            # Convert transformed image to numpy for cropping
            img_array = np.array(transformed)
            img_height, img_width = img_array.shape[:2]

            pieces = []
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = map(int, detection)
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # Get piece class name
                piece_name = None
                for name, class_id in self.piece_classes.items():
                    if class_id == cls_id:
                        piece_name = name
                        break

                if piece_name is None:
                    continue

                # Calculate YOLO format (normalized center x, y, width, height)
                center_x = ((x1 + x2) / 2) / img_width
                center_y = ((y1 + y2) / 2) / img_height
                box_width = (x2 - x1) / img_width
                box_height = (y2 - y1) / img_height

                pieces.append({
                    'class_id': cls_id,
                    'class_name': piece_name,
                    'bbox': [center_x, center_y, box_width, box_height],
                    'confidence': conf,
                    'crop': img_array[y1:y2, x1:x2]
                })

            return pieces

        except Exception as e:
            print(f"  Error processing {board_image_path}: {e}")
            return []

    def save_yolo_format(self, image_array: np.ndarray, pieces: List[Dict],
                         filename_prefix: str):
        """
        Save image and labels in YOLO format.

        Args:
            image_array: Full board image as numpy array (RGB)
            pieces: List of piece dictionaries
            filename_prefix: Prefix for filenames
        """
        # Save image
        img_path = self.images_dir / f"{filename_prefix}.jpg"
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), img_bgr)

        # Save labels
        label_path = self.labels_dir / f"{filename_prefix}.txt"
        with open(label_path, 'w') as f:
            for piece in pieces:
                cls_id = piece['class_id']
                cx, cy, w, h = piece['bbox']
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                self.stats[piece['class_name']] += 1

        self.total_images += 1

    def process_directory(self, source_dir: Path, min_confidence: float = 0.5):
        """
        Process all images in a directory.

        Args:
            source_dir: Directory containing board images
            min_confidence: Minimum detection confidence to include
        """
        image_files = list(source_dir.glob("*.png")) + \
                     list(source_dir.glob("*.jpg")) + \
                     list(source_dir.glob("*.jpeg"))

        print(f"\nFound {len(image_files)} images in {source_dir}")

        detector = BoardDetector()

        for img_idx, img_path in enumerate(image_files):
            print(f"\nProcessing [{img_idx + 1}/{len(image_files)}]: {img_path.name}")

            pieces = self.extract_pieces_from_board(str(img_path), detector)

            # Filter by confidence
            pieces = [p for p in pieces if p['confidence'] >= min_confidence]

            if len(pieces) == 0:
                continue

            print(f"  Found {len(pieces)} pieces (conf >= {min_confidence})")

            # Save in YOLO format
            transformed = detector.four_point_transform(str(img_path),
                                                       detector.detect_corners(str(img_path)))
            self.save_yolo_format(np.array(transformed), pieces,
                                f"board_{img_idx:04d}")

    def print_stats(self):
        """Print collection statistics."""
        print("\n" + "=" * 60)
        print("Data Collection Statistics")
        print("=" * 60)
        print(f"Total images: {self.total_images}")
        print(f"\nPiece counts:")
        for piece_name in sorted(self.stats.keys()):
            count = self.stats[piece_name]
            print(f"  {piece_name}: {count}")
        print(f"\nTotal pieces: {sum(self.stats.values())}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect chess piece data for YOLO training"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing board images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/",
        help="Output directory for collected data (default: dataset/)"
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    print("=" * 60)
    print("Chess Piece Data Collector")
    print("=" * 60)

    collector = ChessPieceDataCollector(Path(args.output))
    collector.process_directory(source_dir, min_confidence=args.min_conf)
    collector.print_stats()

    return 0


if __name__ == "__main__":
    sys.exit(main())
