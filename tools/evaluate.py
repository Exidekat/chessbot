"""
Evaluation Framework for Chess Piece Detection

This script evaluates YOLO model performance with detailed metrics:
- Per-class precision/recall
- Confusion matrix
- Detection accuracy
- Common misclassification patterns

Usage:
    python tools/evaluate.py --model runs/train/chess_pieces/weights/best.pt --data dataset/
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
from typing import List, Dict, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class ChessPieceEvaluator:
    """Evaluate chess piece detection model performance."""

    def __init__(self, model_path: str):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained YOLO model
        """
        self.model = YOLO(model_path)

        # Class names
        self.class_names = {
            0: 'b_bishop', 1: 'b_king', 2: 'b_knight', 3: 'b_pawn',
            4: 'b_queen', 5: 'b_rook', 6: 'w_bishop', 7: 'w_king',
            8: 'w_knight', 9: 'w_pawn', 10: 'w_queen', 11: 'w_rook'
        }

        # Statistics
        self.true_labels = []
        self.pred_labels = []
        self.confidences = []
        self.detection_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    def evaluate_image(self, image_path: str, label_path: str,
                      iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate model on a single image.

        Args:
            image_path: Path to image
            label_path: Path to ground truth label file (YOLO format)
            iou_threshold: IoU threshold for considering detection correct

        Returns:
            Dictionary with evaluation results
        """
        # Load ground truth
        gt_boxes, gt_classes = self._load_yolo_labels(label_path)

        if len(gt_boxes) == 0:
            return {'error': 'No ground truth labels'}

        # Run prediction
        results = self.model.predict(image_path, conf=0.35, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            # All ground truth are false negatives
            for cls in gt_classes:
                self.detection_stats[int(cls)]['fn'] += 1
            return {'detections': 0, 'gt': len(gt_boxes)}

        # Convert predictions to numpy
        pred_boxes = results.boxes.xyxyn.cpu().numpy()  # Normalized xyxy
        pred_classes = results.boxes.cls.cpu().numpy()
        pred_confs = results.boxes.conf.cpu().numpy()

        # Convert YOLO format to xyxy for ground truth
        gt_boxes_xyxy = self._yolo_to_xyxy(gt_boxes)

        # Match predictions to ground truth using IoU
        matched_gt = set()
        matched_pred = set()

        for pred_idx, (pred_box, pred_cls, pred_conf) in enumerate(
                zip(pred_boxes, pred_classes, pred_confs)):

            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes_xyxy, gt_classes)):
                if gt_idx in matched_gt:
                    continue

                iou = self._calculate_iou(pred_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # Match found
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)

                gt_cls = int(gt_classes[best_gt_idx])
                pred_cls_int = int(pred_cls)

                # Record for confusion matrix
                self.true_labels.append(gt_cls)
                self.pred_labels.append(pred_cls_int)
                self.confidences.append(float(pred_conf))

                if gt_cls == pred_cls_int:
                    # True positive
                    self.detection_stats[gt_cls]['tp'] += 1
                else:
                    # Misclassification (FP for pred, FN for gt)
                    self.detection_stats[pred_cls_int]['fp'] += 1
                    self.detection_stats[gt_cls]['fn'] += 1
            else:
                # False positive (no match or low IoU)
                self.detection_stats[int(pred_cls)]['fp'] += 1

        # Unmatched ground truth = false negatives
        for gt_idx, gt_cls in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                self.detection_stats[int(gt_cls)]['fn'] += 1

        return {
            'detections': len(pred_boxes),
            'gt': len(gt_boxes),
            'matched': len(matched_gt)
        }

    def evaluate_dataset(self, images_dir: Path, labels_dir: Path):
        """
        Evaluate model on entire dataset.

        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing label files
        """
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        print(f"\nEvaluating {len(image_files)} images...")

        for idx, img_path in enumerate(image_files):
            label_path = labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            self.evaluate_image(str(img_path), str(label_path))

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images")

    def _load_yolo_labels(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load YOLO format labels."""
        if not Path(label_path).exists():
            return np.array([]), np.array([])

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return np.array([]), np.array([])

        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                data.append([cls, cx, cy, w, h])

        data = np.array(data)
        return data[:, 1:5], data[:, 0]

    def _yolo_to_xyxy(self, yolo_boxes: np.ndarray) -> np.ndarray:
        """Convert YOLO format (cx, cy, w, h) to xyxy format."""
        xyxy = np.zeros_like(yolo_boxes)
        xyxy[:, 0] = yolo_boxes[:, 0] - yolo_boxes[:, 2] / 2  # x1
        xyxy[:, 1] = yolo_boxes[:, 1] - yolo_boxes[:, 3] / 2  # y1
        xyxy[:, 2] = yolo_boxes[:, 0] + yolo_boxes[:, 2] / 2  # x2
        xyxy[:, 3] = yolo_boxes[:, 1] + yolo_boxes[:, 3] / 2  # y2
        return xyxy

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes (xyxy format)."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def print_metrics(self):
        """Print evaluation metrics."""
        print("\n" + "=" * 70)
        print("Per-Class Metrics")
        print("=" * 70)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Count':>8}")
        print("-" * 70)

        for cls_id in sorted(self.detection_stats.keys()):
            stats = self.detection_stats[cls_id]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_name = self.class_names.get(cls_id, f"class_{cls_id}")
            total = tp + fn

            print(f"{class_name:<15} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {total:>8}")

        print("=" * 70)

        # Overall metrics
        total_tp = sum(s['tp'] for s in self.detection_stats.values())
        total_fp = sum(s['fp'] for s in self.detection_stats.values())
        total_fn = sum(s['fn'] for s in self.detection_stats.values())

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
            if (overall_precision + overall_recall) > 0 else 0

        print(f"\nOverall Precision: {overall_precision:.3f}")
        print(f"Overall Recall: {overall_recall:.3f}")
        print(f"Overall F1-Score: {overall_f1:.3f}")
        print(f"Total Detections: {total_tp + total_fp}")
        print(f"Total Ground Truth: {total_tp + total_fn}")

    def plot_confusion_matrix(self, output_path: str = "confusion_matrix.png"):
        """Generate and save confusion matrix plot."""
        if len(self.true_labels) == 0:
            print("No data for confusion matrix")
            return

        cm = confusion_matrix(self.true_labels, self.pred_labels,
                             labels=list(range(12)))

        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.class_names[i] for i in range(12)],
                   yticklabels=[self.class_names[i] for i in range(12)])
        plt.title('Confusion Matrix - Chess Piece Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"\nConfusion matrix saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate chess piece detection model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model (.pt file)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/",
        help="Dataset directory (default: dataset/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results/",
        help="Output directory for results (default: evaluation_results/)"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    dataset_dir = Path(args.data)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: Dataset directories not found in {dataset_dir}")
        print("Expected: images/ and labels/ subdirectories")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Chess Piece Detection Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print("=" * 70)

    # Evaluate
    evaluator = ChessPieceEvaluator(str(model_path))
    evaluator.evaluate_dataset(images_dir, labels_dir)

    # Print metrics
    evaluator.print_metrics()

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(str(output_dir / "confusion_matrix.png"))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
