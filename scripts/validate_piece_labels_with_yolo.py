"""
Validate Piece Labels with YOLO Model

Runs the fine-tuned YOLO model on the training dataset and shows
which pieces are misclassified. This helps identify:
1. Labeling errors in the training data
2. Pieces the model struggles to classify correctly
3. Patterns in misclassification (e.g., pawns vs rooks)

Edit Mode:
    Press 'e' to enter edit mode, then click on misclassified pieces or
    false positives to accept the model's prediction. This is useful when
    the model correctly identifies pieces that were mislabeled in the
    training data.

Usage:
    # Use best_cls.pt (lowest cls_loss - recommended)
    python scripts/validate_piece_labels_with_yolo.py --data data/training/piece_dataset/

    # Use specific model
    python scripts/validate_piece_labels_with_yolo.py --model data/training/runs/piece_finetune/weights/best_cls.pt

    # Show all predictions (not just misclassifications)
    python scripts/validate_piece_labels_with_yolo.py --show-all
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Piece class mapping
PIECE_CLASSES = {
    0: 'b-bishop',
    1: 'b-king',
    2: 'b-knight',
    3: 'b-pawn',
    4: 'b-queen',
    5: 'b-rook',
    6: 'W-bishop',
    7: 'W-king',
    8: 'W-knight',
    9: 'W-pawn',
    10: 'W-queen',
    11: 'W-rook'
}


def load_labels(label_path):
    """Load YOLO format labels from file."""
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append({
                    'class_id': int(parts[0]),
                    'bbox': [float(p) for p in parts[1:]]  # x_center, y_center, width, height (normalized)
                })
    return labels


def save_labels(label_path, labels):
    """Save YOLO format labels to file."""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id = label['class_id']
            bbox = label['bbox']
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in YOLO format (x_center, y_center, w, h).

    Args:
        box1, box2: Lists of [x_center, y_center, width, height] (normalized 0-1)

    Returns:
        IoU value (0-1)
    """
    # Convert to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions_to_labels(predictions, ground_truth, iou_threshold=0.5):
    """
    Match predictions to ground truth labels by IoU.

    Returns:
        List of matches: [(gt_label, pred_label or None, iou)]
        List of unmatched predictions
    """
    matches = []
    unmatched_preds = list(predictions)

    for gt in ground_truth:
        best_match = None
        best_iou = 0.0
        best_idx = -1

        for i, pred in enumerate(unmatched_preds):
            iou = calculate_iou(gt['bbox'], pred['bbox'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = pred
                best_idx = i

        if best_match is not None:
            matches.append((gt, best_match, best_iou))
            unmatched_preds.pop(best_idx)
        else:
            matches.append((gt, None, 0.0))  # Missed detection

    return matches, unmatched_preds


class YOLOValidator:
    """Validate piece labels using YOLO model predictions."""

    def __init__(self, model_path, dataset_path, conf_threshold=0.25):
        """
        Initialize validator.

        Args:
            model_path: Path to YOLO model (.pt file)
            dataset_path: Path to dataset directory (contains images/ and labels/)
            conf_threshold: Confidence threshold for predictions
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.conf_threshold = conf_threshold

        # Load model
        print(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        print("[OK] Model loaded")

        # Setup paths
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset must have images/ and labels/ directories")

        # Load image list
        self.images = self._load_images()
        print(f"[OK] Found {len(self.images)} images")

        # Statistics
        self.stats = {
            'total_gt': 0,
            'total_pred': 0,
            'correct': 0,
            'misclassified': 0,
            'missed': 0,
            'false_positives': 0,
            'confusion': defaultdict(lambda: defaultdict(int))  # confusion[gt][pred] = count
        }

        self.window_name = "YOLO Validation"

        # Edit mode state
        self.edit_mode = False
        self.clickable_boxes = []  # List of (x1, y1, x2, y2, edit_type, data)
        self.current_results = None
        self.edits_made = 0

    def _load_images(self):
        """Load list of images with label files."""
        images = []
        for label_file in sorted(self.labels_dir.glob("*.txt")):
            for ext in ['.png', '.jpg', '.jpeg']:
                image_file = self.images_dir / f"{label_file.stem}{ext}"
                if image_file.exists():
                    images.append({
                        'name': label_file.stem,
                        'image_path': image_file,
                        'label_path': label_file
                    })
                    break
        return images

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks in edit mode."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self.edit_mode:
            return

        # Check if click is inside any clickable box
        for box_x1, box_y1, box_x2, box_y2, edit_type, data in self.clickable_boxes:
            if box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2:
                self._apply_edit(edit_type, data)
                return

    def _apply_edit(self, edit_type, data):
        """
        Apply an edit based on click.

        For misclassifications: change the GT label class to match model prediction
        For false positives: add a new label with model's predicted class
        """
        if self.current_results is None:
            return

        label_path = self.current_results['image_info']['label_path']
        labels = load_labels(label_path)

        if edit_type == 'misclassification':
            gt, pred, iou = data
            # Find the matching label and update its class
            for label in labels:
                if label['bbox'] == gt['bbox'] and label['class_id'] == gt['class_id']:
                    old_name = PIECE_CLASSES.get(label['class_id'], f"cls{label['class_id']}")
                    new_name = PIECE_CLASSES.get(pred['class_id'], f"cls{pred['class_id']}")
                    label['class_id'] = pred['class_id']
                    print(f"  [EDIT] Changed {old_name} -> {new_name}")
                    break

        elif edit_type == 'false_positive':
            pred = data
            # Add a new label for this detection
            new_label = {
                'class_id': pred['class_id'],
                'bbox': pred['bbox']
            }
            labels.append(new_label)
            new_name = PIECE_CLASSES.get(pred['class_id'], f"cls{pred['class_id']}")
            print(f"  [EDIT] Added new label: {new_name}")

        # Save updated labels
        save_labels(label_path, labels)
        self.edits_made += 1

        # Mark that we need to refresh the display
        self._needs_refresh = True

    def run_inference(self, image_path):
        """Run YOLO inference on image."""
        results = self.model(str(image_path), conf=self.conf_threshold, verbose=False)

        predictions = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                img_h, img_w = result.orig_shape
                for box in result.boxes:
                    # Get normalized bbox (YOLO format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    predictions.append({
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'conf': float(box.conf[0].cpu().numpy()),
                        'bbox': [x_center, y_center, width, height]
                    })

        return predictions

    def validate_image(self, image_info):
        """
        Validate a single image.

        Returns:
            dict with validation results
        """
        # Load ground truth
        gt_labels = load_labels(image_info['label_path'])

        # Run inference
        predictions = self.run_inference(image_info['image_path'])

        # Match predictions to ground truth
        matches, false_positives = match_predictions_to_labels(predictions, gt_labels)

        # Analyze results
        results = {
            'image_info': image_info,
            'gt_labels': gt_labels,
            'predictions': predictions,
            'matches': matches,
            'false_positives': false_positives,
            'misclassifications': [],
            'missed': [],
            'correct': []
        }

        for gt, pred, iou in matches:
            if pred is None:
                results['missed'].append(gt)
            elif gt['class_id'] != pred['class_id']:
                results['misclassifications'].append((gt, pred, iou))
            else:
                results['correct'].append((gt, pred, iou))

        return results

    def draw_validation_image(self, results, show_all=False):
        """Draw image with validation results."""
        img = cv2.imread(str(results['image_info']['image_path']))
        if img is None:
            return None

        img_h, img_w = img.shape[:2]

        # Clear clickable boxes for this image
        self.clickable_boxes = []

        # Draw misclassifications in RED
        for gt, pred, iou in results['misclassifications']:
            x_c, y_c, w, h = gt['bbox']
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)

            # Red box for misclassification (thicker border in edit mode)
            thickness = 4 if self.edit_mode else 3
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)

            # Label: GT -> Pred
            gt_name = PIECE_CLASSES.get(gt['class_id'], f"cls{gt['class_id']}")[:6]
            pred_name = PIECE_CLASSES.get(pred['class_id'], f"cls{pred['class_id']}")[:6]
            label = f"{gt_name}->{pred_name}"
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Track clickable box for edit mode
            self.clickable_boxes.append((x1, y1, x2, y2, 'misclassification', (gt, pred, iou)))

        # Draw missed detections in ORANGE
        for gt in results['missed']:
            x_c, y_c, w, h = gt['bbox']
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 3)
            gt_name = PIECE_CLASSES.get(gt['class_id'], f"cls{gt['class_id']}")
            cv2.putText(img, f"MISSED:{gt_name}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

        # Draw false positives in PURPLE
        for pred in results['false_positives']:
            x_c, y_c, w, h = pred['bbox']
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)

            # Thicker border in edit mode
            thickness = 3 if self.edit_mode else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), thickness)
            pred_name = PIECE_CLASSES.get(pred['class_id'], f"cls{pred['class_id']}")
            cv2.putText(img, f"FP:{pred_name}", (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

            # Track clickable box for edit mode
            self.clickable_boxes.append((x1, y1, x2, y2, 'false_positive', pred))

        # Draw correct predictions in GREEN (if show_all)
        if show_all:
            for gt, pred, iou in results['correct']:
                x_c, y_c, w, h = gt['bbox']
                x1 = int((x_c - w/2) * img_w)
                y1 = int((y_c - h/2) * img_h)
                x2 = int((x_c + w/2) * img_w)
                y2 = int((y_c + h/2) * img_h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Add status bar
        status_height = 100
        status_bar = np.zeros((status_height, img.shape[1], 3), dtype=np.uint8)

        # Image name and counts
        name = results['image_info']['name']
        n_correct = len(results['correct'])
        n_misclass = len(results['misclassifications'])
        n_missed = len(results['missed'])
        n_fp = len(results['false_positives'])

        cv2.putText(status_bar, f"{name}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        stats_text = f"Correct: {n_correct} | Misclass: {n_misclass} | Missed: {n_missed} | FP: {n_fp}"
        cv2.putText(status_bar, stats_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Legend
        legend = "RED=misclass (GT->Pred) | ORANGE=missed | PURPLE=false_pos (model found) | GREEN=correct"
        cv2.putText(status_bar, legend, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Controls - different based on edit mode
        if self.edit_mode:
            controls = "EDIT MODE: click to accept model prediction | e=exit edit | q=quit"
            cv2.putText(status_bar, controls, (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            # Edit mode indicator
            cv2.putText(status_bar, f"[EDIT MODE] Edits: {self.edits_made}", (img_w - 200, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            controls = "SPACE=next | b=back | e=edit mode | q=quit"
            cv2.putText(status_bar, controls, (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return np.vstack([img, status_bar])

    def run_validation(self, show_all=False, errors_only=True):
        """
        Run validation on all images.

        Args:
            show_all: Show correct predictions too (green boxes)
            errors_only: Only show images with errors (misclassifications, missed, FP)
        """
        print("\n" + "=" * 60)
        print("YOLO MODEL VALIDATION")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Images: {len(self.images)}")
        print(f"Mode: {'Errors only' if errors_only else 'All images'}")
        print("=" * 60)

        # First pass: collect all results
        print("\nAnalyzing images...")
        all_results = []
        for i, img_info in enumerate(self.images):
            results = self.validate_image(img_info)
            all_results.append(results)

            # Update global stats
            self.stats['total_gt'] += len(results['gt_labels'])
            self.stats['total_pred'] += len(results['predictions'])
            self.stats['correct'] += len(results['correct'])
            self.stats['misclassified'] += len(results['misclassifications'])
            self.stats['missed'] += len(results['missed'])
            self.stats['false_positives'] += len(results['false_positives'])

            # Update confusion matrix
            for gt, pred, iou in results['misclassifications']:
                self.stats['confusion'][gt['class_id']][pred['class_id']] += 1

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(self.images)} images...")

        # Print summary
        self._print_summary()

        # Filter to images with errors if requested
        if errors_only:
            display_results = [r for r in all_results
                             if r['misclassifications'] or r['missed'] or r['false_positives']]
            print(f"\n[INFO] {len(display_results)} images have errors (out of {len(all_results)})")
        else:
            display_results = all_results

        if not display_results:
            print("\n[OK] No errors found! Model classifies all pieces correctly.")
            return

        # Interactive display
        print("\nStarting interactive review...")
        print("Controls: SPACE=next, b=back, e=edit mode, q=quit")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        idx = 0
        self._needs_refresh = False

        while 0 <= idx < len(display_results):
            results = display_results[idx]
            self.current_results = results
            display = self.draw_validation_image(results, show_all=show_all)

            if display is not None:
                cv2.imshow(self.window_name, display)

            key = cv2.waitKey(100) & 0xFF  # 100ms for responsive edit mode

            # Check if we need to refresh after an edit
            if self._needs_refresh:
                self._needs_refresh = False
                # Re-validate this image to reflect the edit
                results = self.validate_image(results['image_info'])
                display_results[idx] = results
                self.current_results = results
                display = self.draw_validation_image(results, show_all=show_all)
                if display is not None:
                    cv2.imshow(self.window_name, display)
                continue

            if key == 255:  # No key pressed (timeout)
                continue
            elif key == ord('q'):
                break
            elif key == ord(' ') or key == ord('n'):
                idx += 1
            elif key == ord('b'):
                idx = max(0, idx - 1)
            elif key == ord('e'):
                self.edit_mode = not self.edit_mode
                mode_str = "ON" if self.edit_mode else "OFF"
                print(f"  Edit mode: {mode_str}")
                # Redraw to show edit mode indicator
                display = self.draw_validation_image(results, show_all=show_all)
                if display is not None:
                    cv2.imshow(self.window_name, display)

        cv2.destroyAllWindows()

        if self.edits_made > 0:
            print(f"\n[OK] Made {self.edits_made} edits to label files.")

    def _print_summary(self):
        """Print validation summary statistics."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        total = self.stats['total_gt']
        correct = self.stats['correct']
        misclass = self.stats['misclassified']
        missed = self.stats['missed']
        fp = self.stats['false_positives']

        print(f"Ground truth labels: {total}")
        print(f"Model predictions:   {self.stats['total_pred']}")
        print()
        print(f"Correct:         {correct:4d} ({100*correct/total:.1f}%)" if total > 0 else "Correct: 0")
        print(f"Misclassified:   {misclass:4d} ({100*misclass/total:.1f}%)" if total > 0 else "Misclassified: 0")
        print(f"Missed:          {missed:4d} ({100*missed/total:.1f}%)" if total > 0 else "Missed: 0")
        print(f"False positives: {fp:4d}")
        print()

        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Classification Accuracy: {accuracy:.1f}%")

        # Print confusion matrix for misclassifications
        if self.stats['confusion']:
            print("\n" + "-" * 40)
            print("MISCLASSIFICATION PATTERNS (GT -> Pred)")
            print("-" * 40)

            confusion_list = []
            for gt_cls, pred_counts in self.stats['confusion'].items():
                for pred_cls, count in pred_counts.items():
                    gt_name = PIECE_CLASSES.get(gt_cls, f"cls{gt_cls}")
                    pred_name = PIECE_CLASSES.get(pred_cls, f"cls{pred_cls}")
                    confusion_list.append((count, gt_name, pred_name))

            # Sort by count descending
            confusion_list.sort(reverse=True)

            for count, gt_name, pred_name in confusion_list[:15]:  # Top 15
                print(f"  {gt_name:10s} -> {pred_name:10s}: {count}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate piece labels by comparing YOLO predictions to ground truth"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/piece_dataset",
        help="Path to dataset directory or data.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model (default: best_cls.pt from training runs)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all predictions including correct ones (green boxes)"
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Show all images, not just ones with errors"
    )

    args = parser.parse_args()

    # Handle data path
    dataset_path = Path(args.data)
    if dataset_path.suffix in ['.yaml', '.yml']:
        dataset_path = dataset_path.parent
        print(f"[Info] Using directory: {dataset_path}")

    if not dataset_path.exists():
        print(f"[X] Dataset not found: {dataset_path}")
        return 1

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        # Try to find best_cls.pt first, then best.pt
        candidates = [
            dataset_path.parent / "runs" / "piece_finetune" / "weights" / "best_cls.pt",
            Path("data/training/runs/piece_finetune/weights/best_cls.pt"),
            Path("data/training/runs/piece_finetune/weights/best.pt"),
            Path("data/best_transformed_detection.pt"),
        ]
        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                print(f"[Info] Using model: {model_path}")
                break

        if model_path is None:
            print("[X] No model found. Please specify with --model")
            print("    Tried: " + ", ".join(str(c) for c in candidates))
            return 1

    if not model_path.exists():
        print(f"[X] Model not found: {model_path}")
        return 1

    # Run validation
    try:
        validator = YOLOValidator(
            model_path=str(model_path),
            dataset_path=dataset_path,
            conf_threshold=args.conf
        )
        validator.run_validation(
            show_all=args.show_all,
            errors_only=not args.all_images
        )
    except Exception as e:
        print(f"\n[X] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
