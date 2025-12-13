"""
Validate Piece Detection Labels

Interactive tool to validate existing YOLO-format piece labels.
Helps catch labeling errors before training.

Features:
- Review labels instance-by-instance
- Model-assisted flagging (predictions vs labels)
- Organize review by class or flagged instances
- Edit/delete incorrect labels
- Generate validation report

Usage:
    # Basic validation (all instances)
    python scripts/validate_piece_labels.py --data data/training/piece_dataset/

    # Model-assisted (flagged instances only)
    python scripts/validate_piece_labels.py \
        --data data/training/piece_dataset/ \
        --model data/best_transformed_detection.pt \
        --mode flagged

    # Review specific class
    python scripts/validate_piece_labels.py \
        --data data/training/piece_dataset/ \
        --class 10  # White queen
"""

import argparse
import sys
from pathlib import Path
import cv2
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Piece class mapping (same as label_pieces.py)
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


class LabelValidator:
    """Validate YOLO-format piece labels interactively."""

    def __init__(self, dataset_path, model_path=None, state_path=None):
        """
        Initialize validator.

        Args:
            dataset_path: Path to dataset directory (contains images/ and labels/)
            model_path: Optional path to YOLO model for prediction comparison
            state_path: Optional path to save/load validation state
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset must have images/ and labels/ directories")

        # Load YOLO model if provided
        self.model = None
        if model_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"[OK] Loaded model: {model_path}")
            except Exception as e:
                print(f"[Warning] Could not load model: {e}")
                self.model = None

        # State file for persistence
        if state_path is None:
            state_path = self.dataset_path / "validation_state.json"
        self.state_path = Path(state_path)

        # Load instances
        print("Loading dataset instances...")
        self.instances = self._load_all_instances()
        print(f"[OK] Loaded {len(self.instances)} instances from {len(set(i['image_path'] for i in self.instances))} images")

        # Validation state
        self.validation_state = {}  # {instance_id: 'correct'|'incorrect'|'skipped'}
        self.corrections = {}       # {instance_id: corrected_label}
        self.flagged_instances = set()  # Instance IDs flagged by model

        # Load previous state if exists
        self._load_state()

        # Current review index
        self.current_idx = 0

        # Window name
        self.window_name = "Validate Labels"

    def _load_all_instances(self):
        """Parse all label files into instance list."""
        instances = []

        for label_file in sorted(self.labels_dir.glob("*.txt")):
            image_file = self.images_dir / f"{label_file.stem}.png"

            if not image_file.exists():
                # Try jpg extension
                image_file = self.images_dir / f"{label_file.stem}.jpg"
                if not image_file.exists():
                    print(f"[Warning] No image for label: {label_file.name}")
                    continue

            # Parse label file
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Create unique instance ID
                    instance_id = f"{label_file.stem}_{line_num}"

                    instances.append({
                        'id': instance_id,
                        'image_path': image_file,
                        'label_path': label_file,
                        'class_id': class_id,
                        'bbox': [x_center, y_center, width, height],
                        'line_number': line_num
                    })

        return instances

    def _load_state(self):
        """Load validation state from file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.validation_state = data.get('validation_state', {})
                    self.corrections = data.get('corrections', {})
                    self.current_idx = data.get('current_idx', 0)
                print(f"[OK] Resumed from previous session (instance {self.current_idx}/{len(self.instances)})")
            except Exception as e:
                print(f"[Warning] Could not load state: {e}")

    def _save_state(self):
        """Save validation state to file."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump({
                    'validation_state': self.validation_state,
                    'corrections': self.corrections,
                    'current_idx': self.current_idx,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[Warning] Could not save state: {e}")

    def flag_suspicious_instances(self, conf_threshold=0.5, iou_threshold=0.5):
        """
        Use model predictions to flag likely labeling errors.

        Args:
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching predictions to labels
        """
        if self.model is None:
            print("[Warning] No model loaded, skipping flagging")
            return

        print("\nFlagging suspicious instances with model predictions...")

        # Group instances by image
        instances_by_image = defaultdict(list)
        for inst in self.instances:
            instances_by_image[str(inst['image_path'])].append(inst)

        flagged_count = 0

        for image_path, image_instances in instances_by_image.items():
            # Run prediction
            try:
                results = self.model.predict(
                    source=image_path,
                    conf=conf_threshold,
                    iou=0.7,
                    verbose=False,
                    device='cpu'
                )

                if len(results) == 0:
                    continue

                result = results[0]

                # Load image to get dimensions
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]

                # Get predictions
                if result.boxes is None or len(result.boxes) == 0:
                    # No detections - flag all instances in this image
                    for inst in image_instances:
                        self.flagged_instances.add(inst['id'])
                        flagged_count += 1
                    continue

                # Convert predictions to same format as labels
                pred_boxes = []
                for box in result.boxes:
                    pred_class = int(box.cls[0])
                    pred_conf = float(box.conf[0])

                    # Get xyxy coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to normalized xywh
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    pred_boxes.append({
                        'class_id': pred_class,
                        'bbox': [x_center, y_center, width, height],
                        'conf': pred_conf
                    })

                # Match each label to predictions
                for inst in image_instances:
                    label_bbox = inst['bbox']
                    label_class = inst['class_id']

                    # Find best matching prediction
                    best_iou = 0
                    best_match = None

                    for pred in pred_boxes:
                        iou = self._calculate_iou(label_bbox, pred['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_match = pred

                    # Flag if:
                    # 1. No matching prediction (IoU < threshold)
                    # 2. Matching prediction has different class
                    # 3. Matching prediction has low confidence
                    should_flag = False

                    if best_iou < iou_threshold:
                        should_flag = True  # No match found
                    elif best_match['class_id'] != label_class:
                        should_flag = True  # Class mismatch
                    elif best_match['conf'] < conf_threshold:
                        should_flag = True  # Low confidence

                    if should_flag:
                        self.flagged_instances.add(inst['id'])
                        flagged_count += 1

            except Exception as e:
                print(f"[Warning] Prediction failed for {Path(image_path).name}: {e}")
                continue

        print(f"[OK] Flagged {flagged_count}/{len(self.instances)} instances ({100*flagged_count/len(self.instances):.1f}%)")

    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bboxes in normalized xywh format.

        Args:
            bbox1, bbox2: [x_center, y_center, width, height] (normalized 0-1)

        Returns:
            IoU value (0-1)
        """
        # Convert to xyxy
        x1_1 = bbox1[0] - bbox1[2] / 2
        y1_1 = bbox1[1] - bbox1[3] / 2
        x2_1 = bbox1[0] + bbox1[2] / 2
        y2_1 = bbox1[1] + bbox1[3] / 2

        x1_2 = bbox2[0] - bbox2[2] / 2
        y1_2 = bbox2[1] - bbox2[3] / 2
        x2_2 = bbox2[0] + bbox2[2] / 2
        y2_2 = bbox2[1] + bbox2[3] / 2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def draw_validation_display(self, instance, show_crop=True):
        """
        Draw validation display with image, bbox, and cropped piece.

        Args:
            instance: Instance dict
            show_crop: Whether to show cropped piece view

        Returns:
            Combined display image
        """
        # Load image
        img = cv2.imread(str(instance['image_path']))
        if img is None:
            return None

        img_h, img_w = img.shape[:2]

        # Convert normalized bbox to pixel coordinates
        x_center, y_center, width, height = instance['bbox']
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        # Draw full image with bbox
        display_img = img.copy()

        # Load all labels for this image to show context
        label_file = instance['label_path']
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                xc, yc, w, h = [float(p) for p in parts[1:]]

                bx1 = int((xc - w / 2) * img_w)
                by1 = int((yc - h / 2) * img_h)
                bx2 = int((xc + w / 2) * img_w)
                by2 = int((yc + h / 2) * img_h)

                # Different color for current instance vs context
                if line_num == instance['line_number']:
                    # Current instance - thick green border
                    color = (0, 255, 0)
                    thickness = 3
                else:
                    # Context - thin colored border
                    color = (0, 0, 255) if cls_id < 6 else (255, 0, 0)
                    thickness = 1

                cv2.rectangle(display_img, (bx1, by1), (bx2, by2), color, thickness)

                # Label for current instance only
                if line_num == instance['line_number']:
                    label = PIECE_CLASSES[cls_id]
                    cv2.putText(display_img, label, (bx1, by1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Extract cropped piece if requested
        if show_crop and x2 > x1 and y2 > y1:
            # Ensure crop is within bounds
            x1_crop = max(0, x1)
            y1_crop = max(0, y1)
            x2_crop = min(img_w, x2)
            y2_crop = min(img_h, y2)

            crop = img[y1_crop:y2_crop, x1_crop:x2_crop].copy()

            # Resize crop to reasonable size (e.g., 200x200)
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                target_size = 200
                crop_h, crop_w = crop.shape[:2]
                scale = min(target_size / crop_w, target_size / crop_h)
                new_w = int(crop_w * scale)
                new_h = int(crop_h * scale)
                crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Create crop canvas
                crop_canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                # Center the crop
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2
                crop_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized

                # Add border
                cv2.rectangle(crop_canvas, (0, 0), (target_size-1, target_size-1), (0, 255, 0), 2)

                # Add label
                class_name = PIECE_CLASSES[instance['class_id']]
                cv2.putText(crop_canvas, class_name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Combine images horizontally
                display_height = max(display_img.shape[0], crop_canvas.shape[0])
                combined = np.zeros((display_height, display_img.shape[1] + target_size + 20, 3), dtype=np.uint8)
                combined[:display_img.shape[0], :display_img.shape[1]] = display_img
                combined[:target_size, display_img.shape[1]+20:] = crop_canvas

                display_img = combined

        # Add status bar
        status_height = 80
        status_bar = np.zeros((status_height, display_img.shape[1], 3), dtype=np.uint8)

        # Progress
        progress = f"Instance {self.current_idx + 1}/{len(self.instances)}"
        cv2.putText(status_bar, progress, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Class and bbox info
        class_name = PIECE_CLASSES[instance['class_id']]
        bbox_str = f"{class_name} | bbox: ({x1},{y1})-({x2},{y2})"
        cv2.putText(status_bar, bbox_str, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Controls
        controls = "y/SPACE=correct | n=incorrect | e=edit | s=skip | u=undo | q=quit | h=help"
        cv2.putText(status_bar, controls, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Flagged indicator
        if instance['id'] in self.flagged_instances:
            cv2.putText(status_bar, "[FLAGGED]", (display_img.shape[1] - 150, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Validation state indicator
        if instance['id'] in self.validation_state:
            state = self.validation_state[instance['id']]
            if state == 'correct':
                cv2.putText(status_bar, "[OK]", (display_img.shape[1] - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif state == 'incorrect':
                cv2.putText(status_bar, "[BAD]", (display_img.shape[1] - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif state == 'skipped':
                cv2.putText(status_bar, "[SKIP]", (display_img.shape[1] - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Combine with status bar
        final_display = np.vstack([display_img, status_bar])

        return final_display

    def validate_instance(self, instance):
        """
        Interactive validation for single instance.

        Returns:
            'correct', 'incorrect', 'skipped', 'quit', or 'undo'
        """
        display = self.draw_validation_display(instance)
        if display is None:
            return 'skipped'

        cv2.imshow(self.window_name, display)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y') or key == 32:  # y or SPACE
                return 'correct'
            elif key == ord('n'):
                return 'incorrect'
            elif key == ord('s'):
                return 'skipped'
            elif key == ord('u'):
                return 'undo'
            elif key == ord('q'):
                return 'quit'
            elif key == ord('h'):
                self._show_help()
                cv2.imshow(self.window_name, display)
            elif key == ord('e'):
                # TODO: Implement edit mode
                print("[Warning] Edit mode not yet implemented")
                continue

    def _show_help(self):
        """Show help overlay."""
        help_text = [
            "VALIDATION CONTROLS",
            "",
            "y / SPACE  - Mark as CORRECT, continue",
            "n          - Mark as INCORRECT, continue",
            "s          - SKIP this instance (review later)",
            "u          - UNDO previous marking, go back",
            "e          - EDIT mode (change class or bbox) [TODO]",
            "q          - QUIT and save progress",
            "h          - Show this help",
            "",
            "GREEN box = Current instance being reviewed",
            "RED box   = Black pieces (context)",
            "BLUE box  = White pieces (context)",
            "",
            "[FLAGGED] = Model prediction mismatch",
            "",
            "Press any key to continue..."
        ]

        # Create help display
        help_img = np.zeros((len(help_text) * 30 + 40, 800, 3), dtype=np.uint8)

        for i, line in enumerate(help_text):
            y = 30 + i * 30
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(help_img, line, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow(self.window_name, help_img)
        cv2.waitKey(0)

    def run_validation(self, mode='all', target_class=None):
        """
        Main validation loop.

        Args:
            mode: 'all', 'flagged', or 'by_class'
            target_class: If mode='by_class', which class to review
        """
        # Filter instances based on mode
        if mode == 'flagged':
            instances_to_review = [inst for inst in self.instances if inst['id'] in self.flagged_instances]
            print(f"\nReviewing {len(instances_to_review)} flagged instances")
        elif mode == 'by_class' and target_class is not None:
            instances_to_review = [inst for inst in self.instances if inst['class_id'] == target_class]
            print(f"\nReviewing {len(instances_to_review)} instances of class {PIECE_CLASSES[target_class]}")
        else:
            instances_to_review = self.instances
            print(f"\nReviewing all {len(instances_to_review)} instances")

        if len(instances_to_review) == 0:
            print("[Warning] No instances to review")
            return

        # Find starting index
        if self.current_idx >= len(instances_to_review):
            self.current_idx = 0

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Validation loop
        while self.current_idx < len(instances_to_review):
            instance = instances_to_review[self.current_idx]

            # Validate
            result = self.validate_instance(instance)

            if result == 'quit':
                print("\n[Quit] Saving progress...")
                break
            elif result == 'undo':
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print(f"[Undo] Back to instance {self.current_idx}")
                continue
            else:
                # Record validation
                self.validation_state[instance['id']] = result
                print(f"[{result.upper()}] {PIECE_CLASSES[instance['class_id']]} in {instance['image_path'].name}")

                # Save state periodically
                if self.current_idx % 10 == 0:
                    self._save_state()

                # Move to next
                self.current_idx += 1

        # Final save
        self._save_state()
        cv2.destroyAllWindows()

        print(f"\n[OK] Validation complete ({self.current_idx}/{len(instances_to_review)} reviewed)")

    def generate_report(self, output_path=None):
        """
        Generate validation report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report dict
        """
        # Count by status
        correct = sum(1 for s in self.validation_state.values() if s == 'correct')
        incorrect = sum(1 for s in self.validation_state.values() if s == 'incorrect')
        skipped = sum(1 for s in self.validation_state.values() if s == 'skipped')
        not_reviewed = len(self.instances) - len(self.validation_state)

        # Count by class
        class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0, 'skipped': 0})

        for inst in self.instances:
            cls_id = inst['class_id']
            class_stats[cls_id]['total'] += 1

            if inst['id'] in self.validation_state:
                status = self.validation_state[inst['id']]
                class_stats[cls_id][status] += 1

        # Build report
        report = {
            'summary': {
                'total_instances': len(self.instances),
                'reviewed': len(self.validation_state),
                'not_reviewed': not_reviewed,
                'correct': correct,
                'incorrect': incorrect,
                'skipped': skipped,
                'accuracy': f"{100 * correct / len(self.validation_state):.1f}%" if len(self.validation_state) > 0 else "N/A"
            },
            'by_class': {}
        }

        for cls_id in sorted(class_stats.keys()):
            stats = class_stats[cls_id]
            report['by_class'][PIECE_CLASSES[cls_id]] = {
                'total': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'skipped': stats['skipped'],
                'not_reviewed': stats['total'] - (stats['correct'] + stats['incorrect'] + stats['skipped']),
                'error_rate': f"{100 * stats['incorrect'] / stats['total']:.1f}%" if stats['total'] > 0 else "N/A"
            }

        # Print report
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        print(f"Total instances: {report['summary']['total_instances']}")
        print(f"Reviewed: {report['summary']['reviewed']} ({100*report['summary']['reviewed']/report['summary']['total_instances']:.1f}%)")
        print(f"  Correct: {report['summary']['correct']}")
        print(f"  Incorrect: {report['summary']['incorrect']}")
        print(f"  Skipped: {report['summary']['skipped']}")
        print(f"Not reviewed: {report['summary']['not_reviewed']}")
        print(f"\nLabel accuracy: {report['summary']['accuracy']}")

        print("\nPer-Class Breakdown:")
        print(f"{'Class':<15} {'Total':>6} {'Correct':>8} {'Incorrect':>10} {'Error Rate':>12}")
        print("-" * 60)

        for cls_name, stats in report['by_class'].items():
            print(f"{cls_name:<15} {stats['total']:>6} {stats['correct']:>8} {stats['incorrect']:>10} {stats['error_rate']:>12}")

        # Save report if requested
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n[OK] Report saved: {output_path}")

        print("=" * 60)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate piece detection labels interactively"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset directory (contains images/ and labels/)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model for prediction comparison (optional)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "flagged", "by_class"],
        help="Validation mode (default: all)"
    )
    parser.add_argument(
        "--class",
        type=int,
        default=None,
        dest="target_class",
        help="Class ID to review (for --mode by_class)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for model flagging (default: 0.5)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to labels (default: 0.5)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save validation report (optional)"
    )

    args = parser.parse_args()

    # Validate inputs
    dataset_path = Path(args.data)
    if not dataset_path.exists():
        print(f"[X] Dataset not found: {dataset_path}")
        return 1

    # Create validator
    try:
        validator = LabelValidator(
            dataset_path=dataset_path,
            model_path=args.model
        )
    except Exception as e:
        print(f"[X] Failed to initialize validator: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Flag suspicious instances if model provided
    if args.model:
        validator.flag_suspicious_instances(
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )

    # Run validation
    try:
        validator.run_validation(
            mode=args.mode,
            target_class=args.target_class
        )
    except Exception as e:
        print(f"\n[X] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate report
    validator.generate_report(output_path=args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
