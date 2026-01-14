"""
Validate Piece Detection Labels

Interactive tool to validate existing YOLO-format piece labels.
Reviews one image at a time with all bounding boxes shown.

Features:
- Review all labels per image at once (much faster than per-instance)
- Edit mode: click to select, change class, add/delete boxes
- Progress tracking with resume capability
- Generate validation report

Usage:
    # Validate all images in dataset
    python scripts/validate_piece_labels.py --data data/training/piece_dataset/
    python scripts/validate_piece_labels.py --data data/training/piece_dataset/data.yaml

Controls:
    y/SPACE = all labels correct, move to next image
    e = enter edit mode (fix incorrect labels)
    s = skip image (review later)
    u = go back to previous image
    q = quit and save progress
    h = show help
"""

import argparse
import sys
from pathlib import Path
import cv2
import json
import numpy as np
from datetime import datetime


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


class LabelValidator:
    """Validate YOLO-format piece labels by image."""

    def __init__(self, dataset_path, state_path=None):
        """
        Initialize validator.

        Args:
            dataset_path: Path to dataset directory (contains images/ and labels/)
            state_path: Optional path to save/load validation state
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset must have images/ and labels/ directories")

        # State file for persistence
        if state_path is None:
            state_path = self.dataset_path / "validation_state.json"
        self.state_path = Path(state_path)

        # Load images list
        print("Loading dataset...")
        self.images = self._load_images()
        print(f"[OK] Found {len(self.images)} labeled images")

        # Validation state: {image_name: 'correct'|'edited'|'skipped'}
        self.validation_state = {}
        self.current_idx = 0

        # Load previous state if exists
        self._load_state()

        # Window name
        self.window_name = "Validate Labels"

    def _load_images(self):
        """Load list of images with their label files."""
        images = []

        for label_file in sorted(self.labels_dir.glob("*.txt")):
            # Find corresponding image
            image_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = self.images_dir / f"{label_file.stem}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break

            if image_file is None:
                print(f"[Warning] No image for label: {label_file.name}")
                continue

            # Count labels
            with open(label_file, 'r') as f:
                label_count = sum(1 for line in f if len(line.strip().split()) == 5)

            images.append({
                'name': label_file.stem,
                'image_path': image_file,
                'label_path': label_file,
                'label_count': label_count
            })

        return images

    def _load_state(self):
        """Load validation state from file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.validation_state = data.get('validation_state', {})
                    self.current_idx = data.get('current_idx', 0)

                reviewed = len(self.validation_state)
                print(f"[OK] Resumed: {reviewed}/{len(self.images)} images reviewed")
            except Exception as e:
                print(f"[Warning] Could not load state: {e}")

    def _save_state(self):
        """Save validation state to file."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump({
                    'validation_state': self.validation_state,
                    'current_idx': self.current_idx,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[Warning] Could not save state: {e}")

    def _load_labels(self, label_path):
        """Load labels from YOLO format file."""
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append({
                        'class_id': int(parts[0]),
                        'bbox': [float(p) for p in parts[1:]]
                    })
        return labels

    def _save_labels(self, label_path, labels):
        """Save labels to YOLO format file."""
        with open(label_path, 'w') as f:
            for label in labels:
                cls_id = label['class_id']
                x_c, y_c, w, h = label['bbox']
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    def draw_image_display(self, image_info):
        """Draw image with all labels and status bar."""
        img = cv2.imread(str(image_info['image_path']))
        if img is None:
            return None

        img_h, img_w = img.shape[:2]
        labels = self._load_labels(image_info['label_path'])

        # Draw all bounding boxes
        for i, label in enumerate(labels):
            x_center, y_center, width, height = label['bbox']
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # Color by piece type (black pieces = red, white pieces = blue)
            cls_id = label['class_id']
            if cls_id < 6:  # Black pieces
                color = (0, 0, 255)  # Red
            else:  # White pieces
                color = (255, 0, 0)  # Blue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Class label
            class_name = PIECE_CLASSES.get(cls_id, f"cls{cls_id}")
            cv2.putText(img, class_name, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Status bar
        status_height = 80
        status_bar = np.zeros((status_height, img.shape[1], 3), dtype=np.uint8)

        # Progress
        progress = f"Image {self.current_idx + 1}/{len(self.images)} | {len(labels)} labels"
        cv2.putText(status_bar, progress, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Image name
        cv2.putText(status_bar, image_info['name'], (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls
        controls = "y/SPACE=correct | e=edit | s=skip | u=back | q=quit | h=help"
        cv2.putText(status_bar, controls, (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Validation state indicator
        state = self.validation_state.get(image_info['name'])
        if state:
            state_colors = {'correct': (0, 255, 0), 'edited': (0, 255, 255), 'skipped': (0, 165, 255)}
            color = state_colors.get(state, (255, 255, 255))
            cv2.putText(status_bar, f"[{state.upper()}]", (img.shape[1] - 120, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return np.vstack([img, status_bar])

    def edit_image(self, image_info):
        """
        Edit mode for correcting labels.

        Returns:
            True if changes were saved, False otherwise
        """
        img = cv2.imread(str(image_info['image_path']))
        if img is None:
            return False

        img_h, img_w = img.shape[:2]
        labels = self._load_labels(image_info['label_path'])

        selected_idx = None
        modified = False
        draw_mode = False
        draw_start = None
        first_digit = None

        def draw_edit_display():
            display = img.copy()

            for i, label in enumerate(labels):
                x_center, y_center, width, height = label['bbox']
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)

                if i == selected_idx:
                    color = (0, 255, 255)  # Yellow for selected
                    thickness = 3
                else:
                    cls_id = label['class_id']
                    color = (0, 0, 255) if cls_id < 6 else (255, 0, 0)
                    thickness = 2

                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

                class_name = PIECE_CLASSES.get(label['class_id'], f"cls{label['class_id']}")
                cv2.putText(display, f"{i}:{class_name}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Status bar
            status_height = 100
            status_bar = np.zeros((status_height, display.shape[1], 3), dtype=np.uint8)

            if draw_mode:
                mode_text = "DRAW MODE - Click and drag to add bbox"
                mode_color = (0, 165, 255)
            elif first_digit is not None:
                mode_text = f"Enter second digit (0-9) for class {first_digit}X"
                mode_color = (255, 165, 0)
            else:
                mode_text = "EDIT MODE - Click to select, then change class or delete"
                mode_color = (0, 255, 0)

            cv2.putText(status_bar, mode_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

            if selected_idx is not None and selected_idx < len(labels):
                sel_class = PIECE_CLASSES.get(labels[selected_idx]['class_id'], "?")
                cv2.putText(status_bar, f"Selected: [{selected_idx}] {sel_class}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            controls = "Click=select | 0-9,0-9=class | d=delete | a=add | ENTER=save | ESC=cancel"
            cv2.putText(status_bar, controls, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Class reference
            ref_x = display.shape[1] - 180
            for cls_id in range(12):
                y_pos = 15 + (cls_id % 6) * 14
                x_pos = ref_x if cls_id < 6 else ref_x + 90
                cv2.putText(status_bar, f"{cls_id:02d}:{PIECE_CLASSES[cls_id][:6]}", (x_pos, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            return np.vstack([display, status_bar])

        def find_bbox_at_point(x, y):
            for i, label in enumerate(labels):
                xc, yc, w, h = label['bbox']
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return i
            return None

        mouse_state = {'x': 0, 'y': 0, 'drawing': False}

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_idx, draw_start, modified
            mouse_state['x'] = x
            mouse_state['y'] = y

            if event == cv2.EVENT_LBUTTONDOWN:
                if draw_mode:
                    draw_start = (x, y)
                    mouse_state['drawing'] = True
                else:
                    selected_idx = find_bbox_at_point(x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                if mouse_state['drawing'] and draw_start:
                    x1, y1 = draw_start
                    x2, y2 = x, y
                    xc = ((x1 + x2) / 2) / img_w
                    yc = ((y1 + y2) / 2) / img_h
                    w = abs(x2 - x1) / img_w
                    h = abs(y2 - y1) / img_h

                    if w > 0.01 and h > 0.01:
                        labels.append({'class_id': 9, 'bbox': [xc, yc, w, h]})
                        selected_idx = len(labels) - 1
                        modified = True

                    draw_start = None
                    mouse_state['drawing'] = False

        cv2.setMouseCallback(self.window_name, mouse_callback)

        while True:
            display = draw_edit_display()

            if draw_mode and draw_start and mouse_state['drawing']:
                x1, y1 = draw_start
                cv2.rectangle(display[:img_h], (x1, y1),
                            (mouse_state['x'], mouse_state['y']), (255, 0, 255), 2)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:  # ESC
                break

            elif key == 13:  # Enter
                if modified:
                    self._save_labels(image_info['label_path'], labels)
                    print(f"[Saved] {len(labels)} labels")
                    cv2.setMouseCallback(self.window_name, lambda *args: None)
                    return True
                break

            elif key == ord('a'):
                draw_mode = not draw_mode
                if not draw_mode:
                    draw_start = None

            elif key == ord('d'):
                if selected_idx is not None and selected_idx < len(labels):
                    del labels[selected_idx]
                    selected_idx = None
                    modified = True

            elif ord('0') <= key <= ord('9'):
                digit = key - ord('0')
                if first_digit is None:
                    first_digit = digit
                else:
                    new_class = first_digit * 10 + digit
                    if 0 <= new_class <= 11 and selected_idx is not None:
                        labels[selected_idx]['class_id'] = new_class
                        modified = True
                        print(f"[Class] -> {PIECE_CLASSES[new_class]}")
                    first_digit = None

        cv2.setMouseCallback(self.window_name, lambda *args: None)
        return modified

    def show_help(self):
        """Show help overlay."""
        help_text = [
            "VALIDATION CONTROLS",
            "",
            "y / SPACE  - All labels CORRECT, next image",
            "e          - EDIT mode (fix labels)",
            "s          - SKIP this image",
            "u          - Go BACK to previous image",
            "q          - QUIT and save progress",
            "h          - Show this help",
            "",
            "EDIT MODE:",
            "  Click      - Select a bounding box",
            "  0-9, 0-9   - Change class (two digits)",
            "  d          - Delete selected box",
            "  a          - Add mode (draw new box)",
            "  ENTER      - Save changes",
            "  ESC        - Cancel",
            "",
            "RED boxes  = Black pieces (b-)",
            "BLUE boxes = White pieces (W-)",
            "",
            "Press any key to continue..."
        ]

        help_img = np.zeros((len(help_text) * 28 + 20, 500, 3), dtype=np.uint8)
        for i, line in enumerate(help_text):
            color = (0, 255, 255) if i == 0 or line == "EDIT MODE:" else (255, 255, 255)
            cv2.putText(help_img, line, (20, 25 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv2.imshow(self.window_name, help_img)
        cv2.waitKey(0)

    def run_validation(self):
        """Main validation loop."""
        if len(self.images) == 0:
            print("[Warning] No images to validate")
            return

        if self.current_idx >= len(self.images):
            self.current_idx = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.current_idx < len(self.images):
            image_info = self.images[self.current_idx]

            display = self.draw_image_display(image_info)
            if display is None:
                self.current_idx += 1
                continue

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("\n[Quit] Saving progress...")
                break

            elif key == ord('y') or key == 32:  # y or SPACE
                self.validation_state[image_info['name']] = 'correct'
                print(f"[OK] {image_info['name']} ({image_info['label_count']} labels)")
                self.current_idx += 1

            elif key == ord('e'):
                if self.edit_image(image_info):
                    self.validation_state[image_info['name']] = 'edited'
                # Refresh display after edit
                continue

            elif key == ord('s'):
                self.validation_state[image_info['name']] = 'skipped'
                print(f"[Skip] {image_info['name']}")
                self.current_idx += 1

            elif key == ord('u'):
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print(f"[Back] Image {self.current_idx + 1}")
                continue

            elif key == ord('h'):
                self.show_help()
                continue

            # Save periodically
            if self.current_idx % 5 == 0:
                self._save_state()

        self._save_state()
        cv2.destroyAllWindows()

        # Print summary
        correct = sum(1 for s in self.validation_state.values() if s == 'correct')
        edited = sum(1 for s in self.validation_state.values() if s == 'edited')
        skipped = sum(1 for s in self.validation_state.values() if s == 'skipped')

        print(f"\n[Done] {len(self.validation_state)}/{len(self.images)} images reviewed")
        print(f"  Correct: {correct} | Edited: {edited} | Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate piece detection labels by image"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset directory or data.yaml"
    )

    args = parser.parse_args()

    dataset_path = Path(args.data)

    # Handle data.yaml path
    if dataset_path.suffix in ['.yaml', '.yml']:
        dataset_path = dataset_path.parent
        print(f"[Info] Using directory: {dataset_path}")

    if not dataset_path.exists():
        print(f"[X] Dataset not found: {dataset_path}")
        return 1

    try:
        validator = LabelValidator(dataset_path)
    except Exception as e:
        print(f"[X] Failed to initialize: {e}")
        return 1

    try:
        validator.run_validation()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"\n[X] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
