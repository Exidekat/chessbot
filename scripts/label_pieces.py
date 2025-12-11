"""
Label Pieces for Piece Detection Training

Interactive tool to label PIECES (bounding boxes) in training photos.
This creates a YOLO-format dataset specifically for fine-tuning the PIECE detection model.

NOTE: This is for PIECE detection only, not corner detection.

Supports 12 piece classes:
  0: b-bishop    1: b-king      2: b-knight
  3: b-pawn      4: b-queen     5: b-rook
  6: W-bishop    7: W-king      8: W-knight
  9: W-pawn     10: W-queen    11: W-rook

Usage:
    python scripts/label_pieces.py --input data/training/piece_photos
"""

import argparse
import sys
from pathlib import Path
import cv2
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Piece class mapping (same as YOLO model training)
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

# Reverse mapping for quick lookup
CLASS_TO_ID = {v: k for k, v in PIECE_CLASSES.items()}


class PieceLabeler:
    """Interactive piece bounding box labeling tool."""

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.display_image = self.image.copy()
        self.boxes = []  # List of (class_id, x1, y1, x2, y2)
        self.current_box = None  # Current box being drawn (x1, y1, x2, y2)
        self.current_class = 9  # Default to white pawn
        self.window_name = "Label Pieces - Draw boxes around pieces"
        self.white_mode = True  # Toggle between white (True) and black (False)

        # Piece class shortcuts (keyboard keys) - lowercase only
        self.shortcuts = {
            'b': 'bishop',
            'k': 'king',
            'n': 'knight',
            'p': 'pawn',
            'q': 'queen',
            'r': 'rook'
        }

        # Mapping from piece type + color to class ID
        self.piece_to_class = {
            ('bishop', False): 0,   # b-bishop
            ('king', False): 1,     # b-king
            ('knight', False): 2,   # b-knight
            ('pawn', False): 3,     # b-pawn
            ('queen', False): 4,    # b-queen
            ('rook', False): 5,     # b-rook
            ('bishop', True): 6,    # W-bishop
            ('king', True): 7,      # W-king
            ('knight', True): 8,    # W-knight
            ('pawn', True): 9,      # W-pawn
            ('queen', True): 10,    # W-queen
            ('rook', True): 11      # W-rook
        }

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for bounding box drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start new box
            self.current_box = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.current_box is not None:
                # Update box end point
                self.current_box[2] = x
                self.current_box[3] = y
                self.draw_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_box is not None:
                # Finish box
                self.current_box[2] = x
                self.current_box[3] = y

                # Only save if box has non-zero area
                x1, y1, x2, y2 = self.current_box
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    # Normalize coordinates (ensure x1 < x2, y1 < y2)
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    self.boxes.append((self.current_class, x1, y1, x2, y2))
                    print(f"  Added: {PIECE_CLASSES[self.current_class]} at ({x1},{y1}) to ({x2},{y2})")

                self.current_box = None
                self.draw_display()

    def draw_display(self):
        """Draw current state on display image."""
        self.display_image = self.image.copy()

        # Draw saved boxes
        for class_id, x1, y1, x2, y2 in self.boxes:
            color = self.get_class_color(class_id)
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = PIECE_CLASSES[class_id]
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(self.display_image, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(self.display_image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw current box being drawn
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            color = self.get_class_color(self.current_class)
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)

        # Draw status
        color_text = "WHITE" if self.white_mode else "BLACK"
        status = f"Mode: {color_text} | Class: {PIECE_CLASSES[self.current_class]} | Boxes: {len(self.boxes)}"
        cv2.putText(self.display_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw instructions
        instructions = "TAB=toggle color | p/r/n/b/q/k=piece type | u=undo | ENTER=save"
        cv2.putText(self.display_image, instructions, (10, self.image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(self.window_name, self.display_image)

    def get_class_color(self, class_id):
        """Get color for piece class (white pieces = blue, black pieces = red)."""
        if class_id < 6:
            return (0, 0, 255)  # Black pieces = red
        else:
            return (255, 0, 0)  # White pieces = blue

    def label(self):
        """Run interactive labeling."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.draw_display()

        print(f"\nLabeling: {self.image_path.name}")
        print("  Draw bounding boxes around pieces")
        print("  Controls:")
        print("    TAB = toggle between WHITE/BLACK mode")
        print("    p/r/n/b/q/k = select piece type (pawn/rook/knight/bishop/queen/king)")
        print("    u = undo last box")
        print("    ENTER = save and continue")
        print("    s = skip this image")
        print("    q = quit labeling")

        while True:
            key = cv2.waitKey(1) & 0xFF

            # Toggle color mode with TAB
            if key == 9:  # TAB key
                self.white_mode = not self.white_mode
                # Update current class to match new color
                piece_type = None
                for ptype, cls_id in self.piece_to_class.items():
                    if cls_id == self.current_class:
                        piece_type = ptype[0]
                        break
                if piece_type:
                    self.current_class = self.piece_to_class[(piece_type, self.white_mode)]
                color_text = "WHITE" if self.white_mode else "BLACK"
                print(f"  Switched to {color_text} mode")
                self.draw_display()
                continue

            # Check for piece type selection
            if key != 255:  # 255 means no key pressed
                try:
                    key_char = chr(key).lower()  # Convert to lowercase
                    if key_char in self.shortcuts:
                        piece_type = self.shortcuts[key_char]
                        self.current_class = self.piece_to_class[(piece_type, self.white_mode)]
                        print(f"  Selected: {PIECE_CLASSES[self.current_class]}")
                        self.draw_display()
                        continue  # Skip other checks
                except ValueError:
                    pass  # Invalid character, ignore

            if key == ord('u'):
                # Undo last box
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"  Removed: {PIECE_CLASSES[removed[0]]}")
                    self.draw_display()

            if key == 13:  # Enter
                cv2.destroyWindow(self.window_name)
                return self.boxes

            if key == ord('s'):
                # Skip this image
                print("  [Skipped]")
                cv2.destroyWindow(self.window_name)
                return None

            if key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None

        return None


def convert_to_yolo_format(boxes, img_width, img_height):
    """
    Convert bounding boxes to YOLO format.

    Args:
        boxes: List of (class_id, x1, y1, x2, y2)
        img_width: Image width
        img_height: Image height

    Returns:
        List of YOLO format strings: "class_id x_center y_center width height"
    """
    yolo_labels = []

    for class_id, x1, y1, x2, y2 in boxes:
        # Calculate center and size
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # Normalize to 0-1
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # YOLO format: class x_center y_center width height (all normalized 0-1)
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_labels


def create_yolo_dataset(input_dir, output_dir):
    """
    Create YOLO dataset from labeled photos.

    Args:
        input_dir: Directory with training photos
        output_dir: Directory for YOLO dataset
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create dataset structure
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))

    if not image_files:
        print(f"[X] No images found in {input_dir}")
        return False

    print("\n" + "=" * 60)
    print(f"PIECE LABELING")
    print("=" * 60)
    print(f"Found {len(image_files)} images")
    print("\nInstructions:")
    print("  1. Draw bounding boxes around each piece")
    print("  2. Press keyboard shortcuts to select piece class:")
    print("     Black: b=bishop, k=king, n=knight, p=pawn, q=queen, r=rook")
    print("     White: B=Bishop, K=King, N=Knight, P=Pawn, Q=Queen, R=Rook")
    print("  3. Press u to undo last box")
    print("  4. Press ENTER to save and move to next image")
    print("  5. Press s to skip image, q to quit")
    print("=" * 60)

    labeled_count = 0
    total_pieces = 0

    for idx, image_file in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file.name}")

        try:
            # Label pieces
            labeler = PieceLabeler(image_file)
            boxes = labeler.label()

            if boxes is None:
                print("  [Skipped]")
                continue

            if len(boxes) == 0:
                print("  [Warning] No pieces labeled")
                continue

            # Get image dimensions
            img = cv2.imread(str(image_file))
            img_height, img_width = img.shape[:2]

            # Convert to YOLO format
            yolo_labels = convert_to_yolo_format(boxes, img_width, img_height)

            # Save image and label
            image_dest = images_dir / image_file.name
            label_dest = labels_dir / (image_file.stem + ".txt")

            # Copy image
            shutil.copy(image_file, image_dest)

            # Write label file
            with open(label_dest, 'w') as f:
                f.write('\n'.join(yolo_labels))

            print(f"  [OK] Saved {len(boxes)} pieces to dataset")
            labeled_count += 1
            total_pieces += len(boxes)

        except Exception as e:
            print(f"  [X] Error: {e}")
            continue

    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images',
        'val': 'images',  # Use same images for validation (small dataset)
        'nc': 12,  # Number of classes
        'names': list(PIECE_CLASSES.values())
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("[OK] Dataset creation complete!")
    print("=" * 60)
    print(f"Labeled images: {labeled_count}/{len(image_files)}")
    print(f"Total pieces: {total_pieces}")
    print(f"Avg pieces/image: {total_pieces/labeled_count:.1f}" if labeled_count > 0 else "")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Config: {yaml_path}")
    print("=" * 60)

    return labeled_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Label pieces in training photos for YOLO fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/piece_photos",
        help="Input directory with training photos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/piece_dataset",
        help="Output directory for YOLO dataset"
    )

    args = parser.parse_args()

    success = create_yolo_dataset(args.input, args.output)

    if success:
        print("\nNext steps:")
        print("  1. Review the dataset in:", args.output)
        print("  2. Fine-tune the model:")
        print(f"     python scripts/finetune_pieces.py --data {args.output}/data.yaml")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
