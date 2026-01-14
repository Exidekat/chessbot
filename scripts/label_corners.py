"""
Label Corners for Corner Detection Training

Interactive tool to label the 4 CORNERS of the chessboard in training photos.
This creates a YOLO-format dataset specifically for fine-tuning the CORNER detection model.

NOTE: This is for CORNER detection only, not piece detection.

Usage:
    python scripts/label_corners.py --input data/training/board_photos
"""

import argparse
import sys
from pathlib import Path
import cv2
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_preprocessing import preprocess_for_corner_detection


class CornerLabeler:
    """Interactive corner labeling tool."""

    def __init__(self, image_path):
        self.image_path = image_path

        # Load and preprocess image (grayscale + CLAHE)
        # This shows the user the same view the model will see during inference
        self.image = preprocess_for_corner_detection(str(image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.display_image = self.image.copy()
        self.corners = []  # List of (x, y) tuples
        self.corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        self.window_name = "Label Corners (Preprocessed) - Click 4 corners: TL, TR, BR, BL"

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            # Add corner
            self.corners.append((x, y))

            # Redraw
            self.draw_corners()

            print(f"  [{len(self.corners)}/4] {self.corner_names[len(self.corners)-1]}: ({x}, {y})")

    def draw_corners(self):
        """Draw current corners on display image."""
        self.display_image = self.image.copy()

        # Draw existing corners
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0)]  # Red, Yellow, Blue, Cyan
        for idx, (x, y) in enumerate(self.corners):
            color = colors[idx]
            cv2.circle(self.display_image, (x, y), 20, color, -1)
            cv2.putText(self.display_image, self.corner_names[idx][:2],
                       (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw status
        status = f"Corners: {len(self.corners)}/4"
        if len(self.corners) < 4:
            status += f" - Click {self.corner_names[len(self.corners)]}"
        else:
            status += " - Press ENTER to save, R to reset"

        cv2.putText(self.display_image, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(self.window_name, self.display_image)

    def label(self):
        """Run interactive labeling."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.draw_corners()

        print(f"\nLabeling: {self.image_path.name}")
        print("  Click corners in order: TL, TR, BR, BL")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                # Reset
                self.corners = []
                self.draw_corners()
                print("  [Reset] Cleared all corners")

            elif key == 13:  # Enter
                if len(self.corners) == 4:
                    cv2.destroyWindow(self.window_name)
                    return self.corners
                else:
                    print(f"  [Warning] Need 4 corners, have {len(self.corners)}")

            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None

        return None


def convert_to_yolo_format(corners, img_width, img_height):
    """
    Convert corner coordinates to YOLO bounding box format.

    Each corner becomes a small bounding box (20x20 pixels) centered on the corner.

    Args:
        corners: List of (x, y) corner coordinates
        img_width: Image width
        img_height: Image height

    Returns:
        List of YOLO format strings: "class_id x_center y_center width height"
    """
    yolo_labels = []

    # Each corner is class 0 (we only have one class: "corner")
    box_size_px = 40  # 40x40 pixel box around each corner

    for x, y in corners:
        # Convert to normalized coordinates
        x_center = x / img_width
        y_center = y / img_height
        width = box_size_px / img_width
        height = box_size_px / img_height

        # YOLO format: class x_center y_center width height (all normalized 0-1)
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

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
    print(f"CORNER LABELING")
    print("=" * 60)
    print(f"Found {len(image_files)} images")
    print("\nInstructions:")
    print("  1. Click 4 corners in order: TL, TR, BR, BL")
    print("  2. Press R to reset if you make a mistake")
    print("  3. Press ENTER to save and move to next image")
    print("  4. Press Q to quit")
    print("=" * 60)

    labeled_count = 0
    skipped_count = 0

    for idx, image_file in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] Processing: {image_file.name}")

        # Check if already labeled
        label_dest = labels_dir / (image_file.stem + ".txt")
        image_dest = images_dir / image_file.name

        if label_dest.exists() and image_dest.exists():
            print(f"  [Already labeled] Skipping...")
            skipped_count += 1
            continue

        try:
            # Label corners
            labeler = CornerLabeler(image_file)
            corners = labeler.label()

            if corners is None:
                print("  [Skipped]")
                continue

            # Get image dimensions
            img = cv2.imread(str(image_file))
            img_height, img_width = img.shape[:2]

            # Convert to YOLO format
            yolo_labels = convert_to_yolo_format(corners, img_width, img_height)

            # Save image and label
            image_dest = images_dir / image_file.name
            label_dest = labels_dir / (image_file.stem + ".txt")

            # Save preprocessed image (not original - training should match inference)
            preprocessed = preprocess_for_corner_detection(str(image_file))
            cv2.imwrite(str(image_dest), preprocessed)

            # Write label file
            with open(label_dest, 'w') as f:
                f.write('\n'.join(yolo_labels))

            print(f"  [OK] Saved to dataset")
            labeled_count += 1

        except Exception as e:
            print(f"  [X] Error: {e}")
            continue

    # Create data.yaml
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images',
        'val': 'images',  # Use same images for validation (small dataset)
        'nc': 1,  # Number of classes
        'names': ['corner']
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print("\n" + "=" * 60)
    print("[OK] Dataset creation complete!")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Already labeled: {skipped_count}")
    print(f"Newly labeled: {labeled_count}")
    print(f"Final dataset size: {labeled_count + skipped_count}")
    print()
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    print(f"Config: {yaml_path}")
    print("=" * 60)

    return labeled_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Label corners in training photos for YOLO fine-tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/board_photos",
        help="Input directory with training photos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/corner_dataset",
        help="Output directory for YOLO dataset"
    )

    args = parser.parse_args()

    success = create_yolo_dataset(args.input, args.output)

    if success:
        print("\nNext steps:")
        print("  1. Review the dataset in:", args.output)
        print("  2. Fine-tune the model:")
        print(f"     python scripts/finetune_corners.py --data {args.output}/data.yaml")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
