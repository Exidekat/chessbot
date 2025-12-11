"""
Analyze Test Board - Debug corner and piece detection

Usage:
    python scripts/analyze_test_board.py --image data/chessboard.png
    python scripts/analyze_test_board.py --image data/chessboard.png --preprocessing
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def analyze_corners(image_path, conf_thresholds=[0.01, 0.05, 0.1]):
    """Analyze corner detection at various confidence levels."""
    print("\n" + "=" * 60)
    print("Corner Detection Analysis")
    print("=" * 60)

    model_path = "data/best_corners.pt"
    if not Path(model_path).exists():
        print(f"✗ Corner model not found: {model_path}")
        return False

    model = YOLO(model_path)
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return False

    print(f"Image: {image_path}")
    print(f"Size: {image.shape[1]}x{image.shape[0]}")

    print("\nCorner Detection Results:")
    print("-" * 60)

    best_result = None
    best_conf = None

    for conf in conf_thresholds:
        results = model.predict(image, conf=conf, verbose=False)
        num_corners = len(results[0].boxes)

        status = ""
        if num_corners == 4:
            status = " ← PERFECT"
            if best_result is None:
                best_result = results
                best_conf = conf
        elif num_corners < 4:
            status = " (too few)"
        else:
            status = " (too many)"

        print(f"Confidence {conf:0.2f}: {num_corners} corners{status}")

        if num_corners > 0:
            for i, box in enumerate(results[0].boxes):
                conf_val = float(box.conf[0])
                xyxy = box.xyxy[0]
                x = (float(xyxy[0]) + float(xyxy[2])) / 2
                y = (float(xyxy[1]) + float(xyxy[3])) / 2
                print(f"    [{i+1}] ({x:.0f}, {y:.0f}) conf={conf_val:.3f}")

    # Save visualization with best result
    if best_result is not None:
        print(f"\n✓ Found 4 corners at confidence {best_conf}")
        output_path = Path("results") / "corner_detection_debug.png"
        output_path.parent.mkdir(exist_ok=True)

        # Draw corners
        img_viz = image.copy()
        for i, box in enumerate(best_result[0].boxes):
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img_viz, (x_center, y_center), 10, (0, 0, 255), -1)
            cv2.putText(img_viz, f"{i+1}", (x_center + 15, y_center),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imwrite(str(output_path), img_viz)
        print(f"Saved visualization: {output_path}")
    else:
        print("\n✗ Could not find exactly 4 corners")
        print("\nSuggestions:")
        print("  - Ensure all 4 board corners are visible")
        print("  - Improve lighting (avoid glare and shadows)")
        print("  - Position camera perpendicular to board")
        print("  - Move camera closer or farther for better framing")

    return best_result is not None


def test_preprocessing(image_path):
    """Test various preprocessing techniques."""
    print("\n" + "=" * 60)
    print("Preprocessing Analysis")
    print("=" * 60)

    image = cv2.imread(str(image_path))
    output_dir = Path("results") / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Original
    cv2.imwrite(str(output_dir / "1_original.png"), image)
    print("Saved: 1_original.png")

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_dir / "2_grayscale.png"), gray)
    print("Saved: 2_grayscale.png")

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(str(output_dir / "3_clahe.png"), enhanced)
    print("Saved: 3_clahe.png")

    # Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(str(output_dir / "4_sharpened.png"), sharpened)
    print("Saved: 4_sharpened.png")

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(output_dir / "5_edges.png"), edges)
    print("Saved: 5_edges.png")

    print(f"\n✓ Preprocessing examples saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze chess board detection")
    parser.add_argument("--image", required=True, help="Path to board image")
    parser.add_argument("--preprocessing", action="store_true", help="Generate preprocessing examples")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"✗ Image not found: {image_path}")
        return 1

    # Analyze corners
    success = analyze_corners(image_path)

    # Preprocessing examples
    if args.preprocessing:
        test_preprocessing(image_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
