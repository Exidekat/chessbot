"""
Collect Piece Training Photos

This script helps you capture multiple photos of your chessboard specifically
for fine-tuning the PIECE DETECTION model (not corner detection).

Purpose: Capture 30+ photos with various board positions to teach the model
to recognize YOUR specific chess pieces (all 12 piece types: 6 white + 6 black).

Usage:
    python scripts/collect_piece_training_photos.py --device /dev/video0 --count 30
"""

import argparse
import sys
from pathlib import Path
import cv2
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_camera_index_from_device(device_path):
    """Extract camera index from device path."""
    try:
        if device_path.startswith('/dev/video'):
            return int(device_path.replace('/dev/video', ''))
    except ValueError:
        pass

    try:
        return int(device_path)
    except ValueError:
        print(f"[X] Cannot parse device path: {device_path}")
        sys.exit(1)


def capture_training_photos(device_path, output_dir, count=30):
    """
    Capture training photos interactively.

    Args:
        device_path: Camera device path
        output_dir: Directory to save photos
        count: Number of photos to capture
    """
    camera_index = get_camera_index_from_device(device_path)

    print(f"\n[TrainingCapture] Opening camera: {device_path}")

    # Configure camera for YUYV format
    print(f"[TrainingCapture] Setting YUYV 1280x720 format...")
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-fmt-video=width=1280,height=720,pixelformat=YUYV"
        ], check=True, capture_output=True, text=True)
        print(f"[TrainingCapture] [OK] Format set to YUYV 1280x720")
    except subprocess.CalledProcessError as e:
        print(f"[TrainingCapture] Warning: Could not set format: {e}")

    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[TrainingCapture] [X] Failed to open camera")
        return False

    # Explicitly set resolution in OpenCV (v4l2-ctl may not persist)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verify resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[TrainingCapture] Camera resolution: {width}x{height}")

    if width != 1280 or height != 720:
        print(f"[TrainingCapture] Warning: Expected 1280x720, got {width}x{height}")
        print(f"[TrainingCapture] Camera may not support YUYV at 720p, using available resolution")

    # Warm up
    print("[TrainingCapture] Warming up camera...")
    for _ in range(5):
        cap.read()

    print("\n" + "=" * 60)
    print("PIECE DETECTION TRAINING PHOTO COLLECTION")
    print("=" * 60)
    print(f"Target: {count} photos")
    print("\nInstructions:")
    print("  - Set up different board positions between each photo")
    print("  - Include ALL piece types in various positions:")
    print("    * Pawns, Knights, Bishops, Rooks, Queens, Kings")
    print("    * Both white and black pieces")
    print("  - Try different arrangements:")
    print("    * Opening positions (e.g., e4, d4, Nf3, etc.)")
    print("    * Mid-game positions (pieces scattered)")
    print("    * Endgame positions (few pieces remaining)")
    print("    * Edge/corner positions (test boundary detection)")
    print("  - Vary lighting conditions if possible")
    print("  - Ensure all 4 CORNERS are visible (for transform)")
    print("\n  Press SPACE to capture a photo")
    print("  Press Q to finish early")
    print("=" * 60)

    captured = 0

    while captured < count:
        ret, frame = cap.read()

        if not ret:
            print("[TrainingCapture] [X] Failed to read frame")
            break

        # Draw status overlay
        display_frame = frame.copy()
        status_text = f"Photos: {captured}/{count} - Press SPACE to capture, Q to quit"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add reminder for variety
        if captured > 0 and captured % 5 == 0:
            reminder_text = "Reminder: Change board position!"
            cv2.putText(display_frame, reminder_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Piece Training Photo Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Capture photo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"board_{captured+1:03d}_{timestamp}.png"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), frame)
            captured += 1

            print(f"[OK] Captured {captured}/{count}: {filename}")

            # Print suggestion for next position
            if captured < count:
                suggestions = [
                    "Try a different opening position",
                    "Add more pieces to the board",
                    "Try an endgame position with fewer pieces",
                    "Move pieces to the edges/corners",
                    "Try a completely different setup",
                    "Include more knights and bishops",
                    "Include more rooks and queens",
                    "Mix piece positions randomly"
                ]
                suggestion = suggestions[captured % len(suggestions)]
                print(f"    Next: {suggestion}")

        elif key == ord('q'):
            print(f"\n[TrainingCapture] Finished early with {captured} photos")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"[OK] Training photo collection complete!")
    print(f"Total photos: {captured}")
    print(f"Saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Label pieces: python scripts/label_pieces.py --input {output_dir}")
    print(f"  2. Fine-tune model: python scripts/finetune_pieces.py")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect training photos for piece detection fine-tuning"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Camera device (e.g., /dev/video0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/piece_photos",
        help="Output directory for photos (default: data/training/piece_photos)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of photos to capture (default: 30)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Piece Detection Training Photo Collection")
    print("=" * 60)
    print(f"Camera: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Target: {args.count} photos")
    print()

    # Capture photos
    success = capture_training_photos(args.device, output_dir, args.count)

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
