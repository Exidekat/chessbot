"""
Collect Corner Training Photos

This script helps you capture multiple photos of your chessboard specifically
for fine-tuning the CORNER DETECTION model (not piece detection).

Purpose: Capture 20+ photos to teach the model to recognize YOUR board's corners.

IMPORTANT: Uses the same 4K MJPEG -> 720p downscale pipeline as inference scripts
to ensure training data matches inference conditions (lighting, exposure, etc.)

Usage:
    python scripts/collect_corner_training_photos.py --device /dev/video0 --count 20
"""

import argparse
import sys
from pathlib import Path
import cv2
import subprocess
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.camera_helpers import (
    get_camera_index_from_device,
    get_default_global_camera,
    get_available_cameras,
    select_camera,
)


def capture_training_photos(device_path, output_dir, count=20):
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
    print("CORNER DETECTION TRAINING PHOTO COLLECTION")
    print("=" * 60)
    print(f"Target: {count} photos")
    print("\nInstructions:")
    print("  - Press SPACE to capture a photo")
    print("  - Press Q to finish early")
    print("  - Try different board positions, angles, and lighting")
    print("  - Make sure all 4 CORNERS are clearly visible in each photo")
    print("  - Empty board or with pieces - both are fine")
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

        cv2.imshow("Corner Training Photo Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Capture photo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"board_{captured+1:03d}_{timestamp}.png"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), frame)
            captured += 1

            print(f"[OK] Captured {captured}/{count}: {filename}")

        elif key == ord('q'):
            print(f"\n[TrainingCapture] Finished early with {captured} photos")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"[OK] Training photo collection complete!")
    print(f"Total photos: {captured}")
    print(f"Saved to: {output_dir}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Label corners: python scripts/label_corners.py --input {output_dir} --output data/training/corner_dataset")
    print(f"  2. Train model: python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml --epochs 100")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect training photos for corner detection fine-tuning"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Camera device (auto-detects WBC-0E01 if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/board_photos",
        help="Output directory for photos (default: data/training/board_photos)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of photos to capture (default: 20)"
    )

    args = parser.parse_args()

    # Camera selection: specified > auto-detect by name > prompt
    if args.device:
        device_path = args.device
        print(f"[OK] Using specified device: {device_path}")
    else:
        device_path = get_default_global_camera()
        if device_path:
            print(f"[OK] Auto-detected global camera: {device_path} (WBC-0E01)")
        else:
            cameras = get_available_cameras()
            if not cameras:
                print("[X] No cameras found!")
                return 1
            elif len(cameras) == 1:
                device_path = cameras[0][0]
                print(f"[OK] Auto-selected camera: {device_path}")
            else:
                print("[WARN] WBC-0E01 not found, prompting for selection...")
                device_path = select_camera(cameras)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Corner Detection Training Photo Collection")
    print("=" * 60)
    print(f"Camera: {device_path}")
    print(f"Output: {output_dir}")
    print(f"Target: {args.count} photos")
    print()

    # Capture photos
    success = capture_training_photos(device_path, output_dir, args.count)

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
