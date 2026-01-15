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

from utils.camera_helpers import (
    get_camera_index_from_device,
    get_default_global_camera,
    get_available_cameras,
    select_camera,
)


def capture_training_photos(device_path, output_dir, count=30, use_yuyv=False):
    """
    Capture training photos interactively.

    Supports two capture modes:
    - Default: 4K MJPEG -> 720p downscale (supersampled, anti-aliased)
    - YUYV: Native 720p YUYV uncompressed (best quality, no JPEG artifacts)

    This matches the exact capture pipeline used in deployment scripts
    (best_move_demo.py, collect_vla_episodes.py) to ensure training data
    has the same characteristics as inference data.

    Args:
        device_path: Camera device path
        output_dir: Directory to save photos
        count: Number of photos to capture
        use_yuyv: If True, use native 720p YUYV instead of 4K MJPEG downscale
    """
    import gc
    import time

    camera_index = get_camera_index_from_device(device_path)

    print(f"\n[TrainingCapture] Opening camera: {device_path}")

    # Configure camera based on capture mode
    if use_yuyv:
        # YUYV mode: Native 720p uncompressed (best quality)
        print(f"[TrainingCapture] Setting YUYV 1280x720 @ 10fps format...")
        try:
            subprocess.run([
                "v4l2-ctl",
                f"--device={device_path}",
                "--set-fmt-video=width=1280,height=720,pixelformat=YUYV",
                "--set-parm=10"
            ], check=True, capture_output=True, text=True)
            print(f"[TrainingCapture] [OK] Format set to YUYV 1280x720 @ 10fps")
        except subprocess.CalledProcessError as e:
            print(f"[TrainingCapture] Warning: Could not set format: {e}")
        capture_mode = "720p YUYV"
        target_width, target_height, target_fps = 1280, 720, 10
        fourcc = cv2.VideoWriter_fourcc(*'YUYV')
    else:
        # Default mode: 4K MJPEG -> 720p downscale
        print(f"[TrainingCapture] Setting MJPEG 3840x2160 @ 30fps format...")
        try:
            subprocess.run([
                "v4l2-ctl",
                f"--device={device_path}",
                "--set-fmt-video=width=3840,height=2160,pixelformat=MJPG",
                "--set-parm=30"
            ], check=True, capture_output=True, text=True)
            print(f"[TrainingCapture] [OK] Format set to MJPEG 3840x2160 @ 30fps")
        except subprocess.CalledProcessError as e:
            print(f"[TrainingCapture] Warning: Could not set format: {e}")
        capture_mode = "4K -> 720p"
        target_width, target_height, target_fps = 3840, 2160, 30
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    # Reset camera to consistent auto settings
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-ctrl=white_balance_automatic=1",
            "--set-ctrl=focus_automatic_continuous=1",
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pass  # Non-critical

    # Open camera with V4L2 backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[TrainingCapture] [X] Failed to open camera")
        return False

    # Explicitly set resolution in OpenCV
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    # Verify resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[TrainingCapture] Camera resolution: {width}x{height} @ {fps}fps")

    if width != target_width or height != target_height:
        print(f"[TrainingCapture] Warning: Expected {target_width}x{target_height}, got {width}x{height}")
        print(f"[TrainingCapture] Attempting to continue with available resolution")

    # Warm up camera (let autofocus/exposure settle)
    print("[TrainingCapture] Warming up camera (1 second)...")
    start = time.time()
    while time.time() - start < 1.0:
        cap.read()

    print("\n" + "=" * 60)
    print("PIECE DETECTION TRAINING PHOTO COLLECTION")
    print("=" * 60)
    print(f"Target: {count} photos")
    print(f"Format: {capture_mode} (matches deployment)")
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
        ret, frame_raw = cap.read()

        if not ret:
            print("[TrainingCapture] [X] Failed to read frame")
            break

        # Get 720p frame (downscale if needed)
        if use_yuyv:
            # YUYV is already 720p, no downscale needed
            frame_720p = frame_raw
        else:
            # Downscale 4K to 720p using LANCZOS4
            frame_720p = cv2.resize(frame_raw, (1280, 720), interpolation=cv2.INTER_LANCZOS4)

        # Draw status overlay on display copy
        display_frame = frame_720p.copy()
        status_text = f"Photos: {captured}/{count} - Press SPACE to capture, Q to quit"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show format info
        format_text = f"{capture_mode} (matches deployment)"
        cv2.putText(display_frame, format_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Add reminder for variety
        if captured > 0 and captured % 5 == 0:
            reminder_text = "Reminder: Change board position!"
            cv2.putText(display_frame, reminder_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Piece Training Photo Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # Save the downscaled 720p frame (NOT the 4K frame)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"board_{captured+1:03d}_{timestamp}.png"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), frame_720p)
            captured += 1

            print(f"[OK] Captured {captured}/{count}: {filename} ({capture_mode})")

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

    # Release camera (GC method for WBC-0E01 compatibility)
    del cap
    gc.collect()
    cv2.destroyAllWindows()
    print("[TrainingCapture] Camera released (GC)")

    print("\n" + "=" * 60)
    print(f"[OK] Training photo collection complete!")
    print(f"Total photos: {captured}")
    print(f"Format: {capture_mode} (matches deployment)")
    print(f"Saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Label pieces: python scripts/label_pieces.py --input {output_dir}")
    print(f"  2. Fine-tune model: python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect training photos for piece detection fine-tuning"
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
        default="data/training/piece_photos",
        help="Output directory for photos (default: data/training/piece_photos)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of photos to capture (default: 30)"
    )
    parser.add_argument(
        "--yuyv",
        action="store_true",
        help="Use native 720p YUYV capture (uncompressed, best quality, 10fps)"
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

    # Check for existing images (we ADD to them, never delete)
    existing_images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))

    print("=" * 60)
    print("Piece Detection Training Photo Collection")
    print("=" * 60)
    print(f"Camera: {device_path}")
    print(f"Output: {output_dir}")
    if existing_images:
        print(f"Existing images: {len(existing_images)} (will be PRESERVED)")
    print(f"New images to capture: {args.count}")
    print(f"Total after capture: {len(existing_images) + args.count}")
    print()

    # Capture photos
    success = capture_training_photos(device_path, output_dir, args.count, use_yuyv=args.yuyv)

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
