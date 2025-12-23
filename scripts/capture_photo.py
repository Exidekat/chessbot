"""
Capture Photo from USB Camera

Smart camera selection:
- If --device specified: Use that device
- If multiple cameras: Prompt user to select
- If single camera: Auto-select
- If no cameras: Exit with error

Usage:
    python scripts/capture_photo.py                          # Auto-detect camera
    python scripts/capture_photo.py --device /dev/video0     # Specific device
    python scripts/capture_photo.py --output custom.png      # Custom filename
"""

import argparse
import sys
import cv2
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device
)


def capture_photo(device_path, output_path, preview=True, width=None, height=None):
    """
    Capture a photo from the specified camera.

    Args:
        device_path: Camera device path
        output_path: Path to save the captured image
        preview: Whether to show preview window
        width: Optional resolution width
        height: Optional resolution height

    Returns:
        bool: True if successful, False otherwise
    """
    camera_index = get_camera_index_from_device(device_path)

    print(f"\nOpening camera: {device_path} (index: {camera_index})")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"✗ Failed to open camera: {device_path}")
        return False

    # Set resolution (defaults to 1920x1080 for good quality without USB bandwidth issues)
    if width is not None and height is not None:
        print(f"Setting resolution: {width}x{height}")
        # Use MJPEG for high resolutions, increase buffer size to avoid corruption
        if width >= 1920 or height >= 1080:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    # Warm up camera (capture more frames for high-res to stabilize)
    print("Warming up camera...")
    warmup_frames = 10 if (width >= 3000 or height >= 2000) else 5
    for _ in range(warmup_frames):
        cap.read()

    # Extra delay for high resolution cameras
    if width >= 3000 or height >= 2000:
        import time
        time.sleep(0.5)

    if preview:
        print("\nPreview mode:")
        print("  - Press SPACE to capture")
        print("  - Press Q to quit")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("✗ Failed to read frame from camera")
                cap.release()
                cv2.destroyAllWindows()
                return False

            # Display preview
            cv2.imshow("Camera Preview - Press SPACE to capture, Q to quit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Capture photo
                cv2.imwrite(str(output_path), frame)
                print(f"\n✓ Photo captured: {output_path}")
                break
            elif key == ord('q'):
                print("\n✗ Cancelled by user")
                cap.release()
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
    else:
        # Auto-capture without preview
        print("Capturing photo...")
        ret, frame = cap.read()

        if not ret:
            print("✗ Failed to read frame from camera")
            cap.release()
            return False

        cv2.imwrite(str(output_path), frame)
        print(f"✓ Photo captured: {output_path}")

    cap.release()
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture photo from USB camera with smart device selection"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific camera device (e.g., /dev/video0 or 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: capture_YYYYMMDD_HHMMSS.png in results/)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture immediately without preview window"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Camera resolution width (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Camera resolution height (default: 1080)"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("USB Camera Photo Capture")
    print("=" * 60)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("results") / f"capture_{timestamp}.png"

    # Ensure results directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which camera to use
    if args.device:
        # User specified device
        device_path = args.device
        print(f"\nUsing specified device: {device_path}")
    else:
        # Auto-detect cameras
        print("\nDetecting available cameras...")
        cameras = get_available_cameras()

        if not cameras:
            print("\n✗ No cameras found!")
            print("\nTroubleshooting:")
            print("  1. Check camera is connected via USB")
            print("  2. Check camera permissions: ls -l /dev/video*")
            print("  3. Add user to video group: sudo usermod -a -G video $USER")
            print("  4. Verify with: v4l2-ctl --list-devices")
            return 1

        if len(cameras) == 1:
            # Auto-select single camera
            device_path = cameras[0][0]
            print(f"\n✓ Auto-selected camera: {device_path} - {cameras[0][1]}")
        else:
            # Multiple cameras - prompt user
            device_path = select_camera(cameras)

    # Capture photo
    print("\n" + "=" * 60)
    success = capture_photo(
        device_path,
        output_path,
        preview=not args.no_preview,
        width=args.width,
        height=args.height
    )
    print("=" * 60)

    if success:
        print(f"\n✓ Photo saved to: {output_path.absolute()}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        return 0
    else:
        print("\n✗ Photo capture failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
