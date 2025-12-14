#!/usr/bin/env python3
"""
Camera Helper Utilities

Shared utilities for camera detection and capture across ChessBot scripts.
Extracted from duplicated code in:
- scripts/best_move_demo.py
- scripts/create_overlay_demo.py
- scripts/virtual_overlay_demo.py
- vla/vla_deploy.py

Functions:
- get_available_cameras(): Detect all USB cameras via v4l2
- select_camera(): Interactive camera selection prompt
- get_camera_index_from_device(): Parse device path to OpenCV index
- capture_4k_downscale(): Capture 4K MJPEG and downscale to 720p
"""

import sys
import subprocess
import time
import gc
from pathlib import Path
from typing import List, Tuple

try:
    import cv2
except ImportError:
    print("[X] OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


def get_available_cameras() -> List[Tuple[str, str]]:
    """
    Detect all available USB cameras.

    Uses v4l2-ctl (Linux) to enumerate video devices. Falls back to brute-force
    device opening if v4l2-ctl is not available.

    Returns:
        List of (device_path, device_info) tuples, e.g.:
        [('/dev/video0', 'WBC-0E01: WBC-0E01'), ('/dev/video2', 'eMeet C950')]

    Example:
        >>> cameras = get_available_cameras()
        >>> if cameras:
        >>>     print(f"Found {len(cameras)} camera(s)")
        >>>     for device_path, device_info in cameras:
        >>>         print(f"  {device_path}: {device_info}")
    """
    cameras = []

    # Method 1: Try v4l2-ctl (Linux standard)
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            current_device_name = None

            for line in lines:
                if line and not line.startswith('\t') and not line.startswith(' '):
                    # Device name line
                    current_device_name = line.strip().rstrip(':')
                elif line.strip().startswith('/dev/video'):
                    # Device path line
                    device_path = line.strip()
                    if current_device_name:
                        cameras.append((device_path, current_device_name))

            return cameras
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: Fallback - try opening /dev/video* devices directly
    print("v4l2-ctl not available, using fallback camera detection...")
    for i in range(10):
        device_path = f"/dev/video{i}"
        if Path(device_path).exists():
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it's a real camera
                ret, _ = cap.read()
                if ret:
                    cameras.append((device_path, f"Camera {i}"))
                cap.release()

    return cameras


def select_camera(cameras: List[Tuple[str, str]]) -> str:
    """
    Prompt user to select a camera from the list.

    Args:
        cameras: List of (device_path, device_info) tuples from get_available_cameras()

    Returns:
        Selected device path (e.g., "/dev/video0")

    Raises:
        SystemExit: If user cancels with Ctrl+C

    Example:
        >>> cameras = get_available_cameras()
        >>> if len(cameras) > 1:
        >>>     device = select_camera(cameras)
        >>> else:
        >>>     device = cameras[0][0]  # Auto-select single camera
    """
    print("\n" + "=" * 60)
    print("Multiple cameras detected:")
    print("=" * 60)

    for idx, (device_path, device_info) in enumerate(cameras, 1):
        print(f"  [{idx}] {device_path} - {device_info}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nSelect camera [1-{len(cameras)}]: ").strip()
            idx = int(choice) - 1

            if 0 <= idx < len(cameras):
                selected = cameras[idx][0]
                print(f"[OK] Selected: {selected}")
                return selected
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(cameras)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n[X] Cancelled by user")
            sys.exit(1)


def get_camera_index_from_device(device_path: str) -> int:
    """
    Extract camera index from device path for OpenCV.

    Args:
        device_path: Path to video device (e.g., "/dev/video0" or "0")

    Returns:
        Camera index for OpenCV VideoCapture (e.g., 0 for /dev/video0)

    Raises:
        SystemExit: If device path cannot be parsed

    Example:
        >>> idx = get_camera_index_from_device("/dev/video2")
        >>> cap = cv2.VideoCapture(idx)
    """
    try:
        # Extract number from /dev/video{N}
        if device_path.startswith('/dev/video'):
            return int(device_path.replace('/dev/video', ''))
    except ValueError:
        pass

    # Fallback: try to parse as integer
    try:
        return int(device_path)
    except ValueError:
        print(f"[X] Cannot parse device path: {device_path}")
        sys.exit(1)


def capture_4k_downscale(device_path: str, output_path: Path) -> bool:
    """
    Capture 4K MJPEG video for 1 second, then downscale best frame to 1280x720.

    This approach uses the camera's 4K MJPEG mode (which produces better quality
    than direct 720p YUYV) and downscales to the 720p resolution required by
    ChessBot's YOLO detection models.

    The function:
    1. Configures camera to 4K MJPEG @ 30fps via v4l2-ctl
    2. Captures frames for 1 second
    3. Selects middle frame (avoids motion blur at start/end)
    4. Downscales to 1280x720 using LANCZOS4 interpolation
    5. Saves as PNG

    Args:
        device_path: Camera device path (e.g., "/dev/video0")
        output_path: Path to save the captured 720p image

    Returns:
        True if successful, False otherwise

    Note:
        This function releases the camera using garbage collection instead of
        cap.release() to work around WBC-0E01 camera quirks (errno=19 issue).

    Example:
        >>> from pathlib import Path
        >>> success = capture_4k_downscale("/dev/video0", Path("board.png"))
        >>> if success:
        >>>     print("Captured successfully!")
    """
    camera_index = get_camera_index_from_device(device_path)

    print(f"\n[CameraCapture] Opening camera: {device_path} (index: {camera_index})")

    # Configure camera for MJPEG 3840x2160 (4K)
    print(f"[CameraCapture] Setting MJPEG 3840x2160 @ 30fps format...")
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-fmt-video=width=3840,height=2160,pixelformat=MJPG",
            "--set-parm=30"
        ], check=True, capture_output=True, text=True)
        print(f"[CameraCapture] [OK] Format set to MJPEG 3840x2160 @ 30fps")
    except subprocess.CalledProcessError as e:
        print(f"[CameraCapture] Warning: Could not set format via v4l2-ctl: {e}")
        print(f"[CameraCapture] Continuing with OpenCV defaults...")

    # Open camera with V4L2 backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[CameraCapture] [X] Failed to open camera: {device_path}")
        return False

    # Explicitly set resolution and FPS in OpenCV
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Verify resolution and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[CameraCapture] Camera resolution: {width}x{height} @ {fps}fps")

    if width != 3840 or height != 2160:
        print(f"[CameraCapture] Warning: Expected 3840x2160, got {width}x{height}")
        print(f"[CameraCapture] Attempting to continue with available resolution")

    # Capture frames for 1 second
    print(f"[CameraCapture] Capturing 4K frames for 1 second...")
    frames = []
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 1.0:
        ret, frame = cap.read()
        if not ret:
            print(f"[CameraCapture] [X] Failed to read frame {frame_count + 1}")
            del cap
            gc.collect()
            return False
        frames.append(frame)
        frame_count += 1

    print(f"[CameraCapture] [OK] Captured {len(frames)} 4K frames")

    # Release camera early (before processing)
    del cap
    gc.collect()
    print("[CameraCapture] [OK] Camera released (GC)")

    if not frames:
        print(f"[CameraCapture] [X] No frames captured")
        return False

    # Use the middle frame (best chance of avoiding motion blur from start/end)
    middle_idx = len(frames) // 2
    best_frame = frames[middle_idx]
    print(f"[CameraCapture] Using frame {middle_idx + 1}/{len(frames)} (middle frame)")

    # Downscale to 1280x720 using high-quality interpolation
    print(f"[CameraCapture] Downscaling from {best_frame.shape[1]}x{best_frame.shape[0]} to 1280x720...")
    downscaled = cv2.resize(best_frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)

    # Save downscaled image
    cv2.imwrite(str(output_path), downscaled)
    print(f"[CameraCapture] [OK] Photo saved: {output_path}")
    print(f"[CameraCapture] Size: {output_path.stat().st_size / 1024:.1f} KB")

    return True


if __name__ == "__main__":
    """Test camera detection and capture."""
    print("=" * 60)
    print("Camera Helpers Test")
    print("=" * 60)
    print()

    # Test camera detection
    print("Testing camera detection...")
    cameras = get_available_cameras()

    if not cameras:
        print("[X] No cameras found!")
        sys.exit(1)

    print(f"[OK] Found {len(cameras)} camera(s):")
    for device_path, device_info in cameras:
        print(f"  {device_path}: {device_info}")

    # Test camera selection
    if len(cameras) > 1:
        print("\nTesting camera selection prompt...")
        selected = select_camera(cameras)
        print(f"[OK] User selected: {selected}")
    else:
        selected = cameras[0][0]
        print(f"[OK] Auto-selected single camera: {selected}")

    # Test capture (optional - requires user confirmation)
    print(f"\nTest capture from {selected}? (y/n): ", end='')
    if input().lower() == 'y':
        test_output = Path("test_capture.png")
        success = capture_4k_downscale(selected, test_output)
        if success:
            print(f"\n[OK] Test capture successful: {test_output.absolute()}")
        else:
            print("\n[X] Test capture failed")
    else:
        print("[SKIP] Capture test skipped")

    print("\n" + "=" * 60)
    print("[OK] Camera helpers test complete!")
    print("=" * 60)
