#!/usr/bin/env python3
"""
Test USB camera stability by capturing multiple frames slowly.
This helps diagnose if the errno=19 issue is related to USB connection drops.
"""

import cv2
import time
import sys
import subprocess

def test_camera_stability(device="/dev/video0", num_frames=5, delay=1.0):
    """
    Test camera by capturing multiple frames with delays.

    Args:
        device: Camera device path
        num_frames: Number of frames to capture
        delay: Delay between captures in seconds
    """
    print("=" * 60)
    print("Camera Stability Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Frames: {num_frames}")
    print(f"Delay: {delay}s between captures")
    print()

    # Set format via v4l2-ctl
    print("Setting YUYV 1280x720 @ 10fps format...")
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device}",
            "--set-fmt-video=width=1280,height=720,pixelformat=YUYV",
            "--set-parm=10"
        ], check=True, capture_output=True, text=True)
        print("[OK] Format set")
    except subprocess.CalledProcessError as e:
        print(f"[Warning] Could not set format: {e}")

    # Get camera index
    camera_index = int(device.replace('/dev/video', ''))

    print(f"\nOpening camera {device} (index {camera_index})...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[X] Failed to open camera")
        return False

    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)

    # Verify
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[OK] Camera opened: {width}x{height} @ {fps}fps")
    print()

    # Capture frames
    successful_captures = 0
    for i in range(num_frames):
        print(f"[{i+1}/{num_frames}] Capturing frame...")
        ret, frame = cap.read()

        if not ret:
            print(f"  [X] Failed to capture frame {i+1}")
            break

        print(f"  [OK] Frame captured: {frame.shape}")
        successful_captures += 1

        if i < num_frames - 1:
            print(f"  Waiting {delay}s...")
            time.sleep(delay)

    print()
    print("=" * 60)
    print(f"Results: {successful_captures}/{num_frames} successful captures")
    print("=" * 60)

    # Test different release methods
    print("\nTesting release methods...")

    print("1. Testing cap.release()...")
    try:
        cap.release()
        print("   [OK] cap.release() succeeded")
    except Exception as e:
        print(f"   [X] cap.release() failed: {e}")

    print()
    print("Test complete!")
    return successful_captures == num_frames


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    success = test_camera_stability(device)
    sys.exit(0 if success else 1)
