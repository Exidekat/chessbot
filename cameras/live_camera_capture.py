"""
Live Camera Capture Module

Provides threaded camera capture for real-time video streaming.
Extracted from virtual_overlay_demo.py for reusability across VLA scripts.
"""

import threading
import gc
import subprocess
from typing import Optional

import cv2
import numpy as np


def get_camera_index_from_device(device_path: str) -> int:
    """
    Extract camera index from device path (e.g., /dev/video0 -> 0).

    Args:
        device_path: Path to video device

    Returns:
        int: Camera index for OpenCV
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
        raise ValueError(f"Invalid device path: {device_path}")


class LiveCameraCapture:
    """
    Continuously capture 720p frames from camera at 30fps.

    Used for virtual camera output with live overlay. Runs capture in a background
    thread with thread-safe frame access and minimal latency (buffer size=1).

    Example:
        >>> capture = LiveCameraCapture("/dev/video7")
        >>> capture.start()
        >>> frame = capture.get_latest_frame()  # Returns latest 720p BGR frame
        >>> capture.stop()
    """

    def __init__(self, device_path: str):
        """
        Initialize live camera capture.

        Args:
            device_path: Camera device path (e.g., /dev/video7)
        """
        self.device_path = device_path
        self.camera_index = get_camera_index_from_device(device_path)
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def start(self):
        """Start continuous capture thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[LiveCapture] Started continuous capture from {self.device_path}")

    def stop(self):
        """Stop continuous capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print(f"[LiveCapture] Stopped continuous capture")

    def _capture_loop(self):
        """
        Continuous capture loop (runs in background thread).

        Configures camera for NATIVE 720p MJPEG at 30fps (no downscaling for speed).
        Updates self.latest_frame with thread-safe access.
        """
        # Configure camera for NATIVE 720p (no downscaling = faster!)
        try:
            subprocess.run([
                "v4l2-ctl",
                f"--device={self.device_path}",
                "--set-fmt-video=width=1280,height=720,pixelformat=MJPG",
                "--set-parm=30"
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            pass  # Continue with OpenCV defaults

        # Open camera
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[LiveCapture] [X] Failed to open camera")
            return

        # Set to 720p MJPEG at 30fps (native resolution - no downscaling needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size for lower latency

        while self.running:
            # Capture frame (no sleep - capture as fast as possible)
            ret, frame = cap.read()
            if ret:
                # No downscaling needed - already 720p!
                # Update latest frame (thread-safe)
                with self.frame_lock:
                    self.latest_frame = frame  # Direct assignment, no copy needed

        del cap
        gc.collect()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent captured frame (thread-safe).

        Returns:
            np.ndarray: Latest 720p BGR frame (1280x720x3) or None if not ready
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
