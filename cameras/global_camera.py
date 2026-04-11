"""
Global Camera Module

Manages the overhead camera stream providing a bird's-eye view of the chessboard.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time


class GlobalCamera:
    """
    Manages the global overhead camera stream.

    This camera provides a top-down view of the entire chessboard
    for board state detection and visual guidance.
    """

    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize the global camera.

        Args:
            camera_id: Camera device ID (default: 0)
            resolution: Desired resolution (width, height)
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """
        Start the camera capture.

        Returns:
            True if successfully started, False otherwise
        """
        if self.running:
            print("[GlobalCamera] Already running")
            return True

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[GlobalCamera] Error: Could not open camera {self.camera_id}")
            self.cap.release()
            self.cap = None
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"[GlobalCamera] Started on device {self.camera_id} @ {self.resolution}")
        return True

    def stop(self):
        """Stop the camera capture."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                print("[GlobalCamera] Warning: capture thread did not stop within timeout")

        if self.cap and not (self.capture_thread and self.capture_thread.is_alive()):
            self.cap.release()
            self.cap = None

        print("[GlobalCamera] Stopped")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                time.sleep(0.001)
            else:
                print("[GlobalCamera] Warning: Failed to capture frame")
                time.sleep(0.1)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera.

        Returns:
            Latest frame as numpy array (BGR), or None if unavailable
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get the current camera resolution.

        Returns:
            (width, height) tuple
        """
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return self.resolution

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self.running

    def save_frame(self, filename: str) -> bool:
        """
        Save the current frame to a file.

        Args:
            filename: Output filename

        Returns:
            True if saved successfully, False otherwise
        """
        frame = self.get_frame()
        if frame is not None:
            cv2.imwrite(filename, frame)
            return True
        return False
