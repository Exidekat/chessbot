"""
Gripper Camera Module

Manages the arm-mounted camera stream providing a close-up view during piece manipulation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time


class GripperCamera:
    """
    Manages the arm-mounted gripper camera stream.

    This camera provides a close-up view for precise piece manipulation
    and verification during robot movements.
    """

    def __init__(self, camera_id: int = 1, resolution: Tuple[int, int] = (640, 480)):
        """
        Initialize the gripper camera.

        Args:
            camera_id: Camera device ID (default: 1)
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
            print("[GripperCamera] Already running")
            return True

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[GripperCamera] Error: Could not open camera {self.camera_id}")
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print(f"[GripperCamera] Started on device {self.camera_id} @ {self.resolution}")
        return True

    def stop(self):
        """Stop the camera capture."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        print("[GripperCamera] Stopped")

    def _capture_loop(self):
        """Background thread for continuous frame capture."""
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print("[GripperCamera] Warning: Failed to capture frame")
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
