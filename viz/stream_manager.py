"""
Stream Manager for Camera Feeds

Encodes camera frames as JPEG for web streaming.
Manages CameraManager integration for all 3 streams.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cameras.camera_manager import CameraManager


class StreamManager:
    """
    Manages camera stream encoding for web delivery.

    Converts OpenCV frames to JPEG bytes for HTTP streaming.
    Integrates with CameraManager for accessing all 3 streams.
    """

    def __init__(self, camera_manager: Optional[CameraManager] = None, jpeg_quality: int = 85):
        """
        Initialize stream manager.

        Args:
            camera_manager: CameraManager instance (creates new if None)
            jpeg_quality: JPEG compression quality (0-100, default: 85)
        """
        self.camera_manager = camera_manager or CameraManager()
        self.jpeg_quality = jpeg_quality

        # JPEG encoding parameters
        self.encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
        ]

    def start(self):
        """Start camera streams."""
        success = self.camera_manager.start_all()
        if success:
            print("[StreamManager] All camera streams started")
        else:
            print("[StreamManager] Warning: Some camera streams failed to start")
        return success

    def stop(self):
        """Stop camera streams."""
        self.camera_manager.stop_all()
        print("[StreamManager] All camera streams stopped")

    def get_global_frame_jpeg(self) -> Optional[bytes]:
        """
        Get current global camera frame as JPEG bytes.

        Returns:
            JPEG-encoded frame bytes, or None if unavailable
        """
        frame = self.camera_manager.get_global_frame()
        if frame is None:
            return None

        return self._encode_frame(frame)

    def get_gripper_frame_jpeg(self) -> Optional[bytes]:
        """
        Get current gripper camera frame as JPEG bytes.

        Returns:
            JPEG-encoded frame bytes, or None if unavailable
        """
        frame = self.camera_manager.get_gripper_frame()
        if frame is None:
            return None

        return self._encode_frame(frame)

    def get_overlay_frame_jpeg(self) -> Optional[bytes]:
        """
        Get current overlay as JPEG bytes.

        Returns:
            JPEG-encoded overlay bytes, or None if unavailable
        """
        frame = self.camera_manager.get_overlay_frame()
        if frame is None:
            return None

        return self._encode_frame(frame)

    def get_all_frames_jpeg(self) -> Tuple[Optional[bytes], Optional[bytes], Optional[bytes]]:
        """
        Get all 3 frames as JPEG bytes.

        Returns:
            Tuple of (global_jpeg, gripper_jpeg, overlay_jpeg)
        """
        return (
            self.get_global_frame_jpeg(),
            self.get_gripper_frame_jpeg(),
            self.get_overlay_frame_jpeg()
        )

    def _encode_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Encode frame as JPEG bytes.

        Args:
            frame: OpenCV frame (BGR or RGB)

        Returns:
            JPEG bytes, or None on encoding error
        """
        try:
            # Ensure frame is in BGR format (OpenCV standard)
            if len(frame.shape) == 2:
                # Grayscale - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                # RGBA - convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            # Assume already BGR if 3 channels

            # Encode as JPEG
            success, encoded = cv2.imencode('.jpg', frame, self.encode_params)

            if not success:
                print("[StreamManager] Failed to encode frame as JPEG")
                return None

            return encoded.tobytes()

        except Exception as e:
            print(f"[StreamManager] Error encoding frame: {e}")
            return None

    def get_placeholder_jpeg(self, width: int = 640, height: int = 480, text: str = "No Signal") -> bytes:
        """
        Generate placeholder image when camera unavailable.

        Args:
            width: Image width
            height: Image height
            text: Text to display

        Returns:
            JPEG bytes of placeholder image
        """
        # Create black image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (200, 200, 200)

        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate position (centered)
        x = (width - text_width) // 2
        y = (height + text_height) // 2

        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

        # Encode as JPEG
        success, encoded = cv2.imencode('.jpg', img, self.encode_params)
        return encoded.tobytes() if success else b''

    def get_stream_status(self) -> dict:
        """
        Get status of all camera streams.

        Returns:
            Dictionary with stream availability status
        """
        return {
            "global_camera": self.camera_manager.global_camera is not None,
            "gripper_camera": self.camera_manager.gripper_camera is not None,
            "overlay_generator": self.camera_manager.overlay_generator is not None
        }
