"""
Camera Manager Module

Consolidates multiple camera streams: global, gripper, and guidance overlay.
Resource-efficient: only fetches overlay from static PNG when flagged by guidance.
"""

import cv2
import numpy as np
import sys
from typing import Optional, Dict, Tuple
from pathlib import Path

from .global_camera import GlobalCamera
from .gripper_camera import GripperCamera
from .overlay_generator import OverlayGenerator

# Add parent directory to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from configs import get_camera_config, get_path_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class CameraManager:
    """
    Manages all camera streams for the chess robot system.

    Consolidates:
    - Global camera: Real-time overhead view
    - Gripper camera: Real-time arm-mounted view
    - Guidance overlay: Static PNG loaded when flagged by guidance system
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the camera manager.

        Args:
            config: Configuration dictionary with camera settings:
                {
                    'global_camera_id': int,
                    'global_resolution': (width, height),
                    'gripper_camera_id': int,
                    'gripper_resolution': (width, height),
                    'overlay_path': str,
                    'overlay_flag_path': str
                }
        """
        config = config or {}

        # Initialize cameras
        self.global_camera = GlobalCamera(
            camera_id=config.get('global_camera_id', 0),
            resolution=config.get('global_resolution', (1280, 720))
        )

        self.gripper_camera = GripperCamera(
            camera_id=config.get('gripper_camera_id', 0),
            resolution=config.get('gripper_resolution', (640, 480))
        )

        # Initialize overlay generator (resource-efficient: only loads when flagged)
        self.overlay_generator = OverlayGenerator(
            overlay_path=config.get('overlay_path', 'data/guidance_overlay.png'),
            flag_path=config.get('overlay_flag_path', 'data/overlay_ready.flag')
        )

        self.running = False

    @classmethod
    def from_config(cls):
        """
        Create CameraManager from YAML configuration file.

        Loads camera settings from configs/current.yaml.

        Returns:
            CameraManager instance configured from YAML

        Example:
            camera_mgr = CameraManager.from_config()
            camera_mgr.start_all()
        """
        if not CONFIG_AVAILABLE:
            print("[CameraManager] Warning: Config module not available, using defaults")
            return cls()

        # Load camera and path config
        camera_config = get_camera_config()
        path_config = get_path_config()

        # Build config dict for initialization
        config = {
            'global_camera_id': camera_config.get('global_camera_id', 0),
            'global_resolution': tuple(camera_config.get('global_resolution', [1280, 720])),
            'gripper_camera_id': camera_config.get('gripper_camera_id', 0),
            'gripper_resolution': tuple(camera_config.get('gripper_resolution', [640, 480])),
            'overlay_path': path_config.get('overlay_image', 'data/guidance_overlay.png'),
            'overlay_flag_path': path_config.get('overlay_flag', 'data/overlay_ready.flag'),
        }

        return cls(config)

    def start(self) -> bool:
        """
        Start all camera streams.

        Returns:
            True if all cameras started successfully, False otherwise
        """
        success = True

        # Start global camera
        if not self.global_camera.start():
            print("[CameraManager] Error: Failed to start global camera")
            success = False

        # Start gripper camera
        if not self.gripper_camera.start():
            print("[CameraManager] Warning: Failed to start gripper camera")
            # Don't fail completely if gripper camera fails
            # success = False

        self.running = success
        if success:
            print("[CameraManager] All cameras started successfully")

        return success

    def stop(self):
        """Stop all camera streams."""
        self.global_camera.stop()
        self.gripper_camera.stop()
        self.running = False
        print("[CameraManager] All cameras stopped")

    def get_global_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the global overhead camera.

        Returns:
            Frame as numpy array (BGR), or None if unavailable
        """
        return self.global_camera.get_frame()

    def get_gripper_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the gripper camera.

        Returns:
            Frame as numpy array (BGR), or None if unavailable
        """
        return self.gripper_camera.get_frame()

    def get_overlay_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest guidance overlay.

        This is resource-efficient: only loads from disk when the guidance
        system has signaled a new overlay is ready via flag file.

        Returns:
            Overlay frame as numpy array (BGR), or None if unavailable
        """
        return self.overlay_generator.get_overlay()

    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Get all available frames at once.

        Returns:
            Dictionary with keys: 'global', 'gripper', 'overlay'
        """
        return {
            'global': self.get_global_frame(),
            'gripper': self.get_gripper_frame(),
            'overlay': self.get_overlay_frame()
        }

    def has_overlay(self) -> bool:
        """Check if a guidance overlay is currently available."""
        return self.overlay_generator.has_overlay()

    def clear_overlay(self):
        """Clear the current guidance overlay."""
        self.overlay_generator.clear_overlay()

    def is_running(self) -> bool:
        """Check if camera manager is running."""
        return self.running

    def get_stream_status(self) -> Dict[str, bool]:
        """
        Get status of all streams.

        Returns:
            Dictionary with keys: 'global', 'gripper', 'overlay'
        """
        return {
            'global': self.global_camera.is_running(),
            'gripper': self.gripper_camera.is_running(),
            'overlay': self.overlay_generator.has_overlay()
        }

    def save_snapshot(self, directory: str = "data/snapshots"):
        """
        Save snapshots from all streams to a directory.

        Args:
            directory: Directory to save snapshots
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save global frame
        global_frame = self.get_global_frame()
        if global_frame is not None:
            cv2.imwrite(str(dir_path / f"global_{timestamp}.png"), global_frame)

        # Save gripper frame
        gripper_frame = self.get_gripper_frame()
        if gripper_frame is not None:
            cv2.imwrite(str(dir_path / f"gripper_{timestamp}.png"), gripper_frame)

        # Save overlay
        overlay = self.get_overlay_frame()
        if overlay is not None:
            cv2.imwrite(str(dir_path / f"overlay_{timestamp}.png"), overlay)

        print(f"[CameraManager] Snapshots saved to {directory}")
