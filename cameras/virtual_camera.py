"""
Virtual Camera Module

Provides virtual camera output via v4l2loopback and ffmpeg.
Extracted from virtual_overlay_demo.py for reusability across VLA scripts.
"""

import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VirtualCamera:
    """
    Output frames to a virtual camera device using ffmpeg pipe to v4l2loopback.

    This approach is more reliable than direct device writing and provides low-latency
    streaming for VLA training. Requires v4l2loopback kernel module.

    Setup (one-time):
        sudo modprobe v4l2loopback devices=1 video_nr=7 \\
            card_label="ChessBot Virtual Cam" exclusive_caps=1

    Example:
        >>> vcam = VirtualCamera("/dev/video7", width=1280, height=720)
        >>> vcam.start()
        >>> vcam.write_frame(frame)  # Queue frame for output (non-blocking)
        >>> vcam.stop()
    """

    def __init__(self, device_path: str = "/dev/video7", width: int = 1280, height: int = 720):
        """
        Initialize virtual camera output.

        Args:
            device_path: v4l2loopback device path (default: /dev/video7)
            width: Output frame width (default: 1280)
            height: Output frame height (default: 720)
        """
        self.device_path = device_path
        self.width = width
        self.height = height
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=1)  # Only 1 frame buffer for minimum latency
        self.ffmpeg_process = None

    def start(self) -> bool:
        """
        Start virtual camera output thread.

        Returns:
            bool: True if successful, False if device not found or ffmpeg fails
        """
        if self.running:
            return True

        # Check if device exists
        if not Path(self.device_path).exists():
            print(f"[VirtualCam] [X] Device not found: {self.device_path}")
            print(f"[VirtualCam] Load v4l2loopback: sudo modprobe v4l2loopback devices=1 video_nr=7 card_label='ChessBot Virtual Cam' exclusive_caps=1")
            return False

        # Start ffmpeg process to write to v4l2loopback device
        # Optimized for LOW LATENCY
        try:
            self.ffmpeg_process = subprocess.Popen([
                'ffmpeg',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', '30',  # 30 fps for smoother, lower latency
                '-i', '-',  # Read from stdin
                '-fflags', 'nobuffer',  # Disable buffering
                '-flags', 'low_delay',  # Low delay mode
                '-probesize', '32',  # Minimal probing
                '-analyzeduration', '0',  # No analysis delay
                '-f', 'v4l2',
                '-pix_fmt', 'yuv420p',
                '-tune', 'zerolatency',  # Zero latency tuning
                self.device_path
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"[VirtualCam] Started ffmpeg output to {self.device_path} (low latency mode)")
        except Exception as e:
            print(f"[VirtualCam] [X] Failed to start ffmpeg: {e}")
            print(f"[VirtualCam] Make sure ffmpeg is installed: sudo apt install ffmpeg")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._output_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        """Stop virtual camera output thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=2.0)
            except:
                self.ffmpeg_process.kill()

        print(f"[VirtualCam] Stopped output")

    def write_frame(self, frame: np.ndarray):
        """
        Queue a frame for output (non-blocking).

        Args:
            frame: BGR frame (numpy array, any resolution - will be resized if needed)
        """
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame if queue is full (maintains low latency)

    def _output_loop(self):
        """Continuous output loop (runs in background thread)."""
        frame_count = 0

        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame = self.frame_queue.get(timeout=0.1)

                # Ensure frame is correct size
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))

                # Debug info on first frame
                if frame_count == 0:
                    print(f"[VirtualCam] Frame shape: {frame.shape}, dtype: {frame.dtype}")
                    print(f"[VirtualCam] Streaming BGR24 frames to ffmpeg...")

                # Write BGR frame directly to ffmpeg stdin
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                frame_count += 1

                if frame_count == 1:
                    print(f"[VirtualCam] First frame written successfully!")

            except queue.Empty:
                continue
            except Exception as e:
                if self.running:  # Only print error if we're still supposed to be running
                    print(f"[VirtualCam] Error writing frame #{frame_count}: {e}")
                break

        print(f"[VirtualCam] Output loop ended, wrote {frame_count} frames")
