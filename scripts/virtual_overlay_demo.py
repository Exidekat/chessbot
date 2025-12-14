"""
Virtual Overlay Demo - Live VLA Training with Virtual Camera

This script creates a virtual camera with real-time overlay visualization:
1. Set up virtual camera output at /dev/video7
2. Capture initial board state and calculate best move
3. Decompose move into stages (capture, movement, promotion, etc.)
4. For each stage:
   - Continuously capture and display live 720p feed with overlay (1 sec refresh)
   - User presses ENTER to save final frame for that stage
5. Output live feed to virtual camera for VLA observation
6. Save stage overlays for VLA training data collection

Requires: v4l2loopback kernel module
    sudo modprobe v4l2loopback devices=1 video_nr=7 card_label="ChessBot Virtual Cam" exclusive_caps=1

Usage:
    python scripts/virtual_overlay_demo.py [--engine ENGINE_PATH] [--device CAMERA_DEVICE]
"""

import argparse
import sys
from pathlib import Path
import chess
import cv2
import subprocess
from datetime import datetime
import time
import gc
import threading
import queue
from typing import Optional
import fcntl
import struct
import os

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator
from guidance.coordinate_mapper import CoordinateMapper
from guidance.highlight_renderer import HighlightRenderer
from PIL import Image
import numpy as np


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


def decompose_move(board: chess.Board, move: chess.Move) -> list:
    """
    Decompose a chess move into stages for overlay generation.

    Each stage is a dict with:
    - description: Human-readable description
    - pickup_square: Square to pick piece from (or None for graveyard)
    - place_square: Square to place piece on (or None for graveyard)
    - piece: Piece type being moved (for graveyard -> board moves)

    Args:
        board: Current board state (before move is applied)
        move: Chess move to decompose

    Returns:
        List of stage dictionaries
    """
    stages = []
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)

    # Get piece being moved
    moving_piece = board.piece_at(move.from_square)
    if not moving_piece:
        return stages

    # Check for capture
    captured_piece = board.piece_at(move.to_square)

    # Check for castling
    is_castling = board.is_castling(move)

    # Check for promotion
    is_promotion = move.promotion is not None

    # CASTLING: King first, then rook
    if is_castling:
        # King move
        stages.append({
            "description": f"Move {moving_piece.symbol().upper()} from {from_square} to {to_square} (castling)",
            "pickup_square": from_square,
            "place_square": to_square,
            "piece": None
        })

        # Determine rook move
        # Kingside castling (e1g1 or e8g8): rook from h file to f file
        # Queenside castling (e1c1 or e8c8): rook from a file to d file
        if to_square in ["g1", "g8"]:  # Kingside
            rank = "1" if moving_piece.color == chess.WHITE else "8"
            rook_from = f"h{rank}"
            rook_to = f"f{rank}"
        else:  # Queenside
            rank = "1" if moving_piece.color == chess.WHITE else "8"
            rook_from = f"a{rank}"
            rook_to = f"d{rank}"

        stages.append({
            "description": f"Move rook from {rook_from} to {rook_to} (castling)",
            "pickup_square": rook_from,
            "place_square": rook_to,
            "piece": None
        })

        return stages

    # PROMOTION WITH CAPTURE
    if is_promotion and captured_piece:
        # Stage 1: Remove captured piece
        piece_name = captured_piece.symbol().upper() if captured_piece.color == chess.WHITE else captured_piece.symbol().lower()
        stages.append({
            "description": f"Remove {piece_name} from {to_square} (captured)",
            "pickup_square": to_square,
            "place_square": None,  # To graveyard
            "piece": None
        })

        # Stage 2: Remove pawn
        pawn_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        stages.append({
            "description": f"Remove {pawn_name} from {from_square}",
            "pickup_square": from_square,
            "place_square": None,  # To graveyard
            "piece": None
        })

        # Stage 3: Place promoted piece
        promotion_piece_type = chess.piece_name(move.promotion)
        promotion_symbol = chess.Piece(move.promotion, moving_piece.color).symbol()
        stages.append({
            "description": f"Place {promotion_symbol} on {to_square} (promotion from graveyard)",
            "pickup_square": None,  # From graveyard
            "place_square": to_square,
            "piece": promotion_symbol
        })

        return stages

    # PROMOTION WITHOUT CAPTURE
    if is_promotion:
        # Stage 1: Remove pawn
        pawn_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        stages.append({
            "description": f"Remove {pawn_name} from {from_square}",
            "pickup_square": from_square,
            "place_square": None,  # To graveyard
            "piece": None
        })

        # Stage 2: Place promoted piece
        promotion_piece_type = chess.piece_name(move.promotion)
        promotion_symbol = chess.Piece(move.promotion, moving_piece.color).symbol()
        stages.append({
            "description": f"Place {promotion_symbol} on {to_square} (promotion from graveyard)",
            "pickup_square": None,  # From graveyard
            "place_square": to_square,
            "piece": promotion_symbol
        })

        return stages

    # CAPTURE (non-promotion)
    if captured_piece:
        # Stage 1: Remove captured piece to graveyard
        piece_name = captured_piece.symbol().upper() if captured_piece.color == chess.WHITE else captured_piece.symbol().lower()
        stages.append({
            "description": f"Remove {piece_name} from {to_square} (captured)",
            "pickup_square": to_square,
            "place_square": None,  # To graveyard
            "piece": None
        })

        # Stage 2: Move attacking piece
        attacker_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        stages.append({
            "description": f"Move {attacker_name} from {from_square} to {to_square}",
            "pickup_square": from_square,
            "place_square": to_square,
            "piece": None
        })

        return stages

    # NORMAL MOVE (no capture, no special moves)
    piece_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
    stages.append({
        "description": f"Move {piece_name} from {from_square} to {to_square}",
        "pickup_square": from_square,
        "place_square": to_square,
        "piece": None
    })

    return stages


def get_available_cameras():
    """
    Detect all available USB cameras.

    Returns:
        list: List of tuples (device_path, device_info)
    """
    cameras = []

    # Method 1: Try v4l2 (Linux)
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

    # Method 2: Fallback - try opening /dev/video* devices
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


def select_camera(cameras):
    """
    Prompt user to select a camera from the list.

    Args:
        cameras: List of (device_path, device_info) tuples

    Returns:
        str: Selected device path
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


def get_camera_index_from_device(device_path):
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
        sys.exit(1)


def capture_4k_downscale(device_path, output_path):
    """
    Capture 4K MJPEG video for 1 second, then downscale best frame to 1280x720.

    This approach uses the camera's 4K MJPEG mode (which produces better quality)
    and downscales to the 720p resolution required by detection models.

    Args:
        device_path: Camera device path
        output_path: Path to save the captured image

    Returns:
        bool: True if successful, False otherwise
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


def capture_yuyv_720p(device_path, output_path):
    """
    DEPRECATED: Direct 720p YUYV capture (kept for fallback).
    Use capture_4k_downscale() instead for better quality.

    Capture a 1280x720 YUYV photo from the specified camera.

    Args:
        device_path: Camera device path
        output_path: Path to save the captured image

    Returns:
        bool: True if successful, False otherwise
    """
    camera_index = get_camera_index_from_device(device_path)

    print(f"\n[CameraCapture] Opening camera: {device_path} (index: {camera_index})")

    # Configure camera for YUYV 1280x720 (uncompressed)
    print(f"[CameraCapture] Setting YUYV 1280x720 @ 10fps format...")
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-fmt-video=width=1280,height=720,pixelformat=YUYV",
            "--set-parm=10"
        ], check=True, capture_output=True, text=True)
        print(f"[CameraCapture] [OK] Format set to YUYV 1280x720 @ 10fps")
    except subprocess.CalledProcessError as e:
        print(f"[CameraCapture] Warning: Could not set format via v4l2-ctl: {e}")
        print(f"[CameraCapture] Continuing with OpenCV defaults...")

    # Open camera with V4L2 backend
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[CameraCapture] [X] Failed to open camera: {device_path}")
        return False

    # Explicitly set resolution and FPS in OpenCV (v4l2-ctl may not persist)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 10)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    # Verify resolution and FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[CameraCapture] Camera resolution: {width}x{height} @ {fps}fps")

    if width != 1280 or height != 720:
        print(f"[CameraCapture] Warning: Expected 1280x720, got {width}x{height}")
        print(f"[CameraCapture] Camera may not support YUYV at 720p, using available resolution")

    # Capture 2 frames (1 for warmup, use the 2nd frame)
    # Reduced to minimize time before power-cycle with YUYV
    print("[CameraCapture] Warming up camera and capturing...")
    frame = None
    for i in range(2):
        ret, frame = cap.read()
        if not ret:
            print(f"[CameraCapture] [X] Failed to read frame {i+1}/2")
            # Note: Not calling cap.release() due to WBC-0E01 quirk
            del cap
            gc.collect()
            return False

    # Save image directly (no transformations - same as training scripts)
    cv2.imwrite(str(output_path), frame)
    print(f"[CameraCapture] [OK] Photo captured: {output_path}")
    print(f"[CameraCapture] Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Note: Not calling cap.release() due to WBC-0E01 quirk causing errno=19
    # Instead, delete reference and force garbage collection
    del cap
    gc.collect()
    print("[CameraCapture] [OK] Camera capture complete (GC release)")
    return True


class LiveCameraCapture:
    """
    Continuously capture 720p frames from camera at 30fps.
    Used for virtual camera output with live overlay.
    """
    def __init__(self, device_path: str):
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
        """Continuous capture loop (runs in background thread)."""
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
        """Get the most recent captured frame (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None


class VirtualCamera:
    """
    Output frames to a virtual camera device using ffmpeg pipe to v4l2loopback.
    This approach is more reliable than direct device writing.
    """
    def __init__(self, device_path: str = "/dev/video7", width: int = 1280, height: int = 720):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=1)  # Only 1 frame buffer for minimum latency
        self.ffmpeg_process = None

    def start(self):
        """Start virtual camera output thread."""
        if self.running:
            return

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
        """Queue a frame for output (non-blocking)."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame if queue is full

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


def rotate_square_for_camera(square: str, camera_position: str, inverse: bool = False) -> str:
    """
    Rotate a chess square notation based on camera position.

    When camera is positioned at 'right', the square 'a1' (bottom-left from top view)
    appears in a different position in the camera view. We need to map the chess
    square to the actual physical square in the camera's perspective.

    Args:
        square: Chess square notation (e.g., "e4")
        camera_position: Camera position ('top', 'right', 'bottom', 'left')
        inverse: If True, convert from camera view back to chess notation

    Returns:
        Rotated square notation
    """
    if camera_position == "top":
        return square  # No rotation

    file = ord(square[0]) - ord('a')  # 0-7
    rank = int(square[1]) - 1  # 0-7

    # Apply rotation based on camera position
    if camera_position == "right":
        # 90° counter-clockwise (or clockwise if inverse)
        if not inverse:
            new_file = rank
            new_rank = 7 - file
        else:
            new_file = 7 - rank
            new_rank = file
    elif camera_position == "bottom":
        # 180° rotation
        new_file = 7 - file
        new_rank = 7 - rank
    elif camera_position == "left":
        # 90° clockwise (or counter-clockwise if inverse)
        if not inverse:
            new_file = 7 - rank
            new_rank = file
        else:
            new_file = rank
            new_rank = 7 - file
    else:
        return square  # Unknown position

    return chr(ord('a') + new_file) + str(new_rank + 1)


def apply_stage_overlay_to_frame(
    frame: np.ndarray,
    stage: dict,
    transformed_image_path: str,
    detector: BoardDetector,
    perspective_matrix: np.ndarray,
    camera_position: str
) -> np.ndarray:
    """
    Apply overlay for a move stage to a live frame (no file I/O).

    Args:
        frame: Live 720p frame (numpy array)
        stage: Stage dictionary from decompose_move()
        transformed_image_path: Path to transformed board image (for coordinate mapping)
        detector: BoardDetector instance for coordinate mapping
        perspective_matrix: Perspective transform matrix (M)
        camera_position: Camera position for rotation handling

    Returns:
        Frame with overlay applied
    """
    if frame is None:
        return None

    # Load transformed image for coordinate mapping
    pil_transformed = Image.open(transformed_image_path)

    # Initialize coordinate mapper
    mapper = CoordinateMapper(board_detector=detector)

    # Calculate inverse perspective matrix
    M_inv = cv2.invert(perspective_matrix)[1]

    # Create overlay with transparency
    overlay = frame.copy()

    # Color definitions (BGR format)
    RED = (0, 0, 255)      # Pickup/remove
    BLUE = (255, 0, 0)     # Place
    ORANGE = (0, 165, 255) # Graveyard (discard)
    PURPLE = (255, 0, 255) # Graveyard piece (promotion)

    # Helper function to transform square bounds back to original image
    def transform_square_to_original(square_name: str):
        x1, y1, x2, y2 = mapper.get_square_bounds(square_name, pil_transformed)
        corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        corners_reshaped = corners.reshape(1, -1, 2)
        original_corners = cv2.perspectiveTransform(corners_reshaped, M_inv)
        return original_corners.reshape(-1, 2).astype(np.int32)

    # Render pickup square (red)
    if stage["pickup_square"] is not None:
        rotated_square = rotate_square_for_camera(stage["pickup_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)
        cv2.fillPoly(overlay, [original_corners], RED)
        cv2.polylines(overlay, [original_corners], isClosed=True, color=RED, thickness=4)

    # Render place square (blue)
    if stage["place_square"] is not None:
        rotated_square = rotate_square_for_camera(stage["place_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)
        cv2.fillPoly(overlay, [original_corners], BLUE)
        cv2.polylines(overlay, [original_corners], isClosed=True, color=BLUE, thickness=4)

    # Render graveyard
    if stage["pickup_square"] is None and stage["piece"] is not None:
        # Taking piece FROM graveyard - highlight specific piece
        target_piece = stage["piece"]
        piece_to_class = {
            'b': 0, 'k': 1, 'n': 2, 'p': 3, 'q': 4, 'r': 5,
            'B': 6, 'K': 7, 'N': 8, 'P': 9, 'Q': 10, 'R': 11
        }
        target_class = piece_to_class.get(target_piece)

        left_edge_transformed = np.array([[[0, pil_transformed.size[1] // 2]]], dtype=np.float32)
        left_edge_original = cv2.perspectiveTransform(left_edge_transformed, M_inv)
        board_left_x = int(left_edge_original[0][0][0])

        if hasattr(detector, 'original_detections') and hasattr(detector, 'original_boxes'):
            detections = detector.original_detections
            boxes = detector.original_boxes
            matching_pieces = []
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection[:4]
                box_center_x = (x1 + x2) / 2
                cls = int(boxes.cls[i].item())
                if box_center_x < board_left_x and cls == target_class:
                    matching_pieces.append((i, detection, boxes.conf[i].item()))

            if matching_pieces:
                matching_pieces.sort(key=lambda x: x[2], reverse=True)
                idx, detection, conf = matching_pieces[0]
                x1, y1, x2, y2 = map(int, detection[:4])
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                cv2.fillPoly(overlay, [corners], PURPLE)
                cv2.polylines(overlay, [corners], isClosed=True, color=PURPLE, thickness=4)
            else:
                graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
                point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
                original_point = cv2.perspectiveTransform(point, M_inv)
                gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
                cv2.circle(overlay, (gx, gy), 40, PURPLE, -1)
                cv2.circle(overlay, (gx, gy), 40, PURPLE, 4)

    elif stage["place_square"] is None:
        graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
        point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
        original_point = cv2.perspectiveTransform(point, M_inv)
        gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
        cv2.circle(overlay, (gx, gy), 40, ORANGE, -1)
        cv2.circle(overlay, (gx, gy), 40, ORANGE, 4)

    # Apply alpha blending
    result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    return result


def generate_stage_overlay(
    stage: dict,
    original_image_path: str,
    transformed_image_path: str,
    output_path: str,
    detector: BoardDetector,
    perspective_matrix: np.ndarray,
    camera_position: str
) -> None:
    """
    Generate overlay for a single move stage on the ORIGINAL 720p image.

    Args:
        stage: Stage dictionary from decompose_move()
        original_image_path: Path to original 720p captured image
        transformed_image_path: Path to transformed board image
        output_path: Path to save overlay image
        detector: BoardDetector instance for coordinate mapping
        perspective_matrix: Perspective transform matrix (M)
        camera_position: Camera position for rotation handling
    """
    # Load ORIGINAL 720p image
    image = cv2.imread(original_image_path)
    if image is None:
        print(f"[X] Failed to load image: {original_image_path}")
        return

    # Load transformed image for coordinate mapping
    pil_transformed = Image.open(transformed_image_path)

    # Initialize coordinate mapper
    mapper = CoordinateMapper(board_detector=detector)

    # Calculate inverse perspective matrix
    # This transforms coordinates from transformed board back to original image
    M_inv = cv2.invert(perspective_matrix)[1]

    # Create overlay with transparency
    overlay = image.copy()

    # Color definitions (BGR format)
    RED = (0, 0, 255)      # Pickup/remove
    BLUE = (255, 0, 0)     # Place
    ORANGE = (0, 165, 255) # Graveyard (discard)
    PURPLE = (255, 0, 255) # Graveyard piece (promotion)

    # Helper function to transform square bounds back to original image
    def transform_square_to_original(square_name: str):
        # Get square bounds on transformed image
        x1, y1, x2, y2 = mapper.get_square_bounds(square_name, pil_transformed)

        # Get all 4 corners of the square
        corners = np.array([
            [x1, y1],  # Top-left
            [x2, y1],  # Top-right
            [x2, y2],  # Bottom-right
            [x1, y2]   # Bottom-left
        ], dtype=np.float32)

        # Transform corners back to original image
        corners_reshaped = corners.reshape(1, -1, 2)
        original_corners = cv2.perspectiveTransform(corners_reshaped, M_inv)
        original_corners = original_corners.reshape(-1, 2).astype(np.int32)

        return original_corners

    # Render pickup square (red)
    if stage["pickup_square"] is not None:
        # Rotate square name to match camera position
        rotated_square = rotate_square_for_camera(stage["pickup_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)

        # Draw filled quadrilateral with transparency
        cv2.fillPoly(overlay, [original_corners], RED)
        # Draw border
        cv2.polylines(overlay, [original_corners], isClosed=True, color=RED, thickness=4)

    # Render place square (blue)
    if stage["place_square"] is not None:
        # Rotate square name to match camera position
        rotated_square = rotate_square_for_camera(stage["place_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)

        # Draw filled quadrilateral with transparency
        cv2.fillPoly(overlay, [original_corners], BLUE)
        # Draw border
        cv2.polylines(overlay, [original_corners], isClosed=True, color=BLUE, thickness=4)

    # Render graveyard
    # If pickup is None -> taking FROM graveyard (highlight specific piece)
    # If place is None -> placing TO graveyard (orange circle)
    if stage["pickup_square"] is None and stage["piece"] is not None:
        # Taking piece FROM graveyard - highlight the specific piece
        # Find the piece in the original detections that is left of the board

        # Load original image to get dimensions
        original_img = cv2.imread(original_image_path)
        img_height, img_width = original_img.shape[:2]

        # Get the piece we're looking for
        target_piece = stage["piece"]

        # Piece class mapping (from BoardDetector.piece_map)
        piece_to_class = {
            'b': 0, 'k': 1, 'n': 2, 'p': 3, 'q': 4, 'r': 5,  # black pieces
            'B': 6, 'K': 7, 'N': 8, 'P': 9, 'Q': 10, 'R': 11  # white pieces
        }
        target_class = piece_to_class.get(target_piece)

        # Find all detections of this piece type that are left of the board
        # Get board left edge by transforming the left edge of the transformed board
        left_edge_transformed = np.array([[[0, pil_transformed.size[1] // 2]]], dtype=np.float32)
        left_edge_original = cv2.perspectiveTransform(left_edge_transformed, M_inv)
        board_left_x = int(left_edge_original[0][0][0])

        # Find matching pieces in graveyard (left of board)
        if hasattr(detector, 'original_detections') and hasattr(detector, 'original_boxes'):
            detections = detector.original_detections
            boxes = detector.original_boxes

            matching_pieces = []
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection[:4]
                box_center_x = (x1 + x2) / 2
                cls = int(boxes.cls[i].item())

                # Check if piece is left of board and matches target class
                if box_center_x < board_left_x and cls == target_class:
                    matching_pieces.append((i, detection, boxes.conf[i].item()))

            if matching_pieces:
                # Use the first matching piece (or highest confidence)
                matching_pieces.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
                idx, detection, conf = matching_pieces[0]

                # Highlight this piece's bounding box with PURPLE
                x1, y1, x2, y2 = map(int, detection[:4])
                corners = np.array([
                    [x1, y1],  # Top-left
                    [x2, y1],  # Top-right
                    [x2, y2],  # Bottom-right
                    [x1, y2]   # Bottom-left
                ], dtype=np.int32)

                # Draw filled quadrilateral with transparency
                cv2.fillPoly(overlay, [corners], PURPLE)
                # Draw border
                cv2.polylines(overlay, [corners], isClosed=True, color=PURPLE, thickness=4)
            else:
                # Fallback: use generic graveyard position
                graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
                point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
                original_point = cv2.perspectiveTransform(point, M_inv)
                gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
                cv2.circle(overlay, (gx, gy), 40, BLUE, -1)
                cv2.circle(overlay, (gx, gy), 40, BLUE, 4)
        else:
            # Fallback if detections not available
            graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
            point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
            original_point = cv2.perspectiveTransform(point, M_inv)
            gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
            cv2.circle(overlay, (gx, gy), 40, BLUE, -1)
            cv2.circle(overlay, (gx, gy), 40, BLUE, 4)

    elif stage["place_square"] is None:
        # Placing TO graveyard (discard) - use orange circle
        graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
        point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
        original_point = cv2.perspectiveTransform(point, M_inv)
        gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
        cv2.circle(overlay, (gx, gy), 40, ORANGE, -1)
        cv2.circle(overlay, (gx, gy), 40, ORANGE, 4)

    # Apply alpha blending (30% overlay, 70% original - more transparent)
    result = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

    # Save result
    cv2.imwrite(output_path, result)
    print(f"[OK] Overlay saved: {output_path}")


def main():
    """Main demo driver for overlay generation."""
    parser = argparse.ArgumentParser(
        description="Create Overlay Demo - Generate stage-by-stage move overlays for VLA training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific camera device (e.g., /dev/video0 or 0). If not specified, auto-detects."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="stockfish",
        help="Path to UCI chess engine (default: stockfish)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Time limit for engine analysis in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--no-bestmove",
        action="store_true",
        help="Skip calculating best move (useful if engine not available)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with visualization output"
    )
    parser.add_argument(
        "--corner-conf",
        type=float,
        default=0.005,
        help="Confidence threshold for corner detection (0.0-1.0, default: 0.005)"
    )
    parser.add_argument(
        "--min-corner-dist",
        type=float,
        default=30.0,
        help="Minimum distance between corners in pixels (default: 30.0)"
    )
    parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        choices=["left", "right", "top", "bottom"],
        help="Camera rotation relative to board (default: right). Use 'top' for no rotation."
    )
    # Legacy aliases for common rotations
    parser.add_argument(
        "--left",
        action="store_const",
        const="left",
        dest="rotation",
        help="Shortcut for --rotation left"
    )
    parser.add_argument(
        "--right",
        action="store_const",
        const="right",
        dest="rotation",
        help="Shortcut for --rotation right (default)"
    )
    parser.add_argument(
        "--turn",
        type=str,
        default=None,
        choices=["white", "black", "w", "b"],
        help="Whose turn to calculate move for (default: white)"
    )
    # Convenient aliases for turn
    parser.add_argument(
        "--white",
        action="store_const",
        const="w",
        dest="turn",
        help="Calculate move for white (default)"
    )
    parser.add_argument(
        "--black",
        action="store_const",
        const="b",
        dest="turn",
        help="Calculate move for black"
    )

    args = parser.parse_args()

    # Default to 'right' if no rotation specified (backward compatibility)
    if args.rotation is None:
        args.rotation = "right"

    # Default to white's turn if not specified
    if args.turn is None:
        args.turn = "w"
    elif args.turn == "white":
        args.turn = "w"
    elif args.turn == "black":
        args.turn = "b"

    # Force debug mode (required for transformed image output)
    args.debug = True

    print("=" * 60)
    print("Create Overlay Demo - VLA Training Data Generation")
    print("=" * 60)
    print(f"Board rotation: {args.rotation}")
    turn_name = "White" if args.turn == "w" else "Black"
    print(f"Calculate move for: {turn_name}")
    print(f"Corner confidence: {args.corner_conf}")
    print(f"Min corner distance: {args.min_corner_dist}")
    print()

    # Step 1: Camera selection and photo capture
    print("=" * 60)
    print("STAGE 1: Camera Capture (4K MJPEG -> 720p Downscale)")
    print("=" * 60)

    # Determine which camera to use
    if args.device:
        # User specified device
        device_path = args.device
        print(f"Using specified device: {device_path}")
    else:
        # Auto-detect cameras
        print("Detecting available cameras...")
        cameras = get_available_cameras()

        if not cameras:
            print("\n[X] No cameras found!")
            print("\nTroubleshooting:")
            print("  1. Check camera is connected via USB")
            print("  2. Check camera permissions: ls -l /dev/video*")
            print("  3. Add user to video group: sudo usermod -a -G video $USER")
            print("  4. Verify with: v4l2-ctl --list-devices")
            return 1

        if len(cameras) == 1:
            # Auto-select single camera
            device_path = cameras[0][0]
            print(f"[OK] Auto-selected camera: {device_path} - {cameras[0][1]}")
        else:
            # Multiple cameras - prompt user
            device_path = select_camera(cameras)

    # Capture photo using 4K MJPEG with downscaling to 720p
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path("data") / f"chessboard_capture_{timestamp}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    success = capture_4k_downscale(device_path, image_path)
    if not success:
        print("\n[X] Photo capture failed")
        return 1

    print("\n" + "=" * 60)
    print("STAGE 2: Board Detection & Move Calculation")
    print("=" * 60)
    print(f"Image: {image_path}")
    print()

    try:
        # Initialize BoardDetector
        print("Initializing BoardDetector...")
        detector = BoardDetector(camera_position=args.rotation)
        print(f"[OK] BoardDetector initialized (rotation: {args.rotation})")
        print()

        # Detect board state
        print(f"Detecting board state from {image_path}...")
        fen, transformed_image = detector.detect_board_state(
            str(image_path),
            corner_conf=args.corner_conf,
            min_corner_distance=args.min_corner_dist,
            debug=args.debug,
            turn=args.turn
        )
        print("[OK] Board state detected")
        print()

        # Create chess board from FEN
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"[X] Invalid FEN: {fen}")
            print("   Creating empty board")
            board = chess.Board(None)

        # Display results
        print("FEN Notation:")
        print("-" * 60)
        print(fen)
        print("-" * 60)
        print()

        print_board(board)

        # Calculate best move if requested
        if not args.no_bestmove:
            print("\n" + "=" * 60)
            print("STAGE 3: Best Move Calculation")
            print("=" * 60)
            print(f"Engine: {args.engine}")
            print(f"Time limit: {args.time}s")
            print()

            try:
                # Initialize MoveCalculator
                calculator = MoveCalculator(engine_path=args.engine)

                # Calculate best move
                best_move = calculator.calculate_best_move(
                    board,
                    time_limit=args.time
                )

                if best_move:
                    print(f"[OK] Best move: {best_move}")
                    print(f"  UCI notation: {best_move.uci()}")

                    # Show the move in algebraic notation
                    san_move = board.san(best_move)
                    print(f"  SAN notation: {san_move}")
                    print()

                    # Decompose move into stages
                    print("\n" + "=" * 60)
                    print("STAGE 4: Move Decomposition & Overlay Generation")
                    print("=" * 60)

                    # Need to use board BEFORE the move for decomposition
                    # (board.push was not called yet in this version)
                    stages = decompose_move(board, best_move)

                    print(f"Move decomposed into {len(stages)} stage(s):")
                    for i, stage in enumerate(stages, 1):
                        print(f"  {i}. {stage['description']}")
                    print()

                    # Transformed image path
                    transformed_path = "data/chessboard_transformed.png"
                    if not Path(transformed_path).exists():
                        print(f"[X] Transformed image not found: {transformed_path}")
                        print("   Make sure --debug flag was used for detection")
                        return 1

                    # Get perspective matrix from detector
                    if not hasattr(detector, 'perspective_matrix'):
                        print(f"[X] Perspective matrix not found in detector")
                        print("   This should not happen - contact developer")
                        return 1

                    print("\n" + "=" * 60)
                    print("STAGE 5: Start Live Virtual Camera & Overlay")
                    print("=" * 60)

                    # Start live camera capture
                    live_capture = LiveCameraCapture(device_path)
                    live_capture.start()

                    # Wait for first frame
                    print("[LiveCapture] Waiting for first frame...")
                    time.sleep(2)

                    # Start virtual camera output
                    virtual_cam = VirtualCamera("/dev/video7", 1280, 720)
                    if not virtual_cam.start():
                        print("[X] Failed to start virtual camera")
                        print("   Run: sudo modprobe v4l2loopback devices=1 video_nr=7 card_label='ChessBot Virtual Cam' exclusive_caps=1")
                        live_capture.stop()
                        return 1

                    print(f"[OK] Virtual camera active at /dev/video7")
                    print(f"")
                    print(f"View low-latency stream with:")
                    print(f"  ffplay -fflags nobuffer -flags low_delay -framedrop /dev/video7")
                    print()

                    # Generate overlay for each stage with user interaction
                    for i, stage in enumerate(stages, 1):
                        print("\n" + "-" * 60)
                        print(f"Stage {i}/{len(stages)}: {stage['description']}")
                        print("-" * 60)

                        # Display stage details
                        if stage["pickup_square"]:
                            print(f"  Pickup: {stage['pickup_square']} (RED)")
                        if stage["place_square"]:
                            print(f"  Place: {stage['place_square']} (BLUE)")
                        if stage["pickup_square"] is None or stage["place_square"] is None:
                            graveyard_type = "PURPLE" if stage["piece"] else "ORANGE"
                            print(f"  Graveyard: Left of board ({graveyard_type})")

                        print(f"\n[LiveFeed] Streaming overlay to /dev/video7...")
                        print("           Live feed updates at ~30fps")
                        print("           Press ENTER to save final frame and continue...")

                        # Use event to signal when user presses ENTER
                        enter_pressed = threading.Event()
                        last_frame = [None]  # Use list to allow modification from thread

                        def wait_for_enter():
                            input()
                            enter_pressed.set()

                        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
                        input_thread.start()

                        # Stream live feed with overlay until user presses ENTER
                        while not enter_pressed.is_set():
                            # Get latest frame from live capture
                            frame = live_capture.get_latest_frame()

                            if frame is not None:
                                # Apply overlay to frame
                                overlayed_frame = apply_stage_overlay_to_frame(
                                    frame,
                                    stage,
                                    transformed_path,
                                    detector,
                                    detector.perspective_matrix,
                                    args.rotation
                                )

                                # Send to virtual camera
                                virtual_cam.write_frame(overlayed_frame)
                                last_frame[0] = overlayed_frame

                            # Minimal sleep to prevent CPU spinning (1ms)
                            time.sleep(0.001)

                        # Save the final frame for this stage
                        overlay_path = str(Path("data") / f"overlay_stage_{i}.png")
                        if last_frame[0] is not None:
                            cv2.imwrite(overlay_path, last_frame[0])
                            print(f"[OK] Final frame saved: {overlay_path}")

                    # Stop live capture and virtual camera
                    live_capture.stop()
                    virtual_cam.stop()

                    print("\n" + "=" * 60)
                    print("[OK] All stage overlays generated!")
                    print("=" * 60)
                    for i in range(1, len(stages) + 1):
                        overlay_path = Path("data") / f"overlay_stage_{i}.png"
                        print(f"  Stage {i}: {overlay_path.absolute()}")

                else:
                    print("[X] No legal moves available")

            except FileNotFoundError:
                print(f"[X] Engine not found at '{args.engine}'")
                print("  Install stockfish or specify engine path with --engine")
                print("  On macOS: brew install stockfish")
                print("  On Ubuntu: sudo apt-get install stockfish")
                print("  On Windows: Download from https://stockfishchess.org/download/")
            except Exception as e:
                print(f"[X] Error calculating best move: {e}")

        print("\n" + "=" * 60)
        print("[OK] Demo completed successfully!")
        print("=" * 60)
        print(f"\nCaptured image saved: {image_path.absolute()}")

        return 0

    except FileNotFoundError as e:
        print(f"[X] Error: {e}")
        print("\nPlease run download.py first to set up the model files:")
        print("  python scripts/download.py")
        return 1
    except Exception as e:
        print(f"[X] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
