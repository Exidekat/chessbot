"""
Best Move Demo - Unified Guidance System

This script demonstrates the new guidance module architecture:
1. Capture fresh YUYV 720p photo from camera
2. Detect board state from the image using BoardDetector
3. Calculate best move using MoveCalculator
4. Display results

This replaces the old main.py but uses the new modular guidance system.

Usage:
    python best_move_demo.py [--engine ENGINE_PATH] [--device CAMERA_DEVICE]
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

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator

# Import shared utilities
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device,
    capture_4k_downscale
)


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


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


def main():
    """Main demo driver."""
    parser = argparse.ArgumentParser(
        description="Best Move Demo - Unified Guidance System with Live Camera Capture"
    )
    parser.add_argument(
        "--global-camera",
        type=str,
        default=None,
        help="Global/overhead camera device (e.g., /dev/video0). If not specified, auto-detects."
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
        default="right",
        choices=["left", "right", "top", "bottom"],
        help="Camera rotation relative to board (default: right)"
    )
    parser.add_argument(
        "--turn",
        type=str,
        default="black",
        choices=["white", "black"],
        help="Whose turn to calculate move for (default: black)"
    )

    args = parser.parse_args()

    # Normalize turn to single letter for internal use
    turn_letter = "w" if args.turn == "white" else "b"

    print("=" * 60)
    print("Best Move Demo - Unified Guidance System")
    print("=" * 60)
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    print(f"Board rotation: {args.rotation}")
    print(f"Calculate move for: {args.turn.capitalize()}")
    if args.debug:
        print(f"Corner confidence: {args.corner_conf}")
        print(f"Min corner distance: {args.min_corner_dist}")
    print()

    # Step 1: Camera selection and photo capture
    print("=" * 60)
    print("STAGE 1: Camera Capture (4K MJPEG -> 720p Downscale)")
    print("=" * 60)

    # Determine which camera to use
    if args.global_camera:
        # User specified device
        device_path = args.global_camera
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
            turn=turn_letter
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

                    # Show board after the move
                    board.push(best_move)
                    print("\nBoard after best move:")
                    print("-" * 60)
                    print(board)
                    print("-" * 60)
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
