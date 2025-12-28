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
from typing import Optional

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator
from guidance.move_decomposer import decompose_move
from guidance.coordinate_mapper import CoordinateMapper
from guidance import rotate_square_for_camera, apply_stage_overlay_to_frame

# Import from cameras module
from cameras import LiveCameraCapture, VirtualCamera
from cameras.live_camera_capture import get_camera_index_from_device

# Import shared camera utilities
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    capture_4k_downscale
)

from PIL import Image
import numpy as np


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


# NOTE: Camera functions (get_available_cameras, select_camera, capture_4k_downscale)
# imported from utils.camera_helpers
# NOTE: rotate_square_for_camera, apply_stage_overlay_to_frame imported from guidance module


def main():
    """Main demo driver for overlay generation."""
    parser = argparse.ArgumentParser(
        description="Create Overlay Demo - Generate stage-by-stage move overlays for VLA training"
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
        default=None,
        choices=["left", "right", "top", "bottom"],
        help="Camera rotation relative to board (default: right). Use 'top' for no rotation."
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
        default="black",
        choices=["white", "black"],
        help="Whose turn to calculate move for (default: black)"
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

    # Normalize turn to single letter for internal use
    turn_letter = "w" if args.turn == "white" else "b"

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
                        print(f"  VLM Prompt: {stage['vlm_prompt']}")

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
