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
from datetime import datetime
import time

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
    capture_4k_downscale,
    capture_720p_yuyv,
    get_default_global_camera,
    DEFAULT_USE_YUYV,
)


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


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
    parser.add_argument(
        "--yuyv",
        action="store_true",
        help="Force native 720p YUYV capture (uncompressed, best quality, 10fps). This is the default."
    )
    parser.add_argument(
        "--mjpeg",
        action="store_true",
        help="Use 4K MJPEG -> 720p downscale (supersampled, 30fps) instead of default YUYV"
    )
    parser.add_argument(
        "--corner-grayscale",
        action="store_true",
        help="Use grayscale+CLAHE for corner detection (default is RGB)"
    )
    parser.add_argument(
        "--piece-grayscale",
        action="store_true",
        help="Use grayscale+CLAHE for piece detection (default is RGB)"
    )

    args = parser.parse_args()

    # Determine capture mode: --mjpeg overrides to MJPEG, --yuyv forces YUYV, else use default
    use_yuyv = DEFAULT_USE_YUYV
    if args.mjpeg:
        use_yuyv = False
    elif args.yuyv:
        use_yuyv = True

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
    capture_mode_str = "720p YUYV Uncompressed" if use_yuyv else "4K MJPEG -> 720p Downscale"
    print(f"STAGE 1: Camera Capture ({capture_mode_str})")
    print("=" * 60)

    # Determine which camera to use: specified > auto-detect by name > prompt
    if args.global_camera:
        # User specified device
        device_path = args.global_camera
        print(f"[OK] Using specified device: {device_path}")
    else:
        # Try auto-detect by name first (WBC-0E01)
        device_path = get_default_global_camera()
        if device_path:
            print(f"[OK] Auto-detected global camera: {device_path} (WBC-0E01)")
        else:
            # Fallback to camera selection
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
                print("[WARN] WBC-0E01 not found, prompting for selection...")
                device_path = select_camera(cameras)

    # Capture photo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path("data") / f"chessboard_capture_{timestamp}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    if use_yuyv:
        success = capture_720p_yuyv(device_path, image_path)
    else:
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
        detector = BoardDetector(
            camera_position=args.rotation,
            use_corner_rgb=not args.corner_grayscale,
            use_piece_rgb=not args.piece_grayscale,
        )
        print(f"[OK] BoardDetector initialized (rotation: {args.rotation})")
        corner_mode = "grayscale+CLAHE" if args.corner_grayscale else "RGB"
        piece_mode = "grayscale+CLAHE" if args.piece_grayscale else "RGB"
        print(f"     Corner preprocessing: {corner_mode}")
        print(f"     Piece preprocessing: {piece_mode}")
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
