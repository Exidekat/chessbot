"""
Create Overlay Demo - VLA Training Data Generation

This script generates stage-by-stage move overlays for VLA training:
1. Capture fresh 1080p MJPEG -> 720p downscale photo from camera
2. Detect board state from the image using BoardDetector
3. Calculate best move using MoveCalculator
4. Decompose move into stages (capture, movement, promotion, etc.)
5. Generate overlay for each stage with user interaction
6. Save stage overlays for VLA training data collection

Usage:
    python scripts/create_overlay_demo.py [--engine ENGINE_PATH] [--device CAMERA_DEVICE]
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
from guidance.move_decomposer import decompose_move
from guidance import rotate_square_for_camera, apply_stage_overlay_to_frame
import numpy as np

# Import shared utilities
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device,
    capture_1080p_downscale,
    get_default_global_camera,
)


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


# NOTE: Camera functions imported from utils.camera_helpers
# NOTE: rotate_square_for_camera imported from guidance module


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

    This is a file-based wrapper around apply_stage_overlay_to_frame() from
    the shared guidance module. Loads image from disk, applies overlay, saves result.

    Args:
        stage: Stage dictionary from decompose_move()
        original_image_path: Path to original 720p captured image
        transformed_image_path: Path to transformed board image
        output_path: Path to save overlay image
        detector: BoardDetector instance for coordinate mapping
        perspective_matrix: Perspective transform matrix (M)
        camera_position: Camera position for rotation handling
    """
    # Load original 720p image
    image = cv2.imread(original_image_path)
    if image is None:
        print(f"[X] Failed to load image: {original_image_path}")
        return

    # Apply overlay using shared module
    result = apply_stage_overlay_to_frame(
        frame=image,
        stage=stage,
        transformed_image_path=transformed_image_path,
        detector=detector,
        perspective_matrix=perspective_matrix,
        camera_position=camera_position
    )

    if result is not None:
        cv2.imwrite(output_path, result)
        print(f"[OK] Overlay saved: {output_path}")
    else:
        print(f"[X] Failed to generate overlay")


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
    print("STAGE 1: Camera Capture (1080p MJPEG -> 720p Downscale)")
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

    # Capture photo using 4K MJPEG with downscaling to 720p
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path("data") / f"chessboard_capture_{timestamp}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    success = capture_1080p_downscale(device_path, image_path)
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

                    # Generate overlay for each stage with user interaction
                    for i, stage in enumerate(stages, 1):
                        print("\n" + "-" * 60)
                        print(f"Stage {i}/{len(stages)}: {stage['description']}")
                        print("-" * 60)

                        # Generate overlay on ORIGINAL 720p image
                        overlay_path = str(Path("data") / f"overlay_stage_{i}.png")
                        generate_stage_overlay(
                            stage,
                            str(image_path),  # Original 720p captured image
                            transformed_path,  # Transformed board for coordinate mapping
                            overlay_path,
                            detector,
                            detector.perspective_matrix,
                            args.rotation
                        )

                        # Display stage details
                        if stage["pickup_square"]:
                            print(f"  Pickup: {stage['pickup_square']} (RED)")
                        if stage["place_square"]:
                            print(f"  Place: {stage['place_square']} (BLUE)")
                        if stage["pickup_square"] is None or stage["place_square"] is None:
                            print(f"  Graveyard: Left of board (ORANGE)")

                        # Wait for user input (except on last stage)
                        if i < len(stages):
                            input("\n  Press ENTER to continue to next stage...")

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
