"""
Best Move Demo - Unified Guidance System

This script demonstrates the new guidance module architecture:
1. Detect board state from an image using BoardDetector
2. Calculate best move using MoveCalculator
3. Display results

This replaces the old main.py but uses the new modular guidance system.

Usage:
    python best_move_demo.py [--image IMAGE_PATH] [--engine ENGINE_PATH]
"""

import argparse
import sys
from pathlib import Path
import chess

# Import from new guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator


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
        description="Best Move Demo - Unified Guidance System"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/chessboard.png",
        help="Path to the chess board image (default: data/chessboard.png)"
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
        default=0.1,
        help="Confidence threshold for corner detection (0.0-1.0, default: 0.1)"
    )
    parser.add_argument(
        "--min-corner-dist",
        type=float,
        default=50.0,
        help="Minimum distance between corners in pixels (default: 50.0)"
    )

    args = parser.parse_args()

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print(f"Please ensure the test image exists at {args.image}")
        return 1

    print("=" * 60)
    print("Best Move Demo - Unified Guidance System")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    if args.debug:
        print(f"Corner confidence: {args.corner_conf}")
        print(f"Min corner distance: {args.min_corner_dist}")
    print()

    try:
        # Initialize BoardDetector
        print("Initializing BoardDetector...")
        detector = BoardDetector()
        print("✓ BoardDetector initialized")
        print()

        # Detect board state
        print(f"Detecting board state from {image_path}...")
        fen, transformed_image = detector.detect_board_state(
            str(image_path),
            corner_conf=args.corner_conf,
            min_corner_distance=args.min_corner_dist,
            debug=args.debug
        )
        print("✓ Board state detected")
        print()

        # Create chess board from FEN
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"✗ Invalid FEN: {fen}")
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
            print("Calculating best move:")
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
                    print(f"✓ Best move: {best_move}")
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
                    print("✗ No legal moves available")

            except FileNotFoundError:
                print(f"✗ Engine not found at '{args.engine}'")
                print("  Install stockfish or specify engine path with --engine")
                print("  On macOS: brew install stockfish")
                print("  On Ubuntu: sudo apt-get install stockfish")
                print("  On Windows: Download from https://stockfishchess.org/download/")
            except Exception as e:
                print(f"✗ Error calculating best move: {e}")

        print("\n" + "=" * 60)
        print("✓ Demo completed successfully!")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease run download.py first to set up the model files:")
        print("  python download.py")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
