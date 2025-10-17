"""
Test driver for the Chess Vision Board State system.

This script demonstrates how to use the BoardState class to:
1. Take a snapshot of a chess board from an image
2. Get the current board state
3. Calculate the best move using a chess engine

Usage:
    python main.py [--image IMAGE_PATH] [--engine ENGINE_PATH]
"""

import argparse
import sys
from pathlib import Path
from board_state import BoardState
import chess


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


def main():
    """Main test driver."""
    parser = argparse.ArgumentParser(
        description="Test the Chess Vision Board State system"
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
        "--output",
        type=str,
        choices=["fen", "board"],
        default="board",
        help="Output format: 'fen' or 'board' (default: board)"
    )
    parser.add_argument(
        "--no-bestmove",
        action="store_true",
        help="Skip calculating best move (useful if engine not available)"
    )

    args = parser.parse_args()

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print(f"Please ensure the test image exists at {args.image}")
        return 1

    print("=" * 60)
    print("Chess Vision Board State - Test Driver")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output format: {args.output}")
    print()

    try:
        # Initialize BoardState
        print("Initializing BoardState...")
        detector = BoardState()
        print("✓ BoardState initialized")
        print()

        # Take snapshot
        print(f"Taking snapshot of {image_path}...")
        result = detector.snapshot(str(image_path), output_format=args.output)
        print("✓ Snapshot complete")
        print()

        # Display results
        if args.output == "fen":
            print("FEN Notation:")
            print("-" * 60)
            print(result)
            print("-" * 60)
            print()

            # Also get board for display
            board = detector.current(output_format="board")
            print_board(board)

        else:  # board
            print_board(result)

            # Also show FEN
            fen = detector.current(output_format="fen")
            print("\nFEN Notation:")
            print("-" * 60)
            print(fen)
            print("-" * 60)

        # Test current() function
        print("\n" + "=" * 60)
        print("Testing current() function:")
        print("=" * 60)
        current_board = detector.current(output_format="board")
        current_fen = detector.current(output_format="fen")
        print(f"✓ Current board retrieved successfully")
        print(f"FEN: {current_fen}")
        print()

        # Calculate best move if requested
        if not args.no_bestmove:
            print("=" * 60)
            print("Calculating best move:")
            print("=" * 60)
            print(f"Engine: {args.engine}")
            print(f"Time limit: {args.time}s")
            print()

            try:
                best_move = detector.bestmove(
                    engine_path=args.engine,
                    time_limit=args.time
                )

                if best_move:
                    print(f"✓ Best move: {best_move}")
                    print(f"  UCI notation: {best_move.uci()}")

                    # Show the move in algebraic notation
                    board = detector.current(output_format="board")
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
            except Exception as e:
                print(f"✗ Error calculating best move: {e}")

        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
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
