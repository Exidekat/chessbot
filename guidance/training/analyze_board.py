"""
Analyze detection results for a standard starting position chess board.

This tool compares detection results against the known starting position
to identify specific detection and classification errors.

Usage:
    python -m guidance.training.analyze_board --image data/chessboardv2.png
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

from guidance.board_detector import BoardDetector
import chess


def get_starting_position_fen():
    """Return the FEN for standard chess starting position."""
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def parse_fen_to_grid(fen: str):
    """Parse FEN string to 8x8 grid."""
    board_fen = fen.split()[0]
    rows = board_fen.split('/')

    grid = {}
    for rank_idx, row in enumerate(rows):
        rank = 8 - rank_idx  # FEN starts from rank 8
        file_idx = 0

        for char in row:
            if char.isdigit():
                file_idx += int(char)
            else:
                file = chr(ord('a') + file_idx)
                square = f"{file}{rank}"
                grid[square] = char
                file_idx += 1

    return grid


def analyze_detections(expected_fen: str, detected_fen: str):
    """
    Compare expected vs detected board state.

    Args:
        expected_fen: FEN of expected position
        detected_fen: FEN of detected position

    Returns:
        Dictionary with analysis results
    """
    expected = parse_fen_to_grid(expected_fen)
    detected = parse_fen_to_grid(detected_fen)

    # Get all squares
    all_squares = [f"{f}{r}" for r in range(1, 9) for f in 'abcdefgh']

    # Track errors
    errors = {
        'missing': [],  # Expected piece but nothing detected
        'false_positive': [],  # Detected piece but should be empty
        'misclassified': [],  # Detected wrong piece type
        'correct': []  # Correctly detected
    }

    # Class confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))

    for square in all_squares:
        expected_piece = expected.get(square, '.')
        detected_piece = detected.get(square, '.')

        if expected_piece != '.':
            if detected_piece == '.':
                errors['missing'].append((square, expected_piece))
            elif detected_piece != expected_piece:
                errors['misclassified'].append((square, expected_piece, detected_piece))
                confusion[expected_piece][detected_piece] += 1
            else:
                errors['correct'].append(square)
                confusion[expected_piece][detected_piece] += 1
        elif detected_piece != '.':
            errors['false_positive'].append((square, detected_piece))

    return errors, confusion


def print_analysis(errors, confusion):
    """Print detailed analysis."""
    print("\n" + "=" * 70)
    print("DETECTION ANALYSIS")
    print("=" * 70)

    # Summary
    total_expected = len(errors['correct']) + len(errors['missing']) + len(errors['misclassified'])
    total_detected = len(errors['correct']) + len(errors['misclassified']) + len(errors['false_positive'])

    print(f"\nSummary:")
    print(f"  Expected pieces: {total_expected}")
    print(f"  Detected pieces: {total_detected}")
    print(f"  Correctly classified: {len(errors['correct'])} ({len(errors['correct'])/total_expected*100:.1f}%)")
    print(f"  Misclassified: {len(errors['misclassified'])} ({len(errors['misclassified'])/total_expected*100:.1f}%)")
    print(f"  Missing (not detected): {len(errors['missing'])} ({len(errors['missing'])/total_expected*100:.1f}%)")
    print(f"  False positives: {len(errors['false_positive'])}")

    # Missing pieces
    if errors['missing']:
        print(f"\n❌ Missing Detections ({len(errors['missing'])} pieces):")
        by_type = defaultdict(list)
        for square, piece in errors['missing']:
            by_type[piece].append(square)

        for piece in sorted(by_type.keys()):
            squares = ', '.join(by_type[piece])
            piece_name = get_piece_name(piece)
            print(f"  {piece_name}: {squares}")

    # Misclassifications
    if errors['misclassified']:
        print(f"\n❌ Misclassifications ({len(errors['misclassified'])} pieces):")

        # Group by error type
        by_error = defaultdict(list)
        for square, expected, detected in errors['misclassified']:
            by_error[(expected, detected)].append(square)

        for (expected, detected) in sorted(by_error.keys()):
            squares = ', '.join(by_error[(expected, detected)])
            exp_name = get_piece_name(expected)
            det_name = get_piece_name(detected)
            count = len(by_error[(expected, detected)])
            print(f"  {exp_name} → {det_name}: {squares} ({count}x)")

    # Confusion matrix
    if confusion:
        print(f"\n📊 Confusion Matrix:")
        print(f"{'Expected':<10} → {'Detected':<10} {'Count':>6}")
        print("-" * 30)

        for expected in sorted(confusion.keys()):
            for detected in sorted(confusion[expected].keys()):
                count = confusion[expected][detected]
                exp_name = get_piece_name(expected)
                det_name = get_piece_name(detected)

                if expected == detected:
                    print(f"  {exp_name:<10} → {det_name:<10} {count:>6} ✓")
                else:
                    print(f"  {exp_name:<10} → {det_name:<10} {count:>6} ❌")

    print("=" * 70)


def get_piece_name(piece: str) -> str:
    """Get human-readable piece name."""
    names = {
        'p': 'b_pawn', 'n': 'b_knight', 'b': 'b_bishop',
        'r': 'b_rook', 'q': 'b_queen', 'k': 'b_king',
        'P': 'w_pawn', 'N': 'w_knight', 'B': 'w_bishop',
        'R': 'w_rook', 'Q': 'w_queen', 'K': 'w_king',
    }
    return names.get(piece, piece)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze detection results for starting position board"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/chessboardv2.png",
        help="Path to board image (default: data/chessboardv2.png)"
    )
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Enable preprocessing (default: disabled)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Chess Board Detection Analysis")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Preprocessing: {'Enabled' if args.preprocessing else 'Disabled'}")
    print("=" * 70)

    # Run detection
    detector = BoardDetector()

    # Run board detection
    try:
        detected_fen, _ = detector.detect_board_state(args.image, debug=False)
    except Exception as e:
        print(f"Error during detection: {e}")
        return 1

    # Get expected FEN
    expected_fen = get_starting_position_fen()

    print(f"\nExpected FEN: {expected_fen}")
    print(f"Detected FEN: {detected_fen}")

    # Analyze
    errors, confusion = analyze_detections(expected_fen, detected_fen)

    # Print analysis
    print_analysis(errors, confusion)

    return 0


if __name__ == "__main__":
    sys.exit(main())
