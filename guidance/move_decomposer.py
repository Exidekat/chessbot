"""
Move Decomposer Module

Decomposes chess moves into stages for robot execution and VLA training.
Generates both technical descriptions and VLM-friendly natural language prompts.
"""

import chess
from typing import List, Dict, Optional


def piece_symbol_to_name(symbol: str) -> str:
    """
    Convert piece symbol to full name with color.

    Args:
        symbol: Chess piece symbol (e.g., 'P', 'p', 'Q', 'q')

    Returns:
        Full piece name (e.g., "white pawn", "black queen")
    """
    piece_names = {
        'p': 'pawn', 'n': 'knight', 'b': 'bishop',
        'r': 'rook', 'q': 'queen', 'k': 'king',
        'P': 'pawn', 'N': 'knight', 'B': 'bishop',
        'R': 'rook', 'Q': 'queen', 'K': 'king'
    }

    color = "white" if symbol.isupper() else "black"
    piece_type = piece_names.get(symbol.lower(), "piece")

    return f"{color} {piece_type}"


def generate_vlm_prompt(stage: Dict) -> str:
    """
    Generate VLM-friendly natural language prompt for a move stage with color conditioning.

    Color conditioning matches the overlay highlight colors:
    - RED: Pickup square
    - BLUE: Place square
    - ORANGE: Graveyard (discard)
    - PURPLE: Graveyard piece (promotion source)

    Args:
        stage: Stage dictionary from decompose_move()

    Returns:
        Natural language prompt for VLM training with color conditioning
    """
    pickup_square = stage["pickup_square"]
    place_square = stage["place_square"]
    piece = stage["piece"]

    # Determine piece name for the prompt
    # For pickup from board, extract from description
    if pickup_square and not piece:
        # Extract piece symbol from description (e.g., "Remove B from f1" -> "B")
        # or "Move P from e2 to e4" -> "P"
        desc = stage["description"]
        if "Remove" in desc:
            # Format: "Remove {piece} from {square}"
            piece_symbol = desc.split()[1]
        elif "Move" in desc:
            # Format: "Move {piece} from {square} to {square}"
            piece_symbol = desc.split()[1]
        else:
            piece_symbol = "piece"

        piece_name = piece_symbol_to_name(piece_symbol)
    elif piece:
        # Taking FROM graveyard
        piece_name = piece_symbol_to_name(piece)
    else:
        piece_name = "piece"

    # Generate prompt based on action type with color conditioning
    if pickup_square and place_square:
        # Board to board move: RED pickup, BLUE place
        return f"Pick up {piece_name} from red {pickup_square} and place on blue {place_square}."
    elif pickup_square and not place_square:
        # Board to graveyard (capture or removal): RED pickup, ORANGE graveyard
        return f"Pick up {piece_name} from red {pickup_square} and place in orange graveyard left of board."
    elif not pickup_square and place_square:
        # Graveyard to board (promotion): PURPLE graveyard, BLUE place
        return f"Pick up {piece_name} from purple graveyard left of board and place on blue {place_square}."
    else:
        # Fallback
        return "Complete the current move."


def decompose_move(board: chess.Board, move: chess.Move) -> List[Dict]:
    """
    Decompose a chess move into stages for robot execution and VLA training.

    Each stage represents a single physical action (pickup + place) that the robot
    must execute. Complex moves like castling and promotion are broken down into
    multiple stages.

    Each stage is a dict with:
    - description: Human-readable technical description
    - vlm_prompt: Natural language prompt for VLM training
    - pickup_square: Square to pick piece from (or None for graveyard)
    - place_square: Square to place piece on (or None for graveyard)
    - piece: Piece type being moved (for graveyard -> board moves)

    Args:
        board: Current board state (before move is applied)
        move: Chess move to decompose

    Returns:
        List of stage dictionaries

    Examples:
        Normal move (e2e4):
            [{"description": "Move P from e2 to e4",
              "vlm_prompt": "Pick up white pawn from red e2 and place on blue e4.",
              "pickup_square": "e2", "place_square": "e4", "piece": None}]

        Capture (Nxe5):
            [{"description": "Remove p from e5 (captured)",
              "vlm_prompt": "Pick up black pawn from red e5 and place in orange graveyard left of board.",
              "pickup_square": "e5", "place_square": None, "piece": None},
             {"description": "Move N from c3 to e5",
              "vlm_prompt": "Pick up white knight from red c3 and place on blue e5.",
              "pickup_square": "c3", "place_square": "e5", "piece": None}]

        Promotion with capture (g2f1Q):
            [{"description": "Remove B from f1 (captured)",
              "vlm_prompt": "Pick up white bishop from red f1 and place in orange graveyard left of board.",
              "pickup_square": "f1", "place_square": None, "piece": None},
             {"description": "Remove p from g2",
              "vlm_prompt": "Pick up black pawn from red g2 and place in orange graveyard left of board.",
              "pickup_square": "g2", "place_square": None, "piece": None},
             {"description": "Place q on f1 (promotion from graveyard)",
              "vlm_prompt": "Pick up black queen from purple graveyard left of board and place on blue f1.",
              "pickup_square": None, "place_square": "f1", "piece": "q"}]
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
        king_stage = {
            "description": f"Move {moving_piece.symbol().upper()} from {from_square} to {to_square} (castling)",
            "pickup_square": from_square,
            "place_square": to_square,
            "piece": None
        }
        king_stage["vlm_prompt"] = generate_vlm_prompt(king_stage)
        stages.append(king_stage)

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

        rook_stage = {
            "description": f"Move rook from {rook_from} to {rook_to} (castling)",
            "pickup_square": rook_from,
            "place_square": rook_to,
            "piece": None
        }
        rook_stage["vlm_prompt"] = generate_vlm_prompt(rook_stage)
        stages.append(rook_stage)

        return stages

    # PROMOTION WITH CAPTURE
    if is_promotion and captured_piece:
        # Stage 1: Remove captured piece
        piece_name = captured_piece.symbol().upper() if captured_piece.color == chess.WHITE else captured_piece.symbol().lower()
        capture_stage = {
            "description": f"Remove {piece_name} from {to_square} (captured)",
            "pickup_square": to_square,
            "place_square": None,  # To graveyard
            "piece": None
        }
        capture_stage["vlm_prompt"] = generate_vlm_prompt(capture_stage)
        stages.append(capture_stage)

        # Stage 2: Remove pawn
        pawn_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        pawn_stage = {
            "description": f"Remove {pawn_name} from {from_square}",
            "pickup_square": from_square,
            "place_square": None,  # To graveyard
            "piece": None
        }
        pawn_stage["vlm_prompt"] = generate_vlm_prompt(pawn_stage)
        stages.append(pawn_stage)

        # Stage 3: Place promoted piece
        promotion_piece_type = chess.piece_name(move.promotion)
        promotion_symbol = chess.Piece(move.promotion, moving_piece.color).symbol()
        promotion_stage = {
            "description": f"Place {promotion_symbol} on {to_square} (promotion from graveyard)",
            "pickup_square": None,  # From graveyard
            "place_square": to_square,
            "piece": promotion_symbol
        }
        promotion_stage["vlm_prompt"] = generate_vlm_prompt(promotion_stage)
        stages.append(promotion_stage)

        return stages

    # PROMOTION WITHOUT CAPTURE
    if is_promotion:
        # Stage 1: Remove pawn
        pawn_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        pawn_stage = {
            "description": f"Remove {pawn_name} from {from_square}",
            "pickup_square": from_square,
            "place_square": None,  # To graveyard
            "piece": None
        }
        pawn_stage["vlm_prompt"] = generate_vlm_prompt(pawn_stage)
        stages.append(pawn_stage)

        # Stage 2: Place promoted piece
        promotion_piece_type = chess.piece_name(move.promotion)
        promotion_symbol = chess.Piece(move.promotion, moving_piece.color).symbol()
        promotion_stage = {
            "description": f"Place {promotion_symbol} on {to_square} (promotion from graveyard)",
            "pickup_square": None,  # From graveyard
            "place_square": to_square,
            "piece": promotion_symbol
        }
        promotion_stage["vlm_prompt"] = generate_vlm_prompt(promotion_stage)
        stages.append(promotion_stage)

        return stages

    # CAPTURE (non-promotion)
    if captured_piece:
        # Stage 1: Remove captured piece to graveyard
        piece_name = captured_piece.symbol().upper() if captured_piece.color == chess.WHITE else captured_piece.symbol().lower()
        capture_stage = {
            "description": f"Remove {piece_name} from {to_square} (captured)",
            "pickup_square": to_square,
            "place_square": None,  # To graveyard
            "piece": None
        }
        capture_stage["vlm_prompt"] = generate_vlm_prompt(capture_stage)
        stages.append(capture_stage)

        # Stage 2: Move attacking piece
        attacker_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
        move_stage = {
            "description": f"Move {attacker_name} from {from_square} to {to_square}",
            "pickup_square": from_square,
            "place_square": to_square,
            "piece": None
        }
        move_stage["vlm_prompt"] = generate_vlm_prompt(move_stage)
        stages.append(move_stage)

        return stages

    # NORMAL MOVE (no capture, no special moves)
    piece_name = moving_piece.symbol().upper() if moving_piece.color == chess.WHITE else moving_piece.symbol().lower()
    normal_stage = {
        "description": f"Move {piece_name} from {from_square} to {to_square}",
        "pickup_square": from_square,
        "place_square": to_square,
        "piece": None
    }
    normal_stage["vlm_prompt"] = generate_vlm_prompt(normal_stage)
    stages.append(normal_stage)

    return stages
