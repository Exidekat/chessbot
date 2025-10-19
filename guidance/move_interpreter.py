"""
Move Interpreter Module

Interprets chess moves and decomposes them into atomic robot actions.
Handles normal moves, captures, castling, en passant, and promotions.
"""

import chess
from typing import List, Dict, Tuple, Optional
from enum import Enum


class MoveType(Enum):
    """Types of chess moves."""
    NORMAL = "normal"
    CAPTURE = "capture"
    CASTLING = "castling"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"


class ActionType(Enum):
    """Types of robot actions."""
    PICKUP = "pickup"
    PLACE = "place"
    GRAVEYARD = "graveyard"


class MoveInterpreter:
    """
    Interprets chess moves and decomposes them into robot action sequences.
    """

    @staticmethod
    def get_move_type(move: chess.Move, board: chess.Board) -> MoveType:
        """
        Determine the type of chess move.

        Args:
            move: Chess move to analyze
            board: Current board state

        Returns:
            MoveType enum
        """
        if board.is_castling(move):
            return MoveType.CASTLING
        elif board.is_en_passant(move):
            return MoveType.EN_PASSANT
        elif board.is_capture(move):
            return MoveType.CAPTURE
        elif move.promotion:
            return MoveType.PROMOTION
        else:
            return MoveType.NORMAL

    @staticmethod
    def is_capture(move: chess.Move, board: chess.Board) -> bool:
        """
        Check if move is a capture.

        Args:
            move: Chess move to check
            board: Current board state

        Returns:
            True if capture, False otherwise
        """
        return board.is_capture(move)

    @staticmethod
    def decompose_castling(move: chess.Move, board: chess.Board) -> Tuple[chess.Move, chess.Move]:
        """
        Decompose castling into two separate moves: king, then rook.

        Args:
            move: Castling move
            board: Current board state

        Returns:
            Tuple of (king_move, rook_move)
        """
        from_square = move.from_square
        to_square = move.to_square

        # Determine castling side
        if to_square > from_square:  # Kingside
            # White or black?
            if chess.square_rank(from_square) == 0:  # White (rank 1)
                rook_from = chess.H1
                rook_to = chess.F1
            else:  # Black (rank 8)
                rook_from = chess.H8
                rook_to = chess.F8
        else:  # Queenside
            if chess.square_rank(from_square) == 0:  # White
                rook_from = chess.A1
                rook_to = chess.D1
            else:  # Black
                rook_from = chess.A8
                rook_to = chess.D8

        king_move = chess.Move(from_square, to_square)
        rook_move = chess.Move(rook_from, rook_to)

        return king_move, rook_move

    @staticmethod
    def get_en_passant_capture_square(move: chess.Move, board: chess.Board) -> chess.Square:
        """
        Get the square of the pawn captured by en passant.

        Note: The captured pawn is NOT on the destination square!

        Args:
            move: En passant move
            board: Current board state

        Returns:
            Square of captured pawn
        """
        # Captured pawn is one rank behind the destination
        to_square = move.to_square
        if board.turn == chess.WHITE:
            # White capturing black pawn
            capture_square = to_square - 8  # One rank down
        else:
            # Black capturing white pawn
            capture_square = to_square + 8  # One rank up

        return capture_square

    @staticmethod
    def get_piece_at(square: chess.Square, board: chess.Board) -> str:
        """
        Get piece symbol at square.

        Args:
            square: Chess square
            board: Current board state

        Returns:
            Piece symbol (e.g., 'Q', 'p') or '.' if empty
        """
        piece = board.piece_at(square)
        return piece.symbol() if piece else '.'

    @staticmethod
    def square_to_name(square: chess.Square) -> str:
        """
        Convert square index to name.

        Args:
            square: Chess square index (0-63)

        Returns:
            Square name (e.g., 'e4')
        """
        return chess.square_name(square)

    @staticmethod
    def decompose_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """
        Decompose chess move into atomic robot actions.

        Args:
            move: Chess move to decompose
            board: Current board state

        Returns:
            List of action dictionaries:
            [
                {
                    "type": "pickup",
                    "square": "e2",
                    "piece": "P",
                    "status": "pending"
                },
                ...
            ]
        """
        move_type = MoveInterpreter.get_move_type(move, board)

        if move_type == MoveType.CASTLING:
            return MoveInterpreter._decompose_castling_move(move, board)
        elif move_type == MoveType.EN_PASSANT:
            return MoveInterpreter._decompose_en_passant_move(move, board)
        elif move_type == MoveType.CAPTURE:
            return MoveInterpreter._decompose_capture_move(move, board)
        elif move_type == MoveType.PROMOTION:
            return MoveInterpreter._decompose_promotion_move(move, board)
        else:  # NORMAL
            return MoveInterpreter._decompose_normal_move(move, board)

    @staticmethod
    def _decompose_normal_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """Decompose normal (non-capture, non-castling) move."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        piece = MoveInterpreter.get_piece_at(move.from_square, board)

        return [
            {
                "type": "pickup",
                "square": from_square,
                "piece": piece,
                "status": "pending"
            },
            {
                "type": "place",
                "square": to_square,
                "piece": piece,
                "status": "pending"
            }
        ]

    @staticmethod
    def _decompose_capture_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """Decompose capture move."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        our_piece = MoveInterpreter.get_piece_at(move.from_square, board)
        captured_piece = MoveInterpreter.get_piece_at(move.to_square, board)

        return [
            {
                "type": "pickup",
                "square": to_square,
                "piece": captured_piece,
                "status": "pending"
            },
            {
                "type": "graveyard",
                "square": None,  # Graveyard has no chess square
                "piece": captured_piece,
                "status": "pending"
            },
            {
                "type": "pickup",
                "square": from_square,
                "piece": our_piece,
                "status": "pending"
            },
            {
                "type": "place",
                "square": to_square,
                "piece": our_piece,
                "status": "pending"
            }
        ]

    @staticmethod
    def _decompose_castling_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """Decompose castling move (king first, then rook)."""
        king_move, rook_move = MoveInterpreter.decompose_castling(move, board)

        king_from = MoveInterpreter.square_to_name(king_move.from_square)
        king_to = MoveInterpreter.square_to_name(king_move.to_square)
        rook_from = MoveInterpreter.square_to_name(rook_move.from_square)
        rook_to = MoveInterpreter.square_to_name(rook_move.to_square)

        king_piece = MoveInterpreter.get_piece_at(king_move.from_square, board)
        rook_piece = MoveInterpreter.get_piece_at(rook_move.from_square, board)

        return [
            # Move king first
            {
                "type": "pickup",
                "square": king_from,
                "piece": king_piece,
                "status": "pending"
            },
            {
                "type": "place",
                "square": king_to,
                "piece": king_piece,
                "status": "pending"
            },
            # Then move rook
            {
                "type": "pickup",
                "square": rook_from,
                "piece": rook_piece,
                "status": "pending"
            },
            {
                "type": "place",
                "square": rook_to,
                "piece": rook_piece,
                "status": "pending"
            }
        ]

    @staticmethod
    def _decompose_en_passant_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """Decompose en passant move."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        capture_square = MoveInterpreter.get_en_passant_capture_square(move, board)
        capture_square_name = MoveInterpreter.square_to_name(capture_square)

        our_piece = MoveInterpreter.get_piece_at(move.from_square, board)
        captured_piece = MoveInterpreter.get_piece_at(capture_square, board)

        return [
            # Pick up opponent's pawn (not on destination square!)
            {
                "type": "pickup",
                "square": capture_square_name,
                "piece": captured_piece,
                "status": "pending"
            },
            {
                "type": "graveyard",
                "square": None,
                "piece": captured_piece,
                "status": "pending"
            },
            # Pick up our pawn
            {
                "type": "pickup",
                "square": from_square,
                "piece": our_piece,
                "status": "pending"
            },
            {
                "type": "place",
                "square": to_square,
                "piece": our_piece,
                "status": "pending"
            }
        ]

    @staticmethod
    def _decompose_promotion_move(move: chess.Move, board: chess.Board) -> List[Dict]:
        """
        Decompose promotion move.

        Note: Promotion may also be a capture!
        For now, we treat it as a normal move with the promoted piece.
        In a real implementation, the robot would need to swap pieces.
        """
        # Check if it's also a capture
        if board.is_capture(move):
            # Capture + promotion
            from_square = MoveInterpreter.square_to_name(move.from_square)
            to_square = MoveInterpreter.square_to_name(move.to_square)
            our_piece = MoveInterpreter.get_piece_at(move.from_square, board)
            captured_piece = MoveInterpreter.get_piece_at(move.to_square, board)

            # Get promoted piece symbol
            promoted_piece = chess.piece_symbol(move.promotion)
            if board.turn == chess.WHITE:
                promoted_piece = promoted_piece.upper()

            return [
                # Remove captured piece
                {
                    "type": "pickup",
                    "square": to_square,
                    "piece": captured_piece,
                    "status": "pending"
                },
                {
                    "type": "graveyard",
                    "square": None,
                    "piece": captured_piece,
                    "status": "pending"
                },
                # Remove pawn
                {
                    "type": "pickup",
                    "square": from_square,
                    "piece": our_piece,
                    "status": "pending"
                },
                {
                    "type": "graveyard",
                    "square": None,
                    "piece": our_piece,
                    "status": "pending",
                    "note": "Pawn to be promoted"
                },
                # Place promoted piece
                {
                    "type": "place",
                    "square": to_square,
                    "piece": promoted_piece,
                    "status": "pending",
                    "note": "Promoted piece - robot must fetch from reserve"
                }
            ]
        else:
            # Simple promotion
            from_square = MoveInterpreter.square_to_name(move.from_square)
            to_square = MoveInterpreter.square_to_name(move.to_square)
            our_piece = MoveInterpreter.get_piece_at(move.from_square, board)

            promoted_piece = chess.piece_symbol(move.promotion)
            if board.turn == chess.WHITE:
                promoted_piece = promoted_piece.upper()

            return [
                # Remove pawn
                {
                    "type": "pickup",
                    "square": from_square,
                    "piece": our_piece,
                    "status": "pending"
                },
                {
                    "type": "graveyard",
                    "square": None,
                    "piece": our_piece,
                    "status": "pending",
                    "note": "Pawn to be promoted"
                },
                # Place promoted piece
                {
                    "type": "place",
                    "square": to_square,
                    "piece": promoted_piece,
                    "status": "pending",
                    "note": "Promoted piece - robot must fetch from reserve"
                }
            ]
