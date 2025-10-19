"""
Move Calculator Module

Calculates best moves using UCI chess engines (e.g., Stockfish).
"""

import chess
import chess.engine
from typing import Optional


class MoveCalculator:
    """
    Calculates optimal chess moves using UCI engine analysis.
    """

    def __init__(self, engine_path: str = "stockfish"):
        """
        Initialize the move calculator.

        Args:
            engine_path: Path to UCI engine executable (e.g., 'stockfish')
        """
        self.engine_path = engine_path

    def calculate_best_move(
        self,
        board: chess.Board,
        time_limit: float = 1.0
    ) -> Optional[chess.Move]:
        """
        Calculate the best move using a UCI engine.

        Args:
            board: Current board state
            time_limit: Time limit in seconds for analysis

        Returns:
            Best move as chess.Move object, or None if no legal moves
        """
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            result = engine.play(
                board,
                chess.engine.Limit(time=time_limit)
            )
            return result.move

    def analyze_position(
        self,
        board: chess.Board,
        time_limit: float = 1.0,
        depth: Optional[int] = None
    ) -> dict:
        """
        Analyze a position and return detailed evaluation.

        Args:
            board: Current board state
            time_limit: Time limit in seconds for analysis
            depth: Optional depth limit for analysis

        Returns:
            Dictionary with analysis results including:
            - best_move: Best move
            - score: Evaluation score
            - pv: Principal variation
        """
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            # Set up limit
            if depth:
                limit = chess.engine.Limit(depth=depth)
            else:
                limit = chess.engine.Limit(time=time_limit)

            # Analyze
            info = engine.analyse(board, limit)

            return {
                "best_move": info.get("pv", [None])[0] if "pv" in info else None,
                "score": info.get("score"),
                "pv": info.get("pv", []),
                "depth": info.get("depth"),
                "nodes": info.get("nodes")
            }
