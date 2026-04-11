"""
Guidance System Orchestrator

High-level orchestrator that coordinates:
- Board detection (BoardDetector)
- Move calculation (MoveCalculator)
- Move decomposition (MoveInterpreter)
- State cache updates
- Overlay generation

This is the main entry point for the guidance system.
"""

import chess
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from PIL import Image

from .board_detector import BoardDetector
from .move_calculator import MoveCalculator
from .move_interpreter import MoveInterpreter
from .coordinate_mapper import CoordinateMapper
from .highlight_renderer import HighlightRenderer
from utils.state_cache import StateCache
from cameras.overlay_generator import OverlayGenerator


class GuidanceSystem:
    """
    High-level orchestrator for the guidance system.

    Coordinates all guidance components and provides simple API for:
    - Detecting board state and calculating best move
    - Updating state cache
    - Generating overlay images
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the guidance system.

        Args:
            config: Configuration dictionary with paths and settings
        """
        config = config or {}

        # Initialize components
        self.detector = BoardDetector(
            corner_model_path=config.get('corner_model', 'data/best_corners.pt'),
            piece_model_path=config.get('piece_model', 'data/best_transformed_detection.pt')
        )

        self.calculator = MoveCalculator(
            engine_path=config.get('engine_path', 'stockfish')
        )

        self.coord_mapper = CoordinateMapper(board_detector=self.detector)
        self.renderer = HighlightRenderer(alpha=0.5, coord_mapper=self.coord_mapper)

        # State cache
        cache_path = config.get('cache_path', 'data/state_cache.json')
        self.cache = StateCache(cache_path)

        # Overlay generator (for signaling cameras)
        overlay_path = config.get('overlay_path', 'data/guidance_overlay.png')
        flag_path = config.get('overlay_flag_path', 'data/overlay_ready.flag')
        self.overlay_gen = OverlayGenerator(overlay_path, flag_path)

    def detect_and_calculate(
        self,
        image_path: str,
        update_cache: bool = True,
        robot_plays_white: bool = False,
        debug: bool = False
    ) -> Tuple[str, Optional[chess.Move], List[Dict]]:
        """
        Detect board state and calculate best move.

        This is the main entry point for guidance pipeline.

        Args:
            image_path: Path to board image
            update_cache: Whether to update state cache (default: True)
            robot_plays_white: Whether robot plays white pieces
            debug: Enable debug output

        Returns:
            Tuple of (FEN, best_move, action_sequence)
        """
        print(f"[GuidanceSystem] Detecting board state from {image_path}")

        # 1. Detect board state
        fen, transformed_image = self.detector.detect_board_state(
            image_path,
            debug=debug
        )

        print(f"[GuidanceSystem] Detected FEN: {fen}")

        # Save transformed image for overlay generation
        transformed_path = "data/chessboard_transformed.png"
        transformed_image.save(transformed_path)

        # 2. Create chess board
        try:
            board = chess.Board(fen)
        except ValueError:
            print(f"[GuidanceSystem] Warning: Invalid FEN, creating empty board")
            board = chess.Board(None)
            if update_cache:
                self.cache.update({
                    "game_state": {
                        "fen": fen,
                        "board_image_path": image_path,
                        "transformed_board_path": transformed_path
                    }
                }, source="guidance")
            return fen, None, []

        # 3. Calculate best move and evaluate position
        print(f"[GuidanceSystem] Calculating best move...")
        best_move = self.calculator.calculate_best_move(board, time_limit=1.0)
        analysis = self.calculator.analyze_position(board, time_limit=1.0)
        evaluation = None
        if analysis.get("score") is not None:
            score = analysis["score"].relative
            if score.is_mate():
                evaluation = f"M{score.mate()}"
            else:
                evaluation = score.score() / 100.0  # centipawns to pawns

        if best_move is None:
            print(f"[GuidanceSystem] No legal moves available")
            if update_cache:
                self.cache.update({
                    "game_state": {
                        "fen": fen,
                        "turn": "white" if board.turn == chess.WHITE else "black",
                        "board_image_path": image_path,
                        "transformed_board_path": transformed_path
                    }
                }, source="guidance")
            return fen, None, []

        # Get move in SAN notation
        best_move_san = board.san(best_move)
        print(f"[GuidanceSystem] Best move: {best_move.uci()} ({best_move_san})")

        # 4. Decompose move into actions
        move_type = MoveInterpreter.get_move_type(best_move, board)
        action_sequence = MoveInterpreter.decompose_move(best_move, board)

        print(f"[GuidanceSystem] Move type: {move_type.value}")
        print(f"[GuidanceSystem] Action sequence: {len(action_sequence)} actions")

        # 5. Update cache if requested
        if update_cache:
            # Determine whose turn it is
            turn = "white" if board.turn == chess.WHITE else "black"
            is_robot_turn = (turn == "white" and robot_plays_white) or \
                           (turn == "black" and not robot_plays_white)

            self.cache.update({
                "game_state": {
                    "fen": fen,
                    "turn": turn,
                    "board_image_path": image_path,
                    "transformed_board_path": transformed_path
                },
                "robot_state": {
                    "is_robot_turn": is_robot_turn,
                    "current_move": best_move.uci(),
                    "move_type": move_type.value,
                    "action_sequence": action_sequence,
                    "action_index": 0,
                    "actions_remaining": len(action_sequence)
                },
                "guidance_state": {
                    "best_move": best_move.uci(),
                    "best_move_san": best_move_san,
                    "evaluation": evaluation
                }
            }, source="guidance")

            print(f"[GuidanceSystem] State cache updated")

        return fen, best_move, action_sequence

    def generate_overlay_from_cache(self) -> bool:
        """
        Generate overlay image from current state cache.

        Reads cache, loads transformed board, renders highlights, and signals cameras.

        Returns:
            True if overlay generated successfully, False otherwise
        """
        print("[GuidanceSystem] Generating overlay from cache...")

        # 1. Read cache
        cache_data = self.cache.get()

        # 2. Load transformed board image
        transformed_path = cache_data.get("game_state", {}).get("transformed_board_path")
        if not transformed_path or not Path(transformed_path).exists():
            print("[GuidanceSystem] Error: Transformed board image not found")
            return False

        try:
            transformed_image = Image.open(transformed_path)
        except Exception as e:
            print(f"[GuidanceSystem] Error loading transformed image: {e}")
            return False

        # 3. Render overlay
        overlay_bgr = self.renderer.render_from_cache(transformed_image, cache_data)

        # 4. Save overlay
        overlay_path = cache_data.get("metadata", {}).get("overlay_path", "data/guidance_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)
        print(f"[GuidanceSystem] Overlay saved to {overlay_path}")

        # 5. Signal cameras (create flag file)
        self.overlay_gen.signal_overlay_ready()
        print(f"[GuidanceSystem] Overlay ready signal sent")

        return True

    def generate_overlay_for_action(
        self,
        action_index: int,
        transformed_image_path: Optional[str] = None
    ) -> bool:
        """
        Generate overlay for a specific action in the sequence.

        Args:
            action_index: Index of action to highlight
            transformed_image_path: Path to transformed board (optional, reads from cache if None)

        Returns:
            True if successful, False otherwise
        """
        # Get cache data
        cache_data = self.cache.get()

        # Load transformed image
        if transformed_image_path is None:
            transformed_image_path = cache_data.get("game_state", {}).get("transformed_board_path")

        if not transformed_image_path or not Path(transformed_image_path).exists():
            print(f"[GuidanceSystem] Error: Transformed board not found")
            return False

        transformed_image = Image.open(transformed_image_path)

        # Get action sequence
        action_sequence = cache_data.get("robot_state", {}).get("action_sequence", [])

        if action_index >= len(action_sequence):
            print(f"[GuidanceSystem] Error: Action index {action_index} out of range")
            return False

        # Render overlay for specific action
        overlay_bgr = self.renderer.render_action_sequence(
            transformed_image,
            action_sequence,
            current_index=action_index
        )

        # Save and signal
        overlay_path = cache_data.get("metadata", {}).get("overlay_path", "data/guidance_overlay.png")
        cv2.imwrite(overlay_path, overlay_bgr)
        self.overlay_gen.signal_overlay_ready()

        print(f"[GuidanceSystem] Overlay generated for action {action_index + 1}/{len(action_sequence)}")

        return True

    def update_robot_action_status(
        self,
        action_index: int,
        status: str,
        holding_piece: Optional[bool] = None
    ):
        """
        Update robot action status in cache.

        Args:
            action_index: Index of action
            status: New status ("pending", "in_progress", "complete", "failed")
            holding_piece: Whether robot is holding a piece (optional)
        """
        # Update action status
        self.cache.set_action_status(action_index, status, source="robot")

        # Update holding_piece if specified
        if holding_piece is not None:
            self.cache.update({
                "robot_state": {"holding_piece": holding_piece}
            }, source="robot")

        print(f"[GuidanceSystem] Action {action_index} status: {status}")

    def advance_to_next_action(self):
        """Advance to next action in sequence."""
        self.cache.advance_action(source="robot")

        current_action = self.cache.get_current_action()
        if current_action:
            print(f"[GuidanceSystem] Advanced to next action: {current_action['type']} {current_action.get('square', 'graveyard')}")
        else:
            print(f"[GuidanceSystem] All actions complete")

    def end_robot_turn(self):
        """Mark robot's turn as complete."""
        self.cache.update({
            "robot_state": {
                "is_robot_turn": False,
                "holding_piece": False
            }
        }, source="user")

        print(f"[GuidanceSystem] Robot turn ended")

    def print_cache_status(self):
        """Print current cache status (for debugging)."""
        self.cache.print_status()
