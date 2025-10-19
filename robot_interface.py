"""
Robot Interface Module for Chess Arm Control

This module provides a complete interface for a robotic chess arm to execute
moves with visual guidance and network-based communication.

Components:
- RobotCommunicator: TCP/UDP network communication with robot
- MoveInterpreter: Parse and analyze chess moves
- CoordinateMapper: Convert chess squares to pixel/physical coordinates
- HighlightRenderer: Draw visual overlays on board images
- MoveExecutor: State machine for move execution

Usage:
    from robot_interface import MoveExecutor
    from board_state import BoardState

    detector = BoardState()
    executor = MoveExecutor(detector, camera_feed)
    success = executor.execute_move(move, board)
"""

import socket
import json
import threading
import time
import cv2
import numpy as np
from PIL import Image
import chess
from typing import Optional, Tuple, List, Dict, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# Data Structures
# ============================================================================

class MoveType(Enum):
    """Types of chess moves."""
    NORMAL = "normal"
    CAPTURE = "capture"
    CASTLING = "castling"
    EN_PASSANT = "en_passant"
    PROMOTION = "promotion"


class ExecutionState(Enum):
    """States for move execution state machine."""
    IDLE = "idle"
    HIGHLIGHT_CAPTURE_PICKUP = "highlight_capture_pickup"
    WAIT_PICKUP_CAPTURE = "wait_pickup_capture"
    HIGHLIGHT_CAPTURE_PLACE = "highlight_capture_place"
    WAIT_PLACE_CAPTURE = "wait_place_capture"
    HIGHLIGHT_PIECE_PICKUP = "highlight_piece_pickup"
    WAIT_PICKUP_PIECE = "wait_pickup_piece"
    HIGHLIGHT_PIECE_PLACE = "highlight_piece_place"
    WAIT_PLACE_PIECE = "wait_place_piece"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class MoveStep:
    """Represents a single step in move execution."""
    action: str  # "pickup" or "place"
    square: Optional[str]  # Chess square (e.g., "e4") or None for graveyard
    piece: str  # Piece being moved
    is_capture: bool
    is_graveyard: bool


@dataclass
class RobotStatus:
    """Robot status information."""
    holding_piece: bool
    ready: bool
    error: Optional[str]
    timestamp: float


# ============================================================================
# Robot Communicator (Network Interface)
# ============================================================================

class RobotCommunicator:
    """
    Handles TCP/UDP network communication with robotic arm.

    Receives status updates from robot and sends action commands.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5555,
                 protocol: str = "tcp", timeout: float = 30.0):
        """
        Initialize robot communicator.

        Args:
            host: Host address to bind server
            port: Port number for communication
            protocol: 'tcp' or 'udp'
            timeout: Timeout in seconds for robot responses
        """
        self.host = host
        self.port = port
        self.protocol = protocol.lower()
        self.timeout = timeout

        # Socket
        self.socket = None
        self.client_socket = None
        self.client_address = None

        # Status
        self.robot_status = RobotStatus(
            holding_piece=False,
            ready=False,
            error=None,
            timestamp=time.time()
        )

        # Thread control
        self.running = False
        self.listen_thread = None
        self.status_lock = threading.Lock()

        # Callbacks
        self.on_status_update: Optional[Callable[[RobotStatus], None]] = None

    def start(self):
        """Start the communication server."""
        if self.running:
            return

        # Create socket
        if self.protocol == "tcp":
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        else:  # UDP
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind and listen
        self.socket.bind((self.host, self.port))
        if self.protocol == "tcp":
            self.socket.listen(1)

        self.socket.settimeout(1.0)  # Non-blocking with timeout

        # Start listening thread
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()

        print(f"[RobotComm] Server started on {self.host}:{self.port} ({self.protocol.upper()})")

    def stop(self):
        """Stop the communication server."""
        self.running = False

        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        print("[RobotComm] Server stopped")

    def _listen_loop(self):
        """Background thread to listen for robot messages."""
        while self.running:
            try:
                if self.protocol == "tcp":
                    self._listen_tcp()
                else:
                    self._listen_udp()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:  # Only log if not shutting down
                    print(f"[RobotComm] Error in listen loop: {e}")
                time.sleep(0.1)

    def _listen_tcp(self):
        """Listen for TCP connections and messages."""
        if not self.client_socket:
            # Accept new connection
            try:
                client, address = self.socket.accept()
                self.client_socket = client
                self.client_address = address
                self.client_socket.settimeout(1.0)
                print(f"[RobotComm] Robot connected from {address}")
            except socket.timeout:
                return

        # Receive data from client
        try:
            data = self.client_socket.recv(4096)
            if not data:
                # Client disconnected
                print("[RobotComm] Robot disconnected")
                self.client_socket.close()
                self.client_socket = None
                return

            self._process_message(data.decode('utf-8'))
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[RobotComm] Error receiving TCP data: {e}")
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def _listen_udp(self):
        """Listen for UDP messages."""
        try:
            data, address = self.socket.recvfrom(4096)
            if address != self.client_address:
                self.client_address = address
                print(f"[RobotComm] Robot identified at {address}")

            self._process_message(data.decode('utf-8'))
        except socket.timeout:
            pass

    def _process_message(self, message: str):
        """
        Process incoming message from robot.

        Expected format:
        {
            "type": "status",
            "holding_piece": bool,
            "ready": bool,
            "error": str | null
        }
        """
        try:
            data = json.loads(message)

            if data.get("type") == "status":
                with self.status_lock:
                    self.robot_status = RobotStatus(
                        holding_piece=data.get("holding_piece", False),
                        ready=data.get("ready", False),
                        error=data.get("error"),
                        timestamp=time.time()
                    )

                # Trigger callback
                if self.on_status_update:
                    self.on_status_update(self.robot_status)

        except json.JSONDecodeError as e:
            print(f"[RobotComm] Invalid JSON received: {e}")
        except Exception as e:
            print(f"[RobotComm] Error processing message: {e}")

    def send_action(self, robot_action: bool, action: str, target_square: Optional[str],
                   coordinates: Tuple[int, int], is_capture: bool, piece: str):
        """
        Send action command to robot.

        Args:
            robot_action: Whether robot should take action
            action: "pickup" or "place"
            target_square: Chess square notation (e.g., "e4") or None for graveyard
            coordinates: (x, y) pixel coordinates
            is_capture: Whether this is a capture move
            piece: Piece being moved (e.g., "Q")
        """
        if not self.client_socket and not self.client_address:
            print("[RobotComm] No robot connected, cannot send action")
            return

        command = {
            "type": "action",
            "robot_action": robot_action,
            "action": action,
            "target_square": target_square,
            "coordinates": {"x": coordinates[0], "y": coordinates[1]},
            "is_capture": is_capture,
            "piece": piece
        }

        message = json.dumps(command) + "\n"

        try:
            if self.protocol == "tcp" and self.client_socket:
                self.client_socket.sendall(message.encode('utf-8'))
            elif self.protocol == "udp" and self.client_address:
                self.socket.sendto(message.encode('utf-8'), self.client_address)

            print(f"[RobotComm] Sent: {action} {target_square or 'graveyard'} @ {coordinates}")

        except Exception as e:
            print(f"[RobotComm] Error sending action: {e}")

    def get_status(self) -> RobotStatus:
        """Get current robot status (thread-safe)."""
        with self.status_lock:
            return self.robot_status

    def wait_for_signal(self, signal_name: str, expected_value: bool,
                       timeout: Optional[float] = None) -> bool:
        """
        Wait for a specific signal to reach expected value.

        Args:
            signal_name: "holding_piece" or "ready"
            expected_value: Expected boolean value
            timeout: Timeout in seconds (None = use default)

        Returns:
            True if signal reached expected value, False if timeout
        """
        timeout = timeout or self.timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_status()

            if signal_name == "holding_piece":
                if status.holding_piece == expected_value:
                    return True
            elif signal_name == "ready":
                if status.ready == expected_value:
                    return True

            # Check for errors
            if status.error:
                print(f"[RobotComm] Robot error: {status.error}")
                return False

            time.sleep(0.05)  # Poll every 50ms

        print(f"[RobotComm] Timeout waiting for {signal_name}={expected_value}")
        return False


# ============================================================================
# Move Interpreter (Chess Logic)
# ============================================================================

class MoveInterpreter:
    """
    Interprets chess moves and extracts execution details.
    """

    @staticmethod
    def get_move_type(move: chess.Move, board: chess.Board) -> MoveType:
        """Determine the type of chess move."""
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
        """Check if move is a capture."""
        return board.is_capture(move)

    @staticmethod
    def decompose_castling(move: chess.Move, board: chess.Board) -> Tuple[chess.Move, chess.Move]:
        """
        Decompose castling into two separate moves: king, then rook.

        Returns:
            (king_move, rook_move)
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
        """Get piece symbol at square (e.g., 'Q', 'p')."""
        piece = board.piece_at(square)
        return piece.symbol() if piece else '.'

    @staticmethod
    def square_to_name(square: chess.Square) -> str:
        """Convert square index to name (e.g., 36 -> 'e5')."""
        return chess.square_name(square)


# ============================================================================
# Coordinate Mapper (Positioning)
# ============================================================================

class CoordinateMapper:
    """
    Maps chess squares to pixel coordinates on board image.
    """

    def __init__(self, board_image: Optional[Image.Image] = None,
                 board_state: Optional[object] = None):
        """
        Initialize coordinate mapper.

        Args:
            board_image: Transformed board image (PIL Image)
            board_state: BoardState instance (for grid calculation)
        """
        self.board_image = board_image
        self.board_state = board_state
        self.grid_cache = None

    def set_board_image(self, image: Image.Image):
        """Update board image."""
        self.board_image = image
        self.grid_cache = None  # Invalidate cache

    def square_to_pixels(self, square: str) -> Tuple[int, int]:
        """
        Convert chess square to pixel coordinates (center of square).

        Args:
            square: Chess square notation (e.g., "e4")

        Returns:
            (x, y) pixel coordinates
        """
        if not self.board_image or not self.board_state:
            raise ValueError("Board image and board_state must be set")

        # Calculate grid if not cached
        if self.grid_cache is None:
            self.grid_cache = self.board_state.calculate_grid(self.board_image)

        x_coords, y_coords = self.grid_cache

        # Parse square
        file = ord(square[0]) - ord('a')  # 0-7
        rank = int(square[1]) - 1  # 0-7

        # Get square bounds
        x1 = x_coords[file]
        x2 = x_coords[file + 1]
        y1 = y_coords[7 - rank]  # Y coordinates go top to bottom
        y2 = y_coords[7 - rank + 1]

        # Return center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    def get_graveyard_coords(self) -> Tuple[int, int]:
        """
        Get coordinates for graveyard area (right side of board).

        Returns:
            (x, y) pixel coordinates for captured piece placement
        """
        if not self.board_image:
            raise ValueError("Board image must be set")

        width, height = self.board_image.size

        # Place graveyard at right edge, centered vertically
        x = width - 50  # 50 pixels from right edge
        y = height // 2

        return x, y

    def get_square_bounds(self, square: str) -> Tuple[int, int, int, int]:
        """
        Get bounding box for a chess square.

        Returns:
            (x1, y1, x2, y2) pixel coordinates
        """
        if not self.grid_cache:
            self.grid_cache = self.board_state.calculate_grid(self.board_image)

        x_coords, y_coords = self.grid_cache

        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1

        x1 = int(x_coords[file])
        x2 = int(x_coords[file + 1])
        y1 = int(y_coords[7 - rank])
        y2 = int(y_coords[7 - rank + 1])

        return x1, y1, x2, y2


# ============================================================================
# Highlight Renderer (Visual Overlay)
# ============================================================================

class HighlightRenderer:
    """
    Renders visual highlights on chess board images.
    """

    # Color schemes (BGR format for OpenCV)
    COLORS = {
        "capture_pickup": (0, 0, 255),    # Red
        "capture_place": (0, 165, 255),   # Orange
        "piece_pickup": (0, 255, 0),      # Green
        "piece_place": (255, 0, 0),       # Blue
    }

    def __init__(self, alpha: float = 0.5):
        """
        Initialize renderer.

        Args:
            alpha: Transparency of highlights (0.0-1.0)
        """
        self.alpha = alpha
        self.current_highlights = []

    def highlight_square(self, image: np.ndarray, bounds: Tuple[int, int, int, int],
                        color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw highlight on a square.

        Args:
            image: Image array (BGR)
            bounds: (x1, y1, x2, y2) square bounds
            color: (B, G, R) color tuple

        Returns:
            Image with highlight
        """
        overlay = image.copy()
        x1, y1, x2, y2 = bounds

        # Draw filled rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Draw border
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

        # Blend with original
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return result

    def highlight_graveyard(self, image: np.ndarray,
                           coords: Tuple[int, int],
                           color: Tuple[int, int, int]) -> np.ndarray:
        """
        Draw highlight on graveyard area.

        Args:
            image: Image array (BGR)
            coords: (x, y) center coordinates
            color: (B, G, R) color tuple

        Returns:
            Image with highlight
        """
        overlay = image.copy()
        x, y = coords

        # Draw circle
        radius = 30
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.circle(overlay, (x, y), radius, color, 3)

        # Blend
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return result

    def render_highlights(self, image: np.ndarray,
                         highlights: List[Dict]) -> np.ndarray:
        """
        Render multiple highlights on image.

        Args:
            image: Base image (BGR)
            highlights: List of highlight specs:
                {
                    "type": "square" | "graveyard",
                    "bounds": (x1, y1, x2, y2) if type="square",
                    "coords": (x, y) if type="graveyard",
                    "color": (B, G, R)
                }

        Returns:
            Image with all highlights
        """
        result = image.copy()

        for h in highlights:
            if h["type"] == "square":
                result = self.highlight_square(result, h["bounds"], h["color"])
            elif h["type"] == "graveyard":
                result = self.highlight_graveyard(result, h["coords"], h["color"])

        return result


# ============================================================================
# Move Executor (State Machine)
# ============================================================================

class MoveExecutor:
    """
    State machine for executing chess moves with robotic arm.
    """

    def __init__(self, board_state: object, robot_comm: Optional[RobotCommunicator] = None):
        """
        Initialize move executor.

        Args:
            board_state: BoardState instance
            robot_comm: RobotCommunicator instance (optional, will create if None)
        """
        self.board_state = board_state
        self.robot_comm = robot_comm or RobotCommunicator()

        self.coord_mapper = CoordinateMapper(board_state=board_state)
        self.renderer = HighlightRenderer(alpha=0.5)

        self.state = ExecutionState.IDLE
        self.current_move = None
        self.current_board = None
        self.camera_feed = None

    def execute_move(self, move: chess.Move, board: chess.Board,
                    camera_feed: Optional[object] = None) -> bool:
        """
        Execute a chess move with the robotic arm.

        Args:
            move: Chess move to execute
            board: Current board state
            camera_feed: OpenCV VideoCapture for live overlay (optional)

        Returns:
            True if move executed successfully, False otherwise
        """
        self.current_move = move
        self.current_board = board
        self.camera_feed = camera_feed

        # Start robot communication if not already started
        if not self.robot_comm.running:
            self.robot_comm.start()

        # Determine move type
        move_type = MoveInterpreter.get_move_type(move, board)

        print(f"\n[MoveExecutor] Executing {move_type.value} move: {move.uci()}")

        # Execute based on type
        if move_type == MoveType.CASTLING:
            return self._execute_castling(move, board)
        elif move_type == MoveType.EN_PASSANT:
            return self._execute_en_passant(move, board)
        elif move_type == MoveType.CAPTURE:
            return self._execute_capture(move, board)
        else:  # NORMAL or PROMOTION
            return self._execute_normal(move, board)

    def _execute_normal(self, move: chess.Move, board: chess.Board) -> bool:
        """Execute a normal move (non-capture, non-castling)."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        piece = MoveInterpreter.get_piece_at(move.from_square, board)

        print(f"[MoveExecutor] Normal move: {piece} from {from_square} to {to_square}")

        # Pickup sequence
        if not self._pickup_piece(from_square, piece, is_capture=False):
            return False

        # Placement sequence
        if not self._place_piece(to_square, piece, is_capture=False):
            return False

        self.state = ExecutionState.COMPLETE
        print("[MoveExecutor] Move complete!")
        return True

    def _execute_capture(self, move: chess.Move, board: chess.Board) -> bool:
        """Execute a capture move."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        our_piece = MoveInterpreter.get_piece_at(move.from_square, board)
        captured_piece = MoveInterpreter.get_piece_at(move.to_square, board)

        print(f"[MoveExecutor] Capture: {our_piece} takes {captured_piece} on {to_square}")

        # 1. Pick up opponent's piece
        if not self._pickup_piece(to_square, captured_piece, is_capture=True):
            return False

        # 2. Place opponent's piece in graveyard
        if not self._place_in_graveyard(captured_piece):
            return False

        # 3. Pick up our piece
        if not self._pickup_piece(from_square, our_piece, is_capture=False):
            return False

        # 4. Place our piece on destination
        if not self._place_piece(to_square, our_piece, is_capture=False):
            return False

        self.state = ExecutionState.COMPLETE
        print("[MoveExecutor] Capture complete!")
        return True

    def _execute_castling(self, move: chess.Move, board: chess.Board) -> bool:
        """Execute castling (king first, then rook)."""
        king_move, rook_move = MoveInterpreter.decompose_castling(move, board)

        print(f"[MoveExecutor] Castling: King {king_move.uci()}, Rook {rook_move.uci()}")

        # Move king first
        if not self._execute_normal(king_move, board):
            return False

        # Move rook
        if not self._execute_normal(rook_move, board):
            return False

        self.state = ExecutionState.COMPLETE
        print("[MoveExecutor] Castling complete!")
        return True

    def _execute_en_passant(self, move: chess.Move, board: chess.Board) -> bool:
        """Execute en passant capture."""
        from_square = MoveInterpreter.square_to_name(move.from_square)
        to_square = MoveInterpreter.square_to_name(move.to_square)
        capture_square = MoveInterpreter.get_en_passant_capture_square(move, board)
        capture_square_name = MoveInterpreter.square_to_name(capture_square)

        our_piece = MoveInterpreter.get_piece_at(move.from_square, board)
        captured_piece = MoveInterpreter.get_piece_at(capture_square, board)

        print(f"[MoveExecutor] En passant: {our_piece} captures {captured_piece} on {capture_square_name}")

        # 1. Pick up opponent's pawn (not on destination square!)
        if not self._pickup_piece(capture_square_name, captured_piece, is_capture=True):
            return False

        # 2. Place in graveyard
        if not self._place_in_graveyard(captured_piece):
            return False

        # 3. Pick up our pawn
        if not self._pickup_piece(from_square, our_piece, is_capture=False):
            return False

        # 4. Place our pawn on destination
        if not self._place_piece(to_square, our_piece, is_capture=False):
            return False

        self.state = ExecutionState.COMPLETE
        print("[MoveExecutor] En passant complete!")
        return True

    def _pickup_piece(self, square: str, piece: str, is_capture: bool) -> bool:
        """Execute pickup sequence for a piece."""
        # Get transformed board image
        # Note: In real use, this would come from camera feed
        # For now, we'll use a placeholder
        board_image = Image.new('RGB', (640, 640), (255, 255, 255))

        # Update coordinate mapper
        self.coord_mapper.set_board_image(board_image)

        # Get coordinates
        coords = self.coord_mapper.square_to_pixels(square)
        bounds = self.coord_mapper.get_square_bounds(square)

        # Determine color based on context
        color_key = "capture_pickup" if is_capture else "piece_pickup"
        color = self.renderer.COLORS[color_key]

        # Highlight
        print(f"[MoveExecutor] Highlighting {square} for pickup ({color_key})")

        # Send action to robot
        self.robot_comm.send_action(
            robot_action=True,
            action="pickup",
            target_square=square,
            coordinates=coords,
            is_capture=is_capture,
            piece=piece
        )

        # Wait for robot to pick up piece
        print(f"[MoveExecutor] Waiting for robot to pick up piece...")
        if not self.robot_comm.wait_for_signal("holding_piece", True):
            self.state = ExecutionState.ERROR
            return False

        print(f"[MoveExecutor] Robot holding piece")
        return True

    def _place_piece(self, square: str, piece: str, is_capture: bool) -> bool:
        """Execute placement sequence for a piece."""
        board_image = Image.new('RGB', (640, 640), (255, 255, 255))
        self.coord_mapper.set_board_image(board_image)

        coords = self.coord_mapper.square_to_pixels(square)

        color = self.renderer.COLORS["piece_place"]

        print(f"[MoveExecutor] Highlighting {square} for placement")

        # Send action
        self.robot_comm.send_action(
            robot_action=True,
            action="place",
            target_square=square,
            coordinates=coords,
            is_capture=is_capture,
            piece=piece
        )

        # Wait for robot to release piece
        print(f"[MoveExecutor] Waiting for robot to place piece...")
        if not self.robot_comm.wait_for_signal("holding_piece", False):
            self.state = ExecutionState.ERROR
            return False

        print(f"[MoveExecutor] Piece placed")
        return True

    def _place_in_graveyard(self, piece: str) -> bool:
        """Place captured piece in graveyard area."""
        board_image = Image.new('RGB', (640, 640), (255, 255, 255))
        self.coord_mapper.set_board_image(board_image)

        coords = self.coord_mapper.get_graveyard_coords()

        color = self.renderer.COLORS["capture_place"]

        print(f"[MoveExecutor] Highlighting graveyard for placement")

        # Send action
        self.robot_comm.send_action(
            robot_action=True,
            action="place",
            target_square=None,  # Graveyard has no chess square
            coordinates=coords,
            is_capture=True,
            piece=piece
        )

        # Wait for robot to release piece
        print(f"[MoveExecutor] Waiting for robot to place in graveyard...")
        if not self.robot_comm.wait_for_signal("holding_piece", False):
            self.state = ExecutionState.ERROR
            return False

        print(f"[MoveExecutor] Piece placed in graveyard")
        return True

    def cleanup(self):
        """Cleanup resources."""
        if self.robot_comm:
            self.robot_comm.stop()
