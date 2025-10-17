"""
Board State Module for Chess Vision System

This module provides functionality to capture and analyze chess board state from images.
It uses a two-stage YOLO pipeline:
1. Corner detection to identify the chessboard
2. Piece detection on the perspective-corrected board
"""

import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO
import chess
import chess.engine
from pathlib import Path
from typing import Optional, Tuple, List


class BoardState:
    """
    Manages chess board state detection and analysis.
    """

    def __init__(
        self,
        corner_model_path: str = "data/best_cornres.pt",
        piece_model_path: str = "data/best_transformed_detection.pt"
    ):
        """
        Initialize the BoardState detector.

        Args:
            corner_model_path: Path to the YOLO corner detection model
            piece_model_path: Path to the YOLO piece detection model
        """
        self.corner_model_path = Path(corner_model_path)
        self.piece_model_path = Path(piece_model_path)

        # Initialize models (lazy loading)
        self._corner_model: Optional[YOLO] = None
        self._piece_model: Optional[YOLO] = None

        # Store last snapshot
        self._last_board_state: Optional[chess.Board] = None
        self._last_fen: Optional[str] = None

        # Piece mapping from class index to notation
        self.piece_map = {
            0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r',  # black pieces
            6: 'B', 7: 'K', 8: 'N', 9: 'P', 10: 'Q', 11: 'R'  # white pieces
        }

    @property
    def corner_model(self) -> YOLO:
        """Lazy load corner detection model."""
        if self._corner_model is None:
            if not self.corner_model_path.exists():
                raise FileNotFoundError(
                    f"Corner detection model not found at {self.corner_model_path}. "
                    "Please run download.py first."
                )
            self._corner_model = YOLO(str(self.corner_model_path))
        return self._corner_model

    @property
    def piece_model(self) -> YOLO:
        """Lazy load piece detection model."""
        if self._piece_model is None:
            if not self.piece_model_path.exists():
                raise FileNotFoundError(
                    f"Piece detection model not found at {self.piece_model_path}. "
                    "Please run download.py first."
                )
            self._piece_model = YOLO(str(self.piece_model_path))
        return self._piece_model

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Order a list of 4 coordinates:
        0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left

        Args:
            pts: Array of 4 points

        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    @staticmethod
    def calculate_iou(box_1: np.ndarray, box_2: np.ndarray) -> float:
        """
        Calculate intersection over union between two polygons.

        Args:
            box_1: First polygon vertices
            box_2: Second polygon vertices

        Returns:
            IoU score
        """
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        intersection = poly_1.intersection(poly_2).area
        union = poly_1.union(poly_2).area
        return intersection / union if union > 0 else 0

    def detect_corners(self, image_path: str) -> np.ndarray:
        """
        Detect the four corners of the chessboard.

        Args:
            image_path: Path to the image file

        Returns:
            Array of 4 corner points ordered [TL, TR, BR, BL]
        """
        results = self.corner_model.predict(
            source=image_path,
            line_thickness=1,
            conf=0.25,
            verbose=False
        )

        boxes = results[0].boxes
        # Move to CPU if on CUDA
        arr = boxes.xywh.cpu().numpy()
        points = arr[:, 0:2]

        corners = self.order_points(points)
        return corners

    def four_point_transform(self, image_path: str, pts: np.ndarray) -> Image.Image:
        """
        Apply perspective transform to get bird's-eye view of the board.

        Args:
            image_path: Path to the image file
            pts: Four corner points

        Returns:
            Transformed PIL Image
        """
        img = Image.open(image_path)
        image = np.asarray(img)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Construct destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute and apply perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return Image.fromarray(warped, "RGB")

    def detect_pieces(self, image: Image.Image) -> Tuple[np.ndarray, object]:
        """
        Detect chess pieces on the transformed board.

        Args:
            image: Transformed board image

        Returns:
            Tuple of (detections array, boxes object)
        """
        # Save image temporarily for YOLO
        temp_path = "data/temp_board.jpg"
        image.save(temp_path)

        results = self.piece_model.predict(
            source=temp_path,
            line_thickness=1,
            conf=0.5,
            augment=False,
            verbose=False
        )

        boxes = results[0].boxes
        detections = boxes.xyxy.cpu().numpy()

        return detections, boxes

    def calculate_grid(self, image: Image.Image) -> Tuple[List[float], List[float]]:
        """
        Calculate the 8x8 grid on the transformed board.

        Args:
            image: Transformed board image

        Returns:
            Tuple of (x_coordinates, y_coordinates) for the grid lines
        """
        width, height = image.size
        corners = np.array([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])

        corners = self.order_points(corners)
        TL, TR, BL, BR = corners[0], corners[1], corners[3], corners[2]

        # Interpolate grid points
        def interpolate(xy0, xy1):
            x0, y0 = xy0
            x1, y1 = xy1
            dx = (x1 - x0) / 8
            dy = (y1 - y0) / 8
            return [(x0 + i * dx, y0 + i * dy) for i in range(9)]

        ptsT = interpolate(TL, TR)
        ptsL = interpolate(TL, BL)

        x_coords = [pt[0] for pt in ptsT]
        y_coords = [pt[1] for pt in ptsL]

        return x_coords, y_coords

    def connect_square_to_detection(
        self,
        detections: np.ndarray,
        boxes: object,
        square: np.ndarray
    ) -> str:
        """
        Match a detected piece to a square using IoU.

        Args:
            detections: Array of detection bounding boxes
            boxes: YOLO boxes object
            square: Square polygon vertices

        Returns:
            Piece notation or 'empty'
        """
        list_of_iou = []

        for detection in detections:
            box_x1, box_y1, box_x2, box_y2 = detection[0], detection[1], detection[2], detection[3]

            # Handle tall pieces by cropping top
            if box_y2 - box_y1 > 60:
                box_complete = np.array([
                    [box_x1, box_y1 + 40],
                    [box_x2, box_y1 + 40],
                    [box_x2, box_y2],
                    [box_x1, box_y2]
                ])
            else:
                box_complete = np.array([
                    [box_x1, box_y1],
                    [box_x2, box_y1],
                    [box_x2, box_y2],
                    [box_x1, box_y2]
                ])

            list_of_iou.append(self.calculate_iou(box_complete, square))

        if not list_of_iou or max(list_of_iou) <= 0.15:
            return 'empty'

        num = list_of_iou.index(max(list_of_iou))
        piece_class = int(boxes.cls[num].item())

        return self.piece_map.get(piece_class, 'empty')

    def create_squares(self, x_coords: List[float], y_coords: List[float]) -> List[List[np.ndarray]]:
        """
        Create all 64 square polygons for the board.

        Args:
            x_coords: X coordinates of vertical grid lines
            y_coords: Y coordinates of horizontal grid lines

        Returns:
            8x8 array of square polygons
        """
        squares = []

        for row in range(8):
            row_squares = []
            for col in range(8):
                square = np.array([
                    [x_coords[col], y_coords[row]],
                    [x_coords[col + 1], y_coords[row]],
                    [x_coords[col + 1], y_coords[row + 1]],
                    [x_coords[col], y_coords[row + 1]]
                ])
                row_squares.append(square)
            squares.append(row_squares)

        return squares

    def board_to_fen(self, board_array: List[List[str]]) -> str:
        """
        Convert board array to FEN notation.

        Args:
            board_array: 8x8 array of piece notations

        Returns:
            FEN string (position only, no move info)
        """
        fen_rows = []

        for row in board_array:
            fen_row = ""
            empty_count = 0

            for square in row:
                if square == 'empty':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += square

            if empty_count > 0:
                fen_row += str(empty_count)

            fen_rows.append(fen_row)

        return '/'.join(fen_rows)

    def snapshot(self, image_path: str, output_format: str = 'fen') -> any:
        """
        Take a snapshot of the board and process it.

        Args:
            image_path: Path to the board image
            output_format: 'fen' or 'board' (python-chess Board object)

        Returns:
            FEN string or chess.Board object
        """
        # Step 1: Detect corners
        corners = self.detect_corners(image_path)

        # Step 2: Apply perspective transform
        transformed_image = self.four_point_transform(image_path, corners)

        # Step 3: Detect pieces
        detections, boxes = self.detect_pieces(transformed_image)

        # Step 4: Calculate grid
        x_coords, y_coords = self.calculate_grid(transformed_image)

        # Step 5: Create squares
        squares = self.create_squares(x_coords, y_coords)

        # Step 6: Match pieces to squares
        board_array = []
        for row in squares:
            row_pieces = []
            for square in row:
                piece = self.connect_square_to_detection(detections, boxes, square)
                row_pieces.append(piece)
            board_array.append(row_pieces)

        # Step 7: Convert to FEN
        fen_position = self.board_to_fen(board_array)

        # Add default FEN components (white to move, no castling, no en passant, etc.)
        full_fen = f"{fen_position} w KQkq - 0 1"

        self._last_fen = full_fen

        # Create chess.Board object
        try:
            self._last_board_state = chess.Board(full_fen)
        except ValueError:
            # If FEN is invalid, create empty board
            self._last_board_state = chess.Board(None)

        if output_format == 'fen':
            return full_fen
        elif output_format == 'board':
            return self._last_board_state
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Use 'fen' or 'board'")

    def current(self, output_format: str = 'board') -> any:
        """
        Get the last snapshot result.

        Args:
            output_format: 'fen' or 'board'

        Returns:
            FEN string or chess.Board object from last snapshot
        """
        if self._last_board_state is None:
            raise RuntimeError("No snapshot taken yet. Call snapshot() first.")

        if output_format == 'fen':
            return self._last_fen
        elif output_format == 'board':
            return self._last_board_state
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Use 'fen' or 'board'")

    def bestmove(
        self,
        engine_path: str = "stockfish",
        time_limit: float = 1.0
    ) -> Optional[chess.Move]:
        """
        Calculate the best move using a UCI engine.

        Args:
            engine_path: Path to UCI engine executable (e.g., 'stockfish')
            time_limit: Time limit in seconds for analysis

        Returns:
            Best move as chess.Move object, or None if no legal moves
        """
        if self._last_board_state is None:
            raise RuntimeError("No snapshot taken yet. Call snapshot() first.")

        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            result = engine.play(
                self._last_board_state,
                chess.engine.Limit(time=time_limit)
            )
            return result.move
