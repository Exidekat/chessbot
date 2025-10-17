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

    @staticmethod
    def filter_duplicate_corners(points: np.ndarray, confidences: np.ndarray,
                                 min_distance: float = 50.0, max_corners: int = 4) -> np.ndarray:
        """
        Filter out duplicate corner detections using confidence-based greedy selection.

        Algorithm:
        1. Sort detections by confidence (highest first)
        2. Greedily select corners that aren't within min_distance of already-selected corners
        3. Stop when we have max_corners corners

        Args:
            points: Array of detected corner points
            confidences: Confidence scores for each point
            min_distance: Minimum distance between corners (pixels)
            max_corners: Maximum number of corners to return (default: 4)

        Returns:
            Filtered array of unique corners (up to max_corners)
        """
        if len(points) == 0:
            return points

        # Sort by confidence (descending)
        sorted_indices = np.argsort(confidences)[::-1]
        sorted_points = points[sorted_indices]
        sorted_confidences = confidences[sorted_indices]

        # Greedily select corners
        selected_corners = []

        for point, conf in zip(sorted_points, sorted_confidences):
            # Check if this point is far enough from all already-selected corners
            is_valid = True
            for selected_corner in selected_corners:
                distance = np.sqrt(np.sum((point - selected_corner) ** 2))
                if distance < min_distance:
                    is_valid = False
                    break

            if is_valid:
                selected_corners.append(point)

                # Stop when we have enough corners
                if len(selected_corners) >= max_corners:
                    break

        return np.array(selected_corners)

    def visualize_raw_detections(self, image_path: str, points: np.ndarray, confidences: np.ndarray,
                                 output_path: str = "data/chessboard_raw_corners.png"):
        """
        Visualize all raw corner detections from YOLO.

        Args:
            image_path: Path to the original image
            points: All detected points
            confidences: Confidence scores for each detection
            output_path: Where to save the visualization
        """
        img = cv2.imread(image_path)
        img_vis = img.copy()

        for i, (pt, conf) in enumerate(zip(points, confidences)):
            x, y = int(pt[0]), int(pt[1])

            # Color based on confidence
            if conf >= 0.5:
                color = (0, 255, 0)  # Green - high confidence
            elif conf >= 0.3:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 0, 255)  # Red - low confidence

            cv2.circle(img_vis, (x, y), 25, color, -1)

            label = f"{i}: {conf:.3f}"
            cv2.putText(img_vis, label, (x + 30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imwrite(output_path, img_vis)
        print(f"  → Raw corner detections saved to {output_path}")

    def detect_corners(self, image_path: str, conf_threshold: float = 0.1,
                      min_distance: float = 50.0, debug: bool = False) -> np.ndarray:
        """
        Detect the four corners of the chessboard.

        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for YOLO detection (0.0-1.0)
            min_distance: Minimum distance between corners in pixels
            debug: Enable debug output

        Returns:
            Array of 4 corner points ordered [TL, TR, BR, BL]

        Raises:
            ValueError: If 4 distinct corners cannot be detected
        """
        results = self.corner_model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False
        )

        boxes = results[0].boxes
        # Move to CPU if on CUDA
        arr = boxes.xywh.cpu().numpy()
        points = arr[:, 0:2]
        confidences = boxes.conf.cpu().numpy()

        if debug:
            print(f"[DEBUG]   Confidence threshold: {conf_threshold}")
            print(f"[DEBUG]   Raw detections: {len(points)}")
            for i, (pt, conf) in enumerate(zip(points, confidences)):
                print(f"[DEBUG]     Detection {i}: {pt} (conf: {conf:.3f})")

            # Save visualization of raw detections
            self.visualize_raw_detections(image_path, points, confidences)

        # Filter duplicate corners using confidence-based greedy selection
        unique_points = self.filter_duplicate_corners(
            points, confidences, min_distance=min_distance, max_corners=4
        )

        if debug:
            print(f"[DEBUG]   Selected corners after filtering (min_distance={min_distance}): {len(unique_points)}")
            if len(unique_points) > 0:
                # Show which detections were selected
                for i, corner in enumerate(unique_points):
                    # Find the original detection index
                    for orig_idx, (pt, conf) in enumerate(zip(points, confidences)):
                        if np.allclose(pt, corner, atol=0.01):
                            print(f"[DEBUG]     Selected corner {i}: Detection {orig_idx} (conf: {conf:.3f})")
                            break

        # Validate we have exactly 4 corners
        if len(unique_points) != 4:
            error_msg = (
                f"Expected 4 corners but found {len(unique_points)} unique corners. "
                f"Raw detections: {len(points)}. "
                f"Try adjusting --corner-conf (current: {conf_threshold}) or --min-corner-dist "
                f"(current: {min_distance}) or check that the entire chessboard is visible."
            )
            raise ValueError(error_msg)

        corners = self.order_points(unique_points)
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

    def visualize_corners(self, image_path: str, corners: np.ndarray, output_path: str = "data/chessboard_corners.png"):
        """
        Visualize detected corners on the original image.

        Args:
            image_path: Path to the original image
            corners: Array of 4 corner points (ordered: TL, TR, BR, BL)
            output_path: Where to save the visualization
        """
        # Load image
        img = cv2.imread(image_path)
        img_vis = img.copy()

        # Corner labels
        labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0)]  # Red, Yellow, Blue, Cyan

        # Draw circles at each corner
        for i, (x, y) in enumerate(corners):
            color = colors[i]
            cv2.circle(img_vis, (int(x), int(y)), 20, color, -1)  # Filled circles

            # Draw label with background
            label = labels[i]
            font_scale = 1.2
            thickness = 3
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Background rectangle
            cv2.rectangle(img_vis, (int(x) + 25, int(y) - label_h - 5),
                         (int(x) + 25 + label_w, int(y) + 5), (0, 0, 0), -1)

            # Text
            cv2.putText(img_vis, label, (int(x) + 25, int(y)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Draw lines connecting corners
        corners_int = corners.astype(int)
        cv2.polylines(img_vis, [corners_int], True, (0, 255, 0), 3)  # Green lines

        cv2.imwrite(output_path, img_vis)
        print(f"  → Corners visualization saved to {output_path}")

    def visualize_grid(self, image: Image.Image, x_coords: List[float], y_coords: List[float],
                       output_path: str = "data/chessboard_grid.png"):
        """
        Visualize the 8x8 grid on the transformed board.

        Args:
            image: Transformed board image
            x_coords: X coordinates of vertical grid lines
            y_coords: Y coordinates of horizontal grid lines
            output_path: Where to save the visualization
        """
        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_vis = img_cv.copy()

        # Draw vertical lines
        for x in x_coords:
            cv2.line(img_vis, (int(x), 0), (int(x), img_vis.shape[0]), (255, 0, 0), 2)

        # Draw horizontal lines
        for y in y_coords:
            cv2.line(img_vis, (0, int(y)), (img_vis.shape[1], int(y)), (255, 0, 0), 2)

        cv2.imwrite(output_path, img_vis)
        print(f"  → Grid visualization saved to {output_path}")

    def get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Get BGR color based on confidence value (0.0 to 1.0).
        Red (low) -> Yellow (medium) -> Green (high)

        Args:
            confidence: Confidence value between 0.0 and 1.0

        Returns:
            BGR color tuple
        """
        if confidence < 0.5:
            # Red to Yellow (0.0 - 0.5)
            # Red (0,0,255) -> Yellow (0,255,255)
            ratio = confidence / 0.5
            return (0, int(255 * ratio), 255)
        else:
            # Yellow to Green (0.5 - 1.0)
            # Yellow (0,255,255) -> Green (0,255,0)
            ratio = (confidence - 0.5) / 0.5
            return (0, 255, int(255 * (1 - ratio)))

    def visualize_detections(self, image: Image.Image, detections: np.ndarray, boxes: object,
                            output_path: str = "data/chessboard_detections.png"):
        """
        Visualize YOLO detections with confidence-based colors.

        Args:
            image: Transformed board image
            detections: Detection bounding boxes
            boxes: YOLO boxes object with confidence and class info
            output_path: Where to save the visualization
        """
        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_vis = img_cv.copy()

        # Get confidences
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection[:4])
            conf = confidences[i]
            cls = int(classes[i])
            piece = self.piece_map.get(cls, '?')

            # Get color based on confidence
            color = self.get_confidence_color(conf)

            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)

            # Draw label with piece and confidence
            label = f"{piece} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw label background
            cv2.rectangle(img_vis, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(img_vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(output_path, img_vis)
        print(f"  → Detection visualization saved to {output_path}")
        print(f"  → Detected {len(detections)} pieces")

    def snapshot(self, image_path: str, output_format: str = 'fen', debug: bool = False,
                corner_conf: float = 0.1, min_corner_distance: float = 50.0) -> any:
        """
        Take a snapshot of the board and process it.

        Args:
            image_path: Path to the board image
            output_format: 'fen' or 'board' (python-chess Board object)
            debug: If True, save debug visualizations at each stage
            corner_conf: Confidence threshold for corner detection (0.0-1.0, default: 0.1)
            min_corner_distance: Minimum distance between corners in pixels (default: 50.0)

        Returns:
            FEN string or chess.Board object
        """
        if debug:
            print("\n[DEBUG] Starting snapshot processing...")

        # Step 1: Detect corners
        if debug:
            print("[DEBUG] Step 1: Detecting corners...")
        corners = self.detect_corners(image_path, conf_threshold=corner_conf,
                                      min_distance=min_corner_distance, debug=debug)
        if debug:
            print(f"[DEBUG]   Final ordered corners: {len(corners)}")
            for i, corner in enumerate(corners):
                labels = ['TL', 'TR', 'BR', 'BL']
                print(f"[DEBUG]     {labels[i]}: {corner}")
            self.visualize_corners(image_path, corners)

        # Step 2: Apply perspective transform
        if debug:
            print("[DEBUG] Step 2: Applying perspective transform...")
        transformed_image = self.four_point_transform(image_path, corners)
        if debug:
            transformed_image.save("data/chessboard_transformed.png")
            print(f"[DEBUG]   Transformed image size: {transformed_image.size}")
            print(f"  → Transformed board saved to data/chessboard_transformed.png")

        # Step 3: Detect pieces
        if debug:
            print("[DEBUG] Step 3: Detecting pieces...")
        detections, boxes = self.detect_pieces(transformed_image)
        if debug:
            print(f"[DEBUG]   Found {len(detections)} detections")
            if len(detections) > 0:
                confidences = boxes.conf.cpu().numpy()
                print(f"[DEBUG]   Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            self.visualize_detections(transformed_image, detections, boxes)

        # Step 4: Calculate grid
        if debug:
            print("[DEBUG] Step 4: Calculating grid...")
        x_coords, y_coords = self.calculate_grid(transformed_image)
        if debug:
            print(f"[DEBUG]   Grid: {len(x_coords)-1}x{len(y_coords)-1} squares")
            self.visualize_grid(transformed_image, x_coords, y_coords)

        # Step 5: Create squares
        if debug:
            print("[DEBUG] Step 5: Creating square polygons...")
        squares = self.create_squares(x_coords, y_coords)

        # Step 6: Match pieces to squares
        if debug:
            print("[DEBUG] Step 6: Matching pieces to squares...")
        board_array = []
        matched_pieces = 0
        for row_idx, row in enumerate(squares):
            row_pieces = []
            for col_idx, square in enumerate(row):
                piece = self.connect_square_to_detection(detections, boxes, square)
                row_pieces.append(piece)
                if piece != 'empty':
                    matched_pieces += 1
                    if debug:
                        square_name = chr(ord('a') + col_idx) + str(8 - row_idx)
                        print(f"[DEBUG]     {square_name}: {piece}")
            board_array.append(row_pieces)

        if debug:
            print(f"[DEBUG]   Matched {matched_pieces} pieces to squares")

        # Step 7: Convert to FEN
        if debug:
            print("[DEBUG] Step 7: Converting to FEN...")
        fen_position = self.board_to_fen(board_array)

        # Add default FEN components (white to move, no castling, no en passant, etc.)
        full_fen = f"{fen_position} w KQkq - 0 1"

        if debug:
            print(f"[DEBUG]   FEN: {full_fen}")
            print("[DEBUG] Snapshot processing complete!\n")

        self._last_fen = full_fen

        # Create chess.Board object
        try:
            self._last_board_state = chess.Board(full_fen)
        except ValueError:
            # If FEN is invalid, create empty board
            if debug:
                print("[DEBUG] WARNING: Invalid FEN, creating empty board")
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
