"""
Board Detection Module

Handles chess board detection using YOLO-based corner and piece detection.
This module provides the core computer vision functionality for the guidance system.
"""

import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO
import chess
from pathlib import Path
from typing import Optional, Tuple, List
from utils.image_preprocessing import preprocess_for_corner_detection, preprocess_for_piece_detection


class BoardDetector:
    """
    Detects and analyzes chess board state using YOLO models.

    Two-stage detection pipeline:
    1. Corner detection to identify the chessboard
    2. Piece detection on the perspective-corrected board
    """

    def __init__(
        self,
        corner_model_path: str = "data/best_corners.pt",
        piece_model_path: str = "data/best_transformed_detection.pt",
        camera_position: str = "right",
        use_corner_rgb: bool = True,
        use_piece_rgb: bool = True,
    ):
        """
        Initialize the BoardDetector.

        Args:
            corner_model_path: Path to the YOLO corner detection model
            piece_model_path: Path to the YOLO piece detection model
            camera_position: Camera position relative to board ('top', 'right', 'bottom', 'left')
            use_corner_rgb: If True (default), use RGB for corner detection.
                           If False, apply grayscale+CLAHE preprocessing.
            use_piece_rgb: If True (default), use RGB for piece detection.
                          If False, apply grayscale+CLAHE preprocessing.
        """
        self.corner_model_path = Path(corner_model_path)
        self.piece_model_path = Path(piece_model_path)
        self.camera_position = camera_position
        self.use_corner_rgb = use_corner_rgb
        self.use_piece_rgb = use_piece_rgb

        # Initialize models (lazy loading)
        self._corner_model: Optional[YOLO] = None
        self._piece_model: Optional[YOLO] = None

        # Transform parameters
        self.top_margin = 0  # Top margin added during perspective transform

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
                                 min_distance: float = 50.0, max_corners: int = 4,
                                 image_width: int = 1280, image_height: int = 720) -> np.ndarray:
        """
        Filter corner detections using quadrant-based selection.

        Algorithm:
        1. Divide image into 4 quadrants (TL, TR, BR, BL)
        2. Select the highest-confidence corner from each quadrant
        3. This ensures we get one corner from each region of the board

        Args:
            points: Array of detected corner points
            confidences: Confidence scores for each point
            min_distance: Minimum distance between corners (pixels) - not used in quadrant method
            max_corners: Maximum number of corners to return (default: 4)
            image_width: Image width for quadrant calculation (default: 1280)
            image_height: Image height for quadrant calculation (default: 720)

        Returns:
            Filtered array of 4 corners (one per quadrant)
        """
        if len(points) == 0:
            return points

        # Calculate image center
        center_x = image_width / 2
        center_y = image_height / 2

        # Assign each detection to a quadrant
        # Quadrants: 0=TL, 1=TR, 2=BR, 3=BL
        quadrants = {0: [], 1: [], 2: [], 3: []}

        for i, point in enumerate(points):
            x, y = point
            conf = confidences[i]

            if x < center_x and y < center_y:
                quadrant = 0  # Top-Left
            elif x >= center_x and y < center_y:
                quadrant = 1  # Top-Right
            elif x >= center_x and y >= center_y:
                quadrant = 2  # Bottom-Right
            else:
                quadrant = 3  # Bottom-Left

            quadrants[quadrant].append((point, conf, i))

        # Select highest-confidence corner from each quadrant
        selected_corners = []

        for quadrant_id in range(4):
            detections = quadrants[quadrant_id]

            if len(detections) == 0:
                # No corner in this quadrant - will cause error later
                continue

            # Sort by confidence (descending) and take the best
            detections.sort(key=lambda x: x[1], reverse=True)
            best_point, best_conf, orig_idx = detections[0]

            selected_corners.append(best_point)

        return np.array(selected_corners)

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
        # Preprocessing depends on use_corner_rgb setting
        if self.use_corner_rgb:
            # RGB mode: use original image directly
            temp_preprocessed_path = image_path
            if debug:
                print(f"[DEBUG]   Preprocessing: RGB mode (no preprocessing)")
        else:
            # Default: grayscale + CLAHE preprocessing
            preprocessed = preprocess_for_corner_detection(image_path)

            # Save preprocessed image to temp file for YOLO
            temp_preprocessed_path = "data/temp_corner_preprocessed.png"
            cv2.imwrite(temp_preprocessed_path, preprocessed)

            if debug:
                print(f"[DEBUG]   Preprocessing: grayscale + CLAHE contrast normalization")
                print(f"[DEBUG]   Preprocessed image saved to {temp_preprocessed_path}")

        results = self.corner_model.predict(
            source=temp_preprocessed_path,
            conf=conf_threshold,
            verbose=False,
            imgsz=1280  # Match training resolution
        )

        boxes = results[0].boxes
        # Move to CPU if on CUDA
        arr = boxes.xywh.cpu().numpy()
        points = arr[:, 0:2]
        confidences = boxes.conf.cpu().numpy()

        # Get image dimensions for quadrant calculation
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size

        if debug:
            print(f"[DEBUG]   Confidence threshold: {conf_threshold}")
            print(f"[DEBUG]   Image dimensions: {img_width}x{img_height}")
            print(f"[DEBUG]   Raw detections: {len(points)}")
            for i, (pt, conf) in enumerate(zip(points, confidences)):
                print(f"[DEBUG]     Detection {i}: {pt} (conf: {conf:.3f})")

            # Save visualization of raw detections
            self.visualize_raw_detections(image_path, points, confidences)

        # Filter corners using quadrant-based selection
        unique_points = self.filter_duplicate_corners(
            points, confidences, min_distance=min_distance, max_corners=4,
            image_width=img_width, image_height=img_height
        )

        if debug:
            print(f"[DEBUG]   Selected corners using quadrant method: {len(unique_points)}")
            if len(unique_points) > 0:
                quadrant_names = ['TL quadrant', 'TR quadrant', 'BR quadrant', 'BL quadrant']
                # Show which detections were selected and from which quadrant
                center_x = img_width / 2
                center_y = img_height / 2

                for i, corner in enumerate(unique_points):
                    # Find the original detection index
                    for orig_idx, (pt, conf) in enumerate(zip(points, confidences)):
                        if np.allclose(pt, corner, atol=0.01):
                            # Determine quadrant
                            x, y = pt
                            if x < center_x and y < center_y:
                                quad_name = quadrant_names[0]
                            elif x >= center_x and y < center_y:
                                quad_name = quadrant_names[1]
                            elif x >= center_x and y >= center_y:
                                quad_name = quadrant_names[2]
                            else:
                                quad_name = quadrant_names[3]

                            print(f"[DEBUG]     Selected corner {i}: Detection {orig_idx} from {quad_name} (conf: {conf:.3f})")
                            break

        # Validate we have exactly 4 corners
        if len(unique_points) != 4:
            error_msg = (
                f"Expected 4 corners but found {len(unique_points)} unique corners. "
                f"Raw detections: {len(points)}. "
                f"Try adjusting conf_threshold (current: {conf_threshold}) or min_distance "
                f"(current: {min_distance}) or check that the entire chessboard is visible."
            )
            raise ValueError(error_msg)

        corners = self.order_points(unique_points)
        return corners

    def get_perspective_matrix(self, image: Image.Image, pts: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
        """
        Calculate perspective transform matrix and output dimensions.

        Args:
            image: Input PIL Image
            pts: Four corner points

        Returns:
            Tuple of (transform_matrix, output_width, output_height, top_margin)
        """
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

        # Ensure minimum dimensions for quality output
        MIN_SIZE = 400
        if maxWidth < MIN_SIZE or maxHeight < MIN_SIZE:
            scale = max(MIN_SIZE / maxWidth, MIN_SIZE / maxHeight)
            maxWidth = int(maxWidth * scale)
            maxHeight = int(maxHeight * scale)

        # Add top margin (1 grid space = 1/8 of board height)
        margin = maxHeight // 8

        # Construct destination points with margin offset
        dst = np.array([
            [0, margin],
            [maxWidth - 1, margin],
            [maxWidth - 1, maxHeight - 1 + margin],
            [0, maxHeight - 1 + margin]
        ], dtype="float32")

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        return M, maxWidth, maxHeight, margin

    def transform_bounding_boxes(self, detections: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Transform bounding box coordinates through perspective warp.

        Args:
            detections: Array of bounding boxes in format (x1, y1, x2, y2)
            transform_matrix: Perspective transformation matrix from cv2.getPerspectiveTransform

        Returns:
            Transformed bounding boxes in same format
        """
        if len(detections) == 0:
            return detections

        transformed_boxes = []

        for box in detections:
            x1, y1, x2, y2 = box[:4]

            # Define the 4 corners of the bounding box
            corners = np.array([
                [x1, y1],  # Top-left
                [x2, y1],  # Top-right
                [x2, y2],  # Bottom-right
                [x1, y2]   # Bottom-left
            ], dtype=np.float32)

            # Apply perspective transform to all 4 corners
            # cv2.perspectiveTransform requires shape (1, N, 2)
            corners_reshaped = corners.reshape(1, -1, 2)
            transformed_corners = cv2.perspectiveTransform(corners_reshaped, transform_matrix)
            transformed_corners = transformed_corners.reshape(-1, 2)

            # Find the new bounding box from transformed corners
            new_x1 = np.min(transformed_corners[:, 0])
            new_y1 = np.min(transformed_corners[:, 1])
            new_x2 = np.max(transformed_corners[:, 0])
            new_y2 = np.max(transformed_corners[:, 1])

            transformed_boxes.append([new_x1, new_y1, new_x2, new_y2])

        return np.array(transformed_boxes)

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
        # Convert to RGB to ensure consistent 3-channel format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.asarray(img)

        # Get transform matrix and dimensions
        M, maxWidth, maxHeight, margin = self.get_perspective_matrix(img, pts)
        self.top_margin = margin

        # Apply perspective transform
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight + margin),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return Image.fromarray(warped, "RGB")

    def preprocess_for_detection(self, image: Image.Image) -> Image.Image:
        """
        Apply grayscale + CLAHE preprocessing to enhance piece detection accuracy.

        This preprocessing improves piece detection by:
        1. Converting to grayscale (removes color variation, focuses on shape)
        2. Applying CLAHE for contrast normalization (enhances piece silhouettes)

        NOTE: While pieces have color (white vs black), grayscale + CLAHE helps the
        model focus on shape features which are more robust to lighting variations.

        IMPORTANT: This preprocessing MUST match the training data preprocessing
        when use_piece_rgb=False. When use_piece_rgb=True, returns original RGB image.

        Args:
            image: Input PIL Image (RGB)

        Returns:
            Preprocessed PIL Image (RGB - 3-channel grayscale for YOLO compatibility)
            or original RGB image if use_piece_rgb=True
        """
        if self.use_piece_rgb:
            # RGB mode: return original image unchanged
            return image

        # Default: Use the centralized preprocessing function for consistency
        preprocessed = preprocess_for_piece_detection(image)
        # Convert BGR (from preprocessing) to RGB for PIL
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, "RGB")

    def detect_pieces(self, image_source, use_preprocessing: bool = False) -> Tuple[np.ndarray, object]:
        """
        Detect chess pieces on an image (either original or transformed).

        Args:
            image_source: Either a PIL Image or path to image file
            use_preprocessing: Whether to apply preprocessing pipeline (default: False for original images)

        Returns:
            Tuple of (detections array, boxes object)
        """
        # Handle both PIL Image and file path inputs
        if isinstance(image_source, str):
            image = Image.open(image_source)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            temp_path = image_source
        else:
            image = image_source
            # Apply preprocessing if requested
            if use_preprocessing:
                image = self.preprocess_for_detection(image)
            # Save image temporarily for YOLO
            temp_path = "data/temp_board.jpg"
            image.save(temp_path, quality=95)

        results = self.piece_model.predict(
            source=temp_path,
            conf=0.50,  # INCREASED from 0.35 - reduce false positives and duplicates
            iou=0.7,    # NMS IoU threshold (raised from 0.5 to reduce duplicate detections)
            augment=False,  # DISABLED - was True, causes duplicates and 2-3x slower
            verbose=False,
            imgsz=1280,  # UPDATED from 640 - match training resolution (1280x720 images)
            device=0  # RTX A5500 GPU for fast inference
        )

        boxes = results[0].boxes
        detections = boxes.xyxy.cpu().numpy()

        # Apply post-processing filters
        detections, boxes = self.post_process_detections(detections, boxes)

        return detections, boxes

    def post_process_detections(self, detections: np.ndarray, boxes: object) -> Tuple[np.ndarray, object]:
        """
        Apply post-processing to filter and improve detection results.

        This addresses pawn misclassification by using higher confidence thresholds
        for pawns compared to other pieces.

        Args:
            detections: Raw detection bounding boxes (N, 4)
            boxes: YOLO boxes object with class and confidence info

        Returns:
            Filtered detections and boxes
        """
        if len(detections) == 0:
            return detections, boxes

        # Class-specific confidence thresholds
        # Pawns and bishops need higher thresholds due to misclassification
        class_thresholds = {
            0: 0.50,  # black bishop - INCREASED (was 0.35)
            1: 0.40,  # black king
            2: 0.45,  # black knight
            3: 0.60,  # black pawn - INCREASED (was 0.45) - reduces false positives
            4: 0.40,  # black queen - lower (underrepresented in training)
            5: 0.45,  # black rook
            6: 0.50,  # white bishop - INCREASED (was 0.35)
            7: 0.40,  # white king
            8: 0.45,  # white knight
            9: 0.60,  # white pawn - INCREASED (was 0.45) - reduces false positives
            10: 0.40, # white queen - lower (underrepresented in training)
            11: 0.45, # white rook
        }

        # Filter detections based on class-specific thresholds
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        keep_indices = []
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            threshold = class_thresholds.get(cls, 0.35)
            if conf >= threshold:
                keep_indices.append(i)

        if len(keep_indices) == 0:
            # Return empty results if all filtered out
            return np.array([]), type('obj', (object,), {'cls': np.array([]), 'conf': np.array([]), 'xyxy': np.array([])})()

        # Filter arrays
        filtered_detections = detections[keep_indices]

        # Create new boxes object with filtered results
        filtered_boxes = type('obj', (object,), {
            'cls': boxes.cls[keep_indices],
            'conf': boxes.conf[keep_indices],
            'xyxy': boxes.xyxy[keep_indices]
        })()

        return filtered_detections, filtered_boxes

    def apply_chess_rules(self, board_array: List[List[str]], confidence_array: List[List[float]], debug: bool = False) -> List[List[str]]:
        """
        Apply chess knowledge to correct implausible piece arrangements.

        Chess piece limits per side:
        - Max 1 king
        - Max 2 queens, rooks, bishops, knights (original pieces)
        - Max 8 pawns

        When excess pieces detected, reclassify based on:
        1. Confidence (lowest confidence pieces reclassified first)
        2. Position heuristics (distance from pawn starting rank as tiebreaker)

        Args:
            board_array: 8x8 array of piece strings (e.g., 'b-rook', 'W-pawn', 'empty')
            confidence_array: 8x8 array of detection confidences (0.0-1.0)
            debug: Print debug information

        Returns:
            Corrected board array
        """
        # Piece limits per side (max count allowed)
        # Using FEN piece types: k=king, q=queen, r=rook, b=bishop, n=knight, p=pawn
        PIECE_LIMITS = {
            'k': 1,
            'q': 2,
            'r': 2,
            'b': 2,
            'n': 2,
            'p': 8
        }

        # FEN piece names for debug output
        PIECE_NAMES = {
            'k': 'king', 'q': 'queen', 'r': 'rook',
            'b': 'bishop', 'n': 'knight', 'p': 'pawn'
        }

        # Count pieces by type and color, track positions with confidence
        # Color: 'black' for lowercase FEN, 'white' for uppercase FEN
        piece_counts = {'black': {}, 'white': {}}
        piece_positions = {'black': {}, 'white': {}}  # (row, col, confidence)

        for row_idx, row in enumerate(board_array):
            for col_idx, piece in enumerate(row):
                if piece == 'empty' or len(piece) != 1:
                    continue
                # FEN notation: lowercase = black, uppercase = white
                if piece.islower():
                    color = 'black'
                    ptype = piece  # Already lowercase
                else:
                    color = 'white'
                    ptype = piece.lower()  # Normalize to lowercase for lookup

                conf = confidence_array[row_idx][col_idx]
                piece_counts[color][ptype] = piece_counts[color].get(ptype, 0) + 1
                if ptype not in piece_positions[color]:
                    piece_positions[color][ptype] = []
                piece_positions[color][ptype].append((row_idx, col_idx, conf))

        if debug:
            print(f"[BoardDetector]   Piece counts before correction:")
            for color in ['black', 'white']:
                for ptype, count in piece_counts[color].items():
                    pname = PIECE_NAMES.get(ptype, ptype)
                    print(f"[BoardDetector]     {color.capitalize()} {pname}: {count}")

        corrections_made = []

        # Fix excess pieces for each color
        for color in ['black', 'white']:
            # Pawn starting rank: black=row 1 (rank 7), white=row 6 (rank 2)
            pawn_rank = 1 if color == 'black' else 6
            # FEN pawn character for this color
            pawn_char = 'p' if color == 'black' else 'P'

            for ptype, max_count in PIECE_LIMITS.items():
                current_count = piece_counts[color].get(ptype, 0)

                if current_count > max_count:
                    excess = current_count - max_count
                    positions = piece_positions[color].get(ptype, [])

                    # Sort by confidence (lowest first), then by distance from pawn rank
                    # This ensures we reclassify the least confident detections first
                    positions_sorted = sorted(positions,
                        key=lambda pos: (pos[2], -abs(pos[0] - pawn_rank)))

                    # Reclassify excess pieces as pawns (most common misclassification)
                    for i in range(excess):
                        if i < len(positions_sorted):
                            row, col, conf = positions_sorted[i]
                            old_piece = board_array[row][col]
                            board_array[row][col] = pawn_char
                            square_name = chr(ord('a') + col) + str(8 - row)
                            old_name = PIECE_NAMES.get(ptype, ptype)
                            corrections_made.append(f"{color} {old_name} -> pawn at {square_name} (conf={conf:.2f})")

        if debug and corrections_made:
            print(f"[BoardDetector]   Chess rules applied {len(corrections_made)} correction(s):")
            for correction in corrections_made:
                print(f"[BoardDetector]     {correction}")

        return board_array

    def calculate_grid(self, image: Image.Image) -> Tuple[List[float], List[float]]:
        """
        Calculate the 8x8 grid on the transformed board.

        Args:
            image: Transformed board image

        Returns:
            Tuple of (x_coordinates, y_coordinates) for the grid lines
        """
        width, height = image.size

        # Account for top margin - the actual board starts at 'top_margin' pixels from top
        # and extends to the bottom of the image
        corners = np.array([
            [0, self.top_margin],              # TL: top-left of actual board
            [width, self.top_margin],          # TR: top-right of actual board
            [0, height],                       # BL: bottom-left
            [width, height]                    # BR: bottom-right
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
        square: np.ndarray,
        used_detections: set = None
    ) -> tuple:
        """
        Match a detected piece to a square using IoU.

        IMPORTANT: Each detection can only be assigned to ONE square.
        This prevents duplicate piece assignments (e.g., one queen bbox
        appearing on two squares).

        Args:
            detections: Array of detection bounding boxes
            boxes: YOLO boxes object
            square: Square polygon vertices
            used_detections: Set of already-used detection indices (optional)

        Returns:
            Tuple of (piece_notation, detection_index)
            - piece_notation: 'r', 'n', 'b', 'q', 'k', 'p' (or uppercase) or 'empty'
            - detection_index: Index of matched detection, or None if empty
        """
        if used_detections is None:
            used_detections = set()

        list_of_overlap = []

        for idx, detection in enumerate(detections):
            # Skip if this detection was already assigned to another square
            if idx in used_detections:
                list_of_overlap.append(0.0)
                continue

            box_x1, box_y1, box_x2, box_y2 = detection[0], detection[1], detection[2], detection[3]

            # Use ONLY the bottom 25% of the bbox for matching
            # This ensures tall pieces (queens, kings) are placed correctly
            # based on their base position, not their tall top portion
            bbox_height = box_y2 - box_y1
            bottom_25_start = box_y1 + (bbox_height * 0.75)  # Start at 75% down

            box_bottom_quarter = np.array([
                [box_x1, bottom_25_start],
                [box_x2, bottom_25_start],
                [box_x2, box_y2],
                [box_x1, box_y2]
            ])

            # Calculate: what fraction of the piece's base is inside this square?
            # If >= 50%, the piece belongs to this square
            poly_base = Polygon(box_bottom_quarter)
            poly_square = Polygon(square)

            if not poly_base.is_valid or poly_base.area == 0:
                list_of_overlap.append(0.0)
                continue

            intersection_area = poly_base.intersection(poly_square).area
            base_coverage = intersection_area / poly_base.area
            list_of_overlap.append(base_coverage)

        # Piece belongs to square if >50% of its base is inside
        if not list_of_overlap or max(list_of_overlap) < 0.5:
            return ('empty', None)

        num = list_of_overlap.index(max(list_of_overlap))
        piece_class = int(boxes.cls[num].item())
        piece_notation = self.piece_map.get(piece_class, 'empty')

        return (piece_notation, num)

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

    def _rotate_2d_array(self, arr: List[List], camera_position: str) -> List[List]:
        """
        Rotate any 2D array based on camera position.

        Args:
            arr: 8x8 2D array
            camera_position: Camera position ('top', 'right', 'bottom', 'left')

        Returns:
            Rotated 8x8 array
        """
        if camera_position == "top":
            return arr
        elif camera_position == "right":
            transposed = [[arr[row][col] for row in range(8)] for col in range(8)]
            return [row[::-1] for row in transposed]
        elif camera_position == "bottom":
            return [row[::-1] for row in arr[::-1]]
        elif camera_position == "left":
            reversed_rows = [row[::-1] for row in arr]
            return [[reversed_rows[row][col] for row in range(8)] for col in range(8)]
        else:
            return arr

    def rotate_board_for_camera_position(self, board_array: List[List[str]]) -> List[List[str]]:
        """
        Rotate board array based on camera position to get correct orientation.

        Camera positions and required rotations:
        - 'top': No rotation (0°)
        - 'right': 90° clockwise
        - 'bottom': 180°
        - 'left': 90° counter-clockwise (270° clockwise)

        Args:
            board_array: 8x8 array of piece notations

        Returns:
            Rotated 8x8 array
        """
        return self._rotate_2d_array(board_array, self.camera_position)

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

    def visualize_placements(self, image: Image.Image, detections: np.ndarray, boxes: object,
                             x_coords: List[float], y_coords: List[float],
                             output_path: str = "data/chessboard_placements.png"):
        """
        Visualize the bottom 25% base rectangles used for piece-to-square matching.

        Shows the actual region used for coverage calculation. A piece is assigned
        to a square if >=50% of its base rectangle is inside that square.

        Args:
            image: Transformed board image
            detections: Detection bounding boxes (transformed)
            boxes: YOLO boxes object with confidence and class info
            x_coords: X coordinates of vertical grid lines
            y_coords: Y coordinates of horizontal grid lines
            output_path: Where to save the visualization
        """
        # Convert PIL to cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_vis = img_cv.copy()

        # Draw grid lines (blue)
        for x in x_coords:
            cv2.line(img_vis, (int(x), 0), (int(x), img_vis.shape[0]), (255, 0, 0), 2)
        for y in y_coords:
            cv2.line(img_vis, (0, int(y)), (img_vis.shape[1], int(y)), (255, 0, 0), 2)

        # Get class info
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
        confidences = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf

        # Draw bottom 25% base rectangle for each detection
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection[:4]

            # Calculate the bottom 25% rectangle (piece base)
            bbox_height = y2 - y1
            bottom_25_start = y1 + (bbox_height * 0.75)

            # Get piece info
            cls = int(classes[i])
            piece = self.piece_map.get(cls, '?')
            conf = confidences[i]

            # Color based on confidence
            color = self.get_confidence_color(conf)

            # Draw semi-transparent filled rectangle for base area
            overlay = img_vis.copy()
            pt1 = (int(x1), int(bottom_25_start))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(overlay, pt1, pt2, color, -1)
            cv2.addWeighted(overlay, 0.4, img_vis, 0.6, 0, img_vis)

            # Draw rectangle outline
            cv2.rectangle(img_vis, pt1, pt2, color, 2)

            # Draw piece label above the base rectangle
            label = piece
            label_x = int(x1) + 2
            label_y = int(bottom_25_start) - 5

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_vis, (label_x - 2, label_y - label_h - 2),
                         (label_x + label_w + 2, label_y + 2), (0, 0, 0), -1)

            # Draw label text
            cv2.putText(img_vis, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(output_path, img_vis)
        print(f"  → Placement visualization saved to {output_path}")

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

    def detect_board_state(self, image_path: str, corner_conf: float = 0.1,
                          min_corner_distance: float = 50.0, debug: bool = False,
                          turn: str = "w") -> Tuple[str, Image.Image]:
        """
        Detect complete board state from an image.

        NEW PIPELINE ORDER:
        1. Detect corners
        2. Detect pieces on ORIGINAL image (trained without perspective correction)
        3. Apply perspective transform to BOTH image and bounding boxes
        4. Calculate grid and match pieces

        Args:
            image_path: Path to the board image
            corner_conf: Confidence threshold for corner detection
            min_corner_distance: Minimum distance between corners in pixels
            debug: Enable debug output

        Returns:
            Tuple of (FEN string, transformed board image)
        """
        # Normalize turn to single letter (handle "white"/"black" inputs)
        if turn.lower() == "white":
            turn = "w"
        elif turn.lower() == "black":
            turn = "b"
        elif turn not in ("w", "b"):
            print(f"[WARNING] Invalid turn '{turn}', defaulting to 'w'")
            turn = "w"

        if debug:
            print("\n[BoardDetector] Starting detection...")

        # Step 1: Detect corners
        if debug:
            print("[BoardDetector] Step 1: Detecting corners...")

        corners = self.detect_corners(image_path, conf_threshold=corner_conf,
                                      min_distance=min_corner_distance, debug=debug)

        if debug:
            print(f"[BoardDetector]   Final ordered corners: {len(corners)}")
            for i, corner in enumerate(corners):
                labels = ['TL', 'TR', 'BR', 'BL']
                print(f"[BoardDetector]     {labels[i]}: {corner}")
            self.visualize_corners(image_path, corners)

        # Step 2: Detect pieces on ORIGINAL image (before transformation)
        if debug:
            print("[BoardDetector] Step 2: Detecting pieces on original image...")
            if self.use_piece_rgb:
                print("[BoardDetector]   Preprocessing: RGB mode (no preprocessing)")
            else:
                print("[BoardDetector]   Preprocessing: grayscale + CLAHE (matches training data)")

        # Load original image for detection
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        # Apply preprocessing (grayscale + CLAHE) to match training data
        detections, boxes = self.detect_pieces(original_image, use_preprocessing=True)

        # Store original detections and boxes for later use (e.g., graveyard piece highlighting)
        self.original_detections = detections
        self.original_boxes = boxes

        if debug:
            print(f"[BoardDetector]   Found {len(detections)} detections")
            if len(detections) > 0:
                confidences = boxes.conf.cpu().numpy()
                print(f"[BoardDetector]   Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            # Visualize detections on ORIGINAL image
            self.visualize_detections(original_image, detections, boxes,
                                     output_path="data/chessboard_detections.png")

        # Step 3: Apply perspective transform to BOTH image and bounding boxes
        if debug:
            print("[BoardDetector] Step 3: Applying perspective transform...")

        # Get transform matrix
        M, maxWidth, maxHeight, margin = self.get_perspective_matrix(original_image, corners)
        self.top_margin = margin
        self.perspective_matrix = M  # Store for later use (e.g., overlay generation)

        # Transform the image
        image_array = np.asarray(original_image)
        warped = cv2.warpPerspective(
            image_array, M, (maxWidth, maxHeight + margin),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        transformed_image = Image.fromarray(warped, "RGB")

        # Transform the bounding boxes
        detections = self.transform_bounding_boxes(detections, M)

        if debug:
            transformed_image.save("data/chessboard_transformed.png")
            print(f"[BoardDetector]   Transformed image size: {transformed_image.size}")
            print(f"  → Transformed board saved to data/chessboard_transformed.png")

        # Step 4: Calculate grid
        if debug:
            print("[BoardDetector] Step 4: Calculating grid...")

        x_coords, y_coords = self.calculate_grid(transformed_image)

        if debug:
            print(f"[BoardDetector]   Grid: {len(x_coords)-1}x{len(y_coords)-1} squares")
            self.visualize_grid(transformed_image, x_coords, y_coords)
            self.visualize_placements(transformed_image, detections, boxes, x_coords, y_coords)

        # Step 5: Create squares
        if debug:
            print("[BoardDetector] Step 5: Creating square polygons...")

        squares = self.create_squares(x_coords, y_coords)

        # Step 6: Match pieces to squares (using transformed detections)
        if debug:
            print("[BoardDetector] Step 6: Matching pieces to squares...")

        board_array = []
        confidence_array = []  # Track confidence for chess rules
        matched_pieces = 0
        used_detections = set()  # Track which detections have been assigned

        for row_idx, row in enumerate(squares):
            row_pieces = []
            row_confidences = []
            for col_idx, square in enumerate(row):
                piece, detection_idx = self.connect_square_to_detection(
                    detections, boxes, square, used_detections
                )
                row_pieces.append(piece)
                # Track confidence for this square
                if piece != 'empty' and detection_idx is not None:
                    conf = float(boxes.conf[detection_idx].cpu().numpy())
                    row_confidences.append(conf)
                    used_detections.add(detection_idx)  # Mark as used
                    matched_pieces += 1
                    if debug:
                        square_name = chr(ord('a') + col_idx) + str(8 - row_idx)
                        print(f"[BoardDetector]     {square_name}: {piece} (detection #{detection_idx}, conf={conf:.2f})")
                else:
                    row_confidences.append(0.0)
            board_array.append(row_pieces)
            confidence_array.append(row_confidences)

        if debug:
            print(f"[BoardDetector]   Matched {matched_pieces} pieces to squares")
            if len(detections) > matched_pieces:
                print(f"[BoardDetector]   Warning: {len(detections) - matched_pieces} detections not assigned to any square")
                # Debug: show unassigned detections
                unassigned = [i for i in range(len(detections)) if i not in used_detections]
                for det_idx in unassigned:
                    det = detections[det_idx]
                    x1, y1, x2, y2 = det[:4]
                    bbox_height = y2 - y1
                    bottom_25_start = y1 + (bbox_height * 0.75)
                    cls = int(boxes.cls[det_idx].item())
                    piece = self.piece_map.get(cls, '?')
                    print(f"[DEBUG] Unassigned #{det_idx} ({piece}): base=({x1:.0f},{bottom_25_start:.0f})-({x2:.0f},{y2:.0f})")
                    # Find best coverage with any square
                    best_coverage = 0
                    best_square = None
                    box_bottom_quarter = np.array([[x1, bottom_25_start], [x2, bottom_25_start], [x2, y2], [x1, y2]])
                    poly_base = Polygon(box_bottom_quarter)
                    if poly_base.is_valid and poly_base.area > 0:
                        for row_idx, row in enumerate(squares):
                            for col_idx, sq in enumerate(row):
                                poly_sq = Polygon(sq)
                                coverage = poly_base.intersection(poly_sq).area / poly_base.area
                                if coverage > best_coverage:
                                    best_coverage = coverage
                                    best_square = (row_idx, col_idx)
                    if best_square:
                        sq_name = chr(ord('a') + best_square[1]) + str(8 - best_square[0])
                        print(f"[DEBUG]   Best coverage: {best_coverage:.1%} in {sq_name} (need >=50%)")

        # Step 7: Rotate board based on camera position
        if debug:
            print(f"[BoardDetector] Step 7: Rotating board for camera position '{self.camera_position}'...")

        board_array = self.rotate_board_for_camera_position(board_array)
        # Apply same rotation to confidence array
        confidence_array = self._rotate_2d_array(confidence_array, self.camera_position)

        if debug:
            print("[BoardDetector]   Board rotated to correct orientation")

        # Step 8: Apply chess rules to fix implausible arrangements
        if debug:
            print("[BoardDetector] Step 8: Applying chess rules...")

        board_array = self.apply_chess_rules(board_array, confidence_array, debug=debug)

        # Step 9: Convert to FEN
        if debug:
            print("[BoardDetector] Step 9: Converting to FEN...")

        fen_position = self.board_to_fen(board_array)
        full_fen = f"{fen_position} {turn} KQkq - 0 1"

        if debug:
            turn_name = "White" if turn == "w" else "Black"
            print(f"[BoardDetector]   FEN: {full_fen}")
            print(f"[BoardDetector]   Turn: {turn_name}")
            print("[BoardDetector] Detection complete!\n")

        return full_fen, transformed_image
