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
        piece_model_path: str = "data/best_transformed_detection.pt"
    ):
        """
        Initialize the BoardDetector.

        Args:
            corner_model_path: Path to the YOLO corner detection model
            piece_model_path: Path to the YOLO piece detection model
        """
        self.corner_model_path = Path(corner_model_path)
        self.piece_model_path = Path(piece_model_path)

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
                f"Try adjusting conf_threshold (current: {conf_threshold}) or min_distance "
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
        # Convert to RGB to ensure consistent 3-channel format (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
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

        # Ensure minimum dimensions for quality output, especially for small input images
        MIN_SIZE = 400
        if maxWidth < MIN_SIZE or maxHeight < MIN_SIZE:
            scale = max(MIN_SIZE / maxWidth, MIN_SIZE / maxHeight)
            maxWidth = int(maxWidth * scale)
            maxHeight = int(maxHeight * scale)

        # Add top margin (1 grid space = 1/8 of board height) to prevent cutting off top pieces
        margin = maxHeight // 8
        self.top_margin = margin

        # Construct destination points with margin offset
        # The board corners map to positions starting at 'margin' pixels from the top
        dst = np.array([
            [0, margin],
            [maxWidth - 1, margin],
            [maxWidth - 1, maxHeight - 1 + margin],
            [0, maxHeight - 1 + margin]
        ], dtype="float32")

        # Compute and apply perspective transform with high-quality interpolation
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight + margin),  # Add margin to output height
            flags=cv2.INTER_CUBIC,  # Higher quality interpolation
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return Image.fromarray(warped, "RGB")

    def preprocess_for_detection(self, image: Image.Image) -> Image.Image:
        """
        Apply GENTLE preprocessing to enhance piece detection accuracy.

        This pipeline addresses two main issues:
        1. Missed detections - through mild contrast enhancement
        2. Pawn misclassification - through subtle color normalization

        Args:
            image: Input PIL Image (RGB)

        Returns:
            Preprocessed PIL Image (RGB)
        """
        # Convert PIL to numpy array
        img_array = np.array(image)

        # Step 1: Apply MILD CLAHE for gentle contrast enhancement
        # Reduced clipLimit from 2.0 to 1.5 and larger tiles to be less aggressive
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
        l_channel = clahe.apply(l_channel)

        # Step 2: SUBTLE color channel normalization
        # Only normalize if there's significant variation (not always needed)
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)

        if a_std > 20:  # Only normalize if there's significant color variation
            a_channel = cv2.normalize(a_channel, None, alpha=50, beta=205,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if b_std > 20:
            b_channel = cv2.normalize(b_channel, None, alpha=50, beta=205,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Merge and convert back to RGB
        lab_enhanced = cv2.merge([l_channel, a_channel, b_channel])
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        # Step 3: VERY MILD sharpening (reduced kernel strength)
        # Blend 70% original + 30% sharpened instead of full sharpening
        kernel_sharpening = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])
        sharpened = cv2.filter2D(rgb_enhanced, -1, kernel_sharpening)
        result = cv2.addWeighted(rgb_enhanced, 0.7, sharpened, 0.3, 0)

        # Convert back to PIL Image (skip denoising - it was too aggressive)
        return Image.fromarray(result, "RGB")

    def detect_pieces(self, image: Image.Image, use_preprocessing: bool = True) -> Tuple[np.ndarray, object]:
        """
        Detect chess pieces on the transformed board.

        Args:
            image: Transformed board image
            use_preprocessing: Whether to apply preprocessing pipeline (default: True)

        Returns:
            Tuple of (detections array, boxes object)
        """
        # Apply preprocessing to improve detection accuracy
        if use_preprocessing:
            image = self.preprocess_for_detection(image)

        # Save image temporarily for YOLO
        temp_path = "data/temp_board.jpg"
        image.save(temp_path, quality=95)  # Higher quality to preserve enhanced details

        results = self.piece_model.predict(
            source=temp_path,
            conf=0.35,  # Lowered from 0.5 to catch more pieces (reduce missed detections)
            iou=0.5,    # NMS IoU threshold for removing duplicate detections
            augment=True,  # Enable test-time augmentation for better detection
            verbose=False,
            imgsz=640,  # Ensure consistent input size
            device='cpu'  # Explicitly set device (will use CUDA if available via PyTorch)
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
        # Pawns (class 3 and 9) require higher confidence to reduce false positives
        class_thresholds = {
            0: 0.35,  # black bishop
            1: 0.35,  # black king
            2: 0.35,  # black knight
            3: 0.45,  # black pawn - HIGHER threshold to reduce misclassification
            4: 0.35,  # black queen
            5: 0.35,  # black rook
            6: 0.35,  # white bishop
            7: 0.35,  # white king
            8: 0.35,  # white knight
            9: 0.45,  # white pawn - HIGHER threshold to reduce misclassification
            10: 0.35, # white queen
            11: 0.35, # white rook
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
                          min_corner_distance: float = 50.0, debug: bool = False) -> Tuple[str, Image.Image]:
        """
        Detect complete board state from an image.

        Args:
            image_path: Path to the board image
            corner_conf: Confidence threshold for corner detection
            min_corner_distance: Minimum distance between corners in pixels
            debug: Enable debug output

        Returns:
            Tuple of (FEN string, transformed board image)
        """
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

        # Step 2: Apply perspective transform
        if debug:
            print("[BoardDetector] Step 2: Applying perspective transform...")

        transformed_image = self.four_point_transform(image_path, corners)

        if debug:
            transformed_image.save("data/chessboard_transformed.png")
            print(f"[BoardDetector]   Transformed image size: {transformed_image.size}")
            print(f"  → Transformed board saved to data/chessboard_transformed.png")

        # Step 3: Detect pieces
        if debug:
            print("[BoardDetector] Step 3: Detecting pieces...")
            print("[BoardDetector]   Preprocessing: ENABLED")

        detections, boxes = self.detect_pieces(transformed_image, use_preprocessing=True)

        if debug:
            print(f"[BoardDetector]   Found {len(detections)} detections")
            if len(detections) > 0:
                confidences = boxes.conf.cpu().numpy()
                print(f"[BoardDetector]   Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
            self.visualize_detections(transformed_image, detections, boxes)

        # Step 4: Calculate grid
        if debug:
            print("[BoardDetector] Step 4: Calculating grid...")

        x_coords, y_coords = self.calculate_grid(transformed_image)

        if debug:
            print(f"[BoardDetector]   Grid: {len(x_coords)-1}x{len(y_coords)-1} squares")
            self.visualize_grid(transformed_image, x_coords, y_coords)

        # Step 5: Create squares
        if debug:
            print("[BoardDetector] Step 5: Creating square polygons...")

        squares = self.create_squares(x_coords, y_coords)

        # Step 6: Match pieces to squares
        if debug:
            print("[BoardDetector] Step 6: Matching pieces to squares...")

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
                        print(f"[BoardDetector]     {square_name}: {piece}")
            board_array.append(row_pieces)

        if debug:
            print(f"[BoardDetector]   Matched {matched_pieces} pieces to squares")

        # Step 7: Convert to FEN
        if debug:
            print("[BoardDetector] Step 7: Converting to FEN...")

        fen_position = self.board_to_fen(board_array)
        full_fen = f"{fen_position} w KQkq - 0 1"

        if debug:
            print(f"[BoardDetector]   FEN: {full_fen}")
            print("[BoardDetector] Detection complete!\n")

        return full_fen, transformed_image
