"""
Frame Overlay Renderer Module

Provides color-coded overlay rendering for chess move guidance.
Extracted from virtual_overlay_demo.py for reusability across VLA scripts.

Color scheme:
- RED: Pickup square (origin)
- BLUE: Place square (destination)
- ORANGE: Graveyard for discarded pieces (captures)
- PURPLE: Graveyard piece for promotion (source)
"""

from typing import Optional
import numpy as np
import cv2
from PIL import Image

from guidance.board_detector import BoardDetector
from guidance.coordinate_mapper import CoordinateMapper, rotate_square_for_camera


def apply_stage_overlay_to_frame(
    frame: np.ndarray,
    stage: dict,
    transformed_image_path: str,
    detector: BoardDetector,
    perspective_matrix: np.ndarray,
    camera_position: str
) -> Optional[np.ndarray]:
    """
    Apply color-coded overlay for a move stage to a live frame (no file I/O).

    Args:
        frame: Live 720p frame (numpy array, BGR format)
        stage: Stage dictionary from decompose_move() with fields:
            - pickup_square: Source square or None for graveyard
            - place_square: Destination square or None for graveyard
            - piece: Piece symbol (for graveyard -> board moves)
        transformed_image_path: Path to transformed board image (for coordinate mapping)
        detector: BoardDetector instance for coordinate mapping
        perspective_matrix: Perspective transform matrix (M)
        camera_position: Camera position for rotation handling

    Returns:
        Frame with overlay applied (BGR numpy array) or None if frame is invalid

    Color scheme:
        - RED (0, 0, 255): Pickup square
        - BLUE (255, 0, 0): Place square
        - ORANGE (0, 165, 255): Graveyard (discard captured piece)
        - PURPLE (255, 0, 255): Graveyard piece (promotion source)
    """
    if frame is None:
        return None

    # Load transformed image for coordinate mapping
    pil_transformed = Image.open(transformed_image_path)

    # Initialize coordinate mapper
    mapper = CoordinateMapper(board_detector=detector)

    # Calculate inverse perspective matrix
    M_inv = cv2.invert(perspective_matrix)[1]

    # Create overlay with transparency
    overlay = frame.copy()

    # Color definitions (BGR format)
    RED = (0, 0, 255)      # Pickup/remove
    BLUE = (255, 0, 0)     # Place
    ORANGE = (0, 165, 255) # Graveyard (discard)
    PURPLE = (255, 0, 255) # Graveyard piece (promotion)

    # Helper function to transform square bounds back to original image
    def transform_square_to_original(square_name: str):
        x1, y1, x2, y2 = mapper.get_square_bounds(square_name, pil_transformed)
        corners = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32)
        corners_reshaped = corners.reshape(1, -1, 2)
        original_corners = cv2.perspectiveTransform(corners_reshaped, M_inv)
        return original_corners.reshape(-1, 2).astype(np.int32)

    # Render pickup square (red)
    if stage["pickup_square"] is not None:
        rotated_square = rotate_square_for_camera(stage["pickup_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)
        cv2.fillPoly(overlay, [original_corners], RED)
        cv2.polylines(overlay, [original_corners], isClosed=True, color=RED, thickness=4)

    # Render place square (blue)
    if stage["place_square"] is not None:
        rotated_square = rotate_square_for_camera(stage["place_square"], camera_position, inverse=True)
        original_corners = transform_square_to_original(rotated_square)
        cv2.fillPoly(overlay, [original_corners], BLUE)
        cv2.polylines(overlay, [original_corners], isClosed=True, color=BLUE, thickness=4)

    # Render graveyard
    if stage["pickup_square"] is None and stage["piece"] is not None:
        # Taking piece FROM graveyard - highlight specific piece
        target_piece = stage["piece"]
        piece_to_class = {
            'b': 0, 'k': 1, 'n': 2, 'p': 3, 'q': 4, 'r': 5,
            'B': 6, 'K': 7, 'N': 8, 'P': 9, 'Q': 10, 'R': 11
        }
        target_class = piece_to_class.get(target_piece)

        left_edge_transformed = np.array([[[0, pil_transformed.size[1] // 2]]], dtype=np.float32)
        left_edge_original = cv2.perspectiveTransform(left_edge_transformed, M_inv)
        board_left_x = int(left_edge_original[0][0][0])

        if hasattr(detector, 'original_detections') and hasattr(detector, 'original_boxes'):
            detections = detector.original_detections
            boxes = detector.original_boxes
            matching_pieces = []
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection[:4]
                box_center_x = (x1 + x2) / 2
                cls = int(boxes.cls[i].item())
                if box_center_x < board_left_x and cls == target_class:
                    matching_pieces.append((i, detection, boxes.conf[i].item()))

            if matching_pieces:
                matching_pieces.sort(key=lambda x: x[2], reverse=True)
                idx, detection, conf = matching_pieces[0]
                x1, y1, x2, y2 = map(int, detection[:4])
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                cv2.fillPoly(overlay, [corners], PURPLE)
                cv2.polylines(overlay, [corners], isClosed=True, color=PURPLE, thickness=4)
            else:
                graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
                point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
                original_point = cv2.perspectiveTransform(point, M_inv)
                gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
                cv2.circle(overlay, (gx, gy), 40, PURPLE, -1)
                cv2.circle(overlay, (gx, gy), 40, PURPLE, 4)

    elif stage["place_square"] is None:
        graveyard_x, graveyard_y = mapper.get_left_graveyard_coords(pil_transformed)
        point = np.array([[[graveyard_x, graveyard_y]]], dtype=np.float32)
        original_point = cv2.perspectiveTransform(point, M_inv)
        gx, gy = int(original_point[0][0][0]), int(original_point[0][0][1])
        cv2.circle(overlay, (gx, gy), 40, ORANGE, -1)
        cv2.circle(overlay, (gx, gy), 40, ORANGE, 4)

    # Apply alpha blending (30% overlay, 70% original)
    result = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    return result
