"""
Highlight Renderer Module

Renders visual highlights on chess board images for robot guidance.
Uses color-coded overlays to indicate pickup, place, capture, and graveyard actions.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional

from .coordinate_mapper import CoordinateMapper


class HighlightRenderer:
    """
    Renders visual highlights on chess board images.

    Color scheme:
    - Green: Pickup piece
    - Blue: Place piece
    - Red: Capture (pickup opponent's piece)
    - Orange: Graveyard placement
    """

    # Color schemes (BGR format for OpenCV)
    COLORS = {
        "pickup": (0, 255, 0),       # Green
        "place": (255, 0, 0),         # Blue
        "capture": (0, 0, 255),       # Red
        "graveyard": (0, 165, 255)    # Orange
    }

    def __init__(self, alpha: float = 0.5, coord_mapper: Optional[CoordinateMapper] = None):
        """
        Initialize highlight renderer.

        Args:
            alpha: Transparency of highlights (0.0-1.0)
            coord_mapper: CoordinateMapper instance (optional)
        """
        self.alpha = alpha
        self.coord_mapper = coord_mapper or CoordinateMapper()

    def render_action_highlight(
        self,
        image: np.ndarray,
        action: Dict,
        transformed_image: Image.Image
    ) -> np.ndarray:
        """
        Render a single action highlight.

        Args:
            image: Image array (BGR) to draw on
            action: Action dict with type, square, piece, status
            transformed_image: Transformed board for coordinate mapping

        Returns:
            Image with highlight rendered
        """
        action_type = action["type"]
        square = action.get("square")

        # Determine color based on action type
        if action_type == "graveyard":
            color = self.COLORS["graveyard"]
            coords = self.coord_mapper.get_graveyard_coords(transformed_image)
            return self._highlight_circle(image, coords, color)

        elif action_type == "pickup":
            # Check if this is a capture (pickup opponent's piece)
            # Heuristic: if square is the destination, it's a capture
            # For now, use different color based on piece capitalization
            if square and action.get("piece", "").islower():
                # Capturing opponent's piece (lowercase = black, usually captured by white)
                color = self.COLORS["capture"]
            else:
                color = self.COLORS["pickup"]

            bounds = self.coord_mapper.get_square_bounds(square, transformed_image)
            return self._highlight_square(image, bounds, color)

        elif action_type == "place":
            color = self.COLORS["place"]
            bounds = self.coord_mapper.get_square_bounds(square, transformed_image)
            return self._highlight_square(image, bounds, color)

        return image

    def _highlight_square(
        self,
        image: np.ndarray,
        bounds: Tuple[int, int, int, int],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
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

        # Draw border (thicker for visibility)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)

        # Blend with original
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return result

    def _highlight_circle(
        self,
        image: np.ndarray,
        coords: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int = 30
    ) -> np.ndarray:
        """
        Draw highlight circle (for graveyard).

        Args:
            image: Image array (BGR)
            coords: (x, y) center coordinates
            color: (B, G, R) color tuple
            radius: Circle radius in pixels

        Returns:
            Image with highlight
        """
        overlay = image.copy()
        x, y = coords

        # Draw filled circle
        cv2.circle(overlay, (x, y), radius, color, -1)

        # Draw border
        cv2.circle(overlay, (x, y), radius, color, 4)

        # Blend
        result = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)

        return result

    def render_from_cache(
        self,
        transformed_image: Image.Image,
        cache: Dict
    ) -> np.ndarray:
        """
        Render highlights from state cache.

        Args:
            transformed_image: Transformed board image
            cache: State cache dictionary

        Returns:
            Overlay image with highlights (BGR numpy array)
        """
        # Convert PIL to OpenCV format
        img_array = np.array(transformed_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Get robot state
        robot_state = cache.get("robot_state", {})
        action_sequence = robot_state.get("action_sequence", [])
        action_index = robot_state.get("action_index", 0)
        is_robot_turn = robot_state.get("is_robot_turn", False)

        # Only render if it's robot's turn
        if not is_robot_turn or len(action_sequence) == 0:
            return img_bgr

        # Render current action
        if action_index < len(action_sequence):
            current_action = action_sequence[action_index]
            img_bgr = self.render_action_highlight(img_bgr, current_action, transformed_image)

        return img_bgr

    def render_action_sequence(
        self,
        transformed_image: Image.Image,
        action_sequence: List[Dict],
        current_index: int = 0
    ) -> np.ndarray:
        """
        Render the current action from a sequence.

        Args:
            transformed_image: Transformed board image
            action_sequence: List of action dictionaries
            current_index: Index of current action to highlight

        Returns:
            Overlay image with current action highlighted (BGR numpy array)
        """
        # Convert PIL to OpenCV format
        img_array = np.array(transformed_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Render current action
        if 0 <= current_index < len(action_sequence):
            action = action_sequence[current_index]
            img_bgr = self.render_action_highlight(img_bgr, action, transformed_image)

        return img_bgr

    def render_all_actions(
        self,
        transformed_image: Image.Image,
        action_sequence: List[Dict]
    ) -> np.ndarray:
        """
        Render ALL actions in sequence (for visualization/debugging).

        Args:
            transformed_image: Transformed board image
            action_sequence: List of action dictionaries

        Returns:
            Overlay image with all actions highlighted (BGR numpy array)
        """
        # Convert PIL to OpenCV format
        img_array = np.array(transformed_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Render each action
        for action in action_sequence:
            img_bgr = self.render_action_highlight(img_bgr, action, transformed_image)

        return img_bgr

    def render_best_move_preview(
        self,
        transformed_image: Image.Image,
        from_square: str,
        to_square: str,
        is_capture: bool = False
    ) -> np.ndarray:
        """
        Render simple best move preview (before decomposition).

        Shows pickup square (green) and destination square (blue).

        Args:
            transformed_image: Transformed board image
            from_square: Origin square (e.g., "e2")
            to_square: Destination square (e.g., "e4")
            is_capture: Whether this is a capture move

        Returns:
            Overlay image with move preview (BGR numpy array)
        """
        # Convert PIL to OpenCV format
        img_array = np.array(transformed_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Highlight from square (green for pickup, red if capturing)
        from_bounds = self.coord_mapper.get_square_bounds(from_square, transformed_image)
        from_color = self.COLORS["capture"] if is_capture else self.COLORS["pickup"]
        img_bgr = self._highlight_square(img_bgr, from_bounds, from_color)

        # Highlight to square (blue for place)
        to_bounds = self.coord_mapper.get_square_bounds(to_square, transformed_image)
        to_color = self.COLORS["place"]
        img_bgr = self._highlight_square(img_bgr, to_bounds, to_color)

        return img_bgr

    def add_text_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.8,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Add text label to image.

        Args:
            image: Image array (BGR)
            text: Text to display
            position: (x, y) position for text
            color: (B, G, R) text color
            font_scale: Font size scale
            thickness: Text thickness

        Returns:
            Image with text added
        """
        # Add background for readability
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        x, y = position
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )

        return image
