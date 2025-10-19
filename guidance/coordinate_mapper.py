"""
Coordinate Mapper Module

Maps chess squares to pixel coordinates on transformed board images.
Uses BoardDetector's grid calculation for accurate square positioning.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional


class CoordinateMapper:
    """
    Maps chess squares to pixel coordinates on transformed board images.
    """

    def __init__(self, board_detector=None):
        """
        Initialize coordinate mapper.

        Args:
            board_detector: BoardDetector instance (optional, for grid calculation)
        """
        self.board_detector = board_detector
        self.grid_cache = {}  # Cache grids by image size

    def square_to_pixels(self, square: str, transformed_image: Image.Image) -> Tuple[int, int]:
        """
        Convert chess square to pixel coordinates (center of square).

        Args:
            square: Chess square notation (e.g., "e4")
            transformed_image: Transformed board image

        Returns:
            (x, y) pixel coordinates of square center
        """
        # Get grid coordinates
        x_coords, y_coords = self._get_grid(transformed_image)

        # Parse square notation
        file = ord(square[0]) - ord('a')  # 0-7 (a-h)
        rank = int(square[1]) - 1  # 0-7 (1-8)

        # Get square bounds
        x1 = x_coords[file]
        x2 = x_coords[file + 1]
        # Y coordinates go top to bottom, rank 8 is at top
        y1 = y_coords[7 - rank]
        y2 = y_coords[7 - rank + 1]

        # Return center
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        return center_x, center_y

    def get_square_bounds(self, square: str, transformed_image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Get bounding box for a chess square.

        Args:
            square: Chess square notation (e.g., "e4")
            transformed_image: Transformed board image

        Returns:
            (x1, y1, x2, y2) pixel coordinates of square bounds
        """
        # Get grid coordinates
        x_coords, y_coords = self._get_grid(transformed_image)

        # Parse square notation
        file = ord(square[0]) - ord('a')
        rank = int(square[1]) - 1

        # Get bounds
        x1 = int(x_coords[file])
        x2 = int(x_coords[file + 1])
        y1 = int(y_coords[7 - rank])
        y2 = int(y_coords[7 - rank + 1])

        return x1, y1, x2, y2

    def get_graveyard_coords(self, transformed_image: Image.Image) -> Tuple[int, int]:
        """
        Get coordinates for graveyard area (off-board piece storage).

        Places graveyard to the right of the board, centered vertically.

        Args:
            transformed_image: Transformed board image

        Returns:
            (x, y) pixel coordinates for graveyard
        """
        width, height = transformed_image.size

        # Place graveyard to the right of board, centered vertically
        # Note: Might extend beyond image bounds - that's okay for overlay
        x = width + 30  # 30 pixels to the right of board
        y = height // 2

        return x, y

    def _get_grid(self, transformed_image: Image.Image) -> Tuple[List[float], List[float]]:
        """
        Get or calculate grid coordinates for transformed image.

        Args:
            transformed_image: Transformed board image

        Returns:
            Tuple of (x_coordinates, y_coordinates) for grid lines
        """
        # Check cache
        img_size = transformed_image.size
        if img_size in self.grid_cache:
            return self.grid_cache[img_size]

        # Calculate grid
        if self.board_detector is not None:
            # Use BoardDetector's grid calculation
            x_coords, y_coords = self.board_detector.calculate_grid(transformed_image)
        else:
            # Fallback: simple linear grid (assumes no margin)
            x_coords, y_coords = self._calculate_simple_grid(transformed_image)

        # Cache result
        self.grid_cache[img_size] = (x_coords, y_coords)

        return x_coords, y_coords

    def _calculate_simple_grid(self, transformed_image: Image.Image) -> Tuple[List[float], List[float]]:
        """
        Calculate simple linear grid (fallback when no BoardDetector).

        Assumes board fills entire image with no margins.

        Args:
            transformed_image: Transformed board image

        Returns:
            Tuple of (x_coordinates, y_coordinates) for grid lines
        """
        width, height = transformed_image.size

        # Simple 8x8 grid
        x_coords = [i * width / 8 for i in range(9)]
        y_coords = [i * height / 8 for i in range(9)]

        return x_coords, y_coords

    def all_squares_to_pixels(self, transformed_image: Image.Image) -> dict:
        """
        Get pixel coordinates for all 64 squares.

        Args:
            transformed_image: Transformed board image

        Returns:
            Dictionary mapping square names to (x, y) pixel coordinates
        """
        coords = {}

        files = "abcdefgh"
        ranks = "12345678"

        for file in files:
            for rank in ranks:
                square = f"{file}{rank}"
                coords[square] = self.square_to_pixels(square, transformed_image)

        return coords

    def pixels_to_square(self, x: int, y: int, transformed_image: Image.Image) -> Optional[str]:
        """
        Convert pixel coordinates to chess square (inverse mapping).

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            transformed_image: Transformed board image

        Returns:
            Chess square notation (e.g., "e4") or None if outside board
        """
        # Get grid coordinates
        x_coords, y_coords = self._get_grid(transformed_image)

        # Find which file (column)
        file = None
        for i in range(8):
            if x_coords[i] <= x < x_coords[i + 1]:
                file = i
                break

        # Find which rank (row)
        rank = None
        for i in range(8):
            if y_coords[i] <= y < y_coords[i + 1]:
                rank = 7 - i  # Invert: y=0 is rank 8
                break

        # Return square or None
        if file is not None and rank is not None:
            return chr(ord('a') + file) + str(rank + 1)
        return None

    def clear_cache(self):
        """Clear the grid cache."""
        self.grid_cache.clear()
