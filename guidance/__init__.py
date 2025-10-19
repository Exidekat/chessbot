"""
Guidance Module

Chess-specific logic using YOLO-based vision.
Includes board detection, move calculation, and visual guidance.
"""

from .board_detector import BoardDetector
from .move_calculator import MoveCalculator

# TODO: These modules will be added as we complete the refactoring
# from .move_interpreter import MoveInterpreter
# from .highlight_renderer import HighlightRenderer
# from .coordinate_mapper import CoordinateMapper
# from .guidance_system import GuidanceSystem

__all__ = [
    'BoardDetector',
    'MoveCalculator',
    # 'MoveInterpreter',
    # 'HighlightRenderer',
    # 'CoordinateMapper',
    # 'GuidanceSystem',
]
