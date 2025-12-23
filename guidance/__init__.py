"""
Guidance Module

Chess-specific logic using YOLO-based vision.
Includes board detection, move calculation, and visual guidance.
"""

from .board_detector import BoardDetector
from .move_calculator import MoveCalculator
from .move_interpreter import MoveInterpreter, MoveType
from .coordinate_mapper import CoordinateMapper
from .highlight_renderer import HighlightRenderer
from .guidance_system import GuidanceSystem
from .frame_overlay_renderer import (
    rotate_square_for_camera,
    apply_stage_overlay_to_frame
)

__all__ = [
    'BoardDetector',
    'MoveCalculator',
    'MoveInterpreter',
    'MoveType',
    'CoordinateMapper',
    'HighlightRenderer',
    'GuidanceSystem',
    'rotate_square_for_camera',
    'apply_stage_overlay_to_frame',
]
