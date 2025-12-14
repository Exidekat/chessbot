"""
Controls Module

Robot hardware control with SO-100 arm integration.
"""

# SO-100 arm control (implemented)
try:
    from .so100_arm import SO100Arm, SO100State
    SO100_AVAILABLE = True
except ImportError:
    SO100_AVAILABLE = False

# Planned ROS components (not yet implemented)
# from .robot_arm import RobotArm
# from .movement import MovementPrimitives
# from .calibration import CalibrationSystem
# from .safety import SafetySystem

__all__ = [
    'SO100Arm',
    'SO100State',
    # 'RobotArm',
    # 'MovementPrimitives',
    # 'CalibrationSystem',
    # 'SafetySystem',
]
