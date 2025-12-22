"""
Controls Module

Robot hardware control with SO-100 arm integration and high-level controller.
"""

# SO-100 arm control (implemented)
try:
    from .so100_arm import SO100Arm, SO100State
    SO100_AVAILABLE = True
except ImportError:
    SO100_AVAILABLE = False

# SO-100 robot controller with Joint 0 stability system
try:
    from .robot_controller import (
        RobotController,
        JointConfig,
        load_joint_configs_for_port,
        load_joint_configs,
        scan_so100_ports,
        get_config_path_for_port,
        get_port_name,
    )
    ROBOT_CONTROLLER_AVAILABLE = True
except ImportError:
    ROBOT_CONTROLLER_AVAILABLE = False

# Planned ROS components (not yet implemented)
# from .robot_arm import RobotArm
# from .movement import MovementPrimitives
# from .calibration import CalibrationSystem
# from .safety import SafetySystem

__all__ = [
    'SO100Arm',
    'SO100State',
    'RobotController',
    'JointConfig',
    'load_joint_configs_for_port',
    'load_joint_configs',
    'scan_so100_ports',
    'get_config_path_for_port',
    'get_port_name',
    # 'RobotArm',
    # 'MovementPrimitives',
    # 'CalibrationSystem',
    # 'SafetySystem',
]
