"""
Configuration Schema and Defaults

Defines the configuration structure, defaults, and validation for the chess robot system.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path


# Default configuration
DEFAULT_CONFIG = {
    "cameras": {
        "global_camera_id": 2,   # WBC-0E01 overhead = /dev/video2 (idx 3 metadata)
        "global_resolution": [1280, 720],
        "gripper_camera_id": 0,  # Innomaker U20CAM capture = /dev/video0 (idx 1 metadata)
        "gripper_resolution": [640, 480],
    },
    "models": {
        "corner_model": "data/best_corners.pt",
        "piece_model": "data/best_transformed_detection.pt",
    },
    "paths": {
        "state_cache": "data/state_cache.json",
        "overlay_image": "data/guidance_overlay.png",
        "overlay_flag": "data/overlay_ready.flag",
        "data_dir": "data",
    },
    "visualization": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "guidance": {
        "robot_plays_white": False,
        "engine_path": "stockfish",
        "engine_time_limit": 1.0,
        "corner_confidence": 0.1,
        "min_corner_distance": 50.0,
    },
}


# Configuration field metadata
CONFIG_METADATA = {
    "cameras.global_camera_id": {
        "type": int,
        "description": "Camera device ID for overhead global view",
        "required": True,
    },
    "cameras.global_resolution": {
        "type": list,
        "description": "Resolution [width, height] for global camera",
        "required": True,
    },
    "cameras.gripper_camera_id": {
        "type": int,
        "description": "Camera device ID for gripper-mounted view",
        "required": True,
    },
    "cameras.gripper_resolution": {
        "type": list,
        "description": "Resolution [width, height] for gripper camera",
        "required": True,
    },
    "models.corner_model": {
        "type": str,
        "description": "Path to YOLO corner detection model (.pt file)",
        "required": True,
    },
    "models.piece_model": {
        "type": str,
        "description": "Path to YOLO piece detection model (.pt file)",
        "required": True,
    },
    "paths.state_cache": {
        "type": str,
        "description": "Path to state cache JSON file",
        "required": True,
    },
    "paths.overlay_image": {
        "type": str,
        "description": "Path to guidance overlay PNG file",
        "required": True,
    },
    "paths.overlay_flag": {
        "type": str,
        "description": "Path to overlay ready flag file",
        "required": True,
    },
    "paths.data_dir": {
        "type": str,
        "description": "Root data directory",
        "required": True,
    },
    "visualization.host": {
        "type": str,
        "description": "Host address for visualization server",
        "required": False,
    },
    "visualization.port": {
        "type": int,
        "description": "Port for visualization server",
        "required": False,
    },
    "guidance.robot_plays_white": {
        "type": bool,
        "description": "Whether robot plays as white",
        "required": False,
    },
    "guidance.engine_path": {
        "type": str,
        "description": "Path to UCI chess engine executable",
        "required": False,
    },
    "guidance.engine_time_limit": {
        "type": float,
        "description": "Time limit (seconds) for engine analysis",
        "required": False,
    },
    "guidance.corner_confidence": {
        "type": float,
        "description": "YOLO confidence threshold for corner detection",
        "required": False,
    },
    "guidance.min_corner_distance": {
        "type": float,
        "description": "Minimum distance (pixels) between corners",
        "required": False,
    },
}


def get_nested(d: Dict, key_path: str, default: Any = None) -> Any:
    """
    Get nested dictionary value using dot notation.

    Args:
        d: Dictionary to query
        key_path: Dot-separated path (e.g., "cameras.global_camera_id")
        default: Default value if key not found

    Returns:
        Value at key_path or default
    """
    keys = key_path.split('.')
    value = d

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_nested(d: Dict, key_path: str, value: Any):
    """
    Set nested dictionary value using dot notation.

    Args:
        d: Dictionary to modify
        key_path: Dot-separated path (e.g., "cameras.global_camera_id")
        value: Value to set
    """
    keys = key_path.split('.')
    current = d

    # Navigate to parent
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set value
    current[keys[-1]] = value


def validate_config(config: Dict) -> List[str]:
    """
    Validate configuration against schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for key_path, metadata in CONFIG_METADATA.items():
        value = get_nested(config, key_path)

        # Check required fields
        if metadata["required"] and value is None:
            errors.append(f"Required field missing: {key_path}")
            continue

        # Type validation
        if value is not None:
            expected_type = metadata["type"]

            if expected_type == int and not isinstance(value, int):
                errors.append(f"{key_path}: Expected int, got {type(value).__name__}")
            elif expected_type == float and not isinstance(value, (int, float)):
                errors.append(f"{key_path}: Expected float, got {type(value).__name__}")
            elif expected_type == str and not isinstance(value, str):
                errors.append(f"{key_path}: Expected str, got {type(value).__name__}")
            elif expected_type == bool and not isinstance(value, bool):
                errors.append(f"{key_path}: Expected bool, got {type(value).__name__}")
            elif expected_type == list and not isinstance(value, list):
                errors.append(f"{key_path}: Expected list, got {type(value).__name__}")

    # Additional validations
    if get_nested(config, "cameras.global_camera_id") == get_nested(config, "cameras.gripper_camera_id"):
        errors.append("cameras.global_camera_id and cameras.gripper_camera_id must be different")

    # Resolution validation
    global_res = get_nested(config, "cameras.global_resolution")
    if global_res and len(global_res) != 2:
        errors.append("cameras.global_resolution must be [width, height]")

    gripper_res = get_nested(config, "cameras.gripper_resolution")
    if gripper_res and len(gripper_res) != 2:
        errors.append("cameras.gripper_resolution must be [width, height]")

    # File path validation (check existence)
    corner_model = get_nested(config, "models.corner_model")
    if corner_model and not Path(corner_model).exists():
        errors.append(f"models.corner_model: File not found: {corner_model}")

    piece_model = get_nested(config, "models.piece_model")
    if piece_model and not Path(piece_model).exists():
        errors.append(f"models.piece_model: File not found: {piece_model}")

    return errors


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to override base with

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
