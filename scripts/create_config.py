"""
Interactive Configuration Wizard

Helps set up chess robot configuration by:
1. Loading existing config if available
2. Detecting cameras via unplug/replug method
3. Prompting for other configuration values
4. Validating and saving config
"""

import cv2
import time
import sys
from pathlib import Path
from typing import List, Optional, Set

# Add configs to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import load_config, save_config, DEFAULT_CONFIG
from configs.config_schema import CONFIG_METADATA, get_nested, set_nested, validate_config


def list_available_cameras(max_cameras: int = 10) -> List[int]:
    """
    List all available camera device IDs.

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of available camera IDs
    """
    available = []

    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            # Try to read a frame to confirm camera works
            ret, _ = cap.read()
            if ret:
                available.append(camera_id)
            cap.release()

    return available


def test_camera(camera_id: int, duration: float = 2.0) -> bool:
    """
    Test camera by displaying live preview.

    Args:
        camera_id: Camera device ID
        duration: Duration (seconds) to display preview

    Returns:
        True if camera works, False otherwise
    """
    print(f"\n[Test] Opening camera {camera_id}...")

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        return False

    print(f"✓ Camera {camera_id} opened successfully")
    print(f"  Displaying preview for {duration} seconds...")
    print("  Press 'q' to close preview early")

    start_time = time.time()
    success = False

    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()

            if not ret:
                print(f"✗ Failed to read frame from camera {camera_id}")
                break

            # Display frame
            cv2.imshow(f"Camera {camera_id} Preview", frame)

            # Check for 'q' key to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            success = True

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return success


def detect_camera_by_removal(camera_name: str) -> Optional[int]:
    """
    Detect camera ID by asking user to unplug and replug.

    Args:
        camera_name: Display name for camera (e.g., "Global Camera")

    Returns:
        Detected camera ID, or None if detection failed
    """
    print("\n" + "=" * 70)
    print(f"{camera_name} Detection")
    print("=" * 70)

    # List cameras before unplug
    print("\nDetecting connected cameras...")
    before_cameras = set(list_available_cameras())

    if not before_cameras:
        print("✗ No cameras detected. Please connect cameras and try again.")
        return None

    print(f"✓ Found {len(before_cameras)} camera(s): {sorted(before_cameras)}")

    # Ask user to unplug
    print(f"\n>> Please UNPLUG the {camera_name} <<")
    input("Press Enter when camera is unplugged...")

    # Wait for device to be removed
    print("Waiting for device removal...")
    time.sleep(1.0)

    # List cameras after unplug
    after_unplug = set(list_available_cameras())

    # Find removed camera
    removed = before_cameras - after_unplug

    if not removed:
        print(f"✗ No camera was removed. Please try again.")
        return None

    if len(removed) > 1:
        print(f"⚠ Multiple cameras removed: {removed}. Using first one.")

    camera_id = sorted(removed)[0]
    print(f"✓ Detected {camera_name} as device ID: {camera_id}")

    # Ask user to replug
    print(f"\n>> Please REPLUG the {camera_name} <<")
    input("Press Enter when camera is replugged...")

    time.sleep(1.0)

    # Verify camera is back
    after_replug = set(list_available_cameras())

    if camera_id not in after_replug:
        print(f"✗ Camera {camera_id} not found after replug. Please check connection.")
        return None

    print(f"✓ {camera_name} reconnected successfully")

    # Test camera with preview
    test_input = input(f"\nTest {camera_name} with live preview? (y/n): ").strip().lower()
    if test_input == 'y':
        if not test_camera(camera_id, duration=3.0):
            print(f"⚠ Camera test failed, but continuing with ID {camera_id}")

    return camera_id


def prompt_for_value(key_path: str, current_value: any, metadata: dict) -> any:
    """
    Prompt user for configuration value.

    Args:
        key_path: Configuration key path (e.g., "cameras.global_camera_id")
        current_value: Current value (if any)
        metadata: Configuration metadata

    Returns:
        New value from user input
    """
    description = metadata.get("description", "")
    value_type = metadata["type"]

    print(f"\n{key_path}")
    print(f"  {description}")

    if current_value is not None:
        print(f"  Current value: {current_value}")

    # Get input
    prompt = "  New value (or press Enter to keep current): " if current_value is not None else "  Value: "
    user_input = input(prompt).strip()

    # Keep current value if empty
    if not user_input and current_value is not None:
        return current_value

    # Parse based on type
    try:
        if value_type == int:
            return int(user_input)
        elif value_type == float:
            return float(user_input)
        elif value_type == bool:
            return user_input.lower() in ['true', 'yes', 'y', '1']
        elif value_type == list:
            # Parse as comma-separated list
            if ',' in user_input:
                return [int(x.strip()) for x in user_input.split(',')]
            else:
                return user_input.split()
        else:  # str
            return user_input

    except ValueError as e:
        print(f"✗ Invalid input: {e}")
        print(f"  Using default: {current_value}")
        return current_value


def run_config_wizard():
    """Run interactive configuration wizard."""
    print("=" * 70)
    print("Chess Robot Configuration Wizard")
    print("=" * 70)

    # Try to load existing config
    config_file = Path("configs/current.yaml")
    if config_file.exists():
        print(f"\n✓ Found existing configuration at {config_file}")
        try:
            existing_config = load_config()
            print("✓ Existing configuration loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading existing config: {e}")
            print("  Starting with defaults")
            existing_config = DEFAULT_CONFIG.copy()
    else:
        print(f"\n✗ No existing configuration found at {config_file}")
        print("  Starting with defaults")
        existing_config = DEFAULT_CONFIG.copy()

    # Create new config (start with existing/defaults)
    new_config = existing_config.copy()

    # Step 1: Camera Detection
    print("\n" + "=" * 70)
    print("STEP 1: Camera Configuration")
    print("=" * 70)

    use_camera_detection = input("\nDetect cameras via unplug/replug method? (y/n): ").strip().lower()

    if use_camera_detection == 'y':
        # Detect global camera
        global_camera_id = detect_camera_by_removal("Global Camera (Overhead)")

        if global_camera_id is not None:
            set_nested(new_config, "cameras.global_camera_id", global_camera_id)

        # Detect gripper camera
        gripper_camera_id = detect_camera_by_removal("Gripper Camera (Arm-Mounted)")

        if gripper_camera_id is not None:
            set_nested(new_config, "cameras.gripper_camera_id", gripper_camera_id)

        # Verify cameras are different
        if global_camera_id == gripper_camera_id:
            print("\n⚠ Warning: Both cameras have the same ID. This is incorrect.")
            print("  Please run the wizard again and ensure you unplug/replug the correct cameras.")

    else:
        # Manual entry
        print("\nEnter camera IDs manually:")
        for key in ["cameras.global_camera_id", "cameras.gripper_camera_id"]:
            current_val = get_nested(new_config, key)
            metadata = CONFIG_METADATA[key]
            new_val = prompt_for_value(key, current_val, metadata)
            set_nested(new_config, key, new_val)

    # Step 2: Camera Resolutions
    print("\n" + "=" * 70)
    print("STEP 2: Camera Resolutions")
    print("=" * 70)

    configure_resolutions = input("\nConfigure camera resolutions? (y/n): ").strip().lower()

    if configure_resolutions == 'y':
        # Global camera resolution
        current_res = get_nested(new_config, "cameras.global_resolution")
        print(f"\nGlobal Camera Resolution")
        print(f"  Current: {current_res[0]}x{current_res[1]}")
        width = input("  Width (press Enter to keep): ").strip()
        height = input("  Height (press Enter to keep): ").strip()

        if width and height:
            set_nested(new_config, "cameras.global_resolution", [int(width), int(height)])

        # Gripper camera resolution
        current_res = get_nested(new_config, "cameras.gripper_resolution")
        print(f"\nGripper Camera Resolution")
        print(f"  Current: {current_res[0]}x{current_res[1]}")
        width = input("  Width (press Enter to keep): ").strip()
        height = input("  Height (press Enter to keep): ").strip()

        if width and height:
            set_nested(new_config, "cameras.gripper_resolution", [int(width), int(height)])

    # Step 3: Other Configuration
    print("\n" + "=" * 70)
    print("STEP 3: Other Configuration")
    print("=" * 70)

    configure_other = input("\nConfigure other settings (paths, models, etc.)? (y/n): ").strip().lower()

    if configure_other == 'y':
        # Model paths
        for key in ["models.corner_model", "models.piece_model"]:
            current_val = get_nested(new_config, key)
            metadata = CONFIG_METADATA[key]
            new_val = prompt_for_value(key, current_val, metadata)
            set_nested(new_config, key, new_val)

        # Guidance settings
        for key in ["guidance.robot_plays_white", "guidance.engine_path", "guidance.engine_time_limit"]:
            current_val = get_nested(new_config, key)
            metadata = CONFIG_METADATA[key]
            new_val = prompt_for_value(key, current_val, metadata)
            set_nested(new_config, key, new_val)

        # Visualization settings
        for key in ["visualization.host", "visualization.port"]:
            current_val = get_nested(new_config, key)
            metadata = CONFIG_METADATA[key]
            new_val = prompt_for_value(key, current_val, metadata)
            set_nested(new_config, key, new_val)

    # Step 4: Validation and Save
    print("\n" + "=" * 70)
    print("STEP 4: Validation and Save")
    print("=" * 70)

    # Validate configuration
    print("\nValidating configuration...")
    errors = validate_config(new_config)

    if errors:
        print("\n✗ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")

        save_anyway = input("\nSave configuration anyway? (y/n): ").strip().lower()
        if save_anyway != 'y':
            print("\n✗ Configuration not saved. Please fix errors and try again.")
            return

    else:
        print("✓ Configuration is valid")

    # Display final configuration
    print("\n" + "=" * 70)
    print("Final Configuration:")
    print("=" * 70)
    print(f"\nCameras:")
    print(f"  Global Camera ID: {get_nested(new_config, 'cameras.global_camera_id')}")
    print(f"  Global Resolution: {get_nested(new_config, 'cameras.global_resolution')}")
    print(f"  Gripper Camera ID: {get_nested(new_config, 'cameras.gripper_camera_id')}")
    print(f"  Gripper Resolution: {get_nested(new_config, 'cameras.gripper_resolution')}")

    print(f"\nModels:")
    print(f"  Corner Model: {get_nested(new_config, 'models.corner_model')}")
    print(f"  Piece Model: {get_nested(new_config, 'models.piece_model')}")

    print(f"\nGuidance:")
    print(f"  Robot Plays White: {get_nested(new_config, 'guidance.robot_plays_white')}")
    print(f"  Engine: {get_nested(new_config, 'guidance.engine_path')}")

    print(f"\nVisualization:")
    print(f"  Host: {get_nested(new_config, 'visualization.host')}")
    print(f"  Port: {get_nested(new_config, 'visualization.port')}")

    # Confirm save
    confirm = input("\nSave this configuration? (y/n): ").strip().lower()

    if confirm == 'y':
        try:
            save_config(new_config, config_file)
            print(f"\n✓ Configuration saved to {config_file}")
            print("\nYou can now run the chess robot system with this configuration.")
        except Exception as e:
            print(f"\n✗ Error saving configuration: {e}")
    else:
        print("\n✗ Configuration not saved")


def main():
    """Main entry point."""
    try:
        run_config_wizard()
        return 0
    except KeyboardInterrupt:
        print("\n\n✗ Configuration wizard interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
