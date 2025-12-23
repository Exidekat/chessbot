#!/usr/bin/env python3
"""
SO-100 Teleoperation Script with Interactive Menu

Features:
1. Direct position control with Joint 0 stability system
2. Port-specific configuration loading with torque safety
3. Interactive menu with options:
   - Exit: Cleanly disconnect all robots
   - Tele-op Leader/Follower: Mirror leader arm movements on follower arm
   - Adjust Home Positions: Home-to-Home tele-op with ability to save new home positions
4. Leader/Follower tele-op at 15Hz control rate
5. Automatic torque release on exit

Torque Safety:
- Robots with port-specific configs: Torque ENABLED, homed to calibrated positions
- Robots without port-specific configs: Torque DISABLED for safety

Usage:
    python scripts/tele_op.py [--config-dir data/]
"""

import argparse
import sys
import time
import csv
import signal
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.so100_arm import SO100Arm, SO100State
from controls.robot_controller import (
    RobotController,
    JointConfig,
    load_joint_configs_for_port,
    scan_so100_ports,
    get_config_path_for_port,
)
from utils.state_cache import StateCache
from utils.keyboard_input import KeyboardInput


# Global list of connected robots for cleanup
_connected_robots: List[RobotController] = []


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_menu_choice(prompt_lines: List[str], num_options: int, keyboard: KeyboardInput) -> Optional[int]:
    """
    Display a menu and get user choice via number key.

    Args:
        prompt_lines: Lines to display as the menu
        num_options: Number of valid options (1 to num_options)
        keyboard: KeyboardInput instance

    Returns:
        int: Selected option (1-indexed), or None if ESC pressed
    """
    while True:
        clear_screen()
        for line in prompt_lines:
            print(line)

        key = keyboard.get_key(timeout=0.1)
        if key == 'ESC':
            return None
        elif key and key.isdigit():
            choice = int(key)
            if 1 <= choice <= num_options:
                return choice
        time.sleep(0.05)


def reset_robots_to_home(robots: List[RobotController]):
    """
    Reset all robots with port-specific configs to home position with torque enabled.
    Disable torque on robots without port-specific configs.
    """
    for robot in robots:
        if robot.config_source == "port-specific":
            # Stop any running control loop
            robot.stop_control_loop()
            # Enable torque
            robot.enable_torque()
            # Set home targets
            robot.set_home_targets()
            # Start control loop
            robot.start_control_loop()
        else:
            # Disable torque for safety
            robot.release_torque()


def build_status_table(robots: List[RobotController]) -> List[str]:
    """Build status table lines for display."""
    lines = []

    # Header
    header = f"{'Robot Port':<20}{'Config':<15}{'Torque':<10}{'Mode':<12}"
    for j in range(1, 7):
        header += f"{'J' + str(j):>8}"

    lines.append("-" * 115)
    lines.append(header)
    lines.append("-" * 115)

    # Robot rows
    for robot in robots:
        positions = robot.get_positions_deg()
        torque_status = "ON" if robot.torque_enabled else "OFF"
        control_mode = "Direct"
        row = f"{robot.port:<20}{robot.config_source:<15}{torque_status:<10}{control_mode:<12}"
        for pos in positions:
            row += f"{pos:>8.1f}"
        lines.append(row)

    lines.append("-" * 115)
    return lines


def run_interactive_menu(robots: List[RobotController], cache: StateCache):
    """
    Run interactive menu system for Stage 2.

    Features:
    - Main menu with status table
    - Exit option
    - Tele-op Leader/Follower mode

    Args:
        robots: List of connected RobotController instances
        cache: StateCache for sharing joint positions with VLA episode collection
    """
    print("\n" + "=" * 80)
    print("Stage 2: Interactive Menu")
    print("=" * 80)

    # Initialize robots to home positions
    reset_robots_to_home(robots)

    with KeyboardInput() as keyboard:
        while True:
            choice = show_main_menu(robots, keyboard)

            if choice is None or choice == 1:
                # Exit
                print("\n\nExiting...")
                return

            elif choice == 2:
                # Tele-op Leader/Follower
                run_teleop_leader_follower(robots, keyboard, cache)
                # After returning, reset robots to home
                reset_robots_to_home(robots)

            elif choice == 3:
                # Adjust Home Positions
                run_adjust_home_positions(robots, keyboard)
                # After returning, reset robots to home
                reset_robots_to_home(robots)


def show_main_menu(robots: List[RobotController], keyboard: KeyboardInput) -> Optional[int]:
    """
    Display main menu with status table and options.

    Returns:
        int: Selected option, or None for exit
    """
    while True:
        clear_screen()
        print("=" * 115)
        print("SO-100 Teleoperation - Main Menu")
        print("=" * 115)
        print()

        # Status table
        for line in build_status_table(robots):
            print(line)

        print()
        print(f"Last updated: {time.strftime('%H:%M:%S')}")
        print()
        print("=" * 115)
        print("OPTIONS:")
        print("  [1] Exit")
        print("  [2] Tele-op Leader/Follower")
        print("  [3] Adjust Home Positions")
        print("=" * 115)
        print("\nPress number key to select option...")

        key = keyboard.get_key(timeout=0.5)
        if key == 'ESC' or key == '1':
            return 1  # Exit
        elif key == '2':
            return 2  # Tele-op
        elif key == '3':
            return 3  # Adjust Home


def select_robot_port(robots: List[RobotController], keyboard: KeyboardInput,
                      title: str, exclude_port: Optional[str] = None) -> Optional[RobotController]:
    """
    Display submenu to select a robot port.

    Args:
        robots: List of connected robots
        keyboard: Keyboard input handler
        title: Menu title (e.g., "Select Leader Arm Port")
        exclude_port: Port to exclude from selection (for follower selection)

    Returns:
        RobotController: Selected robot, or None if cancelled
    """
    available_robots = [r for r in robots if r.port != exclude_port]

    if not available_robots:
        return None

    while True:
        clear_screen()
        print("=" * 60)
        print(title)
        print("=" * 60)
        print()
        print("Available ports:")
        print()

        for i, robot in enumerate(available_robots, start=1):
            config_info = f"({robot.config_source})"
            print(f"  [{i}] {robot.port} {config_info}")

        print()
        print("  [ESC] Cancel - Return to Main Menu")
        print()
        print("=" * 60)
        print("\nPress number key to select port...")

        key = keyboard.get_key(timeout=0.1)
        if key == 'ESC':
            return None
        elif key and key.isdigit():
            choice = int(key)
            if 1 <= choice <= len(available_robots):
                return available_robots[choice - 1]

        time.sleep(0.05)


def show_workspace_warning(keyboard: KeyboardInput, show_mode_selection: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Display workspace clearance warning with optional tele-op mode selection.

    Args:
        keyboard: Keyboard input handler
        show_mode_selection: If True, show mode selection (for option [2])

    Returns:
        Tuple[bool, Optional[str]]: (confirmed, mode)
            - confirmed: True if user confirmed, False if cancelled
            - mode: "home-to-home" or "encpos-to-encpos" (None if cancelled)
    """
    selected_mode = "home-to-home"  # Default

    while True:
        clear_screen()
        print("=" * 70)
        print("!! WARNING - WORKSPACE CLEARANCE !!")
        print("=" * 70)
        print()
        print("  Before starting Tele-op Leader/Follower mode:")
        print()
        print("  1. Ensure the FOLLOWER arm has clear workspace")
        print("  2. Remove any obstacles that could cause collision")
        print("  3. Keep hands away from the follower arm")
        print("  4. The follower will MIRROR the leader's movements")
        print()

        if show_mode_selection:
            print("=" * 70)
            print("  SELECT TELE-OP MODE:")
            print()
            h2h_marker = " [*]" if selected_mode == "home-to-home" else " [ ]"
            e2e_marker = " [*]" if selected_mode == "encpos-to-encpos" else " [ ]"
            print(f"  [1]{h2h_marker} Home-to-Home     (relative to home positions)")
            print(f"  [2]{e2e_marker} EncPos-to-EncPos (direct encoder mirroring)")
            print()

        print("=" * 70)
        print()
        print("  Press [ENTER] to confirm and start")
        if show_mode_selection:
            print("  Press [1] or [2] to change mode")
        print("  Press [ESC] to cancel")
        print()

        key = keyboard.get_key(timeout=0.1)
        if key == 'ESC':
            return False, None
        elif key == 'ENTER':
            return True, selected_mode
        elif show_mode_selection and key == '1':
            selected_mode = "home-to-home"
        elif show_mode_selection and key == '2':
            selected_mode = "encpos-to-encpos"
        time.sleep(0.05)


def run_teleop_leader_follower(robots: List[RobotController], keyboard: KeyboardInput, cache: StateCache):
    """
    Run the tele-op leader/follower mode.

    1. Select leader port
    2. Select follower port
    3. Show workspace warning with mode selection
    4. Run tele-op loop at 15Hz (Home-to-Home or EncPos-to-EncPos)
    5. ESC returns to main menu
    """
    # Step 1: Select leader
    leader = select_robot_port(robots, keyboard, "Select LEADER Arm Port")
    if leader is None:
        return

    # Step 2: Select follower (excluding leader)
    follower = select_robot_port(robots, keyboard, "Select FOLLOWER Arm Port",
                                  exclude_port=leader.port)
    if follower is None:
        return

    # Step 3: Workspace warning with mode selection
    confirmed, teleop_mode = show_workspace_warning(keyboard, show_mode_selection=True)
    if not confirmed:
        return

    # Step 4: Prepare for tele-op
    # Stop any control loops
    leader.stop_control_loop()
    follower.stop_control_loop()

    # Disable torque on leader (freely movable)
    leader.release_torque()

    # Enable torque on follower
    follower.enable_torque()

    # Get home positions for Home-to-Home mode
    leader_home = np.array([c.home_rad for c in leader.joint_configs])
    follower_home = np.array([c.home_rad for c in follower.joint_configs])

    # Step 5: Run tele-op loop at 15Hz
    teleop_active = True
    loop_interval = 1.0 / 15.0  # 15 Hz
    mode_display = "Home-to-Home" if teleop_mode == "home-to-home" else "EncPos-to-EncPos"

    while teleop_active:
        loop_start = time.time()

        # Read leader position
        leader_state = leader.arm.get_state()
        leader_positions = leader_state.joint_positions

        # Calculate follower targets based on mode
        if teleop_mode == "home-to-home":
            # Home-to-Home: Follower Target = Follower Home + (Leader Current - Leader Home)
            leader_delta = leader_positions - leader_home
            follower_targets = follower_home + leader_delta
        else:
            # EncPos-to-EncPos: Direct mirroring
            follower_targets = leader_positions

        # Send to follower (direct position mode)
        for motor_id, target_rad in enumerate(follower_targets, start=1):
            # Convert radians to encoder counts (0-4095)
            encoder_value = int((target_rad / (2 * np.pi)) * 4096) % 4096
            encoder_value = max(0, min(4095, encoder_value))

            # Send position command directly to motor
            packet = [
                *follower.arm.HEADER,
                motor_id,
                0x05,
                follower.arm.INSTR_WRITE,
                follower.arm.REG_GOAL_POSITION,
                encoder_value & 0xFF,
                (encoder_value >> 8) & 0xFF
            ]
            checksum = follower.arm._calculate_checksum(packet[2:])
            packet.append(checksum)
            follower.arm.serial.write(bytes(packet))
            time.sleep(0.002)

        # Update display
        clear_screen()
        print("=" * 80)
        print(f"Live Tele-op - In Progress ({mode_display})")
        print("=" * 80)
        print()
        print(f"  LEADER:   {leader.port}")
        print(f"  FOLLOWER: {follower.port}")
        print(f"  MODE:     {mode_display}")
        print()
        print("-" * 80)

        # Show positions
        leader_deg = np.degrees(leader_positions)
        follower_state = follower.arm.get_state()
        follower_deg = np.degrees(follower_state.joint_positions)
        target_deg = np.degrees(follower_targets)

        # Write follower joint positions to cache for VLA recording
        cache.update_joint_positions(
            positions=follower_state.joint_positions.tolist(),
            gripper_state=float(follower_state.joint_positions[5]),  # Joint 6 is gripper
            source="robot"
        )

        header = f"{'':15}"
        for j in range(1, 7):
            header += f"{'Joint ' + str(j):>12}"
        print(header)
        print("-" * 80)

        leader_row = f"{'LEADER':15}"
        for pos in leader_deg:
            leader_row += f"{pos:>12.1f}"
        print(leader_row)

        follower_row = f"{'FOLLOWER':15}"
        for pos in follower_deg:
            follower_row += f"{pos:>12.1f}"
        print(follower_row)

        target_row = f"{'TARGET':15}"
        for pos in target_deg:
            target_row += f"{pos:>12.1f}"
        print(target_row)

        # Show error (between follower and target)
        error_row = f"{'ERROR':15}"
        for t, f in zip(target_deg, follower_deg):
            error_row += f"{abs(t - f):>12.1f}"
        print(error_row)

        print("-" * 80)
        print()
        print(f"  Control Rate: 15 Hz")
        print(f"  Time: {time.strftime('%H:%M:%S')}")
        print()
        print("=" * 80)
        print("  Press [ESC] to stop and return to Main Menu")
        print("=" * 80)

        # Check for ESC key
        key = keyboard.get_key(timeout=0.01)
        if key == 'ESC':
            teleop_active = False

        # Maintain 15 Hz loop rate
        elapsed = time.time() - loop_start
        sleep_time = loop_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def save_joint_home_to_config(robot: RobotController, joint_idx: int, new_home_rad: float) -> bool:
    """
    Save a new home position for a joint to the robot's config file.

    Args:
        robot: The robot controller
        joint_idx: Joint index (0-5)
        new_home_rad: New home position in radians

    Returns:
        bool: True if save successful
    """
    config_path = get_config_path_for_port(robot.port)

    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return False

    try:
        # Read existing config
        rows = []
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)

        # Update the specific joint's home position
        if joint_idx < len(rows):
            rows[joint_idx]['home_rad'] = f"{new_home_rad:.6f}"
            rows[joint_idx]['home_deg'] = f"{np.degrees(new_home_rad):.2f}"

        # Write back to file
        with open(config_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # Update in-memory config
        robot.joint_configs[joint_idx].home_rad = new_home_rad

        return True

    except Exception as e:
        print(f"[ERROR] Failed to save config: {e}")
        return False


def show_adjust_home_mode_selection(keyboard: KeyboardInput) -> Optional[str]:
    """
    Show mode selection for Adjust Home Positions.

    Returns:
        str: "manual" or "teleop", or None if cancelled
    """
    selected_mode = "manual"  # Default

    while True:
        clear_screen()
        print("=" * 60)
        print("Adjust Home Positions - Select Mode")
        print("=" * 60)
        print()

        manual_marker = " [*]" if selected_mode == "manual" else " [ ]"
        teleop_marker = " [*]" if selected_mode == "teleop" else " [ ]"

        print(f"  [1]{manual_marker} Manual    (manually move joints, save positions)")
        print(f"  [2]{teleop_marker} Tele-op   (use leader arm to guide follower)")
        print()
        print("=" * 60)
        print()
        print("  Press [ENTER] to confirm")
        print("  Press [1] or [2] to change mode")
        print("  Press [ESC] to cancel")
        print()

        key = keyboard.get_key(timeout=0.1)
        if key == 'ESC':
            return None
        elif key == 'ENTER':
            return selected_mode
        elif key == '1':
            selected_mode = "manual"
        elif key == '2':
            selected_mode = "teleop"
        time.sleep(0.05)


def run_adjust_home_manual(robots: List[RobotController], keyboard: KeyboardInput):
    """
    Run Manual mode for adjusting home positions.

    Features:
    - Select target robot
    - Disable torques (robot freely movable)
    - Press 0-5 to save current joint position as new home
    - ESC returns to main menu
    """
    # Step 1: Select target robot
    target = select_robot_port(robots, keyboard, "Select TARGET Robot")
    if target is None:
        return

    # Check target has port-specific config (required for saving)
    if target.config_source != "port-specific":
        clear_screen()
        print("=" * 60)
        print("!! ERROR !!")
        print("=" * 60)
        print()
        print(f"  Target {target.port} does not have a port-specific config.")
        print(f"  Config source: {target.config_source}")
        print()
        print("  To adjust home positions, the target must have a")
        print("  port-specific config file to save changes to.")
        print()
        print("  Run calibration first:")
        print(f"    python scripts/create_so100_config.py --port {target.port}")
        print()
        print("=" * 60)
        print("\n  Press any key to return...")
        keyboard.get_key(timeout=10.0)
        return

    # Step 2: Prepare - stop control loop and disable torque
    target.stop_control_loop()
    target.release_torque()

    # Get current home positions
    target_home = np.array([c.home_rad for c in target.joint_configs])

    # Step 3: Run manual adjustment loop
    active = True
    last_save_message = ""
    save_message_time = 0

    while active:
        # Read current position
        target_state = target.arm.get_state()
        target_positions = target_state.joint_positions

        # Update display
        clear_screen()
        print("=" * 90)
        print("Adjust Home Positions - Manual Mode")
        print("=" * 90)
        print()
        print(f"  TARGET: {target.port}")
        print(f"  TORQUE: DISABLED (freely movable)")
        print()
        print("-" * 90)

        # Show positions
        target_deg = np.degrees(target_positions)
        home_deg = np.degrees(target_home)

        header = f"{'':20}"
        for j in range(6):
            header += f"{'Joint ' + str(j):>11}"
        print(header)
        print("-" * 90)

        current_row = f"{'CURRENT POSITION':20}"
        for pos in target_deg:
            current_row += f"{pos:>11.1f}"
        print(current_row)

        home_row = f"{'HOME POSITION':20}"
        for pos in home_deg:
            home_row += f"{pos:>11.1f}"
        print(home_row)

        # Show difference from home
        diff_row = f"{'DIFF FROM HOME':20}"
        for curr, home in zip(target_deg, home_deg):
            diff_row += f"{curr - home:>+11.1f}"
        print(diff_row)

        print("-" * 90)
        print()
        print(f"  Time: {time.strftime('%H:%M:%S')}")
        print()

        # Show save message if recent
        if last_save_message and (time.time() - save_message_time) < 3.0:
            print(f"  {last_save_message}")
        else:
            last_save_message = ""

        print()
        print("=" * 90)
        print("  Manually move the robot to desired home position.")
        print("  Press [0-5] to save current joint position as new HOME")
        print("  Press [ESC] to return to Main Menu")
        print("=" * 90)

        # Check for key press
        key = keyboard.get_key(timeout=0.1)
        if key == 'ESC':
            active = False
        elif key and key.isdigit():
            joint_idx = int(key)
            if 0 <= joint_idx <= 5:
                # Save current position as new home
                current_pos = target_positions[joint_idx]
                if save_joint_home_to_config(target, joint_idx, current_pos):
                    # Update local home array
                    target_home[joint_idx] = current_pos
                    last_save_message = f"[OK] Joint {joint_idx} home saved: {np.degrees(current_pos):.1f} deg"
                else:
                    last_save_message = f"[ERROR] Failed to save joint {joint_idx}"
                save_message_time = time.time()

        time.sleep(0.05)


def run_adjust_home_teleop(robots: List[RobotController], keyboard: KeyboardInput):
    """
    Run Tele-op mode for adjusting home positions.

    Movement calculation: Follower Target = Follower Home + (Leader Current - Leader Home)

    Features:
    - Home-to-Home relative movement
    - Press 0-5 to save current follower joint position as new home
    - ESC returns to main menu
    """
    # Step 1: Select leader
    leader = select_robot_port(robots, keyboard, "Select LEADER Arm Port")
    if leader is None:
        return

    # Step 2: Select follower (excluding leader)
    follower = select_robot_port(robots, keyboard, "Select FOLLOWER Arm Port",
                                  exclude_port=leader.port)
    if follower is None:
        return

    # Check follower has port-specific config (required for saving)
    if follower.config_source != "port-specific":
        clear_screen()
        print("=" * 60)
        print("!! ERROR !!")
        print("=" * 60)
        print()
        print(f"  Follower {follower.port} does not have a port-specific config.")
        print(f"  Config source: {follower.config_source}")
        print()
        print("  To adjust home positions, the follower must have a")
        print("  port-specific config file to save changes to.")
        print()
        print("  Run calibration first:")
        print(f"    python scripts/create_so100_config.py --port {follower.port}")
        print()
        print("=" * 60)
        print("\n  Press any key to return...")
        keyboard.get_key(timeout=10.0)
        return

    # Step 3: Workspace warning (no mode selection - always Home-to-Home for this option)
    confirmed, _ = show_workspace_warning(keyboard, show_mode_selection=False)
    if not confirmed:
        return

    # Step 4: Prepare for tele-op
    leader.stop_control_loop()
    follower.stop_control_loop()

    # Disable torque on leader (freely movable)
    leader.release_torque()

    # Enable torque on follower
    follower.enable_torque()

    # Get home positions
    leader_home = np.array([c.home_rad for c in leader.joint_configs])
    follower_home = np.array([c.home_rad for c in follower.joint_configs])

    # Step 5: Run Home-to-Home tele-op loop at 15Hz
    teleop_active = True
    loop_interval = 1.0 / 15.0  # 15 Hz
    last_save_message = ""
    save_message_time = 0

    while teleop_active:
        loop_start = time.time()

        # Read leader position
        leader_state = leader.arm.get_state()
        leader_positions = leader_state.joint_positions

        # Calculate follower targets using Home-to-Home formula:
        # Follower Target = Follower Home + (Leader Current - Leader Home)
        leader_delta = leader_positions - leader_home
        follower_targets = follower_home + leader_delta

        # Send to follower (direct position mode)
        for motor_id, target_rad in enumerate(follower_targets, start=1):
            # Convert radians to encoder counts (0-4095)
            encoder_value = int((target_rad / (2 * np.pi)) * 4096) % 4096
            encoder_value = max(0, min(4095, encoder_value))

            # Send position command directly to motor
            packet = [
                *follower.arm.HEADER,
                motor_id,
                0x05,
                follower.arm.INSTR_WRITE,
                follower.arm.REG_GOAL_POSITION,
                encoder_value & 0xFF,
                (encoder_value >> 8) & 0xFF
            ]
            checksum = follower.arm._calculate_checksum(packet[2:])
            packet.append(checksum)
            follower.arm.serial.write(bytes(packet))
            time.sleep(0.002)

        # Read actual follower position
        follower_state = follower.arm.get_state()
        follower_positions = follower_state.joint_positions

        # Update display
        clear_screen()
        print("=" * 90)
        print("Adjust Home Positions - Home-to-Home Tele-op")
        print("=" * 90)
        print()
        print(f"  LEADER:   {leader.port}")
        print(f"  FOLLOWER: {follower.port}")
        print()
        print("-" * 90)

        # Show positions
        leader_deg = np.degrees(leader_positions)
        follower_deg = np.degrees(follower_positions)
        follower_home_deg = np.degrees(follower_home)
        follower_target_deg = np.degrees(follower_targets)

        header = f"{'':20}"
        for j in range(6):
            header += f"{'Joint ' + str(j):>11}"
        print(header)
        print("-" * 90)

        leader_row = f"{'LEADER':20}"
        for pos in leader_deg:
            leader_row += f"{pos:>11.1f}"
        print(leader_row)

        follower_row = f"{'FOLLOWER':20}"
        for pos in follower_deg:
            follower_row += f"{pos:>11.1f}"
        print(follower_row)

        home_row = f"{'FOLLOWER HOME':20}"
        for pos in follower_home_deg:
            home_row += f"{pos:>11.1f}"
        print(home_row)

        target_row = f"{'TARGET':20}"
        for pos in follower_target_deg:
            target_row += f"{pos:>11.1f}"
        print(target_row)

        print("-" * 90)
        print()
        print(f"  Control Rate: 15 Hz | Time: {time.strftime('%H:%M:%S')}")
        print()

        # Show save message if recent
        if last_save_message and (time.time() - save_message_time) < 3.0:
            print(f"  {last_save_message}")
        else:
            last_save_message = ""

        print()
        print("=" * 90)
        print("  Press [0-5] to save current follower joint position as new HOME")
        print("  Press [ESC] to return to Main Menu")
        print("=" * 90)

        # Check for key press
        key = keyboard.get_key(timeout=0.01)
        if key == 'ESC':
            teleop_active = False
        elif key and key.isdigit():
            joint_idx = int(key)
            if 0 <= joint_idx <= 5:
                # Save current follower position as new home
                current_pos = follower_positions[joint_idx]
                if save_joint_home_to_config(follower, joint_idx, current_pos):
                    # Update local follower_home array
                    follower_home[joint_idx] = current_pos
                    last_save_message = f"[OK] Joint {joint_idx} home saved: {np.degrees(current_pos):.1f} deg"
                else:
                    last_save_message = f"[ERROR] Failed to save joint {joint_idx}"
                save_message_time = time.time()

        # Maintain 15 Hz loop rate
        elapsed = time.time() - loop_start
        sleep_time = loop_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def run_adjust_home_positions(robots: List[RobotController], keyboard: KeyboardInput):
    """
    Run Adjust Home Positions with mode selection.

    Modes:
    - Manual: Select target robot, disable torques, manually position and save
    - Tele-op: Use leader arm to guide follower with Home-to-Home movement
    """
    # Show mode selection
    mode = show_adjust_home_mode_selection(keyboard)
    if mode is None:
        return

    if mode == "manual":
        run_adjust_home_manual(robots, keyboard)
    else:  # teleop
        run_adjust_home_teleop(robots, keyboard)


def cleanup_handler(signum, frame):
    """Handle cleanup on exit signal."""
    print("\n\nReceived exit signal, releasing torque...")
    for robot in _connected_robots:
        robot.disconnect()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="SO-100 Teleoperation with Interactive Menu"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="data",
        help="Directory containing SO-100 config CSVs (default: data)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: hold robots at home for 10 seconds then exit (no keyboard input required)"
    )

    args = parser.parse_args()
    config_dir = Path(args.config_dir)

    # Initialize state cache for VLA episode collection
    cache = StateCache("data/state_cache.json")

    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    print("=" * 80)
    print("SO-100 Teleoperation with Interactive Menu")
    print("=" * 80)
    print()

    # Scan for connected robots first
    connected_ports = scan_so100_ports()

    if not connected_ports:
        print("[X] No SO-100 robots found!")
        print("    Check that robots are powered on and connected via USB.")
        return 1

    # Create robot controllers with port-specific configs
    print("\nLoading configurations for each robot...")
    print("Control mode: Direct Position (with Joint 0 stability)")

    robots: List[RobotController] = []
    for port in connected_ports:
        try:
            # Load port-specific config (or fall back to generic/defaults)
            joint_configs, config_source = load_joint_configs_for_port(port, config_dir)

            robot = RobotController(port, joint_configs, config_source)
            if robot.connect():
                robots.append(robot)
                _connected_robots.append(robot)  # Track for cleanup

                # Show home positions for this robot
                print(f"  [{port}] Home positions: ", end="")
                home_degs = [f"{np.degrees(c.home_rad):.1f}" for c in joint_configs]
                print(f"[{', '.join(home_degs)}] deg")
        except Exception as e:
            print(f"  [{port}] Failed to load config: {e}")

    if not robots:
        print("[X] Failed to connect to any robots!")
        return 1

    print(f"\n[OK] Connected to {len(robots)} robot(s)")

    # Show config summary
    print("\nConfiguration summary:")
    for robot in robots:
        status = "ENABLED (has port-specific config)" if robot.config_source == "port-specific" else f"DISABLED ({robot.config_source} config)"
        print(f"  {robot.port}: {status}")

    # Test mode: hold at home for 10 seconds then exit
    if args.test:
        print("\n" + "=" * 80)
        print("TEST MODE: Holding robots at home position for 10 seconds...")
        print("Monitoring joint 0 and 1 for oscillation...")
        print("=" * 80)

        try:
            # Enable torque and home robots with port-specific configs
            for robot in robots:
                if robot.config_source == "port-specific":
                    robot.enable_torque()
                    robot.set_home_targets()
                    robot.start_control_loop()

            # Hold for 10 seconds, printing joint positions at 5Hz
            print("\nTime | Robot        | J0      J1      J2      J3      J4      J5")
            print("-" * 80)

            start_time = time.time()
            sample_interval = 0.2  # 5 Hz sampling
            next_sample = start_time + sample_interval

            while time.time() - start_time < 10:
                if time.time() >= next_sample:
                    elapsed = time.time() - start_time

                    for robot in robots:
                        if robot.config_source == "port-specific":
                            positions = robot.get_positions_deg()
                            port_name = robot.port.split('/')[-1]
                            print(f"{elapsed:4.1f} | {port_name:<12} | ", end="")
                            # Print J0 and J1 with more precision to see oscillation
                            print(f"{positions[0]:7.2f} {positions[1]:7.2f} ", end="")
                            # Print other joints with less precision
                            for pos in positions[2:]:
                                print(f"{pos:7.1f} ", end="")
                            print()

                    next_sample += sample_interval
                else:
                    time.sleep(0.01)  # Small sleep to avoid busy waiting

            print("\n[OK] Test complete!")

        except Exception as e:
            print(f"\nError in test mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            print("\nCleaning up...")
            for robot in robots:
                robot.disconnect()
            print("[OK] All robots disconnected and torque released")

        return 0

    # Interactive mode: wait for user to proceed
    print("\n" + "-" * 80)
    input("Press ENTER to continue to Interactive Menu...")

    # Run interactive menu (handles homing and tele-op)
    try:
        run_interactive_menu(robots, cache)
    except Exception as e:
        print(f"Error in interactive menu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: stop control loops and release torque
        print("\nCleaning up...")
        for robot in robots:
            robot.disconnect()
        print("[OK] All robots disconnected and torque released")

    return 0


if __name__ == "__main__":
    sys.exit(main())


"""
================================================================================
INTERACTIVE MENU FLOW DIAGRAM
================================================================================

  Startup
      |
      +-- Scan for SO-100 robots (/dev/ttyACM0-9)
      +-- Load port-specific configs (or fallback to generic/defaults)
      +-- Connect to robots
      |
      v
  ============================================
  MAIN MENU (robots with port-specific configs
             have torque ENABLED and target home)
  ============================================
      |
      +-- [1] Exit
      |       |
      |       +-- Cleanup: stop control loops, release torque
      |       +-- Disconnect all robots
      |       +-- Exit script
      |
      +-- [2] Tele-op Leader/Follower
      |       |
      |       +-- Select LEADER Arm Port
      |       |       |
      |       |       +-- [1-9] Select from available ports
      |       |       +-- [ESC] Cancel -> Main Menu
      |       |
      |       +-- Select FOLLOWER Arm Port (excludes leader)
      |       |       |
      |       |       +-- [1-9] Select from available ports
      |       |       +-- [ESC] Cancel -> Main Menu
      |       |
      |       +-- Workspace Warning + Mode Selection
      |       |       |
      |       |       +-- [1] Home-to-Home (default)
      |       |       |       Formula: Follower Target = Follower Home + (Leader - Leader Home)
      |       |       |
      |       |       +-- [2] EncPos-to-EncPos
      |       |       |       Formula: Follower Target = Leader Position
      |       |       |
      |       |       +-- [ENTER] Confirm and start
      |       |       +-- [ESC] Cancel -> Main Menu
      |       |
      |       +-- Live Tele-op (15 Hz)
      |               |
      |               +-- Leader: Torque DISABLED (freely movable)
      |               +-- Follower: Torque ENABLED (mirrors leader)
      |               +-- Display: Leader, Follower, Target, Error positions
      |               +-- [ESC] Stop -> Reset robots -> Main Menu
      |
      +-- [3] Adjust Home Positions
              |
              +-- Mode Selection
              |       |
              |       +-- [1] Manual (default)
              |       +-- [2] Tele-op
              |       +-- [ENTER] Confirm
              |       +-- [ESC] Cancel -> Main Menu
              |
              +-- [1] Manual Mode
              |       |
              |       +-- Select TARGET Robot
              |       |       |
              |       |       +-- [1-9] Select from available ports
              |       |       +-- [ESC] Cancel -> Main Menu
              |       |
              |       +-- (Requires port-specific config)
              |       |
              |       +-- Manual Adjustment Screen
              |               |
              |               +-- Target: Torque DISABLED (freely movable)
              |               +-- Display: Current Position, Home Position, Diff from Home
              |               +-- [0-5] Save joint position as new HOME
              |               +-- [ESC] Stop -> Reset robots -> Main Menu
              |
              +-- [2] Tele-op Mode
                      |
                      +-- Select LEADER Arm Port
                      |       |
                      |       +-- [1-9] Select from available ports
                      |       +-- [ESC] Cancel -> Main Menu
                      |
                      +-- Select FOLLOWER Arm Port (excludes leader)
                      |       |
                      |       +-- [1-9] Select from available ports
                      |       +-- [ESC] Cancel -> Main Menu
                      |
                      +-- (Follower requires port-specific config)
                      |
                      +-- Workspace Warning
                      |       |
                      |       +-- [ENTER] Confirm and start
                      |       +-- [ESC] Cancel -> Main Menu
                      |
                      +-- Home-to-Home Tele-op (15 Hz)
                              |
                              +-- Leader: Torque DISABLED (freely movable)
                              +-- Follower: Torque ENABLED (mirrors leader)
                              +-- Formula: Follower Target = Follower Home + (Leader - Leader Home)
                              +-- Display: Leader, Follower, Follower Home, Target positions
                              +-- [0-5] Save follower joint position as new HOME
                              +-- [ESC] Stop -> Reset robots -> Main Menu

================================================================================
TORQUE SAFETY RULES
================================================================================

  Port-Specific Config Found:
      - Torque ENABLED
      - Robot homes to calibrated positions
      - Safe to use for tele-op follower

  Generic/Default Config:
      - Torque DISABLED for safety
      - Robot freely movable by hand
      - Cannot be used as follower (no config to save to)

================================================================================
CONFIG FILE FORMAT (data/so100_config_<port>.csv)
================================================================================

  joint,port,min_rad,max_rad,home_rad,min_deg,max_deg,home_deg,range_deg
  0,/dev/ttyACM0,1.504835,4.844311,3.229030,86.22,277.56,185.01,191.34
  1,/dev/ttyACM0,1.385185,4.445477,3.477535,79.37,254.71,199.25,175.34
  ...

================================================================================
"""
