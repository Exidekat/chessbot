#!/usr/bin/env python3
"""
SO-100 Teleoperation Script with Interactive Menu

Features:
1. Direct position control (default) or Adaptive PID (--adaptive flag)
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
    python scripts/tele_op.py [--config-dir data/] [--adaptive]
"""

import argparse
import sys
import time
import csv
import threading
import signal
import os
import select
import termios
import tty
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.so100_arm import SO100Arm, SO100State


@dataclass
class JointConfig:
    """Configuration for a single joint from calibration."""
    min_rad: float
    max_rad: float
    home_rad: float


def get_port_name(port: str) -> str:
    """Extract port name from device path (e.g., /dev/ttyACM0 -> ttyACM0)."""
    return port.split('/')[-1]


def get_config_path_for_port(port: str, config_dir: Path = Path("data")) -> Path:
    """Get config file path for a specific port."""
    port_name = get_port_name(port)
    return config_dir / f"so100_config_{port_name}.csv"


@dataclass
class AdaptivePIDController:
    """
    Adaptive PID controller that auto-tunes gains during runtime.

    Uses gradient descent on squared error to optimize P, I, D gains.
    """
    # PID gains
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.1

    # Adaptive learning rates
    lr_p: float = 0.001
    lr_i: float = 0.0001
    lr_d: float = 0.0005

    # Gain limits
    kp_min: float = 0.1
    kp_max: float = 10.0
    ki_min: float = 0.0
    ki_max: float = 1.0
    kd_min: float = 0.0
    kd_max: float = 2.0

    # Internal state
    integral: float = 0.0
    prev_error: float = 0.0
    prev_time: float = field(default_factory=time.time)

    # Error history for adaptation
    error_history: List[float] = field(default_factory=list)
    max_history: int = 50

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.error_history.clear()

    def compute(self, target: float, current: float) -> float:
        """
        Compute PID output and adapt gains.

        Args:
            target: Target position (radians)
            current: Current position (radians)

        Returns:
            float: Control output (position delta)
        """
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt < 1e-6:
            return 0.0

        # Calculate error
        error = target - current

        # Update integral (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)

        # Calculate derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID output
        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        # Store error for adaptation
        self.error_history.append(abs(error))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Adapt gains based on error dynamics
        self._adapt_gains(error, derivative, dt)

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        return output

    def _adapt_gains(self, error: float, derivative: float, dt: float):
        """
        Adapt PID gains using gradient descent on error.

        Strategy:
        - Increase Kp if error is large and not oscillating
        - Increase Ki if steady-state error persists
        - Increase Kd if oscillating (large derivative)
        """
        if len(self.error_history) < 10:
            return

        # Compute error metrics
        recent_errors = self.error_history[-10:]
        avg_error = np.mean(recent_errors)
        error_trend = recent_errors[-1] - recent_errors[0]  # Positive = getting worse
        oscillation = np.std(recent_errors)

        # Adapt Kp: increase if error large and stable, decrease if oscillating
        if avg_error > 0.1 and oscillation < 0.05:
            self.kp += self.lr_p * avg_error
        elif oscillation > 0.1:
            self.kp -= self.lr_p * oscillation

        # Adapt Ki: increase if steady-state error persists
        if abs(error_trend) < 0.01 and avg_error > 0.05:
            self.ki += self.lr_i * avg_error

        # Adapt Kd: increase if oscillating, decrease if sluggish
        if oscillation > 0.05:
            self.kd += self.lr_d * oscillation
        elif avg_error > 0.1 and oscillation < 0.02:
            self.kd -= self.lr_d * 0.1

        # Clamp gains to valid range
        self.kp = np.clip(self.kp, self.kp_min, self.kp_max)
        self.ki = np.clip(self.ki, self.ki_min, self.ki_max)
        self.kd = np.clip(self.kd, self.kd_min, self.kd_max)

    def get_gains(self) -> Tuple[float, float, float]:
        """Get current PID gains."""
        return self.kp, self.ki, self.kd


class RobotController:
    """Controller for a single SO-100 robot with adaptive PID per joint."""

    def __init__(self, port: str, joint_configs: List[JointConfig], config_source: str = "defaults", use_adaptive_pid: bool = False):
        self.port = port
        self.arm: Optional[SO100Arm] = None
        self.joint_configs = joint_configs
        self.config_source = config_source  # "port-specific", "generic", or "defaults"
        self.use_adaptive_pid = use_adaptive_pid
        self.pid_controllers = [AdaptivePIDController() for _ in range(6)]
        self.target_positions = np.zeros(6)
        self.connected = False
        self.control_thread: Optional[threading.Thread] = None
        self.running = False
        self.torque_enabled = False

    def connect(self) -> bool:
        """Connect to the robot."""
        self.arm = SO100Arm(port=self.port, baudrate=1000000)
        self.connected = self.arm.connect()
        return self.connected

    def disconnect(self):
        """Disconnect and release torque."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        if self.arm and self.connected:
            self.release_torque()
            self.arm.disconnect()

        self.connected = False

    def enable_torque(self):
        """Enable torque on all motors."""
        if not self.arm or not self.connected:
            return False

        try:
            # Send torque enable command to each motor
            for motor_id in self.arm.MOTOR_IDS:
                packet = [
                    *self.arm.HEADER,
                    motor_id,
                    0x04,  # Length
                    self.arm.INSTR_WRITE,
                    self.arm.REG_TORQUE_ENABLE,
                    0x01  # Enable torque
                ]
                checksum = self.arm._calculate_checksum(packet[2:])
                packet.append(checksum)
                self.arm.serial.write(bytes(packet))
                time.sleep(0.005)

            self.torque_enabled = True
            print(f"[{self.port}] Torque enabled")
            return True
        except Exception as e:
            print(f"[{self.port}] Failed to enable torque: {e}")
            return False

    def release_torque(self):
        """Disable torque on all motors."""
        if not self.arm or not self.connected:
            return

        try:
            # Send torque disable command to each motor
            for motor_id in self.arm.MOTOR_IDS:
                packet = [
                    *self.arm.HEADER,
                    motor_id,
                    0x04,  # Length
                    self.arm.INSTR_WRITE,
                    self.arm.REG_TORQUE_ENABLE,
                    0x00  # Disable torque
                ]
                checksum = self.arm._calculate_checksum(packet[2:])
                packet.append(checksum)
                self.arm.serial.write(bytes(packet))
                time.sleep(0.005)

            self.torque_enabled = False
            print(f"[{self.port}] Torque released")
        except Exception as e:
            print(f"[{self.port}] Failed to release torque: {e}")

    def set_target_positions(self, positions: np.ndarray):
        """Set target positions for all joints."""
        self.target_positions = positions.copy()

    def set_home_targets(self):
        """Set target positions to home positions from config."""
        for i, config in enumerate(self.joint_configs):
            self.target_positions[i] = config.home_rad

    def start_control_loop(self):
        """Start the PID control loop in a background thread."""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

    def _control_loop(self):
        """Background control loop (PID or direct position)."""
        while self.running and self.connected:
            try:
                if self.use_adaptive_pid:
                    # Adaptive PID mode: Compute PID outputs for each joint
                    state = self.arm.get_state()
                    current_positions = state.joint_positions

                    new_positions = np.zeros(6)
                    for i in range(6):
                        output = self.pid_controllers[i].compute(
                            self.target_positions[i],
                            current_positions[i]
                        )
                        # Apply output as position delta
                        new_positions[i] = current_positions[i] + output * 0.1  # Scale down

                        # Wraparound to [0, 2π)
                        new_positions[i] = new_positions[i] % (2 * np.pi)

                    # Send to robot
                    self.arm.move_joints(new_positions.tolist(), speed=0.5, blocking=False)
                else:
                    # Direct position mode: Send target positions directly as encoder values
                    for motor_id, target_rad in enumerate(self.target_positions, start=1):
                        # Convert radians to encoder counts (0-4095)
                        encoder_value = int((target_rad / (2 * np.pi)) * 4096) % 4096

                        # Clamp to valid range
                        encoder_value = max(0, min(4095, encoder_value))

                        # Send position command directly to motor
                        packet = [
                            *self.arm.HEADER,              # 0xFF, 0xFF
                            motor_id,                       # Motor ID
                            0x05,                           # Packet length (Instr + Addr + 2 data + Checksum)
                            self.arm.INSTR_WRITE,           # Write instruction
                            self.arm.REG_GOAL_POSITION,     # 0x2A - Goal position register
                            encoder_value & 0xFF,           # Position low byte
                            (encoder_value >> 8) & 0xFF     # Position high byte
                        ]
                        checksum = self.arm._calculate_checksum(packet[2:])
                        packet.append(checksum)
                        self.arm.serial.write(bytes(packet))
                        time.sleep(0.005)  # Small delay between motors

            except Exception as e:
                pass  # Silently continue on errors

            time.sleep(0.05)  # 20 Hz control rate

    def get_positions_deg(self) -> np.ndarray:
        """Get current joint positions in degrees."""
        if not self.arm or not self.connected:
            return np.zeros(6)

        state = self.arm.get_state()
        return np.degrees(state.joint_positions)


# Global list of connected robots for cleanup
_connected_robots: List[RobotController] = []


def scan_so100_ports() -> List[str]:
    """
    Scan /dev/ttyACM0 through /dev/ttyACM9 for connected SO-100 robots.

    Returns:
        List[str]: List of ports with connected SO-100 robots
    """
    print("Scanning for SO-100 robots...")
    connected_ports = []

    for i in range(10):
        port = f"/dev/ttyACM{i}"
        if not os.path.exists(port):
            continue

        print(f"  Checking {port}...", end=" ", flush=True)

        try:
            arm = SO100Arm(port=port, baudrate=1000000, timeout=0.2)
            if arm.connect():
                print("[OK] SO-100 found")
                connected_ports.append(port)
                arm.disconnect()
            else:
                print("[--] No response")
        except Exception as e:
            print(f"[--] Error: {e}")

    print(f"\nFound {len(connected_ports)} SO-100 robot(s)")
    return connected_ports


def load_joint_configs_for_port(port: str, config_dir: Path = Path("data")) -> Tuple[List[JointConfig], str]:
    """
    Load joint configurations for a specific port.

    Looks for port-specific config first (e.g., so100_config_ttyACM0.csv),
    then falls back to generic config (so100_config.csv), then defaults.

    Args:
        port: Device port (e.g., /dev/ttyACM0)
        config_dir: Directory containing config files

    Returns:
        Tuple[List[JointConfig], str]: (configs, config_source)
            config_source is one of: "port-specific", "generic", "defaults"
    """
    configs = []

    # Try port-specific config first
    port_config_path = get_config_path_for_port(port, config_dir)
    generic_config_path = config_dir / "so100_config.csv"

    config_source = None
    config_path = None

    if port_config_path.exists():
        config_path = port_config_path
        config_source = "port-specific"
        print(f"  [{port}] Using port-specific config: {config_path}")
    elif generic_config_path.exists():
        config_path = generic_config_path
        config_source = "generic"
        print(f"  [{port}] Using generic config: {config_path}")
    else:
        config_source = "defaults"
        print(f"  [{port}] No config found, using defaults (home at midpoint)")
        # Default configs: full range, home at midpoint
        for _ in range(6):
            configs.append(JointConfig(
                min_rad=0.0,
                max_rad=2 * np.pi,
                home_rad=np.pi
            ))
        return configs, config_source

    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(JointConfig(
                min_rad=float(row['min_rad']),
                max_rad=float(row['max_rad']),
                home_rad=float(row['home_rad'])
            ))

    if len(configs) != 6:
        raise ValueError(f"Expected 6 joint configs, got {len(configs)}")

    return configs, config_source


def load_joint_configs(config_path: Path) -> List[JointConfig]:
    """
    Load joint configurations from calibration CSV (legacy function).

    Args:
        config_path: Path to so100_config.csv

    Returns:
        List[JointConfig]: Configuration for each of 6 joints
    """
    configs = []

    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}")
        print("       Using default home positions (midpoint of range)")
        # Default configs: full range, home at midpoint
        for _ in range(6):
            configs.append(JointConfig(
                min_rad=0.0,
                max_rad=2 * np.pi,
                home_rad=np.pi
            ))
        return configs

    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(JointConfig(
                min_rad=float(row['min_rad']),
                max_rad=float(row['max_rad']),
                home_rad=float(row['home_rad'])
            ))

    if len(configs) != 6:
        raise ValueError(f"Expected 6 joint configs, got {len(configs)}")

    return configs


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


class KeyboardInput:
    """Non-blocking keyboard input handler for Linux terminals."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        """Set terminal to raw mode for character-by-character input."""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get a single keypress with timeout.

        Returns:
            str: The key pressed, or None if timeout
            Special keys: 'ESC' for escape, 'ENTER' for enter
        """
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            char = sys.stdin.read(1)
            if char == '\x1b':  # Escape sequence
                # Check for arrow keys or plain escape
                rlist2, _, _ = select.select([sys.stdin], [], [], 0.01)
                if rlist2:
                    # Arrow key or other escape sequence - consume it
                    sys.stdin.read(2)
                    return None
                return 'ESC'
            elif char == '\n' or char == '\r':
                return 'ENTER'
            elif char.isdigit():
                return char
            else:
                return char
        return None


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
        control_mode = "PID" if robot.use_adaptive_pid else "Direct"
        row = f"{robot.port:<20}{robot.config_source:<15}{torque_status:<10}{control_mode:<12}"
        for pos in positions:
            row += f"{pos:>8.1f}"
        lines.append(row)

    lines.append("-" * 115)
    return lines


def run_interactive_menu(robots: List[RobotController]):
    """
    Run interactive menu system for Stage 2.

    Features:
    - Main menu with status table
    - Exit option
    - Tele-op Leader/Follower mode
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
                run_teleop_leader_follower(robots, keyboard)
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


def run_teleop_leader_follower(robots: List[RobotController], keyboard: KeyboardInput):
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


def run_adjust_home_positions(robots: List[RobotController], keyboard: KeyboardInput):
    """
    Run Home-to-Home tele-op mode for adjusting home positions.

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


def cleanup_handler(signum, frame):
    """Handle cleanup on exit signal."""
    print("\n\nReceived exit signal, releasing torque...")
    for robot in _connected_robots:
        robot.disconnect()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="SO-100 Teleoperation with Adaptive PID Control"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="data",
        help="Directory containing SO-100 config CSVs (default: data)"
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable Adaptive PID controller (default: direct position control)"
    )

    args = parser.parse_args()
    config_dir = Path(args.config_dir)

    # Register signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    print("=" * 80)
    print("SO-100 Teleoperation with Adaptive PID Control")
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
    control_mode = "Adaptive PID" if args.adaptive else "Direct Position"
    print(f"Control mode: {control_mode}")

    robots: List[RobotController] = []
    for port in connected_ports:
        try:
            # Load port-specific config (or fall back to generic/defaults)
            joint_configs, config_source = load_joint_configs_for_port(port, config_dir)

            robot = RobotController(port, joint_configs, config_source, use_adaptive_pid=args.adaptive)
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

    # Wait for user to proceed
    print("\n" + "-" * 80)
    input("Press ENTER to continue to Interactive Menu...")

    # Run interactive menu (handles homing and tele-op)
    try:
        run_interactive_menu(robots)
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
