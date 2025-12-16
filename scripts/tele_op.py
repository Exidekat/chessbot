#!/usr/bin/env python3
"""
SO-100 Teleoperation Script with Direct Position or Adaptive PID Control

Features:
1. Direct position control (default) or Adaptive PID (--adaptive flag)
2. Port-specific configuration loading with torque safety
3. Stage 1: Move all joints to home position from calibration config
4. Stage 2: Live status display of all connected SO-100 robots
5. Automatic torque release on exit

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


def display_status_table(robots: List[RobotController]):
    """
    Display live status table of all robots.
    Updates every second until interrupted.
    """
    print("\n" + "=" * 80)
    print("Stage 2: Live Status Display")
    print("=" * 80)
    print("Press Ctrl+C to exit and release torque\n")

    try:
        while True:
            # Build table header
            header = f"{'Robot Port':<20}{'Config':<15}{'Torque':<10}{'Mode':<12}"
            for j in range(1, 7):
                header += f"{'J' + str(j):>8}"

            # Clear and print
            clear_screen()
            print("=" * 115)
            print("SO-100 Teleoperation - Live Status")
            print("=" * 115)
            print("Press Ctrl+C to exit and release torque\n")
            print(header)
            print("-" * 115)

            # Print each robot's status
            for robot in robots:
                positions = robot.get_positions_deg()
                torque_status = "ON" if robot.torque_enabled else "OFF"
                control_mode = "PID" if robot.use_adaptive_pid else "Direct"
                row = f"{robot.port:<20}{robot.config_source:<15}{torque_status:<10}{control_mode:<12}"
                for pos in positions:
                    row += f"{pos:>8.1f}"
                print(row)

            print("-" * 115)
            print(f"\nLast updated: {time.strftime('%H:%M:%S')}")

            # Show PID gains only for enabled robots using adaptive PID
            enabled_adaptive_robots = [r for r in robots if r.torque_enabled and r.use_adaptive_pid]
            if enabled_adaptive_robots:
                print("\nPID Gains (Kp, Ki, Kd) for first enabled robot:")
                for i, pid in enumerate(enabled_adaptive_robots[0].pid_controllers):
                    kp, ki, kd = pid.get_gains()
                    print(f"  Joint {i}: Kp={kp:.3f}, Ki={ki:.4f}, Kd={kd:.3f}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nExiting...")


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

    # Stage 1: Move to home positions (only for robots with port-specific configs)
    print("\n" + "=" * 80)
    print("Stage 1: Moving to Home Positions")
    print("=" * 80)

    for robot in robots:
        if robot.config_source == "port-specific":
            print(f"\n{robot.port}: Port-specific config found")
            print(f"  Enabling torque on all 6 joints...")
            robot.enable_torque()
            print(f"  Setting home targets and starting PID control...")
            robot.set_home_targets()
            robot.start_control_loop()
        else:
            print(f"\n{robot.port}: No port-specific config ({robot.config_source})")
            print(f"  Torque DISABLED for safety - robot will be freely movable")
            robot.release_torque()

    # Wait for homing to complete (only for robots with torque enabled)
    enabled_robots = [r for r in robots if r.config_source == "port-specific"]
    if enabled_robots:
        print("\nRobots with port-specific configs are moving to home positions...")
        print("Waiting 5 seconds for settling...")
        time.sleep(5.0)

    # Show current positions
    print("\nCurrent positions after homing:")
    for robot in robots:
        positions = robot.get_positions_deg()
        status = "ENABLED" if robot.config_source == "port-specific" else "DISABLED"
        print(f"  {robot.port} [{status}]: {[f'{p:.1f}' for p in positions]}")

    # Wait for user to proceed
    print("\n" + "-" * 80)
    input("Press ENTER to continue to Stage 2 (Live Status Display)...")

    # Stage 2: Live status display
    try:
        display_status_table(robots)
    except Exception as e:
        print(f"Error in status display: {e}")
    finally:
        # Cleanup: stop control loops and release torque
        print("\nCleaning up...")
        for robot in robots:
            robot.disconnect()
        print("[OK] All robots disconnected and torque released")

    return 0


if __name__ == "__main__":
    sys.exit(main())
