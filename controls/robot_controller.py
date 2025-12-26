"""
SO-100 Robot Controller with Joint Stability System

Provides high-level control interface for SO-100 robotic arms with:
- Port-specific configuration loading from CSV files
- Direct position control with joint-level stability features
- Stability system for joints 0 and 1 (deadband + smoothing + dynamic torque)
- Multi-robot support with automatic port scanning

Joint Stability System (Joints 0 and 1):
    The base joints (Joint 0 and Joint 1) have mechanical compliance and inertia
    that causes oscillation with naive position control. This is mitigated by:

    1. Low-pass filtering of target positions (5% new, 95% old)
    2. Deadband region (0.1 rad / ~5.7°) where torque is disabled
    3. Dynamic torque enable/disable based on position error
    4. Speed limiting to 50 steps/sec (out of 0-1500 range)

    This three-layer approach prevents the servo's internal controller
    from hunting around the target position, eliminating oscillation.

Configuration System:
    Supports three-tier fallback for robot configuration:
    1. Port-specific: data/so100_config_ttyACM0.csv (calibrated for specific robot)
    2. Generic: data/so100_config.csv (shared across robots)
    3. Defaults: Full range [0, 2π] with home at π (midpoint)

    Only robots with port-specific configs should have torque enabled for safety.

Usage:
    from controls.robot_controller import RobotController, load_joint_configs_for_port

    # Load config and create controller
    configs, source = load_joint_configs_for_port("/dev/ttyACM0")
    robot = RobotController("/dev/ttyACM0", configs, source)

    # Connect and enable
    robot.connect()
    robot.enable_torque()
    robot.set_home_targets()
    robot.start_control_loop()
"""

import csv
import os
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

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
    """
    Get config file path for a specific port.

    Args:
        port: Device port (e.g., /dev/ttyACM0)
        config_dir: Directory containing config files

    Returns:
        Path: Path to port-specific config file
    """
    port_name = get_port_name(port)
    return config_dir / f"so100_config_{port_name}.csv"


class RobotController:
    """Controller for a single SO-100 robot with direct position control."""

    # Joint stability parameters (base joints oscillation fix)
    JOINT_0_DEADBAND = 0.1  # radians (~5.7 degrees)
    JOINT_0_SMOOTHING_ALPHA = 0.05  # Low-pass filter coefficient (only 5% new target)
    JOINT_0_SPEED_LIMIT = 50  # steps/sec (very slow to prevent hunting)

    JOINT_1_DEADBAND = 0.15  # radians (~8.6 degrees) - wider deadband
    JOINT_1_SMOOTHING_ALPHA = 0.2  # Low-pass filter coefficient (20% new target) - faster convergence
    JOINT_1_SPEED_LIMIT = 200  # steps/sec (faster than joint 0)

    JOINT_2_TO_5_SPEED_LIMIT = 500  # steps/sec for joints 2-5 (J3-J6)

    # Joint safety trigger parameters (stuck-joint detection)
    # CRITICAL: Must trigger BEFORE motor board releases all torques on failure
    SAFETY_ERROR_THRESHOLD = 0.2  # radians (~11 degrees) - sensitive threshold
    SAFETY_TIME_THRESHOLD = 0.5   # seconds - fast trigger to prevent board failure
    SAFETY_RECOVERY_TIME = 1.0    # seconds - error must stay low this long to auto-reset
    SAFETY_ENABLED_JOINTS = [5]   # J6 gripper only (for now)

    def __init__(self, port: str, joint_configs: List[JointConfig], config_source: str = "defaults",
                 enable_stability_system: bool = True):
        """
        Initialize robot controller.

        Args:
            port: Serial port device path (e.g., /dev/ttyACM0)
            joint_configs: List of 6 JointConfig objects (one per joint)
            config_source: Source of config ("port-specific", "generic", or "defaults")
            enable_stability_system: Enable joint 0/1 stability features (deadband, smoothing, speed limit)
        """
        self.port = port
        self.arm: Optional[SO100Arm] = None
        self.joint_configs = joint_configs
        self.config_source = config_source  # "port-specific", "generic", or "defaults"
        self.enable_stability_system = enable_stability_system
        self.target_positions = np.zeros(6)
        self.last_sent_positions = np.zeros(6)  # Track last sent positions for joint 0 stability
        self.connected = False
        self.control_thread: Optional[threading.Thread] = None
        self.running = False
        self.torque_enabled = False

        # Safety trigger state (stuck-joint detection)
        self.safety_error_start_time = {}    # {joint_idx: start_time or None}
        self.safety_triggered = {}            # {joint_idx: bool}
        self.safety_recovery_start_time = {}  # {joint_idx: start_time or None}

    def connect(self) -> bool:
        """
        Connect to the robot.

        Returns:
            bool: True if connection successful
        """
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

    def enable_torque(self) -> bool:
        """
        Enable torque on all motors.

        Returns:
            bool: True if successful
        """
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

    def release_torque(self) -> bool:
        """
        Disable torque on all motors with retry logic.

        Returns:
            bool: True if torque was successfully released
        """
        if not self.arm or not self.connected:
            return False

        max_retries = 3
        for attempt in range(max_retries):
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
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{self.port}] Torque release attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.1)
                else:
                    print(f"")
                    print(f"[!!!] CRITICAL: Failed to release torque on {self.port}: {e}")
                    print(f"[!!!] SAFETY WARNING: Robot may still have torque enabled!")
                    print(f"[!!!] Power cycle the robot to release torque.")
                    print(f"")
                    return False

        return False

    def _check_safety_trigger(self, joint_idx: int, current_pos: float, target_pos: float) -> bool:
        """
        Check if joint safety trigger should fire.

        Monitors position error over time for configured joints.
        If error exceeds threshold for too long, triggers protective action.
        Auto-resets when error stays within bounds for SAFETY_RECOVERY_TIME.

        Args:
            joint_idx: Joint index (0-5)
            current_pos: Current joint position in radians
            target_pos: Target joint position in radians

        Returns:
            bool: True if safety triggered (caller should clamp + disable)
        """
        if joint_idx not in self.SAFETY_ENABLED_JOINTS:
            return False

        error = abs(current_pos - target_pos)

        if error > self.SAFETY_ERROR_THRESHOLD:
            # Error exceeds threshold
            self.safety_recovery_start_time[joint_idx] = None  # Reset recovery timer

            if not self.safety_triggered.get(joint_idx, False):
                # Not yet triggered - start/check error timer
                if joint_idx not in self.safety_error_start_time or self.safety_error_start_time[joint_idx] is None:
                    self.safety_error_start_time[joint_idx] = time.time()

                elapsed = time.time() - self.safety_error_start_time[joint_idx]
                if elapsed > self.SAFETY_TIME_THRESHOLD:
                    self.safety_triggered[joint_idx] = True
                    return True
            else:
                # Already triggered - stay triggered
                return True
        else:
            # Error within bounds
            self.safety_error_start_time[joint_idx] = None  # Reset error timer

            if self.safety_triggered.get(joint_idx, False):
                # Currently triggered - check recovery timer
                if joint_idx not in self.safety_recovery_start_time or self.safety_recovery_start_time[joint_idx] is None:
                    self.safety_recovery_start_time[joint_idx] = time.time()

                recovery_elapsed = time.time() - self.safety_recovery_start_time[joint_idx]
                if recovery_elapsed > self.SAFETY_RECOVERY_TIME:
                    # Error stayed low long enough - auto-reset
                    self.safety_triggered[joint_idx] = False
                    self.safety_recovery_start_time[joint_idx] = None
                    return False
                else:
                    # Still in recovery period - stay triggered
                    return True

        return False

    def reset_safety_trigger(self, joint_idx: int = None):
        """
        Reset safety trigger state (call after manual intervention).

        Args:
            joint_idx: Specific joint to reset, or None to reset all
        """
        if joint_idx is None:
            self.safety_error_start_time.clear()
            self.safety_triggered.clear()
            self.safety_recovery_start_time.clear()
        else:
            self.safety_error_start_time[joint_idx] = None
            self.safety_triggered[joint_idx] = False
            self.safety_recovery_start_time[joint_idx] = None

    def set_target_positions(self, positions: np.ndarray):
        """
        Set target positions for all joints.

        Safety-triggered joints are skipped (their targets remain clamped
        until the safety trigger is reset).

        Args:
            positions: Array of 6 target positions in radians
        """
        for i, pos in enumerate(positions):
            # Don't update target for safety-triggered joints
            if not self.safety_triggered.get(i, False):
                self.target_positions[i] = pos

    def set_home_targets(self):
        """Set target positions to home positions from config."""
        for i, config in enumerate(self.joint_configs):
            self.target_positions[i] = config.home_rad

    def start_control_loop(self):
        """Start the control loop in a background thread."""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """Stop the control loop."""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

    def _control_loop(self):
        """
        Background control loop with direct position control.

        Runs at 20 Hz. When stability system is enabled, implements:
        - Low-pass filtering of target positions for joint 0
        - Deadband with dynamic torque disable for joint 0
        - Speed limiting for joints 0 and 1

        When stability system is disabled, all joints use simple position control.
        """
        first_iteration = True

        while self.running and self.connected:
            try:
                if self.enable_stability_system:
                    # Stability mode: special handling for joints 0 and 1
                    state = self.arm.get_state()
                    current_positions = state.joint_positions

                    # Initialize last_sent_positions on first iteration
                    if first_iteration:
                        self.last_sent_positions = current_positions.copy()
                        first_iteration = False

                    for motor_id, target_rad in enumerate(self.target_positions, start=1):
                        joint_idx = motor_id - 1  # Convert to 0-indexed

                        # Check safety trigger for stuck joints (e.g., gripper holding object)
                        if self._check_safety_trigger(joint_idx, current_positions[joint_idx], target_rad):
                            # Safety triggered - clamp target to current position
                            self.target_positions[joint_idx] = current_positions[joint_idx]
                            target_rad = current_positions[joint_idx]

                            # Disable torque on this joint to prevent damage
                            packet = [
                                *self.arm.HEADER,
                                motor_id,
                                0x04,
                                self.arm.INSTR_WRITE,
                                self.arm.REG_TORQUE_ENABLE,
                                0x00  # Disable torque
                            ]
                            checksum = self.arm._calculate_checksum(packet[2:])
                            packet.append(checksum)
                            self.arm.serial.write(bytes(packet))
                            continue  # Skip position update for this joint

                        # Apply smoothing and deadband to joint 0 only
                        if joint_idx == 0:
                            # Joint 0: Low-pass filter and deadband
                            smoothed_target = (self.JOINT_0_SMOOTHING_ALPHA * target_rad +
                                              (1 - self.JOINT_0_SMOOTHING_ALPHA) * self.last_sent_positions[joint_idx])
                            # Check deadband against ACTUAL target, not smoothed target
                            # (smoothed target stays close to current position due to heavy filtering)
                            target_error = abs(current_positions[joint_idx] - target_rad)
                            deadband = self.JOINT_0_DEADBAND

                            if target_error < deadband:
                                # Disable torque - stops oscillation completely
                                packet = [
                                    *self.arm.HEADER,
                                    motor_id,
                                    0x04,
                                    self.arm.INSTR_WRITE,
                                    self.arm.REG_TORQUE_ENABLE,  # 0x28
                                    0x00  # Disable torque
                                ]
                                checksum = self.arm._calculate_checksum(packet[2:])
                                packet.append(checksum)
                                self.arm.serial.write(bytes(packet))
                                continue  # Skip position update

                            # Outside deadband - ensure torque is enabled
                            packet = [
                                *self.arm.HEADER,
                                motor_id,
                                0x04,
                                self.arm.INSTR_WRITE,
                                self.arm.REG_TORQUE_ENABLE,  # 0x28
                                0x01  # Enable torque
                            ]
                            checksum = self.arm._calculate_checksum(packet[2:])
                            packet.append(checksum)
                            self.arm.serial.write(bytes(packet))
                            time.sleep(0.005)

                            final_target = smoothed_target
                        else:
                            # Other joints: no stability system
                            final_target = target_rad

                        # Convert radians to encoder counts (0-4095)
                        encoder_value = int((final_target / (2 * np.pi)) * 4096) % 4096
                        encoder_value = max(0, min(4095, encoder_value))

                        # Determine speed limit for this joint
                        if joint_idx == 0:
                            speed_limit = self.JOINT_0_SPEED_LIMIT
                        elif joint_idx == 1:
                            speed_limit = self.JOINT_1_SPEED_LIMIT
                        else:
                            speed_limit = self.JOINT_2_TO_5_SPEED_LIMIT

                        # Write position
                        packet = [
                            *self.arm.HEADER,
                            motor_id,
                            0x05,
                            self.arm.INSTR_WRITE,
                            self.arm.REG_GOAL_POSITION,  # 0x2A
                            encoder_value & 0xFF,
                            (encoder_value >> 8) & 0xFF
                        ]
                        checksum = self.arm._calculate_checksum(packet[2:])
                        packet.append(checksum)
                        self.arm.serial.write(bytes(packet))
                        time.sleep(0.005)

                        # Write speed limit
                        packet = [
                            *self.arm.HEADER,
                            motor_id,
                            0x05,
                            self.arm.INSTR_WRITE,
                            0x2E,  # Speed register
                            speed_limit & 0xFF,
                            (speed_limit >> 8) & 0xFF
                        ]
                        checksum = self.arm._calculate_checksum(packet[2:])
                        packet.append(checksum)
                        self.arm.serial.write(bytes(packet))

                        time.sleep(0.005)  # Small delay between motors
                        self.last_sent_positions[joint_idx] = final_target

                else:
                    # Simple mode: all joints treated equally
                    # Still need current positions for safety checks
                    state = self.arm.get_state()
                    current_positions = state.joint_positions

                    for motor_id, target_rad in enumerate(self.target_positions, start=1):
                        joint_idx = motor_id - 1

                        # Check safety trigger for stuck joints
                        if self._check_safety_trigger(joint_idx, current_positions[joint_idx], target_rad):
                            # Safety triggered - clamp and disable
                            self.target_positions[joint_idx] = current_positions[joint_idx]
                            packet = [
                                *self.arm.HEADER,
                                motor_id,
                                0x04,
                                self.arm.INSTR_WRITE,
                                self.arm.REG_TORQUE_ENABLE,
                                0x00
                            ]
                            checksum = self.arm._calculate_checksum(packet[2:])
                            packet.append(checksum)
                            self.arm.serial.write(bytes(packet))
                            continue

                        # Convert radians to encoder counts (0-4095)
                        encoder_value = int((target_rad / (2 * np.pi)) * 4096) % 4096
                        encoder_value = max(0, min(4095, encoder_value))

                        # Simple position command
                        packet = [
                            *self.arm.HEADER,
                            motor_id,
                            0x05,
                            self.arm.INSTR_WRITE,
                            self.arm.REG_GOAL_POSITION,  # 0x2A
                            encoder_value & 0xFF,
                            (encoder_value >> 8) & 0xFF
                        ]
                        checksum = self.arm._calculate_checksum(packet[2:])
                        packet.append(checksum)
                        self.arm.serial.write(bytes(packet))
                        time.sleep(0.005)

            except Exception as e:
                pass  # Silently continue on errors

            time.sleep(0.05)  # 20 Hz control rate

    def get_positions_deg(self) -> np.ndarray:
        """
        Get current joint positions in degrees.

        Returns:
            np.ndarray: Array of 6 joint positions in degrees
        """
        if not self.arm or not self.connected:
            return np.zeros(6)

        state = self.arm.get_state()
        return np.degrees(state.joint_positions)


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
