"""
SO-100 Arm Control Module

Provides low-level control interface for the SO-100 robotic arm with 6 Feetech
STS3215 smart servos. Communicates via Feetech serial protocol.

Hardware Specifications:
- 6x Feetech STS3215 smart servos (motor IDs 1-6)
- Serial communication: 1000000 baud (1 Mbps)
- Position resolution: 4096 counts/revolution (12-bit)
- Position register address: 0x38 (56 decimal)
- Feetech protocol with checksum validation

Usage:
    from controls.so100_arm import SO100Arm

    arm = SO100Arm(port="/dev/ttyACM0", baudrate=1000000)
    arm.connect()

    # Get current joint positions (all 6 motors)
    positions = arm.get_joint_positions()

    # Move to target joint positions (all 6 motors)
    arm.move_joints([0.0, 1.5, 3.0, 2.0, 1.0, 0.5])
"""

import serial
import serial.tools.list_ports as list_ports
import struct
import time
import threading
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SO100State:
    """SO-100 arm state."""
    joint_positions: np.ndarray  # [6] motor angles in radians (motors 1-6)
    timestamp: float
    is_moving: bool
    error_code: int


class SO100Arm:
    """
    SO-100 robotic arm control interface.

    Communicates with Feetech STS3215 servos via serial protocol.
    """

    # Feetech STS3215 Protocol Constants
    HEADER = [0xFF, 0xFF]  # Packet header bytes
    INSTR_READ = 0x02       # Read instruction
    INSTR_WRITE = 0x03      # Write instruction

    # Register addresses
    REG_POSITION = 0x38       # Current position (2 bytes, read-only)
    REG_GOAL_POSITION = 0x2A  # Goal position (2 bytes, write)
    REG_TORQUE_ENABLE = 0x28  # Torque enable (1 byte)
    REG_ACCEL = 0x29          # Acceleration limit (1 byte)
    REG_GOAL_SPEED = 0x2E     # Goal speed (2 bytes, write)
    REG_TORQUE_LIMIT = 0x30   # Torque limit (2 bytes, write)

    # Motor IDs (1-6: motors 1-5 are arm joints, motor 6 is gripper)
    MOTOR_IDS = [1, 2, 3, 4, 5, 6]  # All 6 motors (5 arm + 1 gripper)
    NUM_MOTORS = 6                   # Total motors including gripper
    GRIPPER_INDEX = 5                # Gripper is index 5 in joint_positions array (0-indexed)

    # Position resolution: 4096 counts/revolution (12-bit encoder)
    COUNTS_PER_REV = 4096
    COUNTS_TO_RAD = 2 * np.pi / COUNTS_PER_REV  # Conversion factor
    RAD_TO_COUNTS = COUNTS_PER_REV / (2 * np.pi)

    # Joint limits (radians) - based on SO-100 mechanical constraints
    # These prevent self-collision and dangerous positions.
    # Port-specific configs loaded by RobotController may further restrict these.
    JOINT_LIMITS = np.array([
        [np.radians(70),  np.radians(290)],  # Joint 1 (base rotation) - ~220 deg range
        [np.radians(70),  np.radians(290)],  # Joint 2 (shoulder) - ~220 deg range
        [np.radians(70),  np.radians(290)],  # Joint 3 (elbow) - ~220 deg range
        [np.radians(70),  np.radians(290)],  # Joint 4 (wrist pitch) - ~220 deg range
        [np.radians(70),  np.radians(290)],  # Joint 5 (wrist roll) - ~220 deg range
        [np.radians(100), np.radians(260)],  # Joint 6 (gripper) - ~160 deg range
    ])

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baudrate: int = 1000000,  # 1 Mbps for Feetech protocol
        timeout: float = 0.1
    ):
        """
        Initialize SO-100 arm controller.

        Args:
            port: Serial port device path (e.g., /dev/ttyACM0)
            baudrate: Serial communication baud rate (default: 1000000)
            timeout: Serial read timeout in seconds (default: 0.1)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.serial: Optional[serial.Serial] = None
        self.connected = False

        # Lock for serial port access -- shared with RobotController via self.serial_lock
        self.serial_lock = threading.Lock()

        # Current state
        self.state = SO100State(
            joint_positions=np.zeros(6),  # All 6 motors
            timestamp=0.0,
            is_moving=False,
            error_code=0
        )

        # State update thread
        self.state_lock = threading.Lock()
        self.state_thread: Optional[threading.Thread] = None
        self.running = False

        # Track consecutive read failures for disconnect detection
        self._consecutive_read_failures = 0
        self._max_read_failures = 10  # Disconnect after this many consecutive failures

    def connect(self) -> bool:
        """
        Connect to SO-100 via Feetech protocol.

        Returns:
            bool: True if connection successful
        """
        try:
            # Validate port exists
            available_ports = [port.device for port in list_ports.comports()]
            if self.port not in available_ports:
                print(f"[SO100] Port not found: {self.port}")
                print(f"[SO100] Available ports: {available_ports}")
                return False

            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Wait for serial to initialize
            time.sleep(0.1)

            # Verify connection by reading positions from all motors
            print(f"[SO100] Verifying connection to {self.port}...")

            # Try multiple times with delays (motors may need time to respond)
            positions = None
            for attempt in range(3):
                verbose = (attempt == 2)  # Show details on last attempt
                positions = self._read_all_motor_positions(verbose=verbose)
                if positions is not None and len(positions) == self.NUM_MOTORS:
                    break
                print(f"[SO100] Read attempt {attempt + 1}/3 failed, retrying...")
                time.sleep(0.2)

            if positions is not None and len(positions) == self.NUM_MOTORS:
                self.connected = True
                print(f"[SO100] Connected successfully")
                print(f"[SO100] Motor positions (counts): {[int(p) for p in positions]}")
                print(f"[SO100] Motor positions (radians): {[f'{p * self.COUNTS_TO_RAD:.3f}' for p in positions]}")

                # Start state update thread
                self.running = True
                self.state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
                self.state_thread.start()

                return True
            else:
                print(f"[SO100] Failed to verify connection - could not read motor positions")
                self.serial.close()
                return False

        except serial.SerialException as e:
            print(f"[SO100] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from SO-100 motor control board."""
        self.running = False

        if self.state_thread:
            self.state_thread.join(timeout=2.0)

        if self.serial and self.serial.is_open:
            self.serial.close()

        self.connected = False
        print("[SO100] Disconnected")

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.

        Returns:
            np.ndarray: Joint positions in radians [6]
        """
        with self.state_lock:
            return self.state.joint_positions.copy()

    def get_state(self) -> SO100State:
        """
        Get complete arm state.

        Returns:
            SO100State: Current arm state
        """
        with self.state_lock:
            return SO100State(
                joint_positions=self.state.joint_positions.copy(),
                timestamp=self.state.timestamp,
                is_moving=self.state.is_moving,
                error_code=self.state.error_code
            )

    def move_joints(
        self,
        target_positions: List[float],
        speed: float = 1.0,
        blocking: bool = False
    ) -> bool:
        """
        Move to target joint positions.

        Args:
            target_positions: Target joint angles in radians [6]
            speed: Movement speed multiplier (0.0-1.0)
            blocking: If True, wait for movement to complete

        Returns:
            bool: True if command sent successfully
        """
        if not self.connected:
            print("[SO100] Not connected")
            return False

        # Validate positions
        positions = np.array(target_positions, dtype=np.float32)
        if len(positions) != 6:
            print(f"[SO100] Invalid number of joints: {len(positions)} (expected 6)")
            return False

        # Check joint limits
        for i, (pos, limits) in enumerate(zip(positions, self.JOINT_LIMITS)):
            if pos < limits[0] or pos > limits[1]:
                print(f"[SO100] Joint {i} position {pos:.3f} out of bounds {limits}")
                return False

        # Clamp speed
        speed = np.clip(speed, 0.0, 1.0)

        # Send command
        success = self._send_joint_positions(positions)

        if success and blocking:
            # Wait for movement to complete
            self._wait_for_movement()

        return success


    def emergency_stop(self):
        """Immediately stop all movement by disabling torque on all motors."""
        if not self.connected or not self.serial:
            return

        # Feetech protocol has no single emergency stop byte.
        # Disable torque on every motor as fast as possible.
        with self.serial_lock:
            for motor_id in self.MOTOR_IDS:
                try:
                    packet = [
                        *self.HEADER, motor_id, 0x04,
                        self.INSTR_WRITE, self.REG_TORQUE_ENABLE, 0x00,
                    ]
                    checksum = self._calculate_checksum(packet[2:])
                    packet.append(checksum)
                    self.serial.write(bytes(packet))
                except serial.SerialException:
                    pass  # Best-effort: keep trying remaining motors
        print("[SO100] EMERGENCY STOP -- torque disabled on all motors")

    def is_moving(self) -> bool:
        """
        Check if arm is currently moving.

        Returns:
            bool: True if moving
        """
        with self.state_lock:
            return self.state.is_moving

    # Private methods for Feetech protocol communication

    def _calculate_checksum(self, data: List[int]) -> int:
        """
        Calculate Feetech protocol checksum.

        Args:
            data: Packet data (from ID onwards, excluding header)

        Returns:
            int: Checksum byte (one's complement of sum)
        """
        return (~sum(data)) & 0xFF

    def _read_motor_position(self, motor_id: int, verbose: bool = False) -> Optional[int]:
        """
        Read current position from a single motor using Feetech protocol.

        Args:
            motor_id: Motor ID (1-6)
            verbose: Print debug info on failures

        Returns:
            Optional[int]: Position in counts (0-4095), or None if read failed
        """
        try:
            # Construct read packet: [FF FF ID Len Instr Addr DataLen Checksum]
            packet = [
                *self.HEADER,           # 0xFF, 0xFF
                motor_id,                # Motor ID
                0x04,                    # Packet length
                self.INSTR_READ,         # Read instruction
                self.REG_POSITION,       # Position register address
                0x02                     # Data length (2 bytes)
            ]

            # Calculate and append checksum (from ID onwards)
            checksum = self._calculate_checksum(packet[2:])
            packet.append(checksum)

            # All serial I/O under lock to prevent interleaving with control loop
            with self.serial_lock:
                self.serial.reset_input_buffer()
                self.serial.write(bytes(packet))

                # Wait for response
                time.sleep(0.005)  # 5ms delay

                # Read response: [FF FF ID Len Err P_L P_H Checksum]
                if self.serial.in_waiting > 0:
                    response = self.serial.read(self.serial.in_waiting)
                else:
                    response = b''

            if len(response) > 0:

                # Validate response length (minimum 8 bytes for full response)
                if len(response) >= 8:
                    # Check error byte (bit flags)
                    # 0x01 = Voltage Error (warning, position still valid)
                    # 0x04 = Overheating (warning, position still valid)
                    # 0x10 = Checksum Error (fatal - data corrupted)
                    # 0x40 = Instruction Error (fatal - wrong command)
                    error_code = response[4]
                    fatal_errors = 0x10 | 0x40  # Checksum or Instruction errors
                    if error_code & fatal_errors:
                        if verbose:
                            print(f"[SO100] Motor {motor_id}: Fatal error 0x{error_code:02X}, raw bytes: {[hex(b) for b in response]}")
                        return None
                    if error_code != 0 and verbose:
                        print(f"[SO100] Motor {motor_id}: Warning error 0x{error_code:02X} (continuing)")

                    # Extract position (2 bytes: low, high)
                    pos_low = response[5]
                    pos_high = response[6]
                    position = (pos_high << 8) | pos_low

                    # Validate position is in valid range (0-4095 for 12-bit encoder)
                    if position > self.COUNTS_PER_REV - 1:
                        if verbose:
                            print(f"[SO100] Motor {motor_id}: Invalid position {position} (0x{position:04X}), raw bytes: {[hex(b) for b in response]}")
                        return None

                    return position
                else:
                    if verbose:
                        print(f"[SO100] Motor {motor_id}: Short response ({len(response)} bytes): {[hex(b) for b in response]}")
            else:
                if verbose:
                    print(f"[SO100] Motor {motor_id}: No response (0 bytes in buffer)")

            return None

        except (serial.SerialException, IndexError) as e:
            if verbose:
                print(f"[SO100] Motor {motor_id}: Exception: {e}")
            return None

    def _read_all_motor_positions(self, verbose: bool = False) -> Optional[np.ndarray]:
        """
        Read positions from all motors.

        Args:
            verbose: If True, print detailed debug info

        Returns:
            Optional[np.ndarray]: Array of positions in counts [6], or None if any read failed
        """
        positions = []

        for motor_id in self.MOTOR_IDS:
            pos = self._read_motor_position(motor_id, verbose=verbose)
            if pos is None:
                if verbose:
                    print(f"[SO100] Failed to read motor {motor_id}")
                return None  # Fail if any motor doesn't respond
            positions.append(pos)
            if verbose:
                print(f"[SO100] Motor {motor_id}: {pos} counts ({pos * self.COUNTS_TO_RAD:.3f} rad)")

        return np.array(positions, dtype=np.float32)

    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """
        Read current joint positions in radians.

        Returns:
            Optional[np.ndarray]: Joint positions in radians [6], or None if read failed
        """
        positions_counts = self._read_all_motor_positions()

        if positions_counts is not None:
            # Convert counts to radians
            positions_rad = positions_counts * self.COUNTS_TO_RAD
            return positions_rad

        return None

    def _write_motor_position(self, motor_id: int, position_counts: int) -> bool:
        """
        Write goal position to a single motor using Feetech protocol.

        Args:
            motor_id: Motor ID (1-6)
            position_counts: Target position in counts (0-4095)

        Returns:
            bool: True if write successful
        """
        try:
            # Clamp position to valid range
            position_counts = int(np.clip(position_counts, 0, self.COUNTS_PER_REV - 1))

            # Split position into low and high bytes
            pos_low = position_counts & 0xFF
            pos_high = (position_counts >> 8) & 0xFF

            # Construct write packet: [FF FF ID Len Instr Addr P_L P_H Checksum]
            packet = [
                *self.HEADER,              # 0xFF, 0xFF
                motor_id,                   # Motor ID
                0x05,                       # Packet length
                self.INSTR_WRITE,           # Write instruction
                self.REG_GOAL_POSITION,     # Goal position register
                pos_low,                    # Position low byte
                pos_high                    # Position high byte
            ]

            # Calculate and append checksum
            checksum = self._calculate_checksum(packet[2:])
            packet.append(checksum)

            # Send packet (under serial lock)
            with self.serial_lock:
                self.serial.write(bytes(packet))
            return True

        except (serial.SerialException, ValueError) as e:
            print(f"[SO100] Write motor {motor_id} failed: {e}")
            return False

    def _send_joint_positions(self, positions: np.ndarray) -> bool:
        """
        Send target joint positions to all motors.

        Args:
            positions: Target joint positions in radians [6]

        Returns:
            bool: True if all writes successful
        """
        try:
            # Convert radians to counts
            positions_counts = (positions * self.RAD_TO_COUNTS).astype(int)

            # Write to each motor
            for motor_id, pos_counts in zip(self.MOTOR_IDS, positions_counts):
                success = self._write_motor_position(motor_id, pos_counts)
                if not success:
                    return False
                time.sleep(0.002)  # Small delay between writes

            return True

        except Exception as e:
            print(f"[SO100] Send positions failed: {e}")
            return False


    def _state_update_loop(self):
        """Background thread to update arm state."""
        while self.running:
            # Read joint positions (all 6 motors)
            positions = self._read_joint_positions()

            if positions is not None:
                self._consecutive_read_failures = 0
                with self.state_lock:
                    # Check if moving (position changed significantly)
                    position_change = np.linalg.norm(positions - self.state.joint_positions)
                    self.state.is_moving = position_change > 0.01

                    self.state.joint_positions = positions
                    self.state.timestamp = time.time()
            else:
                self._consecutive_read_failures += 1
                if self._consecutive_read_failures >= self._max_read_failures:
                    print(f"[SO100] Lost connection ({self._consecutive_read_failures} consecutive read failures)")
                    self.connected = False
                    break

            time.sleep(0.05)  # 20 Hz update rate

    def _wait_for_movement(self, timeout: float = 10.0):
        """Wait for arm movement to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if not self.is_moving():
                time.sleep(0.1)  # Extra settling time
                return

            time.sleep(0.05)

        print("[SO100] Movement timeout")


def test_so100_connection(port: str = "/dev/ttyACM0"):
    """
    Test SO-100 arm connection and basic commands.

    Args:
        port: Serial port device (default: /dev/ttyACM0)
    """
    print("=" * 60)
    print("SO-100 Arm Connection Test (Feetech Protocol)")
    print("=" * 60)

    # Create arm instance with Feetech settings
    arm = SO100Arm(port=port, baudrate=1000000, timeout=0.1)

    print("\n1. Connecting...")
    if not arm.connect():
        print("[X] Connection failed")
        print("\nTroubleshooting:")
        print("  - Check that SO-100 is powered on")
        print("  - Verify correct serial port (use 'ls /dev/ttyACM* /dev/ttyUSB*')")
        print("  - Ensure user has serial port permissions (add to 'dialout' group)")
        return

    print("\n2. Reading current state...")
    state = arm.get_state()
    print(f"   Joint positions (rad): {state.joint_positions}")
    print(f"   Joint positions (deg): {np.degrees(state.joint_positions)}")
    print(f"   Gripper (joint 5): {np.degrees(state.joint_positions[arm.GRIPPER_INDEX]):.1f} deg")
    print(f"   Moving: {state.is_moving}")

    print("\n3. Testing small movement...")
    print("   WARNING: Arm will move! Ensure workspace is clear.")
    input("   Press ENTER to continue or Ctrl+C to abort...")

    current = arm.get_joint_positions()
    target = current.copy()
    target[0] += 0.05  # Move base joint 0.05 radians (~3 degrees)
    print(f"   Current: {np.degrees(current)} deg")
    print(f"   Target:  {np.degrees(target)} deg")

    if arm.move_joints(target, speed=0.3, blocking=True):
        print("[OK] Movement complete")
        final = arm.get_joint_positions()
        print(f"   Final:   {np.degrees(final)} deg")
    else:
        print("[X] Movement failed")

    arm.disconnect()

    print("=" * 60)
    print("[OK] Test complete")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Get port from command line or use default
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    test_so100_connection(port)
