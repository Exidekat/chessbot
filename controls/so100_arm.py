"""
SO-100 Arm Control Module

Provides low-level control interface for the SO-100 robotic arm with 6 motors
and positional encoders. Communicates with the motor control board via serial.

Hardware Specifications:
- 6 motors with positional encoders
- Serial communication protocol
- Joint position control
- Real-time position feedback

Usage:
    from controls.so100_arm import SO100Arm

    arm = SO100Arm(port="/dev/ttyUSB0", baudrate=115200)
    arm.connect()

    # Get current joint positions
    positions = arm.get_joint_positions()

    # Move to target joint positions
    arm.move_joints([0.0, -0.5, 0.5, 0.0, 0.5, 0.0])

    # Get gripper state
    gripper_open = arm.get_gripper_state()

    # Control gripper
    arm.set_gripper(0.5)  # 0.0 = closed, 1.0 = open
"""

import serial
import struct
import time
import threading
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SO100State:
    """SO-100 arm state."""
    joint_positions: np.ndarray  # [6] joint angles in radians
    gripper_position: float  # 0.0 (closed) to 1.0 (open)
    timestamp: float
    is_moving: bool
    error_code: int


class SO100Arm:
    """
    SO-100 robotic arm control interface.

    Communicates with the motor control board via serial protocol.
    """

    # Protocol constants
    CMD_GET_POSITIONS = 0x01
    CMD_SET_POSITIONS = 0x02
    CMD_GET_GRIPPER = 0x03
    CMD_SET_GRIPPER = 0x04
    CMD_EMERGENCY_STOP = 0xFF

    # Joint limits (radians)
    JOINT_LIMITS = np.array([
        [-np.pi, np.pi],      # Joint 1 (base rotation)
        [-np.pi/2, np.pi/2],  # Joint 2 (shoulder)
        [-np.pi/2, np.pi/2],  # Joint 3 (elbow)
        [-np.pi, np.pi],      # Joint 4 (wrist roll)
        [-np.pi/2, np.pi/2],  # Joint 5 (wrist pitch)
        [-np.pi, np.pi]       # Joint 6 (wrist yaw)
    ])

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout: float = 1.0
    ):
        """
        Initialize SO-100 arm controller.

        Args:
            port: Serial port device path
            baudrate: Serial communication baud rate
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.serial: Optional[serial.Serial] = None
        self.connected = False

        # Current state
        self.state = SO100State(
            joint_positions=np.zeros(6),
            gripper_position=0.0,
            timestamp=0.0,
            is_moving=False,
            error_code=0
        )

        # State update thread
        self.state_lock = threading.Lock()
        self.state_thread: Optional[threading.Thread] = None
        self.running = False

    def connect(self) -> bool:
        """
        Connect to SO-100 motor control board.

        Returns:
            bool: True if connection successful
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Wait for board to initialize
            time.sleep(0.5)

            # Verify connection by reading current positions
            positions = self._read_joint_positions()
            if positions is not None:
                self.connected = True
                print(f"[SO100] Connected to {self.port}")
                print(f"[SO100] Current positions: {positions}")

                # Start state update thread
                self.running = True
                self.state_thread = threading.Thread(target=self._state_update_loop, daemon=True)
                self.state_thread.start()

                return True
            else:
                print(f"[SO100] Failed to verify connection")
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
                gripper_position=self.state.gripper_position,
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
        success = self._send_joint_positions(positions, speed)

        if success and blocking:
            # Wait for movement to complete
            self._wait_for_movement()

        return success

    def set_gripper(self, position: float, blocking: bool = False) -> bool:
        """
        Set gripper position.

        Args:
            position: Gripper position (0.0 = closed, 1.0 = open)
            blocking: If True, wait for movement to complete

        Returns:
            bool: True if command sent successfully
        """
        if not self.connected:
            print("[SO100] Not connected")
            return False

        # Clamp position
        position = np.clip(position, 0.0, 1.0)

        # Send command
        success = self._send_gripper_command(position)

        if success and blocking:
            time.sleep(0.5)  # Gripper movement time

        return success

    def get_gripper_state(self) -> float:
        """
        Get current gripper position.

        Returns:
            float: Gripper position (0.0 = closed, 1.0 = open)
        """
        with self.state_lock:
            return self.state.gripper_position

    def emergency_stop(self):
        """Immediately stop all movement."""
        if not self.connected:
            return

        # Send emergency stop command
        try:
            self.serial.write(bytes([self.CMD_EMERGENCY_STOP]))
            print("[SO100] EMERGENCY STOP")
        except serial.SerialException as e:
            print(f"[SO100] Emergency stop failed: {e}")

    def is_moving(self) -> bool:
        """
        Check if arm is currently moving.

        Returns:
            bool: True if moving
        """
        with self.state_lock:
            return self.state.is_moving

    # Private methods for serial communication

    def _read_joint_positions(self) -> Optional[np.ndarray]:
        """Read current joint positions from control board."""
        try:
            # Send read command
            self.serial.write(bytes([self.CMD_GET_POSITIONS]))

            # Read response (6 floats = 24 bytes)
            data = self.serial.read(24)
            if len(data) != 24:
                return None

            # Unpack positions
            positions = struct.unpack('6f', data)
            return np.array(positions, dtype=np.float32)

        except serial.SerialException as e:
            print(f"[SO100] Read positions failed: {e}")
            return None

    def _send_joint_positions(self, positions: np.ndarray, speed: float) -> bool:
        """Send target joint positions to control board."""
        try:
            # Pack command: CMD + 6 floats (positions) + 1 float (speed)
            data = struct.pack('B6ff', self.CMD_SET_POSITIONS, *positions, speed)
            self.serial.write(data)
            return True

        except serial.SerialException as e:
            print(f"[SO100] Send positions failed: {e}")
            return False

    def _send_gripper_command(self, position: float) -> bool:
        """Send gripper position command."""
        try:
            # Pack command: CMD + 1 float
            data = struct.pack('Bf', self.CMD_SET_GRIPPER, position)
            self.serial.write(data)
            return True

        except serial.SerialException as e:
            print(f"[SO100] Send gripper failed: {e}")
            return False

    def _state_update_loop(self):
        """Background thread to update arm state."""
        while self.running:
            # Read joint positions
            positions = self._read_joint_positions()

            if positions is not None:
                with self.state_lock:
                    # Check if moving (position changed significantly)
                    position_change = np.linalg.norm(positions - self.state.joint_positions)
                    self.state.is_moving = position_change > 0.01

                    self.state.joint_positions = positions
                    self.state.timestamp = time.time()

            # Read gripper state (less frequently)
            # TODO: Implement gripper state reading

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


def test_so100_connection():
    """Test SO-100 arm connection and basic commands."""
    print("=" * 60)
    print("SO-100 Arm Connection Test")
    print("=" * 60)

    arm = SO100Arm(port="/dev/ttyUSB0", baudrate=115200)

    print("\n1. Connecting...")
    if not arm.connect():
        print("[X] Connection failed")
        return

    print("\n2. Reading current state...")
    state = arm.get_state()
    print(f"   Joint positions: {state.joint_positions}")
    print(f"   Gripper: {state.gripper_position}")
    print(f"   Moving: {state.is_moving}")

    print("\n3. Testing small movement...")
    current = arm.get_joint_positions()
    target = current.copy()
    target[0] += 0.1  # Move base joint 0.1 radians
    print(f"   Target: {target}")

    if arm.move_joints(target, speed=0.3, blocking=True):
        print("[OK] Movement complete")
        final = arm.get_joint_positions()
        print(f"   Final position: {final}")
    else:
        print("[X] Movement failed")

    print("\n4. Testing gripper...")
    arm.set_gripper(1.0, blocking=True)  # Open
    print(f"   Gripper opened: {arm.get_gripper_state()}")

    arm.set_gripper(0.0, blocking=True)  # Close
    print(f"   Gripper closed: {arm.get_gripper_state()}")

    print("\n5. Disconnecting...")
    arm.disconnect()

    print("=" * 60)
    print("[OK] Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_so100_connection()
