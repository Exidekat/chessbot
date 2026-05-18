"""
Chesster Arm Control Module

Low-level control for the Chesster arm: 5 Feetech STS3215 smart servos
(4 arm joints + 1 gripper). Identical serial protocol to the SO-101
(see controls/so100_arm.py); the differences are motor count, joint
layout (wrist_roll removed), and joint limits.

Motor IDs:
    1 = shoulder_pan
    2 = shoulder_lift
    3 = elbow_flex
    4 = wrist_flex
    5 = gripper

Usage:
    from controls.chesster_arm import ChessterArm

    arm = ChessterArm(port="/dev/ttyACM0", baudrate=1000000)
    arm.connect()
    positions = arm.get_joint_positions()  # 5-element array
    arm.move_joints([0.0, 1.5, 3.0, 2.0, 0.5])
"""

from __future__ import annotations

import numpy as np

from controls.so100_arm import SO100Arm, SO100State


class ChessterArm(SO100Arm):
    """Chesster arm controller (5 motors). Inherits Feetech serial protocol
    from SO100Arm and overrides the motor/joint constants."""

    # Physical wiring on Chesster, verified empirically by wiggling each
    # motor: motor 2 is the gripper (NOT an arm joint), motor 5 is the
    # shoulder_lift (NOT the gripper). See controls/chesster_kinematics
    # JOINT_NAMES_BY_INDEX for the full mapping. joint_positions follows
    # motor-ID order.
    MOTOR_IDS = [1, 2, 3, 4, 5]
    NUM_MOTORS = 5
    GRIPPER_INDEX = 1  # joint_positions[1] = motor ID 2 = gripper

    # Joint limits (radians) per motor index -- ordering MUST match
    # JOINT_NAMES_BY_INDEX. Arm joints get the wide Feetech encoder
    # range minus a 5-degree wraparound margin; the gripper gets a
    # tighter range matching its recorded mechanical span.
    JOINT_LIMITS = np.array([
        [np.radians(5),   np.radians(355)],  # motor 1: elbow_flex
        [np.radians(80),  np.radians(310)],  # motor 2: gripper
        [np.radians(5),   np.radians(355)],  # motor 3: shoulder_pan
        [np.radians(5),   np.radians(355)],  # motor 4: wrist_flex
        [np.radians(5),   np.radians(355)],  # motor 5: shoulder_lift
    ])

    def __init__(self, port: str = "/dev/ttyACM0",
                 baudrate: int = 1000000, timeout: float = 0.1):
        # Reproduces SO100Arm.__init__ but resizes the state array to NUM_MOTORS=5.
        # SO100Arm hardcodes np.zeros(6), so direct super().__init__() then
        # in-place resize would be brittle; mirror the body instead.
        import serial as _serial  # noqa: F401  (validates pyserial install)
        import threading

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.serial = None
        self.connected = False
        self.serial_lock = threading.Lock()

        self.state = SO100State(
            joint_positions=np.zeros(self.NUM_MOTORS),
            timestamp=0.0,
            is_moving=False,
            error_code=0,
        )

        # State update thread
        self.state_lock = threading.Lock()
        self.state_thread = None
        self.running = False

        # Disconnect detection. Bumped from SO100Arm's default of 10 to 50
        # because Chesster's trajectory writes at 10 Hz contend with the
        # state-reader thread's reads on the Feetech bus; transient bursts
        # of read failures during fast motion shouldn't trigger a phantom
        # disconnect that kills mid-move. 50 misses at 10 Hz polling still
        # detects real cable-unplug events within ~5 seconds.
        self._consecutive_read_failures = 0
        self._max_read_failures = 50

    # ---------------------------------------------------------------
    # State reader pause/resume. Bus contention between the state thread
    # and trajectory writes causes burst read failures during fast motion;
    # callers can pause reads for the duration of a trajectory.
    # ---------------------------------------------------------------

    def pause_state_reader(self):
        """Stop the background state thread. Subsequent get_state() returns
        the last cached values until resume_state_reader() is called."""
        if self.running:
            self.running = False
            if self.state_thread is not None:
                self.state_thread.join(timeout=2.0)
                self.state_thread = None

    def resume_state_reader(self):
        """Restart the background state thread."""
        import threading
        if not self.connected or self.running:
            return
        # Reset failure counter so a pause across motion doesn't trip
        # disconnect detection right after resume.
        self._consecutive_read_failures = 0
        self.running = True
        self.state_thread = threading.Thread(
            target=self._state_update_loop, daemon=True,
        )
        self.state_thread.start()

    def refresh_state_sync(self) -> bool:
        """Synchronously read all motor positions into self.state.

        Useful right after connect()/enable_torque() before the background
        state thread has had time to populate self.state.joint_positions.
        Returns True on success.
        """
        if not self.connected or self.serial is None:
            return False
        import time as _time
        positions = self._read_joint_positions()
        if positions is None:
            return False
        with self.state_lock:
            self.state.joint_positions = positions
            self.state.timestamp = _time.time()
        return True

    # ---------------------------------------------------------------
    # Motion overrides (SO100Arm.move_joints hardcodes 6 motors)
    # ---------------------------------------------------------------

    def move_joints(self, target_positions, speed: float = 1.0,
                    blocking: bool = False) -> bool:
        if not self.connected:
            print("[Chesster] Not connected")
            return False
        positions = np.array(target_positions, dtype=np.float32)
        if len(positions) != self.NUM_MOTORS:
            print(f"[Chesster] Invalid number of joints: {len(positions)} "
                  f"(expected {self.NUM_MOTORS})")
            return False
        for i, (pos, limits) in enumerate(zip(positions, self.JOINT_LIMITS)):
            if pos < limits[0] or pos > limits[1]:
                print(f"[Chesster] Joint {i} position {pos:.3f} out of bounds {limits}")
                return False
        success = self._send_joint_positions(positions)
        if success and blocking:
            self._wait_for_movement()
        return success

    # ---------------------------------------------------------------
    # Torque control (mirrors RobotController.enable_torque/release_torque
    # so the Chesster execute script does not need RobotController, which
    # is hardcoded to 6 motors).
    # ---------------------------------------------------------------

    def _write_torque(self, motor_id: int, enable: bool) -> bool:
        import serial as _serial
        try:
            packet = [
                *self.HEADER, motor_id, 0x04,
                self.INSTR_WRITE, self.REG_TORQUE_ENABLE, 0x01 if enable else 0x00,
            ]
            checksum = self._calculate_checksum(packet[2:])
            packet.append(checksum)
            with self.serial_lock:
                self.serial.write(bytes(packet))
            return True
        except _serial.SerialException as e:
            print(f"[Chesster] Torque write failed motor {motor_id}: {e}")
            return False

    def enable_torque(self) -> bool:
        """Enable torque on all motors."""
        if not self.connected or self.serial is None:
            return False
        import time as _time
        ok = True
        for motor_id in self.MOTOR_IDS:
            ok &= self._write_torque(motor_id, True)
            _time.sleep(0.005)
        return ok

    def release_torque(self) -> bool:
        """Disable torque on all motors."""
        if not self.connected or self.serial is None:
            return False
        import time as _time
        ok = True
        for motor_id in self.MOTOR_IDS:
            ok &= self._write_torque(motor_id, False)
            _time.sleep(0.005)
        return ok
