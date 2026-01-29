"""
ZMQ-based Robot State Publisher/Subscriber for Low-Latency IPC.

Enables real-time position sharing between tele_op.py and collect_vla_episodes.py
with <1ms latency (vs ~10-50ms for file-based state_cache).

Protocol:
    - Publisher (tele_op.py): Sends JSON messages with joint positions, port, timestamp
    - Subscriber (collect_vla_episodes.py): Receives positions in real-time for recording

Message Format:
    {
        "joint_positions": [j0, j1, j2, j3, j4, j5],  # radians
        "robot_port": "/dev/ttyACM0",
        "timestamp": 1234567890.123,
        "gripper_state": 0.5
    }

Usage:
    # In tele_op.py (publisher)
    from utils.robot_state_publisher import RobotStatePublisher
    publisher = RobotStatePublisher()
    publisher.start()
    publisher.publish(joint_positions, robot_port)
    publisher.stop()

    # In collect_vla_episodes.py (subscriber)
    from utils.robot_state_publisher import RobotStateSubscriber
    subscriber = RobotStateSubscriber()
    subscriber.start()
    state = subscriber.get_latest()  # Non-blocking, returns None if no message
    subscriber.stop()
"""

import json
import time
import threading
from typing import Optional, List, Dict, Any

import numpy as np

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False


# Default ZMQ endpoint (TCP on localhost)
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"


class RobotStatePublisher:
    """
    Publishes robot state via ZMQ PUB socket.

    Used by tele_op.py to broadcast joint positions at control loop rate.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT):
        """
        Initialize publisher.

        Args:
            endpoint: ZMQ endpoint (default: tcp://127.0.0.1:5555)
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq not installed. Run: pip install pyzmq")

        self.endpoint = endpoint
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.running = False

    def start(self):
        """Start the publisher socket."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self.endpoint)
        self.running = True

        # Small delay to allow subscribers to connect
        time.sleep(0.1)

        print(f"[ZMQ] Publisher started on {self.endpoint}")

    def stop(self):
        """Stop the publisher socket."""
        self.running = False

        if self.socket:
            self.socket.close()
            self.socket = None

        if self.context:
            self.context.term()
            self.context = None

        print("[ZMQ] Publisher stopped")

    def publish(
        self,
        joint_positions: List[float],
        robot_port: str,
        gripper_state: float = 0.0
    ):
        """
        Publish robot state.

        Args:
            joint_positions: List of 6 joint angles in radians
            robot_port: Robot serial port (e.g., "/dev/ttyACM0")
            gripper_state: Gripper position (joint 6 in radians)
        """
        if not self.running or not self.socket:
            return

        message = {
            "joint_positions": list(joint_positions),
            "robot_port": robot_port,
            "timestamp": time.time(),
            "gripper_state": gripper_state
        }

        try:
            self.socket.send_string(json.dumps(message), zmq.NOBLOCK)
        except zmq.Again:
            pass  # No subscribers, message dropped (expected)


class RobotStateSubscriber:
    """
    Subscribes to robot state via ZMQ SUB socket.

    Used by collect_vla_episodes.py to receive joint positions in real-time.
    Runs a background thread to continuously receive messages.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout_ms: int = 100):
        """
        Initialize subscriber.

        Args:
            endpoint: ZMQ endpoint (default: tcp://127.0.0.1:5555)
            timeout_ms: Socket receive timeout in milliseconds
        """
        if not ZMQ_AVAILABLE:
            raise ImportError("pyzmq not installed. Run: pip install pyzmq")

        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.running = False

        # Latest state (thread-safe)
        self._latest_state: Optional[Dict[str, Any]] = None
        self._state_lock = threading.Lock()

        # Background receiver thread
        self._receiver_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the subscriber socket and receiver thread."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

        self.running = True
        self._receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._receiver_thread.start()

        print(f"[ZMQ] Subscriber connected to {self.endpoint}")

    def stop(self):
        """Stop the subscriber socket and receiver thread."""
        self.running = False

        if self._receiver_thread:
            self._receiver_thread.join(timeout=1.0)
            self._receiver_thread = None

        if self.socket:
            self.socket.close()
            self.socket = None

        if self.context:
            self.context.term()
            self.context = None

        print("[ZMQ] Subscriber stopped")

    def _receiver_loop(self):
        """Background thread that continuously receives messages."""
        while self.running:
            try:
                message_str = self.socket.recv_string()
                message = json.loads(message_str)

                with self._state_lock:
                    self._latest_state = message

            except zmq.Again:
                # Timeout, no message received
                pass
            except json.JSONDecodeError:
                pass  # Invalid message, ignore
            except Exception:
                if self.running:
                    pass  # Socket closed during shutdown

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest robot state.

        Returns:
            Dict with keys: joint_positions, robot_port, timestamp, gripper_state
            Or None if no message received yet
        """
        with self._state_lock:
            return self._latest_state.copy() if self._latest_state else None

    def get_joint_positions(self) -> Optional[List[float]]:
        """
        Get the latest joint positions.

        Returns:
            List of 6 joint angles in radians, or None if no message received
        """
        state = self.get_latest()
        return state["joint_positions"] if state else None

    def get_robot_port(self) -> Optional[str]:
        """
        Get the robot port from the publisher.

        Returns:
            Robot serial port string, or None if no message received
        """
        state = self.get_latest()
        return state["robot_port"] if state else None

    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        """
        Wait for first message from publisher.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected (received at least one message), False if timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.get_latest() is not None:
                return True
            time.sleep(0.05)
        return False
