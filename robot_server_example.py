"""
Robot Server Example

This is example code for the robot-side implementation.
Robot developers should adapt this code for their specific hardware.

The robot should:
1. Connect to chessbot server (TCP or UDP)
2. Send status updates (holding_piece, ready, error)
3. Receive action commands (pickup, place)
4. Execute physical movements based on coordinates

Usage:
    python robot_server_example.py --host 192.168.1.100 --port 5555
"""

import socket
import json
import time
import argparse
import sys
import threading


class RobotClient:
    """
    Example robot client that communicates with chessbot.

    Adapt this for your robot's hardware/controller.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, protocol: str = "tcp"):
        """
        Initialize robot client.

        Args:
            host: Chessbot server IP address
            port: Server port number
            protocol: 'tcp' or 'udp'
        """
        self.host = host
        self.port = port
        self.protocol = protocol.lower()

        # Socket
        self.socket = None

        # Robot state
        self.holding_piece = False
        self.ready = True
        self.error = None

        # Thread control
        self.running = False
        self.send_thread = None
        self.receive_thread = None

        # Hardware interface (PLACEHOLDER - replace with your hardware)
        self.arm = MockRobotArm()

    def connect(self) -> bool:
        """
        Connect to chessbot server.

        Returns:
            True if connected successfully
        """
        try:
            if self.protocol == "tcp":
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                print(f"[Robot] Connected to {self.host}:{self.port} (TCP)")
            else:  # UDP
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"[Robot] Ready to communicate with {self.host}:{self.port} (UDP)")

            self.socket.settimeout(1.0)
            return True

        except Exception as e:
            print(f"[Robot] Connection failed: {e}")
            return False

    def start(self):
        """Start robot client (threads for send/receive)."""
        if not self.connect():
            return False

        self.running = True

        # Start status update thread
        self.send_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.send_thread.start()

        # Start command receive thread
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()

        print("[Robot] Client started")
        return True

    def stop(self):
        """Stop robot client."""
        self.running = False

        if self.send_thread:
            self.send_thread.join(timeout=2.0)
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)

        if self.socket:
            self.socket.close()

        print("[Robot] Client stopped")

    def _status_update_loop(self):
        """Send periodic status updates to chessbot."""
        while self.running:
            try:
                self._send_status()
                time.sleep(0.1)  # Send status 10 times per second
            except Exception as e:
                if self.running:
                    print(f"[Robot] Error sending status: {e}")
                time.sleep(1.0)

    def _send_status(self):
        """Send current status to chessbot."""
        status = {
            "type": "status",
            "holding_piece": self.holding_piece,
            "ready": self.ready,
            "error": self.error
        }

        message = json.dumps(status) + "\n"

        try:
            if self.protocol == "tcp":
                self.socket.sendall(message.encode('utf-8'))
            else:  # UDP
                self.socket.sendto(message.encode('utf-8'), (self.host, self.port))

        except Exception as e:
            if self.running:
                print(f"[Robot] Error sending status: {e}")

    def _receive_loop(self):
        """Receive action commands from chessbot."""
        while self.running:
            try:
                if self.protocol == "tcp":
                    data = self.socket.recv(4096)
                    if not data:
                        print("[Robot] Connection closed by chessbot")
                        self.running = False
                        break
                    self._process_command(data.decode('utf-8'))
                else:  # UDP
                    data, _ = self.socket.recvfrom(4096)
                    self._process_command(data.decode('utf-8'))

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Robot] Error receiving command: {e}")
                time.sleep(0.1)

    def _process_command(self, message: str):
        """
        Process action command from chessbot.

        Command format:
        {
            "type": "action",
            "robot_action": bool,
            "action": "pickup" | "place",
            "target_square": "e4" | null,
            "coordinates": {"x": 250, "y": 300},
            "is_capture": bool,
            "piece": "Q"
        }
        """
        try:
            command = json.loads(message.strip())

            if command.get("type") != "action":
                return

            if not command.get("robot_action"):
                # Robot action flag is False, ignore
                return

            action = command.get("action")
            coords = command.get("coordinates", {})
            x, y = coords.get("x", 0), coords.get("y", 0)
            square = command.get("target_square")
            piece = command.get("piece")

            print(f"\n[Robot] Received command: {action} {square or 'graveyard'} at ({x}, {y})")

            # Execute action with your robot hardware
            if action == "pickup":
                self._execute_pickup(x, y, piece)
            elif action == "place":
                self._execute_place(x, y, piece)

        except json.JSONDecodeError as e:
            print(f"[Robot] Invalid JSON command: {e}")
        except Exception as e:
            print(f"[Robot] Error processing command: {e}")
            self.error = str(e)

    def _execute_pickup(self, x: int, y: int, piece: str):
        """
        Execute pickup action.

        PLACEHOLDER: Replace with your robot's actual pickup code.

        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            piece: Piece being picked up (e.g., "Q")
        """
        print(f"[Robot] Picking up {piece} at ({x}, {y})...")

        # Convert pixel coordinates to physical coordinates
        # This depends on your camera calibration and robot workspace
        physical_x, physical_y = self._pixels_to_physical(x, y)

        # Move arm to position (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.move_to(physical_x, physical_y)

        # Lower arm (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.lower()

        # Activate gripper (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.grip()

        # Raise arm (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.raise_arm()

        # Update status
        self.holding_piece = True

        print(f"[Robot] Pickup complete (holding_piece=True)")

    def _execute_place(self, x: int, y: int, piece: str):
        """
        Execute placement action.

        PLACEHOLDER: Replace with your robot's actual placement code.

        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            piece: Piece being placed (e.g., "Q")
        """
        print(f"[Robot] Placing {piece} at ({x}, {y})...")

        # Convert coordinates
        physical_x, physical_y = self._pixels_to_physical(x, y)

        # Move to position (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.move_to(physical_x, physical_y)

        # Lower arm (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.lower()

        # Release gripper (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.release()

        # Raise arm (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.raise_arm()

        # Move to home position (REPLACE WITH YOUR HARDWARE CODE)
        self.arm.move_to_home()

        # Update status
        self.holding_piece = False

        print(f"[Robot] Placement complete (holding_piece=False)")

    def _pixels_to_physical(self, pixel_x: int, pixel_y: int) -> tuple:
        """
        Convert pixel coordinates to physical robot coordinates.

        PLACEHOLDER: Implement camera calibration here.

        This typically involves:
        1. Camera calibration matrix
        2. Workspace transformation
        3. Scaling based on camera height/angle

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels

        Returns:
            (physical_x, physical_y) in robot's coordinate system (e.g., mm)
        """
        # EXAMPLE ONLY - Replace with your calibration
        # Assume board is 400mm x 400mm, image is 640x640 pixels
        scale = 400.0 / 640.0  # mm per pixel

        physical_x = pixel_x * scale
        physical_y = pixel_y * scale

        return physical_x, physical_y


class MockRobotArm:
    """
    Mock robot arm for demonstration.

    REPLACE THIS with your actual robot hardware interface.

    Examples of what to replace with:
    - Serial commands to Arduino/Raspberry Pi
    - ROS messages
    - Servo control libraries
    - Robotic arm SDK (e.g., Dobot, UR, etc.)
    """

    def __init__(self):
        print("[MockArm] Initialized (REPLACE WITH REAL HARDWARE)")

    def move_to(self, x: float, y: float):
        """Move arm to position."""
        print(f"  [MockArm] Moving to ({x:.1f}, {y:.1f})")
        time.sleep(0.5)  # Simulate movement time

    def lower(self):
        """Lower arm to picking height."""
        print("  [MockArm] Lowering arm")
        time.sleep(0.3)

    def raise_arm(self):
        """Raise arm to safe height."""
        print("  [MockArm] Raising arm")
        time.sleep(0.3)

    def grip(self):
        """Activate gripper."""
        print("  [MockArm] Gripping piece")
        time.sleep(0.2)

    def release(self):
        """Release gripper."""
        print("  [MockArm] Releasing piece")
        time.sleep(0.2)

    def move_to_home(self):
        """Move to home position."""
        print("  [MockArm] Moving to home")
        time.sleep(0.5)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Example robot client for chessbot"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Chessbot server IP address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port (default: 5555)"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["tcp", "udp"],
        default="tcp",
        help="Protocol to use (default: tcp)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ROBOT CLIENT EXAMPLE")
    print("=" * 70)
    print(f"Connecting to: {args.host}:{args.port} ({args.protocol.upper()})")
    print("=" * 70)
    print("\nNOTE: This is example code with mock hardware.")
    print("Replace MockRobotArm with your actual robot interface!")
    print("=" * 70)

    # Create and start robot client
    robot = RobotClient(host=args.host, port=args.port, protocol=args.protocol)

    if not robot.start():
        print("\n✗ Failed to start robot client")
        return 1

    print("\n✓ Robot client running")
    print("Waiting for commands from chessbot...")
    print("Press Ctrl+C to stop\n")

    try:
        while robot.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping robot client...")

    robot.stop()
    print("\n✓ Robot client stopped")

    return 0


if __name__ == "__main__":
    sys.exit(main())
