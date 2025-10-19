"""
Robot Interface Demo Script

Demonstrates the robot interface functionality with simulated robot responses.
Useful for testing without a physical robot.

Usage:
    # Test with simulated robot
    python robot_demo.py --simulate

    # Test with real robot (must be running robot server)
    python robot_demo.py --image data/chessboardv2.png
"""

import argparse
import sys
import time
import threading
import chess
from board_state import BoardState
from robot_interface import MoveExecutor, RobotCommunicator


class SimulatedRobot:
    """
    Simulates robot responses for testing.

    Automatically responds to robot actions with appropriate signals.
    """

    def __init__(self, communicator: RobotCommunicator, pickup_delay: float = 2.0,
                 place_delay: float = 2.0):
        """
        Initialize simulated robot.

        Args:
            communicator: RobotCommunicator instance to monitor
            pickup_delay: Simulated time to pick up piece (seconds)
            place_delay: Simulated time to place piece (seconds)
        """
        self.communicator = communicator
        self.pickup_delay = pickup_delay
        self.place_delay = place_delay
        self.running = False
        self.thread = None

        self.last_action = None
        self.holding_piece = False

    def start(self):
        """Start simulated robot."""
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
        print("[SimRobot] Simulated robot started")

    def stop(self):
        """Stop simulated robot."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("[SimRobot] Simulated robot stopped")

    def _simulation_loop(self):
        """Background thread that simulates robot behavior."""
        # Send initial ready signal
        self._send_status(ready=True, holding_piece=False)

        while self.running:
            time.sleep(0.1)

    def _send_status(self, ready: bool, holding_piece: bool, error: str = None):
        """Send status update to chessbot."""
        # We simulate this by directly updating the communicator's status
        # In real scenario, robot would send this over network
        import json

        status_msg = json.dumps({
            "type": "status",
            "ready": ready,
            "holding_piece": holding_piece,
            "error": error
        })

        # Simulate receiving this message
        self.communicator._process_message(status_msg)

    def simulate_action(self, action: str):
        """
        Simulate response to an action command.

        Args:
            action: "pickup" or "place"
        """
        if action == "pickup":
            print(f"[SimRobot] Simulating pickup (delay: {self.pickup_delay}s)...")
            time.sleep(self.pickup_delay)
            self.holding_piece = True
            self._send_status(ready=True, holding_piece=True)
            print("[SimRobot] Piece picked up")

        elif action == "place":
            print(f"[SimRobot] Simulating placement (delay: {self.place_delay}s)...")
            time.sleep(self.place_delay)
            self.holding_piece = False
            self._send_status(ready=True, holding_piece=False)
            print("[SimRobot] Piece placed")


def test_move_interpreter():
    """Test the MoveInterpreter class."""
    from robot_interface import MoveInterpreter

    print("\n" + "=" * 70)
    print("Testing MoveInterpreter")
    print("=" * 70)

    board = chess.Board()

    # Test normal move
    move = chess.Move.from_uci("e2e4")
    move_type = MoveInterpreter.get_move_type(move, board)
    print(f"e2e4: {move_type.value}")

    # Execute move
    board.push(move)

    # Test capture
    board.push(chess.Move.from_uci("d7d5"))
    capture_move = chess.Move.from_uci("e4d5")
    move_type = MoveInterpreter.get_move_type(capture_move, board)
    is_capture = MoveInterpreter.is_capture(capture_move, board)
    print(f"e4d5: {move_type.value}, is_capture={is_capture}")

    # Test castling
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle_move = chess.Move.from_uci("e1g1")
    move_type = MoveInterpreter.get_move_type(castle_move, board)
    print(f"e1g1 (castling): {move_type.value}")

    king_move, rook_move = MoveInterpreter.decompose_castling(castle_move, board)
    print(f"  Decomposed: King {king_move.uci()}, Rook {rook_move.uci()}")

    print("=" * 70)


def test_robot_communication(simulate: bool = True):
    """
    Test robot communication.

    Args:
        simulate: If True, use simulated robot. If False, wait for real robot.
    """
    print("\n" + "=" * 70)
    print("Testing Robot Communication")
    print("=" * 70)

    # Create communicator
    comm = RobotCommunicator(host="127.0.0.1", port=5556)
    comm.start()

    if simulate:
        # Create simulated robot
        sim_robot = SimulatedRobot(comm, pickup_delay=1.0, place_delay=1.0)
        sim_robot.start()

        # Test pickup sequence
        print("\nTest 1: Pickup action")
        comm.send_action(
            robot_action=True,
            action="pickup",
            target_square="e4",
            coordinates=(320, 320),
            is_capture=False,
            piece="P"
        )

        # Simulate robot response
        sim_robot.simulate_action("pickup")

        # Check status
        status = comm.get_status()
        print(f"Status: holding_piece={status.holding_piece}, ready={status.ready}")

        # Test place sequence
        print("\nTest 2: Place action")
        comm.send_action(
            robot_action=True,
            action="place",
            target_square="e5",
            coordinates=(320, 256),
            is_capture=False,
            piece="P"
        )

        # Simulate robot response
        sim_robot.simulate_action("place")

        status = comm.get_status()
        print(f"Status: holding_piece={status.holding_piece}, ready={status.ready}")

        sim_robot.stop()

    else:
        print("\nWaiting for robot to connect on port 5556...")
        print("Robot should send status messages and respond to actions")
        print("Press Ctrl+C to exit")

        try:
            while True:
                time.sleep(1.0)
                status = comm.get_status()
                print(f"Robot status: holding={status.holding_piece}, ready={status.ready}")
        except KeyboardInterrupt:
            print("\nStopping...")

    comm.stop()
    print("=" * 70)


def test_move_execution(simulate: bool = True, image_path: str = None):
    """
    Test full move execution with robot.

    Args:
        simulate: Use simulated robot
        image_path: Path to board image (if None, skip board detection)
    """
    print("\n" + "=" * 70)
    print("Testing Move Execution")
    print("=" * 70)

    # Create BoardState
    detector = BoardState()

    # Take snapshot if image provided
    if image_path:
        print(f"\nProcessing board image: {image_path}")
        try:
            detector.snapshot(image_path, debug=False)
            board = detector.current(output_format="board")
            print(f"Board state:\n{board}")
        except Exception as e:
            print(f"Error processing image: {e}")
            print("Using default starting position instead")
            board = chess.Board()
            detector._last_board_state = board
    else:
        print("\nUsing default starting position")
        board = chess.Board()
        detector._last_board_state = board

    # Create executor
    executor = detector.get_robot_executor(host="127.0.0.1", port=5557)

    if simulate:
        # Create simulated robot
        sim_robot = SimulatedRobot(executor.robot_comm, pickup_delay=0.5, place_delay=0.5)
        sim_robot.start()

        # Monkey-patch the wait_for_signal to auto-simulate
        original_wait = executor.robot_comm.wait_for_signal

        def auto_simulate_wait(signal_name, expected_value, timeout=None):
            # Trigger simulation based on last action
            if signal_name == "holding_piece":
                if expected_value:  # Waiting for pickup
                    sim_robot.simulate_action("pickup")
                else:  # Waiting for placement
                    sim_robot.simulate_action("place")

            return original_wait(signal_name, expected_value, timeout)

        executor.robot_comm.wait_for_signal = auto_simulate_wait

    # Test moves
    test_moves = [
        ("e2e4", "Normal move"),
        ("e7e5", "Normal move"),
        ("f1c4", "Normal move (bishop)"),
        ("b8c6", "Normal move (knight)"),
        ("d1h5", "Normal move (queen)"),
        ("g8f6", "Normal move (knight)"),
        ("h5f7", "Capture (checkmate!)"),
    ]

    for move_uci, description in test_moves:
        try:
            move = chess.Move.from_uci(move_uci)

            if not board.is_legal(move):
                print(f"\nSkipping illegal move: {move_uci}")
                continue

            print(f"\n--- Executing move: {move_uci} ({description}) ---")

            success = executor.execute_move(move, board)

            if success:
                board.push(move)
                print(f"✓ Move executed successfully")
                print(f"Board after move:\n{board}\n")
            else:
                print(f"✗ Move execution failed")
                break

            time.sleep(0.5)  # Pause between moves

        except Exception as e:
            print(f"Error executing move {move_uci}: {e}")
            break

    # Cleanup
    if simulate:
        sim_robot.stop()
    executor.cleanup()

    print("=" * 70)


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="Robot interface demo and testing"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["interpreter", "communication", "execution", "all"],
        default="all",
        help="Which component to test (default: all)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Use simulated robot (default: True)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to board image for testing"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ROBOT INTERFACE DEMO")
    print("=" * 70)
    print(f"Mode: {'Simulated Robot' if args.simulate else 'Real Robot'}")
    print("=" * 70)

    try:
        if args.test in ["interpreter", "all"]:
            test_move_interpreter()

        if args.test in ["communication", "all"]:
            test_robot_communication(simulate=args.simulate)

        if args.test in ["execution", "all"]:
            test_move_execution(simulate=args.simulate, image_path=args.image)

        print("\n✓ All tests completed!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
