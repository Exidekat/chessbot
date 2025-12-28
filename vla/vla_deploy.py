"""
VLA Deploy - π₀.₅ Inference for Chess Robot Control

This script deploys the Physical Intelligence π₀.₅ model for real-time chess robot control:
1. Loads base π₀.₅ weights (or fine-tuned checkpoint if provided)
2. Performs board detection and move calculation (same as virtual_overlay_demo.py)
3. Generates color-conditioned VLM prompts for each move stage
4. Uses ChessBot Virtual Cam (720p) + Gripper Cam (224x224) as video input
5. User presses ENTER to hand control to VLA, ENTER again to stop
6. Repeats for all stages of the best move

Usage:
    python vla/vla_deploy.py [--checkpoint CHECKPOINT_PATH] [--device CAMERA_DEVICE]
"""

import argparse
import sys
from pathlib import Path
import chess
import cv2
import subprocess
from datetime import datetime
import time
import gc
import threading
from typing import Optional
import numpy as np

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator
from guidance.coordinate_mapper import CoordinateMapper
from guidance.move_decomposer import decompose_move
from PIL import Image

# Import shared utilities
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device,
    capture_4k_downscale
)

# Import VLA model loading
from vla.vla_load_model import load_pi0_model

# Import SO-100 arm control
try:
    from controls.so100_arm import SO100Arm, SO100State
    SO100_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] SO-100 control not available: {e}")
    SO100_AVAILABLE = False

# Import torch for tensor operations
try:
    import torch
    print("[OK] PyTorch import successful")
except ImportError as e:
    print(f"[X] Failed to import PyTorch: {e}")
    print("    Run: conda activate ltx && pip install torch")
    sys.exit(1)

# Helper function for preprocessing observations
def preprocess_observation(obs_dict):
    """Convert numpy observation dict to torch tensors."""
    processed = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            processed[key] = torch.from_numpy(value)
        else:
            processed[key] = value
    return processed


def print_board(board: chess.Board):
    """Pretty print a chess board."""
    print("\n" + "=" * 40)
    print("Current Board Position:")
    print("=" * 40)
    print(board)
    print("=" * 40)


# VLA-specific camera capture functions
# (Single-frame real-time capture for inference, not batch capture for detection)

def capture_720p_frame(device_path):
    """
    Capture a single 720p frame from camera.

    Args:
        device_path: Camera device path

    Returns:
        numpy.ndarray: Captured frame (720p)
    """
    camera_index = get_camera_index_from_device(device_path)

    # Configure camera for 720p MJPEG
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-fmt-video=width=1280,height=720,pixelformat=MJPG",
            "--set-parm=30"
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pass  # Continue with OpenCV defaults

    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[X] Failed to open camera: {device_path}")
        return None

    # Set to 720p MJPEG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Capture frame
    ret, frame = cap.read()
    del cap
    gc.collect()

    if not ret:
        print(f"[X] Failed to capture frame from {device_path}")
        return None

    return frame


def capture_gripper_frame(device_path):
    """
    Capture a single 224x224 frame from gripper camera.

    Args:
        device_path: Camera device path

    Returns:
        numpy.ndarray: Captured frame (224x224)
    """
    camera_index = get_camera_index_from_device(device_path)

    # Configure camera for 640x480 (we'll resize to 224x224)
    try:
        subprocess.run([
            "v4l2-ctl",
            f"--device={device_path}",
            "--set-fmt-video=width=640,height=480,pixelformat=MJPG",
            "--set-parm=30"
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        pass

    # Open camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[X] Failed to open gripper camera: {device_path}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Capture frame
    ret, frame = cap.read()
    del cap
    gc.collect()

    if not ret:
        print(f"[X] Failed to capture frame from gripper camera")
        return None

    # Resize to 224x224 for VLA input
    frame_224 = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return frame_224


class Pi0VLA:
    """Wrapper for Physical Intelligence π₀.₅ model with SO-100 integration."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        robot_arm: Optional[SO100Arm] = None
    ):
        """
        Initialize π₀.₅ model.

        Args:
            checkpoint_path: Path to fine-tuned checkpoint (None = base weights)
            device: Device to run model on ("cuda" or "cpu")
            robot_arm: SO100Arm instance for robot control
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.robot_arm = robot_arm

        print("\n" + "=" * 60)
        print("Loading π₀.₅ Model")
        print("=" * 60)

        # Load π₀.₅ model and tokenizer using shared loader
        self.policy, self.tokenizer = load_pi0_model(
            checkpoint_path=checkpoint_path,
            device=device
        )

        if robot_arm:
            print(f"Robot: SO-100 connected")
        else:
            print(f"Robot: None (actions will not be executed)")

        print("=" * 60)
        print("[OK] π₀.₅ model loaded successfully")
        print("=" * 60)

    def predict_action(
        self,
        global_frame: np.ndarray,
        gripper_frame: np.ndarray,
        language_prompt: str,
        robot_state: Optional[SO100State] = None
    ) -> dict:
        """
        Predict robot action from observation.

        Args:
            global_frame: Global camera frame (720p)
            gripper_frame: Gripper camera frame (224x224)
            language_prompt: Natural language instruction
            robot_state: Current SO-100 state

        Returns:
            dict: Predicted action
                - joint_positions: [6] target joint positions in radians
                - gripper: 0.0-1.0 gripper position
                - confidence: float model confidence
        """
        # Prepare raw observation using LeRobot canonical pattern
        # Step 1: Ensure frames are in (H, W, 3) uint8 format
        if len(global_frame.shape) == 3 and global_frame.shape[0] == 3:
            global_frame = np.transpose(global_frame, (1, 2, 0))
        if len(gripper_frame.shape) == 3 and gripper_frame.shape[0] == 3:
            gripper_frame = np.transpose(gripper_frame, (1, 2, 0))

        # Resize to expected resolution (will be resized again by processor if needed)
        global_frame_resized = cv2.resize(global_frame, (224, 224))
        gripper_frame_resized = cv2.resize(gripper_frame, (224, 224))

        # Ensure uint8 dtype
        if global_frame_resized.dtype != np.uint8:
            global_frame_resized = (global_frame_resized * 255).astype(np.uint8)
        if gripper_frame_resized.dtype != np.uint8:
            gripper_frame_resized = (gripper_frame_resized * 255).astype(np.uint8)

        # Get current robot state
        # SO-100 has 6 motors total: motors 1-5 are arm joints, motor 6 is gripper
        # joint_positions is [6] array containing all motors
        if robot_state:
            state_6d = robot_state.joint_positions.astype(np.float32)  # [6] all motors
        else:
            # Placeholder state when no robot connected
            state_6d = np.zeros(6, dtype=np.float32)
            state_6d[5] = 0.5  # Gripper (motor 6) at 50% open

        # Pad to 32D as expected by π₀ model config
        state_32d = np.zeros(32, dtype=np.float32)
        state_32d[:6] = state_6d  # First 6 dims = SO-100 state (5 arm + 1 gripper)

        # Step 2: Create raw observation dict using model's expected key names
        # Model expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
        # We have: top camera (global) and wrist camera (gripper)
        # Map our cameras to model's expected names
        raw_observation = {
            "observation.images.base_0_rgb": global_frame_resized,       # Top/global camera
            "observation.images.left_wrist_0_rgb": gripper_frame_resized, # Wrist camera
            "observation.images.right_wrist_0_rgb": gripper_frame_resized, # Duplicate for now
            "observation.state": state_32d,                               # 32D state
        }

        # Step 3: Convert to PyTorch tensors
        observation = preprocess_observation(raw_observation)

        # Step 4: Add batch dimension and move to device
        for key in observation.keys():
            if isinstance(observation[key], torch.Tensor):
                # Add batch dimension if not present
                if observation[key].ndim == 3:  # Images (H, W, C) -> (1, C, H, W)
                    observation[key] = observation[key].permute(2, 0, 1).unsqueeze(0)
                elif observation[key].ndim == 1:  # State (D,) -> (1, D)
                    observation[key] = observation[key].unsqueeze(0)
                # Move to device and convert to float
                observation[key] = observation[key].to(self.device).float()
                # Normalize images to [0, 1]
                if "images" in key:
                    observation[key] = observation[key] / 255.0

        # Step 5: Tokenize language prompt and add to observation
        tokens = self.tokenizer(
            language_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length
        )
        observation["observation.language.tokens"] = tokens["input_ids"].to(self.device)
        # Convert attention mask to boolean (model expects bool, not Long)
        observation["observation.language.attention_mask"] = tokens["attention_mask"].to(self.device).bool()

        # Step 6: Run inference using select_action API
        with torch.inference_mode():
            action_tensor = self.policy.select_action(observation)

        # Extract action as numpy array
        if isinstance(action_tensor, torch.Tensor):
            action = action_tensor.cpu().numpy()
        else:
            action = np.array(action_tensor)

        # Handle batch dimension if present
        if len(action.shape) > 1:
            action = action[0]

        # Map action to SO-100 control
        # π₀ outputs normalized action deltas (typically -1 to 1 range)
        # We need to:
        # 1. Extract first 6 dimensions (SO-100 joints)
        # 2. Scale from normalized space to radians
        # 3. Apply as delta to current position

        if len(action) >= 6:
            # Extract 6D action deltas (normalized, likely -1 to 1)
            action_deltas_normalized = action[:6]

            # Scale to radians: assuming actions are in [-1, 1], map to small delta range
            # Use conservative scaling: ±0.1 radians (~5.7 degrees) per step
            max_delta_rad = 0.1
            action_deltas_rad = action_deltas_normalized * max_delta_rad

            # Apply delta to current state
            predicted_joints = state_6d + action_deltas_rad

            # Wraparound to [0, 2π) using modulo (continuous rotation joints)
            predicted_joints = predicted_joints % (2 * np.pi)
        else:
            # Fallback: keep current positions
            print(f"[VLA WARN] Action dimension mismatch: got {len(action)}, expected 6")
            predicted_joints = state_6d

        predicted_action = {
            "joint_positions": predicted_joints,
            "confidence": 1.0
        }

        return predicted_action

    def execute_action(self, action: dict, speed: float = 0.3) -> bool:
        """
        Execute predicted action on SO-100 arm.

        Args:
            action: Action dict from predict_action()
            speed: Movement speed (0.0-1.0)

        Returns:
            bool: True if execution successful
        """
        if not self.robot_arm:
            print("[VLA] No robot arm connected, skipping execution")
            return False

        # Send joint position command (all 6 motors)
        success = self.robot_arm.move_joints(
            action["joint_positions"],
            speed=speed,
            blocking=False  # Non-blocking for continuous control
        )

        return success


def vla_control_loop(
    vla: Pi0VLA,
    global_camera: str,
    gripper_camera: str,
    language_prompt: str,
    duration: float = 30.0
):
    """
    Run VLA control loop for a single stage.

    Args:
        vla: Pi0VLA model instance (with SO-100 arm)
        global_camera: Global camera device path
        gripper_camera: Gripper camera device path
        language_prompt: VLM prompt for this stage
        duration: Maximum duration for control (seconds)
    """
    print("\n" + "=" * 60)
    print("VLA CONTROL ACTIVE")
    print("=" * 60)
    print(f"Prompt: {language_prompt}")
    print(f"Press ENTER to stop VLA control...")
    print("=" * 60)

    # Open cameras once and keep them open
    global_cap = cv2.VideoCapture(get_camera_index_from_device(global_camera), cv2.CAP_V4L2)
    global_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    global_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    global_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    global_cap.set(cv2.CAP_PROP_FPS, 30)

    gripper_cap = cv2.VideoCapture(get_camera_index_from_device(gripper_camera), cv2.CAP_V4L2)
    gripper_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    gripper_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    gripper_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Thread-safe stop flag
    stop_event = threading.Event()

    def wait_for_enter():
        input()
        stop_event.set()

    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    start_time = time.time()
    frame_count = 0
    action_count = 0

    try:
        while not stop_event.is_set():
            loop_start = time.time()

            # Check timeout
            if time.time() - start_time > duration:
                print("\n[VLA] Control timeout reached")
                break

            # Capture frames (fast - cameras already open)
            ret1, global_frame = global_cap.read()
            ret2, gripper_frame = gripper_cap.read()

            if not ret1 or not ret2:
                continue

            # Resize gripper to 224x224
            gripper_frame = cv2.resize(gripper_frame, (224, 224))

            # Get current robot state
            robot_state = None
            if vla.robot_arm:
                robot_state = vla.robot_arm.get_state()

            # Get VLA action prediction
            action = vla.predict_action(
                global_frame,
                gripper_frame,
                language_prompt,
                robot_state=robot_state
            )

            # Execute action on SO-100
            if vla.execute_action(action, speed=0.5):
                action_count += 1

            # Print status every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                hz = frame_count / elapsed if elapsed > 0 else 0
                if robot_state:
                    print(f"[VLA] Frame {frame_count}: {hz:.1f} Hz, "
                          f"joints[0-2]=[{robot_state.joint_positions[0]:.2f}, "
                          f"{robot_state.joint_positions[1]:.2f}, "
                          f"{robot_state.joint_positions[2]:.2f}]")

            frame_count += 1

            # Maintain ~15 Hz loop rate
            loop_time = time.time() - loop_start
            sleep_time = max(0, (1.0/15) - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Clean up cameras
        global_cap.release()
        gripper_cap.release()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("VLA CONTROL STOPPED")
    print("=" * 60)
    print(f"Duration: {elapsed:.2f}s")
    print(f"Frames processed: {frame_count}")
    print(f"Actions executed: {action_count}")
    print(f"Control rate: {frame_count/elapsed:.1f} Hz")
    print("=" * 60)


def main():
    """Main VLA deployment driver."""
    parser = argparse.ArgumentParser(
        description="VLA Deploy - π₀.₅ inference for chess robot control"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (default: base π₀.₅ weights)"
    )
    parser.add_argument(
        "--global-camera",
        type=str,
        default=None,
        help="Global/overhead camera device (e.g., /dev/video4). If not specified, auto-detects."
    )
    parser.add_argument(
        "--gripper-camera",
        type=str,
        default="/dev/video0",
        help="Gripper camera device (default: /dev/video0 - eMeet C950)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="stockfish",
        help="Path to UCI chess engine (default: stockfish)"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Time limit for engine analysis in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--corner-conf",
        type=float,
        default=0.005,
        help="Confidence threshold for corner detection (default: 0.005)"
    )
    parser.add_argument(
        "--min-corner-dist",
        type=float,
        default=30.0,
        help="Minimum distance between corners in pixels (default: 30.0)"
    )
    parser.add_argument(
        "--rotation",
        type=str,
        default="right",
        choices=["left", "right", "top", "bottom"],
        help="Camera rotation relative to board (default: right)"
    )
    parser.add_argument(
        "--turn",
        type=str,
        default="black",
        choices=["white", "black"],
        help="Whose turn to calculate move for (default: black)"
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/ttyUSB0",
        help="SO-100 serial port (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run without robot connection (visualization only)"
    )

    args = parser.parse_args()

    # Normalize turn argument to single letter for internal use
    turn_letter = "w" if args.turn == "white" else "b"

    print("=" * 60)
    print("VLA Deploy - π₀.₅ Chess Robot Control")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint or 'Base π₀.₅ weights'}")
    print(f"Board rotation: {args.rotation}")
    print(f"Turn: {args.turn.capitalize()}")
    print()

    # Step 1: Initialize SO-100 robot arm
    print("=" * 60)
    print("STAGE 1: Initialize SO-100 Robot Arm")
    print("=" * 60)

    robot_arm = None
    if not args.no_robot and SO100_AVAILABLE:
        try:
            robot_arm = SO100Arm(port=args.robot_port, baudrate=1000000)  # Feetech protocol: 1 Mbps
            if robot_arm.connect():
                print(f"[OK] SO-100 connected on {args.robot_port}")
                state = robot_arm.get_state()
                print(f"[OK] Current joint positions: {state.joint_positions}")
            else:
                print(f"[X] Failed to connect to SO-100")
                robot_arm = None
        except Exception as e:
            print(f"[X] SO-100 connection error: {e}")
            robot_arm = None
    else:
        if args.no_robot:
            print("[WARN] Running without robot (--no-robot flag)")
        else:
            print("[WARN] SO-100 control not available")

    # Step 2: Initialize π₀.₅ model
    print("\n" + "=" * 60)
    print("STAGE 2: Initialize π₀.₅ Model")
    print("=" * 60)

    vla = Pi0VLA(
        checkpoint_path=args.checkpoint,
        device="cuda",
        robot_arm=robot_arm
    )

    # Step 3: Camera selection
    print("\n" + "=" * 60)
    print("STAGE 3: Camera Selection")
    print("=" * 60)

    # Detect cameras
    cameras = get_available_cameras()
    if not cameras:
        print("[X] No cameras found!")
        return 1

    # Global camera selection
    if args.global_camera:
        global_camera = args.global_camera
        print(f"[OK] Global camera: {global_camera} (specified)")
    elif len(cameras) == 1:
        global_camera = cameras[0][0]
        print(f"[OK] Global camera: {global_camera} (auto-selected)")
    else:
        global_camera = select_camera(cameras)

    # Gripper camera
    gripper_camera = args.gripper_camera
    print(f"[OK] Gripper camera: {gripper_camera}")

    # Step 4: Capture initial board state
    print("\n" + "=" * 60)
    print("STAGE 4: Board Detection & Move Calculation")
    print("=" * 60)

    # Capture photo for board detection
    print("Capturing board image...")
    board_frame = capture_720p_frame(global_camera)
    if board_frame is None:
        print("[X] Failed to capture board image")
        return 1

    # Save for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path("data") / f"vla_board_{timestamp}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), board_frame)
    print(f"[OK] Board image saved: {image_path}")

    # Detect board state
    print("Detecting board state...")
    detector = BoardDetector(camera_position=args.rotation)
    fen, transformed_image = detector.detect_board_state(
        str(image_path),
        corner_conf=args.corner_conf,
        min_corner_distance=args.min_corner_dist,
        debug=True,
        turn=turn_letter
    )
    print("[OK] Board state detected")
    print()

    # Create chess board from FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        print(f"[X] Invalid FEN: {fen}")
        board = chess.Board(None)

    print("FEN Notation:")
    print("-" * 60)
    print(fen)
    print("-" * 60)
    print()

    print_board(board)

    # Calculate best move
    print("\n" + "=" * 60)
    print("STAGE 5: Best Move Calculation")
    print("=" * 60)

    try:
        calculator = MoveCalculator(engine_path=args.engine)
        best_move = calculator.calculate_best_move(board, time_limit=args.time)

        if best_move:
            print(f"[OK] Best move: {best_move}")
            print(f"  UCI notation: {best_move.uci()}")
            san_move = board.san(best_move)
            print(f"  SAN notation: {san_move}")
            print()

            # Decompose move into stages
            stages = decompose_move(board, best_move)
            print(f"Move decomposed into {len(stages)} stage(s):")
            for i, stage in enumerate(stages, 1):
                print(f"  {i}. {stage['description']}")
            print()

            # Step 5: VLA control for each stage
            print("\n" + "=" * 60)
            print("STAGE 6: VLA-Controlled Execution")
            print("=" * 60)
            print()

            for i, stage in enumerate(stages, 1):
                print("\n" + "-" * 60)
                print(f"Stage {i}/{len(stages)}: {stage['description']}")
                print("-" * 60)
                print(f"  VLM Prompt: {stage['vlm_prompt']}")
                print()

                # Wait for user to press ENTER to start VLA control
                input("Press ENTER to hand control to VLA...")

                # Run VLA control loop
                vla_control_loop(
                    vla,
                    global_camera,
                    gripper_camera,
                    stage['vlm_prompt'],
                    duration=30.0  # 30 second timeout
                )

                print()

            print("\n" + "=" * 60)
            print("[OK] All stages completed!")
            print("=" * 60)

        else:
            print("[X] No legal moves available")

    except FileNotFoundError:
        print(f"[X] Engine not found at '{args.engine}'")
        print("  Install stockfish or specify engine path with --engine")
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if robot_arm:
        print("\nDisconnecting from SO-100...")
        robot_arm.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
