"""
VLA Deploy - Multi-Model Inference for Chess Robot Control

This script deploys VLA models (PI0 or SmolVLA) for real-time chess robot control:
1. Loads base π₀.₅ weights (or fine-tuned checkpoint if provided)
2. Performs board detection and move calculation (same as virtual_overlay_demo.py)
3. Generates color-conditioned VLM prompts for each move stage
4. Uses ChessBot Virtual Cam (720p) + Gripper Cam (224x224) as video input
5. User presses ENTER to hand control to VLA, ENTER again to stop
6. Repeats for all stages of the best move

Usage:
    # Deploy PI0 base model
    python scripts/vla_deploy.py --model pi0

    # Deploy SmolVLA base model
    python scripts/vla_deploy.py --model smolvla

    # Deploy fine-tuned PI0 checkpoint
    python scripts/vla_deploy.py --model pi0 --checkpoint checkpoints/chess_pi0/best.pt

    # Deploy with specific cameras
    python scripts/vla_deploy.py --model pi0 --global-camera /dev/video2 --robot-port /dev/ttyACM0
"""

import argparse
import sys
import termios
from pathlib import Path
import chess
import cv2
import subprocess
from datetime import datetime
import time
import gc
import threading
from typing import Optional, Dict, Any, Tuple, NamedTuple
import numpy as np

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from guidance module
from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator
from guidance.coordinate_mapper import CoordinateMapper
from guidance.move_decomposer import decompose_move
from guidance import apply_stage_overlay_to_frame
from PIL import Image

# Import virtual camera for overlay streaming
from cameras import VirtualCamera

# Import shared utilities
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device,
    capture_4k_downscale,
    get_default_global_camera,
    get_default_gripper_camera,
)
from utils.keyboard_input import KeyboardInput
from controls.robot_controller import (
    RobotController,
    load_joint_configs_for_port,
)

# Import VLA model loading (multi-model support)
from vla.models import load_vla_model, list_models

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


class VLAWrapper:
    """Unified wrapper for VLA models (PI0, SmolVLA) with SO-100 integration."""

    def __init__(
        self,
        model_name: str = "pi0",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        robot_arm: Optional[SO100Arm] = None,
        robot_controller: Optional[RobotController] = None
    ):
        """
        Initialize VLA model.

        Args:
            model_name: Model type ("pi0" or "smolvla")
            checkpoint_path: Path to fine-tuned checkpoint (None = base weights)
            device: Device to run model on ("cuda" or "cpu")
            robot_arm: SO100Arm instance for robot control (deprecated, use robot_controller)
            robot_controller: RobotController instance for position-controlled robot
        """
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.robot_arm = robot_arm
        self.robot_controller = robot_controller

        print("\n" + "=" * 60)
        print(f"Loading VLA Model ({model_name.upper()})")
        print("=" * 60)

        # Load model and tokenizer using factory
        self.model_wrapper, self.tokenizer = load_vla_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device
        )
        self.policy = self.model_wrapper.policy

        if robot_controller:
            print(f"Robot: SO-100 via RobotController (position-controlled)")
        elif robot_arm:
            print(f"Robot: SO-100 direct control")
        else:
            print(f"Robot: None (actions will not be executed)")

        print("=" * 60)
        print("[OK] VLA model loaded successfully")
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

        Uses model-specific preprocessing via the model wrapper, which handles:
        - PI0: 224x224 images, base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb keys, 32D state
        - SmolVLA: 256x256 images, camera1/camera2/camera3 keys, 6D state

        Args:
            global_frame: Global camera frame (720p, with overlay applied)
            gripper_frame: Gripper camera frame (224x224)
            language_prompt: Natural language instruction (VLM prompt)
            robot_state: Current SO-100 state

        Returns:
            dict: Predicted action
                - joint_positions: [6] target joint positions in radians
                - confidence: float model confidence

        """
        # Get current robot state as numpy array
        if robot_state:
            current_state = robot_state.joint_positions.astype(np.float32)
        else:
            # Placeholder state when no robot connected
            current_state = np.zeros(6, dtype=np.float32)
            current_state[5] = 0.5  # Gripper at 50% open

        # Use model wrapper's predict_action which handles:
        # - Model-specific image size (224x224 for PI0, 256x256 for SmolVLA)
        # - Model-specific camera keys (base_0_rgb vs camera1)
        # - Model-specific state dimension (32D for PI0, 6D for SmolVLA)
        # - Preprocessing, tokenization, and inference
        result = self.model_wrapper.predict_action(
            observation=self.model_wrapper.preprocess_observation(
                global_frame=global_frame,
                gripper_frame=gripper_frame,
                robot_state=current_state
            ),
            language_prompt=language_prompt,
            current_state=current_state
        )

        # DEBUG: Print raw model output
        print(f"[DEBUG] Raw model output: {result['raw_action']}")
        print(f"[DEBUG] Current robot state: {current_state}")

        predicted_joints = result["joint_positions"]

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
        # Prefer RobotController (position-controlled with continuous holding)
        if self.robot_controller:
            self.robot_controller.set_target_positions(action["joint_positions"])
            return True

        # Fallback to direct SO100Arm control
        if self.robot_arm:
            success = self.robot_arm.move_joints(
                action["joint_positions"],
                speed=speed,
                blocking=False  # Non-blocking for continuous control
            )
            return success

        print("[VLA] No robot connected, skipping execution")
        return False

    def get_robot_state(self) -> Optional[SO100State]:
        """Get current robot state from controller or arm."""
        if self.robot_controller:
            return self.robot_controller.arm.get_state()
        elif self.robot_arm:
            return self.robot_arm.get_state()
        return None

    def reset_to_home(self):
        """Reset robot to home position (only works with RobotController)."""
        if self.robot_controller:
            self.robot_controller.set_home_targets()
            print("[VLA] Robot reset to home position")


def vla_control_loop(
    vla: VLAWrapper,
    global_camera: str,
    gripper_camera: str,
    language_prompt: str,
    stage: Dict[str, Any],
    detector: BoardDetector,
    perspective_matrix: Optional[np.ndarray],
    camera_rotation: str,
    virtual_cam: Optional[VirtualCamera],
    duration: float = 50.0
):
    """
    Run VLA control loop for a single stage.

    Args:
        vla: Pi0VLA model instance (with SO-100 arm)
        global_camera: Global camera device path
        gripper_camera: Gripper camera device path
        language_prompt: VLM prompt for this stage
        stage: Full stage dict for overlay rendering
        detector: BoardDetector instance for coordinate mapping
        perspective_matrix: Perspective transform matrix from detection
        camera_rotation: Camera rotation setting
        virtual_cam: VirtualCamera instance for overlay streaming
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

    # Path to transformed board image (generated during detection)
    transformed_image_path = "data/chessboard_transformed.png"

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

            # Apply stage overlay to global frame (REQUIRED - matches training data format)
            # Training uses overlayed frames with color conditioning, so inference MUST too
            # DO NOT fall back to raw frame - model was trained with overlays
            if perspective_matrix is None:
                print("[X] FATAL: perspective_matrix is None - cannot apply overlay")
                print("    VLA was trained with color-conditioned overlays. Cannot deploy without them.")
                break

            overlayed_frame = apply_stage_overlay_to_frame(
                frame=global_frame,
                stage=stage,
                transformed_image_path=transformed_image_path,
                detector=detector,
                perspective_matrix=perspective_matrix,
                camera_position=camera_rotation
            )

            if overlayed_frame is None:
                # Overlay failed - skip this frame, don't use raw frame
                print("[WARN] Overlay failed for frame, skipping...")
                frame_count += 1
                continue

            # Use overlayed frame for VLA input (REQUIRED - matches training)
            vla_input_frame = overlayed_frame

            # Stream to virtual camera for visualization
            if virtual_cam:
                virtual_cam.write_frame(vla_input_frame)

            # Resize gripper to 224x224
            gripper_frame = cv2.resize(gripper_frame, (224, 224))

            # Get current robot state
            robot_state = vla.get_robot_state()

            # Get VLA action prediction using overlayed frame (matches training data)
            action = vla.predict_action(
                vla_input_frame,  # Use overlayed frame, not raw
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
    if elapsed > 0:
        print(f"Control rate: {frame_count/elapsed:.1f} Hz")
    print("=" * 60)

    # Flush stdin to clear any pending input from the daemon thread
    # This prevents the double-ENTER issue when control exits due to timeout
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass  # Ignore if not a TTY


class MoveResult(NamedTuple):
    """Result of execute_move()."""
    success: bool           # True if move executed successfully
    restart_requested: bool  # True if user pressed R to recapture


def execute_move(
    vla: VLAWrapper,
    detector: BoardDetector,
    calculator: MoveCalculator,
    global_camera: str,
    gripper_camera: str,
    virtual_cam: Optional[VirtualCamera],
    turn_letter: str,
    camera_rotation: str,
    corner_conf: float,
    min_corner_dist: float,
    time_limit: float
) -> MoveResult:
    """
    Execute a single chess move: detect board, calculate best move, execute all stages.

    Args:
        vla: VLAWrapper model instance
        detector: BoardDetector instance
        calculator: MoveCalculator instance
        global_camera: Global camera device path
        gripper_camera: Gripper camera device path
        virtual_cam: VirtualCamera for overlay streaming (or None)
        turn_letter: 'w' or 'b' for whose turn
        camera_rotation: Camera rotation setting
        corner_conf: Corner detection confidence threshold
        min_corner_dist: Minimum corner distance in pixels
        time_limit: Engine time limit in seconds

    Returns:
        MoveResult with success and restart_requested flags
    """
    print("\n" + "=" * 60)
    print("BOARD DETECTION & MOVE CALCULATION")
    print("=" * 60)

    # Capture photo for board detection
    print("Capturing board image...")
    board_frame = capture_720p_frame(global_camera)
    if board_frame is None:
        print("[X] Failed to capture board image")
        return MoveResult(success=False, restart_requested=False)

    # Save for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = Path("data") / f"vla_board_{timestamp}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), board_frame)
    print(f"[OK] Board image saved: {image_path}")

    # Detect board state
    print("Detecting board state...")
    try:
        fen, transformed_image = detector.detect_board_state(
            str(image_path),
            corner_conf=corner_conf,
            min_corner_distance=min_corner_dist,
            debug=True,
            turn=turn_letter
        )
    except Exception as e:
        print(f"[X] Board detection failed: {e}")
        return MoveResult(success=False, restart_requested=False)

    print("[OK] Board state detected")

    # Get perspective matrix from detector (stored during detection)
    # This is REQUIRED for color-conditioned overlays during VLA control
    perspective_matrix = detector.perspective_matrix
    if perspective_matrix is None:
        print("[X] FATAL: No perspective matrix from board detection!")
        print("    VLA requires color-conditioned overlays which need the perspective transform.")
        print("    This usually means corner detection failed. Check debug images.")
        return MoveResult(success=False, restart_requested=False)

    # Create chess board from FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        print(f"[X] Invalid FEN: {fen}")
        return MoveResult(success=False, restart_requested=False)

    print()
    print("FEN Notation:")
    print("-" * 60)
    print(fen)
    print("-" * 60)
    print()
    print_board(board)

    # Calculate best move
    print("\n" + "=" * 60)
    print("BEST MOVE CALCULATION")
    print("=" * 60)

    try:
        best_move = calculator.calculate_best_move(board, time_limit=time_limit)

        if not best_move:
            print("[X] No legal moves available")
            return MoveResult(success=False, restart_requested=False)

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

        # Execute each stage with VLA control
        print("\n" + "=" * 60)
        print("VLA-CONTROLLED EXECUTION")
        print("=" * 60)
        print()

        for i, stage in enumerate(stages, 1):
            print("\n" + "-" * 60)
            print(f"Stage {i}/{len(stages)}: {stage['description']}")
            print("-" * 60)
            print(f"  VLM Prompt: {stage['vlm_prompt']}")
            print()

            # Wait for user to press ENTER to start VLA control, or R to recapture
            user_input = input("Press ENTER to hand control to VLA (or R to recapture board)...")
            if user_input.strip().lower() == 'r':
                print("\n[INFO] Recapturing board...")
                return MoveResult(success=False, restart_requested=True)

            # Run VLA control loop
            vla_control_loop(
                vla=vla,
                global_camera=global_camera,
                gripper_camera=gripper_camera,
                language_prompt=stage['vlm_prompt'],
                stage=stage,
                detector=detector,
                perspective_matrix=perspective_matrix,
                camera_rotation=camera_rotation,
                virtual_cam=virtual_cam,
                duration=50.0  # 50 second timeout per stage
            )

            print()

        print("\n" + "=" * 60)
        print("[OK] All stages completed!")
        print("=" * 60)
        return MoveResult(success=True, restart_requested=False)

    except FileNotFoundError:
        print(f"[X] Chess engine not found")
        return MoveResult(success=False, restart_requested=False)
    except Exception as e:
        print(f"[X] Error during move execution: {e}")
        import traceback
        traceback.print_exc()
        return MoveResult(success=False, restart_requested=False)


def main():
    """Main VLA deployment driver."""
    parser = argparse.ArgumentParser(
        description="VLA Deploy - Multi-model inference for chess robot control"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pi0",
        choices=list_models(),
        help="Model type to deploy (default: pi0)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (default: base model weights)"
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
        default=None,
        help="Gripper camera device (auto-detects eMeet C950 if not specified)"
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
    print("VLA Deploy - Chess Robot Control")
    print("=" * 60)

    # Clear GPU memory and check availability before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # GB
        total_mem = torch.cuda.mem_get_info()[1] / (1024**3)  # GB
        print(f"[INFO] GPU Memory: {free_mem:.1f} GB free / {total_mem:.1f} GB total")

        if free_mem < 15.0:
            print("[WARN] Low GPU memory! Other processes may be using the GPU.")
            print("       Run 'nvidia-smi' to check for lingering processes.")
    else:
        print("[WARN] CUDA not available, will use CPU (slow)")
    print()
    print(f"Model: {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint or f'Base {args.model.upper()} weights'}")
    print(f"Board rotation: {args.rotation}")
    print(f"Turn: {args.turn.capitalize()}")
    print()

    # Step 1: Initialize SO-100 robot arm with RobotController
    print("=" * 60)
    print("STAGE 1: Initialize SO-100 Robot Arm")
    print("=" * 60)

    robot_controller = None
    if not args.no_robot and SO100_AVAILABLE:
        try:
            # Load port-specific joint configs for calibrated home positions
            config_dir = Path("data")
            joint_configs, config_source = load_joint_configs_for_port(args.robot_port, config_dir)

            print(f"[INFO] Config source: {config_source}")
            home_degs = [f"{np.degrees(c.home_rad):.1f}" for c in joint_configs]
            print(f"[INFO] Home positions: [{', '.join(home_degs)}] deg")

            # Create RobotController for position-controlled operation
            robot_controller = RobotController(
                port=args.robot_port,
                joint_configs=joint_configs,
                config_source=config_source,
                enable_stability_system=True  # Enable joint 0/1 stability
            )

            if robot_controller.connect():
                print(f"[OK] SO-100 connected on {args.robot_port}")

                # Enable torque and start control loop targeting home
                robot_controller.enable_torque()
                robot_controller.set_home_targets()
                robot_controller.start_control_loop()
                print(f"[OK] Control loop started, robot holding at home position")

                state = robot_controller.arm.get_state()
                print(f"[OK] Current joint positions: {np.degrees(state.joint_positions)}")
            else:
                print(f"[X] Failed to connect to SO-100")
                robot_controller = None
        except Exception as e:
            print(f"[X] SO-100 connection error: {e}")
            import traceback
            traceback.print_exc()
            robot_controller = None
    else:
        if args.no_robot:
            print("[WARN] Running without robot (--no-robot flag)")
        else:
            print("[WARN] SO-100 control not available")

    # Step 2: Camera selection and validation (BEFORE model loading)
    print("\n" + "=" * 60)
    print("STAGE 2: Camera Selection")
    print("=" * 60)

    # Global camera selection: specified > auto-detect by name > prompt
    if args.global_camera:
        global_camera = args.global_camera
        print(f"[OK] Global camera: {global_camera} (specified)")
    else:
        global_camera = get_default_global_camera()
        if global_camera:
            print(f"[OK] Global camera: {global_camera} (auto-detected WBC-0E01)")
        else:
            # Fallback to user prompt
            cameras = get_available_cameras()
            if not cameras:
                print("[X] No cameras found!")
                if robot_arm:
                    robot_arm.disconnect()
                return 1
            elif len(cameras) == 1:
                global_camera = cameras[0][0]
                print(f"[OK] Global camera: {global_camera} (auto-selected)")
            else:
                print("[WARN] WBC-0E01 not found, prompting for selection...")
                global_camera = select_camera(cameras)

    # Gripper camera selection: specified > auto-detect by name > error
    if args.gripper_camera:
        gripper_camera = args.gripper_camera
        print(f"[OK] Gripper camera: {gripper_camera} (specified)")
    else:
        gripper_camera = get_default_gripper_camera()
        if gripper_camera:
            print(f"[OK] Gripper camera: {gripper_camera} (auto-detected eMeet C950)")
        else:
            print("[X] Gripper camera (eMeet C950) not found!")
            print("    Use --gripper-camera to specify manually")
            if robot_arm:
                robot_arm.disconnect()
            return 1

    # Validate BOTH cameras work before loading model (fail fast)
    # Both are required: global for overlay detection, gripper for VLA input
    print("\n[INFO] Validating cameras...")

    test_frame = capture_720p_frame(global_camera)
    if test_frame is None:
        print(f"[X] Global camera {global_camera} not working!")
        if robot_controller:
            robot_controller.disconnect()
        return 1
    print(f"[OK] Global camera validated: {test_frame.shape}")

    gripper_test = capture_gripper_frame(gripper_camera)
    if gripper_test is None:
        print(f"[X] Gripper camera {gripper_camera} not working!")
        print("    Both cameras are REQUIRED for VLA deployment.")
        if robot_controller:
            robot_controller.disconnect()
        return 1
    print(f"[OK] Gripper camera validated: {gripper_test.shape}")

    # Step 3: Initialize virtual camera for overlay streaming
    print("\n" + "=" * 60)
    print("STAGE 3: Virtual Camera Output")
    print("=" * 60)

    virtual_cam = None
    try:
        virtual_cam = VirtualCamera(
            device_path="/dev/video7",
            width=1280,
            height=720
        )
        if virtual_cam.start():
            print("[OK] Virtual camera started: /dev/video7")
        else:
            print("[INFO] Virtual camera not available (visualization only)")
            print("       Overlays are still applied to VLA input - this is OK")
            virtual_cam = None
    except Exception as e:
        print(f"[INFO] Virtual camera not available: {e}")
        print("       Overlays are still applied to VLA input - this is OK")
        virtual_cam = None

    # Step 4: Initialize VLA model (GPU-intensive)
    print("\n" + "=" * 60)
    print("STAGE 4: Initialize VLA Model")
    print("=" * 60)

    vla = VLAWrapper(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        robot_controller=robot_controller
    )

    # Step 5: Initialize guidance components
    print("\n" + "=" * 60)
    print("STAGE 5: Initialize Guidance System")
    print("=" * 60)

    detector = BoardDetector(camera_position=args.rotation)
    print(f"[OK] BoardDetector initialized (rotation: {args.rotation})")

    try:
        calculator = MoveCalculator(engine_path=args.engine)
        print(f"[OK] MoveCalculator initialized (engine: {args.engine})")
    except FileNotFoundError:
        print(f"[X] Chess engine not found at '{args.engine}'")
        print("  Install stockfish or specify engine path with --engine")
        if virtual_cam:
            virtual_cam.stop()
        if robot_controller:
            robot_controller.disconnect()
        return 1

    # Ready for game loop
    print("\n" + "=" * 60)
    print("[READY] VLA Chess Robot Ready for Play")
    print("=" * 60)
    print(f"  Robot plays: {args.turn.capitalize()}")
    print(f"  Engine: {args.engine}")
    print(f"  Global camera: {global_camera}")
    print(f"  Gripper camera: {gripper_camera}")
    print(f"  Virtual camera: {'enabled' if virtual_cam else 'disabled'}")
    if robot_controller:
        print(f"  Robot: {args.robot_port} (holding at home position)")
    else:
        print(f"  Robot: None (--no-robot mode)")
    print()
    print("  Press ENTER to capture board and calculate move")
    print("  Press Ctrl+C to exit")
    print("=" * 60)

    # Main game loop
    move_count = 0
    try:
        while True:
            # Robot holds at home position while waiting (control loop running)
            input("\nPress ENTER to start next move...")

            # Loop for board capture with restart support
            while True:
                move_count += 1
                print(f"\n[MOVE {move_count}]")

                result = execute_move(
                    vla=vla,
                    detector=detector,
                    calculator=calculator,
                    global_camera=global_camera,
                    gripper_camera=gripper_camera,
                    virtual_cam=virtual_cam,
                    turn_letter=turn_letter,
                    camera_rotation=args.rotation,
                    corner_conf=args.corner_conf,
                    min_corner_dist=args.min_corner_dist,
                    time_limit=args.time
                )

                if result.restart_requested:
                    # Decrement move count since we're retrying
                    move_count -= 1
                    print("\n[INFO] Recapturing board for new detection...")
                    continue
                else:
                    # Move completed (success or failure), exit inner loop
                    break

            # Reset robot to home position after move completes
            vla.reset_to_home()

            if result.success:
                print("\n[OK] Move completed successfully!")
            else:
                print("\n[WARN] Move failed or was aborted")

            print("\n" + "-" * 60)
            print(f"Moves executed: {move_count}")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\n[INFO] Exiting...")

    finally:
        # Cleanup
        if virtual_cam:
            print("Stopping virtual camera...")
            virtual_cam.stop()
        if robot_controller:
            print("Disconnecting from SO-100...")
            robot_controller.disconnect()
        print("[OK] Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
