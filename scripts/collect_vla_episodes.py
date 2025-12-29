#!/usr/bin/env python3
"""
VLA Episode Collection Script

Passive recording of tele-op sessions for VLA training data.
Reads robot joint positions from state_cache.json (written by tele_op.py).

Requirements:
- tele_op.py running in another terminal (controls robot)
- v4l2loopback loaded (for /dev/video7 virtual camera output)
- Physical cameras connected:
  - Global camera: WBC-0E01 at /dev/video4
  - Gripper camera: eMeet C950 at /dev/video1

Usage:
    # Terminal 1: Start tele-op
    python scripts/tele_op.py

    # Terminal 2: Start episode collection (default cameras)
    python scripts/collect_vla_episodes.py --output data/episodes/

    # With specific cameras
    python scripts/collect_vla_episodes.py --output data/episodes/ \\
        --global-camera /dev/video4 \\
        --gripper-camera /dev/video1 \\
        --virtual-camera /dev/video7
"""

import argparse
import time
import cv2
import chess
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cameras import LiveCameraCapture, VirtualCamera
from guidance import (
    BoardDetector,
    MoveCalculator,
    apply_stage_overlay_to_frame
)
from guidance.move_decomposer import decompose_move
from utils.state_cache import StateCache
from utils.keyboard_input import KeyboardInput
from utils.camera_helpers import get_camera_index_from_device

# Try to import LeRobot (optional dependency)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARNING] LeRobot not installed. Install with: pip install lerobot")

# Constants
DEFAULT_FPS = 15
FRAME_INTERVAL = 1.0 / DEFAULT_FPS  # 0.0667 seconds
POSITION_TOLERANCE_DEG = 2.0  # Degrees tolerance for position mismatch warning
POSITION_TOLERANCE_RAD = POSITION_TOLERANCE_DEG * (np.pi / 180.0)  # ~0.0349 radians

# LeRobot dataset features schema
# Note: gripper is joint_5 in joint_positions (no separate gripper_state)
DATASET_FEATURES = {
    "observation.global_camera": {
        "dtype": "video",
        "shape": (720, 1280, 3),  # H, W, C (BGR from global camera)
        "names": ["height", "width", "channels"]
    },
    "observation.gripper_camera": {
        "dtype": "video",
        "shape": (360, 640, 3),  # Gripper view (eMeet C950 actual resolution)
        "names": ["height", "width", "channels"]
    },
    "observation.joint_positions": {
        "dtype": "float32",
        "shape": (6,),  # 6 SO-100 joints in radians (joint_5 is gripper)
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),  # Copy of joint_positions (action cloning)
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    },
    "language_instruction": {
        "dtype": "string",
        "shape": (1,),  # VLM prompt from move_decomposer
        "names": None
    }
}


class EpisodeRecorder:
    """
    Episode recorder for VLA training data collection.

    Records synchronized video (global + gripper cameras) + robot joint positions
    from tele-op sessions at 15 FPS in LeRobot dataset format.
    """

    def __init__(
        self,
        output_dir: str,
        global_camera_device: str = "/dev/video4",
        gripper_camera_device: str = "/dev/video1",
        virtual_camera_device: str = "/dev/video7",
        fps: int = DEFAULT_FPS,
        engine_path: str = "stockfish",
        camera_rotation: str = "right",
        turn: str = "black",
        use_lerobot: bool = True
    ):
        """
        Initialize episode recorder.

        Args:
            output_dir: Directory to save episodes
            global_camera_device: Physical global camera device (WBC-0E01)
            gripper_camera_device: Physical gripper camera device (eMeet C950)
            virtual_camera_device: Virtual camera output device (v4l2loopback)
            fps: Recording frame rate (default: 15)
            engine_path: Path to UCI chess engine
            camera_rotation: Camera rotation ('top', 'right', 'bottom', 'left')
            turn: Whose turn to calculate move for ('white' or 'black')
            use_lerobot: Whether to use LeRobot dataset format
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_camera_device = global_camera_device
        self.gripper_camera_device = gripper_camera_device
        self.virtual_camera_device = virtual_camera_device
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.camera_rotation = camera_rotation
        self.turn = "w" if turn == "white" else "b"  # Normalize to single letter
        self.use_lerobot = use_lerobot and LEROBOT_AVAILABLE

        # Initialize components
        self.cache = StateCache("data/state_cache.json")
        self.keyboard = KeyboardInput()

        # Initialize guidance components
        self.detector = BoardDetector(
            corner_model_path="data/best_corners.pt",
            piece_model_path="data/best_transformed_detection.pt"
        )
        self.calculator = MoveCalculator(engine_path=engine_path)

        # Camera components (initialized in start_cameras())
        self.global_cam_capture: Optional[LiveCameraCapture] = None
        self.gripper_cam: Optional[cv2.VideoCapture] = None
        self.virtual_cam: Optional[VirtualCamera] = None

        # Dataset (initialized in _initialize_dataset())
        self.dataset: Optional[Any] = None
        self.episode_count = 0

        # State
        self.perspective_matrix: Optional[np.ndarray] = None
        self.transformed_image_path = "data/chessboard_transformed.png"

        print(f"[OK] EpisodeRecorder initialized")
        print(f"     Output: {self.output_dir}")
        print(f"     FPS: {self.fps}")
        print(f"     LeRobot: {'enabled' if self.use_lerobot else 'disabled (fallback to raw files)'}")

    def _initialize_dataset(self):
        """Initialize or load LeRobot dataset."""
        if not self.use_lerobot:
            print("[INFO] LeRobot disabled, using raw file storage")
            return

        try:
            # LeRobot v0.4 uses meta/info.json
            lerobot_meta = self.output_dir / "meta" / "info.json"
            repo_id = f"local/chess_vla_{self.output_dir.name}"

            if lerobot_meta.exists():
                print(f"[INFO] Loading existing dataset from {self.output_dir}")
                self.dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=str(self.output_dir),
                    download_videos=False
                )
                self.episode_count = self.dataset.num_episodes
            else:
                # Handle existing directory - LeRobot.create() fails if directory exists
                if self.output_dir.exists():
                    import shutil
                    print(f"[INFO] Removing existing directory: {self.output_dir}")
                    shutil.rmtree(self.output_dir)

                print(f"[INFO] Creating new dataset at {self.output_dir}")
                self.dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=str(self.output_dir),
                    fps=self.fps,
                    features=DATASET_FEATURES,
                    use_videos=True  # Enable MP4 encoding
                )
                self.episode_count = 0

            print(f"[OK] Dataset ready: {self.episode_count} existing episodes")

        except Exception as e:
            print(f"[ERROR] Failed to initialize LeRobot dataset: {e}")
            print("[INFO] Falling back to raw file storage")
            self.use_lerobot = False

    def start_cameras(self) -> bool:
        """
        Initialize cameras and virtual camera output.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Start live capture from virtual camera (overlay stream)
            print(f"[INFO] Starting global camera capture: {self.global_camera_device}")
            self.global_cam_capture = LiveCameraCapture(self.global_camera_device)
            self.global_cam_capture.start()

            # Give camera time to initialize
            time.sleep(1.0)

            # Check if capture is working
            test_frame = self.global_cam_capture.get_latest_frame()
            if test_frame is None:
                print(f"[ERROR] Failed to capture from {self.global_camera_device}")
                return False

            print(f"[OK] Global camera ready: {test_frame.shape}")

            # Start gripper camera
            print(f"[INFO] Starting gripper camera: {self.gripper_camera_device}")
            gripper_index = get_camera_index_from_device(self.gripper_camera_device)
            self.gripper_cam = cv2.VideoCapture(gripper_index)

            # Configure gripper camera (224x224 for eMeet C950)
            self.gripper_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
            self.gripper_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
            self.gripper_cam.set(cv2.CAP_PROP_FPS, 30)

            # Test gripper camera
            ret, test_gripper = self.gripper_cam.read()
            if not ret:
                print(f"[ERROR] Failed to capture from gripper camera")
                return False

            print(f"[OK] Gripper camera ready: {test_gripper.shape}")

            # Initialize virtual camera output (for streaming overlays)
            print(f"[INFO] Starting virtual camera output: {self.virtual_camera_device}")
            self.virtual_cam = VirtualCamera(
                device_path=self.virtual_camera_device,
                width=1280,
                height=720
            )

            if not self.virtual_cam.start():
                print("[ERROR] Failed to start virtual camera output")
                return False

            print("[OK] Virtual camera ready for overlay streaming")

            return True

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def stop_cameras(self):
        """Stop all cameras."""
        if self.global_cam_capture:
            self.global_cam_capture.stop()

        if self.gripper_cam:
            self.gripper_cam.release()

        if self.virtual_cam:
            self.virtual_cam.stop()

        print("[OK] Cameras stopped")

    def _detect_and_decompose_move(self, board_image_path: str) -> Optional[Dict]:
        """
        Detect board state, calculate best move, and decompose into stages.

        Args:
            board_image_path: Path to board image

        Returns:
            Dictionary with 'board', 'move', 'stages', or None if detection fails
        """
        try:
            # Detect board state (returns FEN string and transformed image)
            print("[INFO] Detecting board state...")
            fen, transformed_image = self.detector.detect_board_state(
                board_image_path,
                debug=True,  # Save visualizations
                turn=self.turn
            )

            if fen is None:
                print("[ERROR] Board detection failed")
                return None

            # Create chess.Board from FEN
            try:
                board = chess.Board(fen)
            except ValueError as e:
                print(f"[ERROR] Invalid FEN: {fen}")
                return None

            # Get perspective matrix from detector (stored during detection)
            self.perspective_matrix = self.detector.perspective_matrix

            print(f"[OK] Board detected: {board.fen()}")

            # Calculate best move
            print("[INFO] Calculating best move...")
            move = self.calculator.calculate_best_move(board)

            if move is None:
                print("[ERROR] Move calculation failed")
                return None

            print(f"[OK] Best move: {move.uci()} ({board.san(move)})")

            # Decompose move into stages
            print("[INFO] Decomposing move into stages...")
            stages = decompose_move(board, move)

            print(f"[OK] Move decomposed into {len(stages)} stage(s)")
            for i, stage in enumerate(stages):
                print(f"     Stage {i+1}: {stage['description']}")

            return {
                "board": board,
                "move": move,
                "stages": stages
            }

        except Exception as e:
            print(f"[ERROR] Detection/decomposition failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_and_stream_overlay(self, stage: Dict):
        """
        Generate overlay for stage and stream to virtual camera.

        Args:
            stage: Stage dictionary from decompose_move()
        """
        try:
            # Get live frame from global camera
            frame = self.global_cam_capture.get_latest_frame()

            if frame is None:
                print("[WARNING] No frame available for overlay")
                return

            # Apply stage overlay
            overlayed_frame = apply_stage_overlay_to_frame(
                frame=frame,
                stage=stage,
                transformed_image_path=self.transformed_image_path,
                detector=self.detector,
                perspective_matrix=self.perspective_matrix,
                camera_position=self.camera_rotation
            )

            if overlayed_frame is not None:
                # Stream to virtual camera
                self.virtual_cam.write_frame(overlayed_frame)
            else:
                print("[WARNING] Overlay generation failed, streaming raw frame")
                self.virtual_cam.write_frame(frame)

        except Exception as e:
            print(f"[ERROR] Overlay streaming failed: {e}")

    def _record_stage(self, stage: Dict, stage_index: int, total_stages: int) -> Optional[List[Dict]]:
        """
        Record one stage at 15 FPS until user presses ENTER.

        Args:
            stage: Stage dictionary from decompose_move()
            stage_index: Current stage index (0-based)
            total_stages: Total number of stages

        Returns:
            List of frame dictionaries or None if recording fails
        """
        print(f"\n{'='*60}")
        print(f"Stage {stage_index + 1}/{total_stages}")
        print(f"{'='*60}")
        print(f"Description: {stage['description']}")
        print(f"VLM Prompt: {stage['vlm_prompt']}")
        print(f"{'='*60}\n")

        # Generate and stream overlay
        print("[INFO] Streaming overlay to virtual camera...")
        self._generate_and_stream_overlay(stage)

        # Wait for user to press ENTER to start recording
        print("\n[PROMPT] Press ENTER to START recording...")
        while True:
            key = self.keyboard.get_key(timeout=0.1)
            if key == 'ENTER':
                break
            # Continue streaming overlay while waiting
            self._generate_and_stream_overlay(stage)

        print(f"[INFO] Recording stage {stage_index + 1} at {self.fps} FPS...")
        print("[PROMPT] Press ENTER to STOP recording")

        recording_active = True
        episode_frames = []
        start_time = time.time()
        frame_count = 0

        try:
            while recording_active:
                frame_start = time.time()

                # Capture synchronized data
                global_frame = self.global_cam_capture.get_latest_frame()
                ret, gripper_frame = self.gripper_cam.read()

                # Read robot state from cache (NO serial conflict!)
                # Note: gripper is joint_5 in joint_positions
                joint_positions = self.cache.get("robot_state.joint_positions")

                # Validate data
                if global_frame is None:
                    print("[WARNING] No global frame, skipping")
                    continue

                if not ret or gripper_frame is None:
                    print("[WARNING] No gripper frame, using black placeholder")
                    gripper_frame = np.zeros((224, 224, 3), dtype=np.uint8)

                if joint_positions is None or len(joint_positions) != 6:
                    print("[WARNING] Invalid joint positions, using zeros")
                    joint_positions = [0.0] * 6

                # Apply overlay to global frame
                overlayed_frame = apply_stage_overlay_to_frame(
                    frame=global_frame,
                    stage=stage,
                    transformed_image_path=self.transformed_image_path,
                    detector=self.detector,
                    perspective_matrix=self.perspective_matrix,
                    camera_position=self.camera_rotation
                )

                if overlayed_frame is None:
                    overlayed_frame = global_frame

                # Stream to virtual camera
                self.virtual_cam.write_frame(overlayed_frame)

                # Store frame
                episode_frames.append({
                    "global_frame": overlayed_frame.copy(),
                    "gripper_frame": gripper_frame.copy(),
                    "joint_positions": np.array(joint_positions, dtype=np.float32),
                    "timestamp": time.time() - start_time,
                    "vlm_prompt": stage["vlm_prompt"]
                })

                frame_count += 1

                # Check for ENTER key to stop
                key = self.keyboard.get_key(timeout=0.001)
                if key == 'ENTER':
                    recording_active = False

                # Maintain FPS timing
                elapsed = time.time() - frame_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -0.01:  # More than 10ms behind
                    print(f"[WARNING] Frame {frame_count} behind schedule by {-sleep_time:.3f}s")

            duration = time.time() - start_time
            actual_fps = frame_count / duration if duration > 0 else 0

            print(f"\n[OK] Recording stopped")
            print(f"     Frames: {frame_count}")
            print(f"     Duration: {duration:.2f}s")
            print(f"     Actual FPS: {actual_fps:.2f}")

            # Check for position mismatch between start and end
            if len(episode_frames) >= 2:
                start_positions = episode_frames[0]["joint_positions"]
                end_positions = episode_frames[-1]["joint_positions"]
                mismatched_joints = []

                for i in range(len(start_positions)):
                    diff_rad = abs(start_positions[i] - end_positions[i])
                    diff_deg = diff_rad * (180.0 / np.pi)
                    if diff_rad > POSITION_TOLERANCE_RAD:
                        mismatched_joints.append((i, diff_deg))

                if mismatched_joints:
                    print(f"\n[WARNING] Position mismatch detected (>{POSITION_TOLERANCE_DEG} deg):")
                    for joint_idx, diff_deg in mismatched_joints:
                        print(f"          Joint {joint_idx}: {diff_deg:.1f} degrees difference")

            return episode_frames

        except KeyboardInterrupt:
            print("\n[INFO] Recording interrupted by user")
            return None
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_episode_lerobot(self, all_frames: List[Dict], move_info: Dict):
        """
        Save episode to LeRobot dataset.

        Args:
            all_frames: List of all frame dictionaries from all stages
            move_info: Dictionary with 'board', 'move', 'stages'
        """
        try:
            episode_index = self.episode_count

            print(f"\n[INFO] Writing episode {episode_index} to LeRobot dataset...")

            for frame_idx, frame_data in enumerate(all_frames):
                self.dataset.add_frame({
                    "observation.global_camera": frame_data["global_frame"],
                    "observation.gripper_camera": frame_data["gripper_frame"],
                    "observation.joint_positions": frame_data["joint_positions"],
                    "action": frame_data["joint_positions"],  # Copy for action cloning
                    "language_instruction": frame_data["vlm_prompt"],
                    "task": frame_data["vlm_prompt"]  # Required by LeRobot v0.4
                })

            # Finalize episode (encode videos, write Parquet)
            self.dataset.save_episode()

            print(f"[OK] Episode {episode_index} saved: {len(all_frames)} frames")
            self.episode_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to save episode to LeRobot: {e}")
            import traceback
            traceback.print_exc()

    def _save_episode_raw(self, all_frames: List[Dict], move_info: Dict):
        """
        Save episode as raw files (fallback if LeRobot not available).

        Args:
            all_frames: List of all frame dictionaries from all stages
            move_info: Dictionary with 'board', 'move', 'stages'
        """
        try:
            episode_index = self.episode_count
            episode_dir = self.output_dir / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[INFO] Saving episode {episode_index} to {episode_dir}...")

            # Save frames as images
            for frame_idx, frame_data in enumerate(all_frames):
                cv2.imwrite(
                    str(episode_dir / f"global_{frame_idx:06d}.png"),
                    frame_data["global_frame"]
                )
                cv2.imwrite(
                    str(episode_dir / f"gripper_{frame_idx:06d}.png"),
                    frame_data["gripper_frame"]
                )

            # Save metadata
            import json
            metadata = {
                "episode_index": episode_index,
                "frame_count": len(all_frames),
                "fps": self.fps,
                "move": move_info["move"].uci(),
                "fen": move_info["board"].fen(),
                "stages": move_info["stages"],
                "joint_positions": [frame["joint_positions"].tolist() for frame in all_frames],
                "timestamps": [frame["timestamp"] for frame in all_frames]
            }

            with open(episode_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Episode {episode_index} saved: {len(all_frames)} frames")
            self.episode_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to save raw episode: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main episode collection loop."""
        print(f"\n{'='*60}")
        print("VLA Episode Collection")
        print(f"{'='*60}\n")

        # Initialize dataset
        self._initialize_dataset()

        # Start cameras
        if not self.start_cameras():
            print("[ERROR] Camera initialization failed, exiting")
            return

        # Use keyboard context manager to enable raw terminal input
        with self.keyboard:
            try:
                print("\n[INFO] Episode collection ready")
                print("[PROMPT] Press ENTER to capture board and start new episode (Ctrl+C to quit)...")

                while True:
                    # Wait for user to press ENTER
                    key = self.keyboard.get_key(timeout=0.1)
                    if key != 'ENTER':
                        continue

                    # Capture board image
                    board_image_path = "data/board_capture.png"
                    frame = self.global_cam_capture.get_latest_frame()

                    if frame is None:
                        print("[ERROR] Failed to capture board image")
                        continue

                    cv2.imwrite(board_image_path, frame)
                    print(f"[OK] Board image captured: {board_image_path}")

                    # Detect and decompose move
                    move_info = self._detect_and_decompose_move(board_image_path)

                    if move_info is None:
                        print("[ERROR] Failed to process board, try again")
                        continue

                    # Record each stage
                    all_frames = []
                    episode_aborted = False

                    for stage_idx, stage in enumerate(move_info["stages"]):
                        # Retry loop for each stage
                        while True:
                            frames = self._record_stage(stage, stage_idx, len(move_info["stages"]))

                            if frames is None:
                                print("[WARNING] Stage recording failed, skipping episode")
                                episode_aborted = True
                                break

                            # Ask if user wants to keep or retry
                            print("\n[PROMPT] Keep this recording? (ENTER=Keep, R=Retry, Q=Abort episode)")
                            while True:
                                key = self.keyboard.get_key(timeout=0.1)
                                if key == 'ENTER':
                                    # Keep recording, move to next stage
                                    all_frames.extend(frames)
                                    break
                                elif key in ('r', 'R'):
                                    # Retry this stage
                                    print("[INFO] Retrying stage...")
                                    break
                                elif key in ('q', 'Q'):
                                    # Abort entire episode
                                    print("[INFO] Aborting episode...")
                                    episode_aborted = True
                                    break

                            # Check if we should keep this recording or retry
                            if key == 'ENTER':
                                break  # Exit retry loop, move to next stage
                            elif episode_aborted:
                                break  # Exit retry loop, abort episode
                            # Otherwise key was 'r'/'R', continue retry loop

                        if episode_aborted:
                            break  # Exit stage loop

                    if episode_aborted or all_frames is None or len(all_frames) == 0:
                        print("[WARNING] Episode aborted or no frames recorded, skipping")
                        print("[PROMPT] Press ENTER to start next episode (Ctrl+C to quit)...")
                        continue

                    # Save episode
                    if self.use_lerobot:
                        self._save_episode_lerobot(all_frames, move_info)
                    else:
                        self._save_episode_raw(all_frames, move_info)

                    # Summary
                    print(f"\n{'='*60}")
                    print(f"Episode Collection Summary")
                    print(f"{'='*60}")
                    print(f"Total episodes: {self.episode_count}")
                    print(f"Last episode frames: {len(all_frames)}")
                    print(f"Output directory: {self.output_dir}")
                    print(f"{'='*60}\n")

                    print("[PROMPT] Press ENTER to start next episode (Ctrl+C to quit)...")

            except KeyboardInterrupt:
                print("\n\n[INFO] Collection stopped by user")

            finally:
                self.stop_cameras()
                print("\n[OK] Episode collection complete")
                print(f"     Total episodes: {self.episode_count}")
                print(f"     Output: {self.output_dir}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Collect VLA training episodes from tele-op sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (requires tele_op.py running in another terminal)
    python scripts/collect_vla_episodes.py --output data/episodes/

    # With specific cameras
    python scripts/collect_vla_episodes.py --output data/episodes/ \\
        --global-camera /dev/video4 \\
        --gripper-camera /dev/video1 \\
        --virtual-camera /dev/video7

    # Custom FPS and engine
    python scripts/collect_vla_episodes.py --output data/episodes/ \\
        --fps 30 \\
        --engine /usr/local/bin/stockfish

    # Without LeRobot (raw file storage)
    python scripts/collect_vla_episodes.py --output data/episodes/ --no-lerobot
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/episodes",
        help="Output directory for episodes (default: data/episodes)"
    )

    parser.add_argument(
        "--global-camera",
        type=str,
        default="/dev/video4",
        help="Physical global camera device (WBC-0E01, default: /dev/video4)"
    )

    parser.add_argument(
        "--gripper-camera",
        type=str,
        default="/dev/video1",
        help="Physical gripper camera device (eMeet C950, default: /dev/video1)"
    )

    parser.add_argument(
        "--virtual-camera",
        type=str,
        default="/dev/video7",
        help="Virtual camera output device (v4l2loopback, default: /dev/video7)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Recording frame rate (default: {DEFAULT_FPS})"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="stockfish",
        help="Path to UCI chess engine (default: stockfish)"
    )

    parser.add_argument(
        "--rotation",
        type=str,
        choices=["top", "right", "bottom", "left"],
        default="right",
        help="Camera rotation (default: right)"
    )

    parser.add_argument(
        "--turn",
        type=str,
        choices=["white", "black"],
        default="black",
        help="Whose turn to calculate move for (default: black)"
    )

    parser.add_argument(
        "--no-lerobot",
        action="store_true",
        help="Disable LeRobot dataset format (use raw files)"
    )

    args = parser.parse_args()

    # Check if LeRobot is required but not available
    if not args.no_lerobot and not LEROBOT_AVAILABLE:
        print("[ERROR] LeRobot not installed. Install with: pip install lerobot")
        print("[INFO] Or use --no-lerobot flag for raw file storage")
        return 1

    # Create recorder
    recorder = EpisodeRecorder(
        output_dir=args.output,
        global_camera_device=args.global_camera,
        gripper_camera_device=args.gripper_camera,
        virtual_camera_device=args.virtual_camera,
        fps=args.fps,
        engine_path=args.engine,
        camera_rotation=args.rotation,
        turn=args.turn,
        use_lerobot=not args.no_lerobot
    )

    # Run collection loop
    recorder.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
