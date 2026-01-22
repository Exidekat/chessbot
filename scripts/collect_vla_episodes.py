#!/usr/bin/env python3
"""
VLA Episode Collection Script (LeRobot Format)

Records tele-op sessions for VLA training in LeRobot v3.0 dataset format.
Reads robot joint positions from state_cache.json (written by tele_op.py).

Output Format:
    LeRobot v3.0 dataset with:
    - observation.images.global: 1280x720 global camera video
    - observation.images.gripper: Native resolution gripper camera video (camera-agnostic)
    - observation.state: 6-DOF joint positions (radians)
    - action: 6-DOF target joint positions

Requirements:
- tele_op.py running in another terminal (controls robot)
- v4l2loopback loaded (for /dev/video7 virtual camera output)
- Physical cameras connected:
  - Global camera: WBC-0E01
  - Gripper camera: eMeet C950

Usage:
    # Terminal 1: Start tele-op
    python scripts/tele_op.py

    # Terminal 2: Start episode collection
    python scripts/collect_vla_episodes.py --output data/lerobot_episodes/

    # With specific cameras
    python scripts/collect_vla_episodes.py --output data/lerobot_episodes/ \\
        --global-camera /dev/video4 \\
        --gripper-camera /dev/video1
"""

import argparse
import json
import queue
import shutil
import tempfile
import threading
import time
import cv2
import chess
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
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
from utils.camera_helpers import (
    get_camera_index_from_device,
    get_default_global_camera,
    get_default_gripper_camera_with_name,
    get_available_cameras,
    select_camera,
    capture_4k_downscale,
    capture_720p_yuyv,
    DEFAULT_USE_YUYV,
)

# Configure HuggingFace for offline/local usage
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Disable HuggingFace Hub access
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# SVT-AV1 encoder logging (set before LeRobot import)
# Will be updated based on --debug flag in main()
# 0=quiet, 1=error, 2=warning, 3=info, 4=debug
os.environ["SVT_LOG"] = "0"  # Default: quiet (no encoder spam)

# Import LeRobot (required dependency)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"[X] LeRobot not installed: {e}")
    print("    Install with: pip install lerobot")
    sys.exit(1)

# Constants
DEFAULT_FPS = 15
FRAME_INTERVAL = 1.0 / DEFAULT_FPS  # 0.0667 seconds
POSITION_TOLERANCE_DEG = 2.0  # Degrees tolerance for position mismatch warning
POSITION_TOLERANCE_RAD = POSITION_TOLERANCE_DEG * (np.pi / 180.0)  # ~0.0349 radians

# LeRobot dataset features schema (v3.0 format)
# Uses standard LeRobot naming: observation.images.*, observation.state, action
# Note: gripper is joint_5 in joint_positions (no separate gripper_state)
DATASET_FEATURES = {
    # Global camera (1280x720)
    "observation.images.global": {
        "dtype": "video",
        "shape": (720, 1280, 3),  # H, W, C
        "names": ["height", "width", "channels"]
    },
    # Gripper camera (native resolution - camera-agnostic)
    # LeRobot dynamically infers actual shape from first frame when use_videos=True
    # Common resolutions: 640x480, 1280x720, 1920x1080
    "observation.images.gripper": {
        "dtype": "video",
        "shape": (480, 640, 3),  # H, W, C - placeholder, actual shape from camera
        "names": ["height", "width", "channels"]
    },
    # Robot state (6 DOF joint positions in radians)
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    },
    # Action (6 DOF target joint positions)
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
    },
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
        use_yuyv: bool = False,
    ):
        """
        Initialize episode recorder.

        Args:
            output_dir: Directory to save episodes (LeRobot dataset)
            global_camera_device: Physical global camera device (WBC-0E01)
            gripper_camera_device: Physical gripper camera device (eMeet C950)
            virtual_camera_device: Virtual camera output device (v4l2loopback)
            fps: Recording frame rate (default: 15)
            engine_path: Path to UCI chess engine
            camera_rotation: Camera rotation ('top', 'right', 'bottom', 'left')
            turn: Whose turn to calculate move for ('white' or 'black')
        """
        self.output_dir = Path(output_dir)

        self.global_camera_device = global_camera_device
        self.gripper_camera_device = gripper_camera_device
        self.virtual_camera_device = virtual_camera_device
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.camera_rotation = camera_rotation
        self.turn = "w" if turn == "white" else "b"  # Normalize to single letter
        self.use_yuyv = use_yuyv

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

        # Background saving thread
        self.save_queue: queue.Queue = queue.Queue()
        self.save_thread: Optional[threading.Thread] = None
        self.save_thread_running = False
        self.save_lock = threading.Lock()  # Protects episode_count and dataset access

        print(f"[OK] EpisodeRecorder initialized")
        print(f"     Output: {self.output_dir}")
        print(f"     FPS: {self.fps}")
        print(f"     Format: LeRobot v3.0")

    def _recover_corrupted_dataset(self) -> bool:
        """
        Scan for and remove corrupted parquet files before loading dataset.

        This runs before LeRobotDataset() is called to prevent load failures
        from corrupted parquet files (caused by OOM kills during save).

        Returns:
            True if recovery was performed, False if dataset was clean
        """
        data_dir = self.output_dir / "data"
        meta_episodes_dir = self.output_dir / "meta" / "episodes"

        if not data_dir.exists():
            return False

        corrupted_files = []
        valid_frame_count = 0
        valid_episode_indices = set()

        # Scan data parquet files
        for parquet_file in sorted(data_dir.rglob("*.parquet")):
            try:
                table = pq.read_table(str(parquet_file))
                valid_frame_count += len(table)
                # Track which episodes are valid
                if "episode_index" in table.column_names:
                    for idx in table.column("episode_index").to_pylist():
                        valid_episode_indices.add(idx)
            except Exception as e:
                print(f"[RECOVERY] Corrupted data file: {parquet_file.name}")
                corrupted_files.append(parquet_file)

        # Scan episode metadata parquet files
        if meta_episodes_dir.exists():
            for parquet_file in sorted(meta_episodes_dir.rglob("*.parquet")):
                try:
                    pq.read_table(str(parquet_file))
                except Exception:
                    print(f"[RECOVERY] Corrupted metadata file: {parquet_file.name}")
                    corrupted_files.append(parquet_file)

        if not corrupted_files:
            return False

        # Remove corrupted files
        print(f"\n[RECOVERY] Found {len(corrupted_files)} corrupted parquet file(s)")
        for f in corrupted_files:
            print(f"  Removing: {f}")
            f.unlink()

        # Update info.json with corrected counts
        info_path = self.output_dir / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)

                old_episodes = info.get("total_episodes", 0)
                old_frames = info.get("total_frames", 0)

                info["total_episodes"] = len(valid_episode_indices)
                info["total_frames"] = valid_frame_count

                # Update splits if present
                if "splits" in info:
                    info["splits"] = {"train": f"[0:{len(valid_episode_indices)}]"}

                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)

                print(f"  Updated info.json: {old_episodes} → {len(valid_episode_indices)} episodes, "
                      f"{old_frames} → {valid_frame_count} frames")
            except Exception as e:
                print(f"[RECOVERY] Warning: Could not update info.json: {e}")

        print(f"[RECOVERY] Dataset repaired. Continuing with load...\n")
        return True

    def _initialize_dataset(self):
        """Initialize or load LeRobot dataset."""
        # First, recover any corrupted parquet files from previous OOM crashes
        if self.output_dir.exists():
            self._recover_corrupted_dataset()

        # LeRobot v3.0 uses meta/info.json
        lerobot_meta = self.output_dir / "meta" / "info.json"
        repo_id = f"local/chess_vla_{self.output_dir.name}"

        if lerobot_meta.exists():
            print(f"[INFO] Loading existing dataset from {self.output_dir}")
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(self.output_dir),
            )
            self.episode_count = self.dataset.num_episodes
            # Ensure clean buffer state for recording new episodes
            # (previous session may have left corrupted buffer if save failed)
            self.dataset.episode_buffer = None
        else:
            # Handle existing directory - LeRobot.create() fails if directory exists
            if self.output_dir.exists():
                import shutil
                print(f"[INFO] Removing existing directory: {self.output_dir}")
                shutil.rmtree(self.output_dir)

            print(f"[INFO] Creating new LeRobot dataset at {self.output_dir}")
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=str(self.output_dir),
                fps=self.fps,
                features=DATASET_FEATURES,
                use_videos=True  # Enable MP4 encoding
            )
            self.episode_count = 0

        print(f"[OK] Dataset ready: {self.episode_count} existing episodes")

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

            # Configure gripper camera - capture at native resolution (camera-agnostic)
            # Resizing is deferred to training/deploy pipelines for flexibility
            # Request common resolution; camera will use closest supported
            self.gripper_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.gripper_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.gripper_cam.set(cv2.CAP_PROP_FPS, 30)

            # Test gripper camera
            ret, test_gripper = self.gripper_cam.read()
            if not ret:
                print(f"[ERROR] Failed to capture from gripper camera")
                return False

            # Log actual capture resolution (no resize in collection)
            print(f"[OK] Gripper camera ready: {test_gripper.shape} (native resolution)")

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

    def start_save_thread(self):
        """Start background save worker thread."""
        if self.save_thread is not None and self.save_thread.is_alive():
            return

        self.save_thread_running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        print("[OK] Background save thread started")

    def stop_save_thread(self):
        """Stop background save worker thread, waiting for queue to drain."""
        if self.save_thread is None:
            return

        # Signal thread to stop
        self.save_thread_running = False

        # Wait for queue to drain
        remaining = self.save_queue.qsize()
        if remaining > 0:
            print(f"[INFO] Waiting for {remaining} episode(s) to save...")

        # Block until queue is empty
        self.save_queue.join()

        # Wait for thread to finish
        self.save_thread.join(timeout=5.0)
        self.save_thread = None
        print("[OK] Background save thread stopped")

    def _save_worker(self):
        """Background worker that saves episodes from queue."""
        while self.save_thread_running or not self.save_queue.empty():
            try:
                # Get next episode to save (with timeout to check running flag)
                item = self.save_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                frame_paths, stage_info, episode_index = item
                self._save_episode_lerobot_impl(frame_paths, stage_info, episode_index)

            except Exception as e:
                print(f"[ERROR] Background save failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.save_queue.task_done()

    def queue_episode(self, frame_paths: List[Path], stage_info: Dict) -> int:
        """
        Queue episode for background saving.

        Args:
            frame_paths: List of temp file paths containing frame data
            stage_info: Stage dictionary with 'description', 'vlm_prompt', etc.

        Returns:
            Episode index assigned to this episode
        """
        with self.save_lock:
            episode_index = self.episode_count
            self.episode_count += 1

        self.save_queue.put((frame_paths, stage_info, episode_index))
        print(f"[INFO] Queued episode {episode_index} ({len(frame_paths)} frames, queue size: {self.save_queue.qsize()})")
        return episode_index

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

    def _record_stage(
        self,
        stage: Dict,
        stage_index: int,
        total_stages: int,
        board: chess.Board,
        move: chess.Move
    ) -> Optional[List[Path]]:
        """
        Record one stage at 15 FPS until user presses ENTER.

        Frames are streamed to temp files on disk to prevent OOM during long
        recordings. Memory usage stays constant (~10 MB) regardless of duration.

        Args:
            stage: Stage dictionary from decompose_move()
            stage_index: Current stage index (0-based)
            total_stages: Total number of stages
            board: Current chess board state (before move)
            move: The move being executed

        Returns:
            List of temp file paths containing frame data, or None if recording fails
        """
        print(f"\n{'='*60}")
        print(f"Stage {stage_index + 1}/{total_stages}")
        print(f"{'='*60}")
        print(f"Description: {stage['description']}")
        print(f"VLM Prompt: {stage['vlm_prompt']}")
        print(f"{'='*60}")

        # Display board before and after move
        print("\nCurrent Board Position:")
        print("=" * 40)
        print(board)
        print("=" * 40)

        # Show board after move
        board_after = board.copy()
        board_after.push(move)
        print(f"\nBoard after {move.uci()} ({board.san(move)}):")
        print("-" * 40)
        print(board_after)
        print("-" * 40)
        print()

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

        # Flush gripper camera buffer to avoid stale/corrupted frames at start
        # Read and discard several frames to clear the buffer
        for _ in range(5):
            self.gripper_cam.read()

        print(f"[INFO] Recording stage {stage_index + 1} at {self.fps} FPS...")
        print("[PROMPT] Press ENTER to STOP recording")

        recording_active = True
        frame_paths: List[Path] = []
        start_time = time.time()
        frame_count = 0

        # Create temp directory for streaming frames to disk
        # This prevents OOM by keeping memory usage constant (~10 MB)
        temp_dir = Path(tempfile.mkdtemp(prefix="chessbot_frames_"))
        first_positions = None
        last_positions = None

        try:
            while recording_active:
                frame_start = time.time()

                # Capture synchronized data
                global_frame = self.global_cam_capture.get_latest_frame()
                ret, gripper_frame = self.gripper_cam.read()

                # Refresh cache from disk to get latest joint positions from tele_op.py
                self.cache.refresh()

                # Read robot state from cache (NO serial conflict!)
                # Note: gripper is joint_5 in joint_positions
                joint_positions = self.cache.get("robot_state.joint_positions")

                # Validate data
                if global_frame is None:
                    print("[WARNING] No global frame, skipping")
                    continue

                if not ret or gripper_frame is None:
                    print("[WARNING] No gripper frame, skipping")
                    continue
                # NO RESIZE - keep at native camera resolution (camera-agnostic)
                # Resizing is deferred to training/deploy pipelines

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

                # Stream frame to temp file instead of accumulating in memory
                # This keeps memory usage constant regardless of recording length
                temp_path = temp_dir / f"frame_{frame_count:06d}.npz"
                np.savez_compressed(
                    temp_path,
                    global_frame=overlayed_frame,
                    gripper_frame=gripper_frame,
                    joint_positions=np.array(joint_positions, dtype=np.float32),
                    timestamp=np.array([time.time() - start_time], dtype=np.float32),
                    vlm_prompt=np.array([stage["vlm_prompt"]])
                )
                frame_paths.append(temp_path)

                # Track first/last positions for mismatch check
                positions_array = np.array(joint_positions, dtype=np.float32)
                if first_positions is None:
                    first_positions = positions_array
                last_positions = positions_array

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
            print(f"     Temp files: {temp_dir}")

            # Check for position mismatch between start and end
            if first_positions is not None and last_positions is not None:
                mismatched_joints = []

                for i in range(len(first_positions)):
                    diff_rad = abs(first_positions[i] - last_positions[i])
                    diff_deg = diff_rad * (180.0 / np.pi)
                    if diff_rad > POSITION_TOLERANCE_RAD:
                        mismatched_joints.append((i, diff_deg))

                if mismatched_joints:
                    print(f"\n[WARNING] Position mismatch detected (>{POSITION_TOLERANCE_DEG} deg):")
                    for joint_idx, diff_deg in mismatched_joints:
                        print(f"          Joint {joint_idx}: {diff_deg:.1f} degrees difference")

            return frame_paths

        except KeyboardInterrupt:
            print("\n[INFO] Recording interrupted by user")
            # Clean up temp files on interrupt
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
        except Exception as e:
            print(f"[ERROR] Recording failed: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp files on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def _save_episode_lerobot_impl(self, frame_paths: List[Path], stage_info: Dict, episode_index: int):
        """
        Save single-stage episode to LeRobot dataset (called from background thread).

        Reads frames from temp files on disk, adds them to LeRobot, then cleans up.
        This streaming approach keeps memory usage low even for long episodes.

        Args:
            frame_paths: List of temp file paths containing frame data
            stage_info: Stage dictionary with 'description', 'vlm_prompt', etc.
            episode_index: Pre-assigned episode index
        """
        from PIL import Image

        temp_dir = frame_paths[0].parent if frame_paths else None

        try:
            vlm_prompt = stage_info.get("vlm_prompt", "")

            print(f"[SAVE] Writing episode {episode_index} to LeRobot dataset ({len(frame_paths)} frames)...")

            # Pre-load next frame's joint positions for action calculation
            # We need to peek ahead, so load them in pairs
            next_positions = None

            with self.save_lock:
                # Safety check: Reset corrupted episode buffer if needed
                # LeRobot's save_episode() pops 'size' before clearing the buffer,
                # so if video encoding fails, the buffer is left corrupted (exists but missing 'size')
                if (self.dataset.episode_buffer is not None and
                    "size" not in self.dataset.episode_buffer):
                    print("[WARN] Resetting corrupted episode buffer from previous failed save")
                    self.dataset.episode_buffer = None

                for frame_idx, frame_path in enumerate(frame_paths):
                    # Load frame data from temp file
                    data = np.load(frame_path)

                    global_frame = data["global_frame"]
                    gripper_frame = data["gripper_frame"]
                    joint_positions = data["joint_positions"]

                    # Convert BGR (OpenCV) to RGB for LeRobot video encoding
                    global_rgb = cv2.cvtColor(global_frame, cv2.COLOR_BGR2RGB)
                    gripper_rgb = cv2.cvtColor(gripper_frame, cv2.COLOR_BGR2RGB)

                    # Convert numpy arrays to PIL Images for LeRobot
                    global_pil = Image.fromarray(global_rgb)
                    gripper_pil = Image.fromarray(gripper_rgb)

                    # Action: next joint position (or current for last frame)
                    if frame_idx < len(frame_paths) - 1:
                        # Peek at next frame for action
                        next_data = np.load(frame_paths[frame_idx + 1])
                        action = next_data["joint_positions"]
                    else:
                        action = joint_positions

                    self.dataset.add_frame({
                        "observation.images.global": global_pil,
                        "observation.images.gripper": gripper_pil,
                        "observation.state": joint_positions,
                        "action": action,
                        "task": vlm_prompt  # Required by LeRobot
                    })

                    # Delete temp file after processing to free disk space
                    frame_path.unlink()

                # Finalize episode (encode videos, write Parquet)
                self.dataset.save_episode()

            print(f"[SAVE] Episode {episode_index} saved: {len(frame_paths)} frames")

        except Exception as e:
            print(f"[ERROR] Failed to save episode {episode_index} to LeRobot: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up temp directory (should be empty, but ensure cleanup)
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

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

        # Start background save thread
        self.start_save_thread()

        # Use keyboard context manager to enable raw terminal input
        with self.keyboard:
            try:
                print("\n[INFO] Episode collection ready")
                print("[INFO] Each stage is saved as a separate episode")
                print("[PROMPT] Press ENTER to capture board and start new move (Ctrl+C to quit)...")

                while True:
                    # Wait for user to press ENTER
                    key = self.keyboard.get_key(timeout=0.1)
                    if key != 'ENTER':
                        continue

                    # Capture board image using 4K MJPEG -> 720p downscale
                    # This matches the training data pipeline for consistent detection
                    board_image_path = Path("data/board_capture.png")

                    # Stop live capture to release camera for 4K capture
                    print("[INFO] Stopping live capture for board detection...")
                    self.global_cam_capture.stop()
                    time.sleep(0.5)  # Allow camera to fully release

                    # Capture board image for detection
                    if self.use_yuyv:
                        success = capture_720p_yuyv(self.global_camera_device, board_image_path)
                        capture_mode = "720p YUYV"
                    else:
                        success = capture_4k_downscale(self.global_camera_device, board_image_path)
                        capture_mode = "4K -> 720p"

                    # Restart live capture for episode recording
                    print("[INFO] Restarting live capture...")
                    self.global_cam_capture.start()
                    time.sleep(1.0)  # Allow camera to reinitialize

                    if not success:
                        print("[ERROR] Failed to capture board image")
                        continue

                    print(f"[OK] Board image captured: {board_image_path} ({capture_mode})")

                    # Detect and decompose move
                    move_info = self._detect_and_decompose_move(str(board_image_path))

                    if move_info is None:
                        print("[ERROR] Failed to process board, try again")
                        continue

                    # Record each stage as its own episode
                    stages_saved = 0
                    move_aborted = False

                    for stage_idx, stage in enumerate(move_info["stages"]):
                        # Retry loop for each stage
                        while True:
                            frames = self._record_stage(
                                stage,
                                stage_idx,
                                len(move_info["stages"]),
                                move_info["board"],
                                move_info["move"]
                            )

                            if frames is None:
                                print("[WARNING] Stage recording failed")
                                move_aborted = True
                                break

                            # Ask if user wants to keep or retry
                            print("\n[PROMPT] Keep this recording? (ENTER=Keep, R=Retry, Q=Abort move)")
                            while True:
                                key = self.keyboard.get_key(timeout=0.1)
                                if key == 'ENTER':
                                    # Keep recording - queue for background save
                                    self.queue_episode(frames, stage)
                                    stages_saved += 1
                                    break
                                elif key in ('r', 'R'):
                                    # Retry this stage
                                    print("[INFO] Retrying stage...")
                                    break
                                elif key in ('q', 'Q'):
                                    # Abort remaining stages
                                    print("[INFO] Aborting remaining stages...")
                                    move_aborted = True
                                    break

                            # Check if we should keep this recording or retry
                            if key == 'ENTER':
                                break  # Exit retry loop, move to next stage
                            elif move_aborted:
                                break  # Exit retry loop, abort move
                            # Otherwise key was 'r'/'R', continue retry loop

                        if move_aborted:
                            break  # Exit stage loop

                    # Summary for this move
                    print(f"\n{'='*60}")
                    print(f"Move Collection Summary")
                    print(f"{'='*60}")
                    print(f"Stages saved: {stages_saved}/{len(move_info['stages'])}")
                    print(f"Total episodes: {self.episode_count}")
                    print(f"Save queue: {self.save_queue.qsize()} pending")
                    print(f"Output directory: {self.output_dir}")
                    print(f"{'='*60}\n")

                    print("[PROMPT] Press ENTER to start next move (Ctrl+C to quit)...")

            except KeyboardInterrupt:
                print("\n\n[INFO] Collection stopped by user")

            finally:
                # Stop save thread (waits for queue to drain)
                self.stop_save_thread()
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
    python scripts/collect_vla_episodes.py --output data/lerobot_episodes/

    # With specific cameras
    python scripts/collect_vla_episodes.py --output data/lerobot_episodes/ \\
        --global-camera /dev/video4 \\
        --gripper-camera /dev/video1 \\
        --virtual-camera /dev/video7

    # Custom FPS and engine
    python scripts/collect_vla_episodes.py --output data/lerobot_episodes/ \\
        --fps 30 \\
        --engine /usr/local/bin/stockfish
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/lerobot_episodes",
        help="Output directory for LeRobot dataset (default: data/lerobot_episodes)"
    )

    parser.add_argument(
        "--global-camera",
        type=str,
        default=None,
        help="Physical global camera device (auto-detects WBC-0E01 if not specified)"
    )

    parser.add_argument(
        "--gripper-camera",
        type=str,
        default=None,
        help="Physical gripper camera device (auto-detects eMeet C950 if not specified)"
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
        "--yuyv",
        action="store_true",
        help="Force native 720p YUYV capture for board detection (uncompressed, best quality, 10fps). This is the default."
    )
    parser.add_argument(
        "--mjpeg",
        action="store_true",
        help="Use 4K MJPEG -> 720p downscale for board detection instead of default YUYV"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose SVT-AV1 encoder output)"
    )

    args = parser.parse_args()

    # Set SVT-AV1 logging level based on debug flag
    if args.debug:
        os.environ["SVT_LOG"] = "3"  # info level

    # Determine capture mode: --mjpeg overrides to MJPEG, --yuyv forces YUYV, else use default
    use_yuyv = DEFAULT_USE_YUYV
    if args.mjpeg:
        use_yuyv = False
    elif args.yuyv:
        use_yuyv = True

    # Camera selection: specified > auto-detect by name > prompt
    print("\n" + "=" * 60)
    print("Camera Selection")
    print("=" * 60)

    # Global camera
    if args.global_camera:
        global_camera = args.global_camera
        print(f"[OK] Global camera: {global_camera} (specified)")
    else:
        global_camera = get_default_global_camera()
        if global_camera:
            print(f"[OK] Global camera: {global_camera} (auto-detected WBC-0E01)")
        else:
            cameras = get_available_cameras()
            if not cameras:
                print("[X] No cameras found!")
                return 1
            elif len(cameras) == 1:
                global_camera = cameras[0][0]
                print(f"[OK] Global camera: {global_camera} (auto-selected)")
            else:
                print("[WARN] WBC-0E01 not found, prompting for selection...")
                global_camera = select_camera(cameras)

    # Gripper camera
    if args.gripper_camera:
        gripper_camera = args.gripper_camera
        print(f"[OK] Gripper camera: {gripper_camera} (specified)")
    else:
        gripper_result = get_default_gripper_camera_with_name()
        if gripper_result:
            gripper_camera, gripper_name = gripper_result
            print(f"[OK] Gripper camera: {gripper_camera} (auto-detected {gripper_name})")
        else:
            print("[X] No gripper camera found!")
            print("    Searched for: eMeet C950, USB2.0_CAM1")
            print("    Use --gripper-camera to specify manually")
            return 1

    # Create recorder
    capture_mode_str = "720p YUYV" if use_yuyv else "4K MJPEG -> 720p"
    print(f"[OK] Board capture mode: {capture_mode_str}")

    recorder = EpisodeRecorder(
        output_dir=args.output,
        global_camera_device=global_camera,
        gripper_camera_device=gripper_camera,
        virtual_camera_device=args.virtual_camera,
        fps=args.fps,
        engine_path=args.engine,
        camera_rotation=args.rotation,
        turn=args.turn,
        use_yuyv=use_yuyv,
    )

    # Run collection loop
    recorder.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
