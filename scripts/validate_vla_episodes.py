#!/usr/bin/env python3
"""
VLA Episode Validation Tool

Review, playback, filter, and execute collected VLA episodes.
Supports both LeRobot dataset format and raw file storage.

Features:
- Episode information display (metadata, statistics)
- Video playback with controls (pause, speed, seek)
- Action playback on physical SO-100 robot
- Quality control (mark good/bad, delete bad episodes)
- Export filtered dataset

Usage:
    # Interactive review (default dataset: data/lerobot_episodes/)
    python scripts/validate_vla_episodes.py

    # Export good episodes only
    python scripts/validate_vla_episodes.py --export data/episodes_filtered/

    # List all episodes
    python scripts/validate_vla_episodes.py --list

    # Execute episode on robot hardware
    python scripts/validate_vla_episodes.py --execute 0

    # Execute at half speed (safety mode)
    python scripts/validate_vla_episodes.py --execute 0 --speed 0.5

    # Dry run (simulate without robot)
    python scripts/validate_vla_episodes.py --execute 0 --dry-run
"""

import argparse
import json
import os
import shutil
import time

# Suppress Qt font warnings from OpenCV before importing cv2
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import LeRobot (optional dependency)
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# Try to import robot control (optional dependency for action playback)
try:
    from controls.robot_controller import RobotController, load_joint_configs_for_port
    from controls.so100_arm import SO100State
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False


class EpisodeValidator:
    """
    Episode validator for VLA training data quality control.

    Supports both LeRobot dataset format and raw file storage.
    Provides episode information display, video playback, and quality control.
    """

    # Playback speed options
    SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(self, dataset_path: str, use_lerobot: bool = True):
        """
        Initialize episode validator.

        Args:
            dataset_path: Path to episode dataset directory
            use_lerobot: Whether to use LeRobot dataset format
        """
        self.dataset_path = Path(dataset_path)
        self.use_lerobot = use_lerobot and LEROBOT_AVAILABLE

        # Dataset
        self.dataset: Optional[Any] = None
        self.episodes: List[Dict] = []
        self.episode_qualities: Dict[int, str] = {}  # episode_idx -> "good" or "bad"

        # Playback state
        self.current_speed_idx = 2  # Default 1.0x
        self.paused = False

        # Robot control for action playback (optional)
        self.robot_controller: Optional['RobotController'] = None
        self.robot_port: Optional[str] = None

        # Load dataset
        self._load_dataset()

        print(f"[OK] EpisodeValidator initialized")
        print(f"     Dataset: {self.dataset_path}")
        print(f"     Episodes: {len(self.episodes)}")
        print(f"     Format: {'LeRobot' if self.use_lerobot else 'Raw files'}")

    def _load_dataset(self):
        """Load dataset and enumerate episodes."""
        # LeRobot v0.4 uses meta/info.json instead of meta.json
        lerobot_meta = (self.dataset_path / "meta" / "info.json").exists()
        if self.use_lerobot and lerobot_meta:
            self._load_lerobot_dataset()
        else:
            self._load_raw_dataset()

        # Load quality annotations if they exist
        quality_file = self.dataset_path / "episode_qualities.json"
        if quality_file.exists():
            with open(quality_file, 'r') as f:
                loaded = json.load(f)
                self.episode_qualities = {int(k): v for k, v in loaded.items()}
            print(f"[INFO] Loaded quality annotations for {len(self.episode_qualities)} episodes")

    def _load_lerobot_dataset(self):
        """Load LeRobot v3.0 format dataset by reading parquet files directly."""
        try:
            import pandas as pd

            # Read info.json for fps and metadata
            info_path = self.dataset_path / "meta" / "info.json"
            with open(info_path, 'r') as f:
                info = json.load(f)

            fps = info.get("fps", 15)
            total_episodes = info.get("total_episodes", 0)

            # Read episodes from meta/episodes parquet files
            episodes_dir = self.dataset_path / "meta" / "episodes"
            episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))

            all_episodes = []
            for ep_file in episode_files:
                try:
                    ep_df = pd.read_parquet(ep_file)
                    for _, row in ep_df.iterrows():
                        ep_idx = int(row["episode_index"])
                        frame_count = int(row["length"])
                        # 'tasks' column contains list of task strings directly
                        tasks_list = row.get("tasks", [])
                        if isinstance(tasks_list, (list, np.ndarray)) and len(tasks_list) > 0:
                            task = str(tasks_list[0])
                        else:
                            task = ""

                        all_episodes.append({
                            "index": ep_idx,
                            "frame_count": frame_count,
                            "fps": fps,
                            "duration": frame_count / fps if fps > 0 else 0,
                            "task": task,
                            "format": "lerobot",
                            # Store chunk/file info for frame loading
                            "data_chunk_index": int(row.get("data/chunk_index", 0)),
                            "data_file_index": int(row.get("data/file_index", 0)),
                            "dataset_from_index": int(row.get("dataset_from_index", 0)),
                            "dataset_to_index": int(row.get("dataset_to_index", frame_count)),
                            # Video info for global camera
                            "videos/observation.images.global/chunk_index": int(row.get("videos/observation.images.global/chunk_index", 0)),
                            "videos/observation.images.global/file_index": int(row.get("videos/observation.images.global/file_index", 0)),
                            "videos/observation.images.global/from_timestamp": float(row.get("videos/observation.images.global/from_timestamp", 0)),
                            "videos/observation.images.global/to_timestamp": float(row.get("videos/observation.images.global/to_timestamp", 0)),
                            # Video info for gripper camera
                            "videos/observation.images.gripper/chunk_index": int(row.get("videos/observation.images.gripper/chunk_index", 0)),
                            "videos/observation.images.gripper/file_index": int(row.get("videos/observation.images.gripper/file_index", 0)),
                            "videos/observation.images.gripper/from_timestamp": float(row.get("videos/observation.images.gripper/from_timestamp", 0)),
                            "videos/observation.images.gripper/to_timestamp": float(row.get("videos/observation.images.gripper/to_timestamp", 0)),
                        })
                except Exception as e:
                    print(f"[WARNING] Failed to read episode file {ep_file}: {e}")

            # Sort by episode index
            self.episodes = sorted(all_episodes, key=lambda x: x["index"])

            # Store info for later use
            self._lerobot_info = info

            # Pre-load valid data parquet files (skip corrupted)
            self._load_data_parquets()

        except Exception as e:
            print(f"[ERROR] Failed to load LeRobot v3.0 dataset: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Falling back to raw file detection")
            self._load_raw_dataset()

    def _load_data_parquets(self):
        """Pre-load data parquet files, skipping corrupted ones."""
        import pandas as pd

        self._data_frames = {}  # (chunk_idx, file_idx) -> DataFrame
        self._corrupted_files = set()

        data_dir = self.dataset_path / "data"
        data_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

        for data_file in data_files:
            # Parse chunk and file indices from path
            parts = data_file.parts
            chunk_idx = int(parts[-2].replace("chunk-", ""))
            file_idx = int(parts[-1].replace("file-", "").replace(".parquet", ""))

            try:
                df = pd.read_parquet(data_file)
                self._data_frames[(chunk_idx, file_idx)] = df
            except Exception as e:
                print(f"[WARNING] Corrupted parquet file {data_file}: {e}")
                self._corrupted_files.add((chunk_idx, file_idx))

    def _load_video_frames(self, episode: Dict, video_key: str, target_shape: Optional[tuple] = None) -> Optional[List[np.ndarray]]:
        """
        Load video frames for an episode from LeRobot v3.0 format using pyav.

        Args:
            episode: Episode metadata dict
            video_key: Video key (e.g., "observation.images.global")
            target_shape: Optional (H, W, C) shape to resize frames to

        Returns:
            List of video frames as numpy arrays (BGR), or None if unavailable
        """
        try:
            import av

            # Get video chunk/file info from episode metadata
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"

            chunk_idx = episode.get(chunk_key, episode.get("data_chunk_index", 0))
            file_idx = episode.get(file_key, episode.get("data_file_index", 0))

            # Construct video path
            video_path = self.dataset_path / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

            if not video_path.exists():
                return None

            # Open video with pyav (supports AV1 codec)
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Get timestamp range for this episode
            from_ts = episode.get(f"videos/{video_key}/from_timestamp", 0)
            to_ts = episode.get(f"videos/{video_key}/to_timestamp", None)
            fps = episode.get("fps", 15)

            # Calculate frame range
            from_frame = int(from_ts * fps)
            to_frame = int(to_ts * fps) if to_ts is not None else None

            frames = []
            frame_idx = 0

            for frame in container.decode(stream):
                # Skip frames before our range
                if frame_idx < from_frame:
                    frame_idx += 1
                    continue

                # Stop if we've passed our range
                if to_frame is not None and frame_idx >= to_frame:
                    break

                # Convert to numpy array (BGR)
                img = frame.to_ndarray(format='bgr24')

                # Resize to target shape if specified
                if target_shape is not None and len(target_shape) >= 2:
                    target_h, target_w = target_shape[0], target_shape[1]
                    if img.shape[0] != target_h or img.shape[1] != target_w:
                        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                frames.append(img)
                frame_idx += 1

            container.close()

            return frames if frames else None

        except Exception as e:
            # Log error for debugging
            print(f"[WARNING] Failed to load video {video_key}: {e}")
            return None

    def _load_raw_dataset(self):
        """Load raw file format dataset."""
        self.use_lerobot = False

        # Find all episode directories
        episode_dirs = sorted(self.dataset_path.glob("episode_*"))

        for ep_dir in episode_dirs:
            metadata_file = ep_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                self.episodes.append({
                    "index": metadata.get("episode_index", 0),
                    "path": ep_dir,
                    "frame_count": metadata.get("frame_count", 0),
                    "fps": metadata.get("fps", 15),
                    "duration": metadata.get("frame_count", 0) / metadata.get("fps", 15),
                    "move": metadata.get("move", "unknown"),
                    "fen": metadata.get("fen", ""),
                    "stages": metadata.get("stages", []),
                    "joint_positions": metadata.get("joint_positions", []),
                    "timestamps": metadata.get("timestamps", []),
                    "task": metadata.get("vlm_prompt", metadata.get("task", "")),  # Support both new and old format
                    "description": metadata.get("description", ""),
                    "format": "raw"
                })

            except Exception as e:
                print(f"[WARNING] Failed to load {ep_dir}: {e}")

    def get_episode_count(self) -> int:
        """Get total number of episodes."""
        return len(self.episodes)

    def get_episode_info(self, episode_index: int) -> Optional[Dict]:
        """
        Get detailed information for an episode.

        Args:
            episode_index: Episode index

        Returns:
            Dictionary with episode metadata and statistics
        """
        # Find episode
        episode = None
        for ep in self.episodes:
            if ep["index"] == episode_index:
                episode = ep
                break

        if episode is None:
            return None

        info = {
            "index": episode["index"],
            "frame_count": episode["frame_count"],
            "fps": episode["fps"],
            "duration": episode["duration"],
            "format": episode["format"],
            "quality": self.episode_qualities.get(episode_index, "unreviewed")
        }

        # Add format-specific info
        if episode["format"] == "raw":
            info["move"] = episode.get("move", "unknown")
            info["fen"] = episode.get("fen", "")
            info["stages"] = episode.get("stages", [])

            # Calculate joint position statistics
            joint_positions = episode.get("joint_positions", [])
            if joint_positions:
                jp_array = np.array(joint_positions)
                info["joint_stats"] = {
                    "min": jp_array.min(axis=0).tolist(),
                    "max": jp_array.max(axis=0).tolist(),
                    "mean": jp_array.mean(axis=0).tolist(),
                    "std": jp_array.std(axis=0).tolist()
                }

        return info

    def display_episode_info(self, episode_index: int):
        """
        Display episode information in terminal.

        Args:
            episode_index: Episode index to display
        """
        info = self.get_episode_info(episode_index)

        if info is None:
            print(f"[ERROR] Episode {episode_index} not found")
            return

        print(f"\n{'='*60}")
        print(f"Episode {info['index']} Information")
        print(f"{'='*60}")

        print(f"Format: {info['format']}")
        print(f"Quality: {info['quality']}")
        print(f"Frames: {info['frame_count']}")
        print(f"FPS: {info['fps']}")
        print(f"Duration: {info['duration']:.2f}s")

        if info['format'] == 'raw':
            print(f"\nMove: {info.get('move', 'N/A')}")
            print(f"FEN: {info.get('fen', 'N/A')}")

            stages = info.get('stages', [])
            if stages:
                print(f"\nStages ({len(stages)}):")
                for i, stage in enumerate(stages):
                    desc = stage.get('description', 'N/A') if isinstance(stage, dict) else str(stage)
                    print(f"  {i+1}. {desc}")

            joint_stats = info.get('joint_stats')
            if joint_stats:
                print(f"\nJoint Position Statistics:")
                print(f"  Min:  {[f'{v:.3f}' for v in joint_stats['min']]}")
                print(f"  Max:  {[f'{v:.3f}' for v in joint_stats['max']]}")
                print(f"  Mean: {[f'{v:.3f}' for v in joint_stats['mean']]}")
                print(f"  Std:  {[f'{v:.3f}' for v in joint_stats['std']]}")

        print(f"{'='*60}\n")

    def _connect_robot(self, port: Optional[str] = None) -> bool:
        """
        Connect to SO-100 robot for action playback.

        Args:
            port: Serial port (auto-detect/prompt if None)

        Returns:
            bool: True if connection successful
        """
        if not ROBOT_AVAILABLE:
            print("[ERROR] Robot control not available - controls module not found")
            return False

        # Auto-detect port if not specified
        if port is None:
            available_ports = []
            for i in range(10):
                test_port = f"/dev/ttyACM{i}"
                if os.path.exists(test_port):
                    available_ports.append(test_port)

            if len(available_ports) == 0:
                print("[ERROR] No robot port found (checked /dev/ttyACM0-9)")
                return False
            elif len(available_ports) == 1:
                port = available_ports[0]
                print(f"[INFO] Auto-selected robot port: {port}")
            else:
                # Multiple ports found - prompt user to select
                print(f"\n[INFO] Multiple robot ports found:")
                for i, p in enumerate(available_ports):
                    # Check for port-specific config
                    port_name = p.split('/')[-1]
                    config_path = Path("data") / f"so100_config_{port_name}.csv"
                    config_status = "(has config)" if config_path.exists() else "(no config)"
                    print(f"  [{i}] {p} {config_status}")

                while True:
                    try:
                        selection = input(f"Select port [0-{len(available_ports)-1}]: ").strip()
                        idx = int(selection)
                        if 0 <= idx < len(available_ports):
                            port = available_ports[idx]
                            break
                        else:
                            print(f"[ERROR] Invalid selection. Enter 0-{len(available_ports)-1}")
                    except ValueError:
                        print(f"[ERROR] Invalid input. Enter a number 0-{len(available_ports)-1}")

        # Check if already connected to same port
        if self.robot_controller and self.robot_port == port:
            print(f"[INFO] Robot already connected on {port}")
            return True

        # Disconnect existing if different port
        if self.robot_controller:
            self._disconnect_robot()

        try:
            # Load joint configs for this port
            config_dir = Path("data")
            joint_configs, config_source = load_joint_configs_for_port(port, config_dir)

            print(f"[INFO] Config source: {config_source}")

            # Create controller with stability system enabled
            self.robot_controller = RobotController(
                port=port,
                joint_configs=joint_configs,
                config_source=config_source,
                enable_stability_system=True
            )

            if not self.robot_controller.connect():
                print(f"[ERROR] Failed to connect to robot on {port}")
                self.robot_controller = None
                return False

            self.robot_port = port
            print(f"[OK] Robot connected on {port}")

            # Enable torque and start control loop
            # IMPORTANT: Set home targets BEFORE starting control loop
            # Otherwise robot immediately moves to [0,0,0,0,0,0] radians (min limits)
            self.robot_controller.enable_torque()
            self.robot_controller.set_home_targets()
            self.robot_controller.start_control_loop()
            print(f"[OK] Control loop started (holding at home position)")

            return True

        except Exception as e:
            print(f"[ERROR] Robot connection failed: {e}")
            self.robot_controller = None
            return False

    def _disconnect_robot(self):
        """Disconnect from robot."""
        if self.robot_controller:
            print("[INFO] Disconnecting robot...")
            self.robot_controller.disconnect()
            self.robot_controller = None
            self.robot_port = None

    def execute_episode(
        self,
        episode_index: int,
        speed_multiplier: float = 1.0,
        dry_run: bool = False,
        go_to_start: bool = True,
        robot_port: Optional[str] = None
    ) -> bool:
        """
        Execute episode actions on physical robot.

        Replays the recorded action values (commanded positions) from the episode
        at the original FPS, scaled by speed_multiplier.

        Args:
            episode_index: Episode to execute
            speed_multiplier: Playback speed (0.5 = half speed, 2.0 = double)
            dry_run: If True, only simulate without sending commands
            go_to_start: Move to first position before starting
            robot_port: Robot serial port (auto-detect if None)

        Controls during execution:
            SPACE: Pause/resume
            Q/ESC: Abort execution

        Returns:
            True if completed, False if aborted
        """
        # Load episode frames (contains action values)
        frames = self._load_episode_frames(episode_index)
        if frames is None or len(frames) == 0:
            print(f"[ERROR] No frames found for episode {episode_index}")
            return False

        info = self.get_episode_info(episode_index)
        if info is None:
            print(f"[ERROR] Episode {episode_index} not found")
            return False

        fps = info.get("fps", 15)
        duration = len(frames) / fps if fps > 0 else 0

        print(f"\n{'='*60}")
        print(f"Episode Execution: Episode {episode_index}")
        print(f"{'='*60}")
        print(f"Frames: {len(frames)}")
        print(f"FPS: {fps}")
        print(f"Duration: {duration:.2f}s")
        print(f"Speed: {speed_multiplier}x")
        print(f"Mode: {'DRY RUN (no robot commands)' if dry_run else 'LIVE EXECUTION'}")
        print(f"{'='*60}")

        # Connect to robot if not dry run
        if not dry_run:
            if not self._connect_robot(robot_port):
                print("[ERROR] Cannot execute - robot not available")
                return False

        # Extract action values from frames
        # Use 'action' if available (commanded positions), fall back to 'joint_positions' (observed)
        actions = []
        for frame in frames:
            # LeRobot format stores action separately
            action = frame.get("action")
            if action is None:
                # Fall back to joint_positions (raw format or observation.state)
                action = frame.get("joint_positions", [0.0] * 6)
            if hasattr(action, 'tolist'):
                action = action.tolist()
            actions.append(np.array(action, dtype=np.float32))

        if len(actions) == 0:
            print("[ERROR] No action data found in episode")
            return False

        # Show first/last actions for confirmation
        print(f"\nFirst action: {[f'{v:.3f}' for v in actions[0]]}")
        print(f"Last action:  {[f'{v:.3f}' for v in actions[-1]]}")

        # Confirm execution
        print(f"\n[WARNING] Robot will move through {len(actions)} positions!")
        response = input("Proceed with execution? (yes/no): ").strip().lower()
        if response != "yes":
            print("[INFO] Execution cancelled")
            return False

        # Move to start position if requested
        if go_to_start and not dry_run:
            print("\n[INFO] Moving to start position at 0.1x speed...")
            start_action = actions[0]

            # Move slowly to start
            self.robot_controller.set_target_positions(start_action)

            # Wait for robot to reach position (with timeout)
            print(f"  Target: {[f'{v:.2f}' for v in start_action]}")
            time.sleep(3.0)  # Give robot time to reach start

            input("Robot at start position. Press ENTER to begin playback...")

        # Create display window for visual feedback
        cv2.namedWindow("Episode Execution", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Episode Execution", 800, 200)

        # Execute actions
        frame_interval = (1.0 / fps) / speed_multiplier
        paused = False
        aborted = False
        frame_idx = 0
        last_frame_time = time.time()

        print(f"\n[INFO] Executing episode (press SPACE to pause, Q/ESC to abort)")

        try:
            while frame_idx < len(actions):
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n[INFO] Execution aborted by user")
                    aborted = True
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")

                # Skip if paused
                if paused:
                    time.sleep(0.05)
                    continue

                # Check timing
                elapsed = time.time() - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(max(0, frame_interval - elapsed - 0.001))
                    continue

                last_frame_time = time.time()

                # Get current action
                action = actions[frame_idx]

                # Send to robot if not dry run
                if not dry_run:
                    self.robot_controller.set_target_positions(action)

                # Create status display
                status_img = np.zeros((200, 800, 3), dtype=np.uint8)
                progress = (frame_idx + 1) / len(actions)

                # Progress bar
                bar_width = 700
                bar_height = 30
                bar_x = 50
                bar_y = 30
                cv2.rectangle(status_img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                cv2.rectangle(status_img, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 200, 0), -1)
                cv2.rectangle(status_img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

                # Text info
                cv2.putText(status_img, f"Episode {episode_index}", (50, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(status_img, f"Frame {frame_idx + 1}/{len(actions)} ({progress*100:.1f}%)", (50, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(status_img, f"Speed: {speed_multiplier}x  {'DRY RUN' if dry_run else 'LIVE'}", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255) if dry_run else (150, 255, 150), 1)
                cv2.putText(status_img, "SPACE: pause | Q/ESC: abort", (50, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

                cv2.imshow("Episode Execution", status_img)

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[INFO] Execution interrupted")
            aborted = True

        finally:
            cv2.destroyWindow("Episode Execution")

        if not aborted:
            print(f"\n[OK] Execution complete ({frame_idx} frames)")

        # Offer to return to home
        if not dry_run and self.robot_controller:
            response = input("\nReturn to home position? (y/n): ").strip().lower()
            if response == 'y':
                print("[INFO] Moving to home position...")
                self.robot_controller.set_home_targets()
                time.sleep(2.0)
                print("[OK] Robot at home position")

        return not aborted

    def _load_episode_frames(self, episode_index: int) -> Optional[List[Dict]]:
        """
        Load all frames for an episode.

        Args:
            episode_index: Episode index

        Returns:
            List of frame dictionaries with 'global_frame', 'gripper_frame', 'joint_positions', 'action'
        """
        # Find episode
        episode = None
        for ep in self.episodes:
            if ep["index"] == episode_index:
                episode = ep
                break

        if episode is None:
            return None

        frames = []

        # Get task from episode metadata
        episode_task = episode.get("task", "")

        if episode["format"] == "raw":
            ep_dir = episode["path"]

            for frame_idx in range(episode["frame_count"]):
                global_path = ep_dir / f"global_{frame_idx:06d}.png"
                gripper_path = ep_dir / f"gripper_{frame_idx:06d}.png"

                global_frame = cv2.imread(str(global_path)) if global_path.exists() else None
                gripper_frame = cv2.imread(str(gripper_path)) if gripper_path.exists() else None

                joint_positions = episode.get("joint_positions", [])
                jp = joint_positions[frame_idx] if frame_idx < len(joint_positions) else [0.0] * 6

                timestamps = episode.get("timestamps", [])
                ts = timestamps[frame_idx] if frame_idx < len(timestamps) else frame_idx / episode["fps"]

                frames.append({
                    "global_frame": global_frame,
                    "gripper_frame": gripper_frame,
                    "joint_positions": jp,
                    "timestamp": ts,
                    "task": episode_task
                })

        elif self.use_lerobot and hasattr(self, '_data_frames'):
            # Load from LeRobot v3.0 format (direct parquet reading)
            try:
                # Get episode's data location
                chunk_idx = episode.get("data_chunk_index", 0)
                file_idx = episode.get("data_file_index", 0)
                from_idx = episode.get("dataset_from_index", 0)
                to_idx = episode.get("dataset_to_index", episode["frame_count"])

                # Check if data file is corrupted
                if (chunk_idx, file_idx) in self._corrupted_files:
                    print(f"[WARNING] Episode {episode_index} data file is corrupted, skipping")
                    return None

                # Get the data frame
                data_df = self._data_frames.get((chunk_idx, file_idx))
                if data_df is None:
                    print(f"[WARNING] Data file not found for episode {episode_index}")
                    return None

                # Filter rows for this episode
                ep_data = data_df[data_df["episode_index"] == episode_index]

                # Try to load video frames if available
                # Get expected shapes from dataset info (if available)
                global_shape = None
                gripper_shape = None
                if hasattr(self, '_lerobot_info') and 'features' in self._lerobot_info:
                    features = self._lerobot_info['features']
                    if 'observation.images.global' in features:
                        global_shape = tuple(features['observation.images.global'].get('shape', []))
                    if 'observation.images.gripper' in features:
                        gripper_shape = tuple(features['observation.images.gripper'].get('shape', []))

                global_video_frames = self._load_video_frames(episode, "observation.images.global", target_shape=global_shape)
                gripper_video_frames = self._load_video_frames(episode, "observation.images.gripper", target_shape=gripper_shape)

                for i, (_, row) in enumerate(ep_data.iterrows()):
                    # Get observation state (joint positions - what was measured)
                    joint_pos = row.get("observation.state", [0.0] * 6)
                    if hasattr(joint_pos, 'tolist'):
                        joint_pos = joint_pos.tolist()

                    # Get action (commanded positions - what was sent to robot)
                    action = row.get("action", None)
                    if action is not None and hasattr(action, 'tolist'):
                        action = action.tolist()

                    # Get frames from video if available
                    global_cam = global_video_frames[i] if global_video_frames and i < len(global_video_frames) else None
                    gripper_cam = gripper_video_frames[i] if gripper_video_frames and i < len(gripper_video_frames) else None

                    timestamp = row.get("timestamp", i / episode["fps"])
                    if hasattr(timestamp, 'item'):
                        timestamp = timestamp.item()

                    frames.append({
                        "global_frame": global_cam,
                        "gripper_frame": gripper_cam,
                        "joint_positions": joint_pos,
                        "action": action,  # Commanded positions for playback
                        "timestamp": float(timestamp),
                        "task": episode_task
                    })

            except Exception as e:
                print(f"[ERROR] Failed to load LeRobot v3.0 frames: {e}")
                import traceback
                traceback.print_exc()
                return None

        return frames

    def playback_episode(self, episode_index: int, start_speed: float = 1.0) -> bool:
        """
        Playback episode with interactive controls.

        Args:
            episode_index: Episode index to playback
            start_speed: Initial playback speed

        Returns:
            True if playback completed, False if aborted

        Controls:
            SPACE: Pause/resume
            LEFT/RIGHT: Jump +/-1 second
            UP/DOWN: Speed up/down
            Q/ESC: Quit playback
        """
        frames = self._load_episode_frames(episode_index)

        if frames is None or len(frames) == 0:
            print(f"[ERROR] No frames to playback for episode {episode_index}")
            return False

        info = self.get_episode_info(episode_index)
        fps = info["fps"] if info else 15

        # Set initial speed
        self.current_speed_idx = self.SPEED_OPTIONS.index(start_speed) if start_speed in self.SPEED_OPTIONS else 2
        self.paused = False

        # Create display window
        cv2.namedWindow("Episode Playback", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Episode Playback", 1320, 400)

        frame_idx = 0
        last_frame_time = time.time()

        print(f"\n[INFO] Playing episode {episode_index}")
        print(f"       Controls: SPACE=pause, LEFT/RIGHT=seek, UP/DOWN=speed, Q=quit")

        try:
            while True:
                current_speed = self.SPEED_OPTIONS[self.current_speed_idx]
                frame_interval = 1.0 / (fps * current_speed) if not self.paused else 0.1

                # Wait for next frame
                elapsed = time.time() - last_frame_time
                if elapsed < frame_interval:
                    # Process key input while waiting
                    key = cv2.waitKey(int((frame_interval - elapsed) * 1000)) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                # Handle key input
                if key == ord('q') or key == 27:  # Q or ESC
                    print("[INFO] Playback stopped")
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"[INFO] {'Paused' if self.paused else 'Resumed'}")
                elif key == 82 or key == ord('w'):  # UP arrow or W
                    if self.current_speed_idx < len(self.SPEED_OPTIONS) - 1:
                        self.current_speed_idx += 1
                        print(f"[INFO] Speed: {self.SPEED_OPTIONS[self.current_speed_idx]}x")
                elif key == 84 or key == ord('s'):  # DOWN arrow or S
                    if self.current_speed_idx > 0:
                        self.current_speed_idx -= 1
                        print(f"[INFO] Speed: {self.SPEED_OPTIONS[self.current_speed_idx]}x")
                elif key == 83 or key == ord('d'):  # RIGHT arrow or D
                    # Jump forward 1 second
                    frame_idx = min(frame_idx + fps, len(frames) - 1)
                    print(f"[INFO] Seek: +1s (frame {frame_idx})")
                elif key == 81 or key == ord('a'):  # LEFT arrow or A
                    # Jump backward 1 second
                    frame_idx = max(frame_idx - fps, 0)
                    print(f"[INFO] Seek: -1s (frame {frame_idx})")

                # Update frame if not paused
                if not self.paused and elapsed >= frame_interval:
                    frame_idx += 1
                    if frame_idx >= len(frames):
                        print("[INFO] Playback complete")
                        break
                    last_frame_time = time.time()

                # Get current frame data
                frame_data = frames[frame_idx]

                # Create display frame
                display = self._create_playback_display(
                    frame_data,
                    episode_index,
                    frame_idx,
                    len(frames),
                    fps,
                    current_speed,
                    self.paused
                )

                cv2.imshow("Episode Playback", display)

        except KeyboardInterrupt:
            print("\n[INFO] Playback interrupted")

        finally:
            cv2.destroyAllWindows()

        return True

    def _create_playback_display(
        self,
        frame_data: Dict,
        episode_index: int,
        frame_idx: int,
        total_frames: int,
        fps: int,
        speed: float,
        paused: bool
    ) -> np.ndarray:
        """
        Create combined display frame for playback.

        Args:
            frame_data: Frame dictionary with camera frames and joint positions
            episode_index: Current episode index
            frame_idx: Current frame index
            total_frames: Total number of frames
            fps: Frame rate
            speed: Current playback speed
            paused: Whether playback is paused

        Returns:
            Combined display frame (BGR numpy array)
        """
        # Get frames
        global_frame = frame_data.get("global_frame")
        gripper_frame = frame_data.get("gripper_frame")
        joint_positions = frame_data.get("joint_positions", [0.0] * 6)
        timestamp = frame_data.get("timestamp", 0.0)
        task = frame_data.get("task", "")

        # Create placeholder if frames are missing
        if global_frame is None:
            global_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(global_frame, "No global frame", (200, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

        if gripper_frame is None:
            gripper_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.putText(gripper_frame, "No gripper", (40, 112),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)

        # Resize frames for display
        global_resized = cv2.resize(global_frame, (640, 360))
        gripper_resized = cv2.resize(gripper_frame, (360, 360))

        # Create info panel (wider to fit task text)
        info_panel = np.zeros((360, 320, 3), dtype=np.uint8)

        # Add episode info
        y_offset = 25
        cv2.putText(info_panel, f"Episode: {episode_index}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        cv2.putText(info_panel, f"Frame: {frame_idx}/{total_frames-1}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        cv2.putText(info_panel, f"Time: {timestamp:.2f}s", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        status = f"[PAUSED] " if paused else ""
        cv2.putText(info_panel, f"{status}Speed: {speed}x", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if paused else (255, 255, 255), 1)
        y_offset += 25

        # Add task/VLM prompt (with word wrapping)
        if task:
            cv2.putText(info_panel, "Task:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
            y_offset += 15

            # Wrap task text (approx 35 chars per line at 0.35 font scale)
            max_chars = 38
            words = task.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            # Display up to 3 lines
            for line in lines[:3]:
                cv2.putText(info_panel, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 150), 1)
                y_offset += 12
            y_offset += 5

        # Add joint positions (compact)
        cv2.putText(info_panel, "Joints:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 15

        for i, jp in enumerate(joint_positions[:6]):
            try:
                jp_val = float(jp)
            except (TypeError, ValueError):
                jp_val = 0.0
            cv2.putText(info_panel, f"J{i}: {jp_val:6.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 255, 150), 1)
            y_offset += 12

        # Add controls (compact)
        y_offset += 8
        cv2.putText(info_panel, "SPACE:pause A/D:seek W/S:spd Q:quit", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Combine frames horizontally
        display = np.hstack([global_resized, gripper_resized, info_panel])

        return display

    def mark_episode(self, episode_index: int, quality: str):
        """
        Mark episode quality.

        Args:
            episode_index: Episode index
            quality: Quality label ("good", "bad", or "unreviewed")
        """
        if quality not in ["good", "bad", "unreviewed"]:
            print(f"[ERROR] Invalid quality: {quality}")
            return

        self.episode_qualities[episode_index] = quality
        self._save_qualities()
        print(f"[OK] Episode {episode_index} marked as: {quality}")

    def _save_qualities(self):
        """Save quality annotations to file."""
        quality_file = self.dataset_path / "episode_qualities.json"
        with open(quality_file, 'w') as f:
            json.dump(self.episode_qualities, f, indent=2)

    def _rebuild_lerobot_after_deletion(self, deleted_indices: List[int]):
        """
        Rebuild LeRobot parquet files after episode deletion.

        This removes deleted episodes from:
        - data/chunk-*/file-*.parquet (frame data)
        - meta/episodes/chunk-*/file-*.parquet (episode metadata)
        - meta/info.json (totals)

        Args:
            deleted_indices: List of episode indices that were deleted
        """
        import pandas as pd

        if not deleted_indices:
            return

        deleted_set = set(deleted_indices)
        print(f"[INFO] Rebuilding LeRobot dataset (removing {len(deleted_indices)} episodes)...")

        # 1. Rebuild data parquet files
        data_dir = self.dataset_path / "data"
        data_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

        for data_file in data_files:
            try:
                df = pd.read_parquet(data_file)
                original_len = len(df)

                # Filter out deleted episodes
                df = df[~df["episode_index"].isin(deleted_set)]

                if len(df) < original_len:
                    if len(df) == 0:
                        # Delete empty file
                        data_file.unlink()
                        print(f"[INFO] Removed empty data file: {data_file.name}")
                    else:
                        # Rewrite filtered data
                        df.to_parquet(data_file, index=False)
                        print(f"[INFO] Updated {data_file.name}: {original_len} -> {len(df)} rows")
            except Exception as e:
                print(f"[WARNING] Failed to update {data_file}: {e}")

        # 2. Rebuild meta/episodes parquet files
        episodes_dir = self.dataset_path / "meta" / "episodes"
        ep_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))

        for ep_file in ep_files:
            try:
                df = pd.read_parquet(ep_file)
                original_len = len(df)

                # Filter out deleted episodes
                df = df[~df["episode_index"].isin(deleted_set)]

                if len(df) < original_len:
                    if len(df) == 0:
                        # Delete empty file
                        ep_file.unlink()
                        print(f"[INFO] Removed empty episode file: {ep_file.name}")
                    else:
                        # Rewrite filtered data
                        df.to_parquet(ep_file, index=False)
                        print(f"[INFO] Updated {ep_file.name}: {original_len} -> {len(df)} episodes")
            except Exception as e:
                print(f"[WARNING] Failed to update {ep_file}: {e}")

        # 3. Update meta/info.json
        info_path = self.dataset_path / "meta" / "info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)

                old_total = info.get("total_episodes", 0)
                info["total_episodes"] = old_total - len(deleted_indices)

                # Recalculate total frames if present
                if "total_frames" in info:
                    # Read remaining data to count frames
                    total_frames = 0
                    for data_file in data_dir.glob("chunk-*/file-*.parquet"):
                        try:
                            df = pd.read_parquet(data_file)
                            total_frames += len(df)
                        except:
                            pass
                    info["total_frames"] = total_frames

                with open(info_path, 'w') as f:
                    json.dump(info, f, indent=2)

                print(f"[INFO] Updated info.json: {old_total} -> {info['total_episodes']} episodes")

            except Exception as e:
                print(f"[WARNING] Failed to update info.json: {e}")

        # 4. Clean up empty chunk directories
        for subdir in ["data", "videos/observation.images.global", "videos/observation.images.gripper", "meta/episodes"]:
            target_dir = self.dataset_path / subdir
            if target_dir.exists():
                for chunk_dir in target_dir.glob("chunk-*"):
                    if chunk_dir.is_dir() and not any(chunk_dir.iterdir()):
                        chunk_dir.rmdir()
                        print(f"[INFO] Removed empty directory: {chunk_dir}")

        print("[OK] LeRobot dataset rebuild complete")

    def delete_bad_episodes(self, confirm: bool = True) -> int:
        """
        Delete all episodes marked as bad.

        Args:
            confirm: Whether to ask for confirmation

        Returns:
            Number of episodes deleted
        """
        bad_episodes = [idx for idx, q in self.episode_qualities.items() if q == "bad"]

        if not bad_episodes:
            print("[INFO] No bad episodes to delete")
            return 0

        print(f"\n[WARNING] About to delete {len(bad_episodes)} episode(s): {bad_episodes}")

        if confirm:
            response = input("Are you sure? (yes/no): ")
            if response.lower() != "yes":
                print("[INFO] Deletion cancelled")
                return 0

        deleted = 0
        deleted_lerobot_indices = []  # Track LeRobot episodes for parquet rebuild

        for ep_idx in bad_episodes:
            # Find episode
            episode = None
            for ep in self.episodes:
                if ep["index"] == ep_idx:
                    episode = ep
                    break

            if episode is None:
                continue

            if episode["format"] == "raw":
                ep_dir = episode["path"]
                try:
                    shutil.rmtree(ep_dir)
                    print(f"[OK] Deleted episode {ep_idx}: {ep_dir}")
                    deleted += 1
                except Exception as e:
                    print(f"[ERROR] Failed to delete {ep_dir}: {e}")

            elif episode["format"] == "lerobot":
                # For LeRobot format, delete the video files for this episode
                try:
                    videos_deleted = 0
                    for video_key in ["observation.images.global", "observation.images.gripper"]:
                        chunk_key = f"videos/{video_key}/chunk_index"
                        file_key = f"videos/{video_key}/file_index"
                        chunk_idx = episode.get(chunk_key, 0)
                        file_idx = episode.get(file_key, 0)

                        video_path = self.dataset_path / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
                        if video_path.exists():
                            video_path.unlink()
                            videos_deleted += 1

                    print(f"[OK] Deleted episode {ep_idx} ({videos_deleted} video files)")
                    deleted += 1
                    deleted_lerobot_indices.append(ep_idx)

                except Exception as e:
                    print(f"[ERROR] Failed to delete LeRobot episode {ep_idx}: {e}")

            # Remove from qualities
            del self.episode_qualities[ep_idx]

        # Save updated qualities
        self._save_qualities()

        # Rebuild LeRobot parquet files if any LeRobot episodes were deleted
        if deleted_lerobot_indices:
            self._rebuild_lerobot_after_deletion(deleted_lerobot_indices)

        # Reload dataset
        self.episodes = []
        self._load_dataset()

        print(f"\n[OK] Deleted {deleted} episode(s)")
        return deleted

    def export_filtered_dataset(self, output_path: str, quality_filter: str = "good") -> int:
        """
        Export filtered episodes to new location.

        Args:
            output_path: Output directory path
            quality_filter: Quality to export ("good", "bad", "all")

        Returns:
            Number of episodes exported
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get episodes to export
        if quality_filter == "all":
            export_indices = [ep["index"] for ep in self.episodes]
        else:
            export_indices = [idx for idx, q in self.episode_qualities.items() if q == quality_filter]

        if not export_indices:
            print(f"[INFO] No episodes matching filter: {quality_filter}")
            return 0

        print(f"\n[INFO] Exporting {len(export_indices)} episode(s) to {output_dir}")

        exported = 0

        for ep_idx in export_indices:
            # Find episode
            episode = None
            for ep in self.episodes:
                if ep["index"] == ep_idx:
                    episode = ep
                    break

            if episode is None:
                continue

            if episode["format"] == "raw":
                src_dir = episode["path"]
                dst_dir = output_dir / f"episode_{exported:06d}"

                try:
                    shutil.copytree(src_dir, dst_dir)

                    # Update metadata with new index
                    metadata_file = dst_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata["episode_index"] = exported
                        metadata["original_index"] = ep_idx
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)

                    print(f"[OK] Exported episode {ep_idx} -> {exported}")
                    exported += 1

                except Exception as e:
                    print(f"[ERROR] Failed to export episode {ep_idx}: {e}")

        print(f"\n[OK] Exported {exported} episode(s) to {output_dir}")
        return exported

    def list_episodes(self):
        """List all episodes with basic info."""
        print(f"\n{'='*90}")
        print(f"{'Idx':>5} | {'Frames':>7} | {'Duration':>10} | {'Quality':>10} | Task")
        print(f"{'='*90}")

        for ep in self.episodes:
            idx = ep["index"]
            frames = ep["frame_count"]
            duration = f"{ep['duration']:.2f}s"
            quality = self.episode_qualities.get(idx, "unreviewed")
            # Use task for LeRobot format, move for raw format
            task = ep.get("task", ep.get("move", "N/A"))

            print(f"{idx:>5} | {frames:>7} | {duration:>10} | {quality:>10} | {task}")

        print(f"{'='*90}")
        print(f"Total: {len(self.episodes)} episodes\n")

    def _parse_indices(self, args: List[str]) -> List[int]:
        """
        Parse episode indices from command arguments.

        Supports multiple formats:
        - Single: "17"
        - Multiple: "17 18 19 20"
        - Range: "17-38"
        - Python list: "[17, 18, 19]" or "17, 18, 19"

        Args:
            args: List of string arguments after the command

        Returns:
            List of parsed integer indices
        """
        indices = []

        # Join args and clean up Python list syntax
        combined = " ".join(args)
        combined = combined.replace("[", "").replace("]", "").replace(",", " ")

        for part in combined.split():
            part = part.strip()
            if not part:
                continue

            # Check for range syntax (e.g., "17-38")
            if "-" in part and not part.startswith("-"):
                try:
                    start, end = part.split("-", 1)
                    start, end = int(start), int(end)
                    if start <= end:
                        indices.extend(range(start, end + 1))
                    else:
                        indices.extend(range(start, end - 1, -1))
                except ValueError:
                    print(f"[WARNING] Invalid range: {part}")
            else:
                try:
                    indices.append(int(part))
                except ValueError:
                    print(f"[WARNING] Invalid index: {part}")

        return indices

    def run_interactive_review(self):
        """Run interactive episode review loop."""
        print(f"\n{'='*60}")
        print("VLA Episode Validator - Interactive Review")
        print(f"{'='*60}\n")

        if len(self.episodes) == 0:
            print("[ERROR] No episodes found in dataset")
            return

        print("Commands:")
        print("  l            - List all episodes")
        print("  i <idx>      - Show episode info")
        print("  p <idx>      - Playback episode (video only)")
        print("  x <idx>      - Execute episode on robot (requires hardware)")
        print("  x <idx> -s   - Execute at half speed (safety mode)")
        print("  x <idx> -d   - Dry run (simulate without robot)")
        print("  g <indices>  - Mark episode(s) as good (e.g., g 5 or g 1-10 or g 1 2 3)")
        print("  b <indices>  - Mark episode(s) as bad (e.g., b 5 or b 17-38 or b 17 18 19)")
        print("  d            - Delete all bad episodes")
        print("  e <path>     - Export good episodes to path")
        print("  q            - Quit")
        print()

        while True:
            try:
                cmd = input("validator> ").strip()

                if not cmd:
                    continue

                parts = cmd.split()
                action = parts[0].lower()

                if action == 'q' or action == 'quit':
                    print("[INFO] Exiting validator")
                    break

                elif action == 'l' or action == 'list':
                    self.list_episodes()

                elif action == 'i' or action == 'info':
                    if len(parts) < 2:
                        print("[ERROR] Usage: i <episode_index>")
                        continue
                    ep_idx = int(parts[1])
                    self.display_episode_info(ep_idx)

                elif action == 'p' or action == 'play':
                    if len(parts) < 2:
                        print("[ERROR] Usage: p <episode_index>")
                        continue
                    ep_idx = int(parts[1])
                    self.playback_episode(ep_idx)

                elif action == 'x' or action == 'exec' or action == 'execute':
                    if len(parts) < 2:
                        print("[ERROR] Usage: x <episode_index> [-s|-d]")
                        print("  -s : half speed (0.5x)")
                        print("  -d : dry run (no robot commands)")
                        continue
                    ep_idx = int(parts[1])
                    # Parse flags
                    speed = 1.0
                    dry_run = False
                    for flag in parts[2:]:
                        if flag == '-s':
                            speed = 0.5
                        elif flag == '-d':
                            dry_run = True
                    self.execute_episode(ep_idx, speed_multiplier=speed, dry_run=dry_run)

                elif action == 'g' or action == 'good':
                    if len(parts) < 2:
                        print("[ERROR] Usage: g <idx> or g <idx1> <idx2> ... or g <start>-<end>")
                        continue
                    indices = self._parse_indices(parts[1:])
                    for ep_idx in indices:
                        self.mark_episode(ep_idx, "good")

                elif action == 'b' or action == 'bad':
                    if len(parts) < 2:
                        print("[ERROR] Usage: b <idx> or b <idx1> <idx2> ... or b <start>-<end>")
                        continue
                    indices = self._parse_indices(parts[1:])
                    for ep_idx in indices:
                        self.mark_episode(ep_idx, "bad")

                elif action == 'd' or action == 'delete':
                    self.delete_bad_episodes()

                elif action == 'e' or action == 'export':
                    if len(parts) < 2:
                        print("[ERROR] Usage: e <output_path>")
                        continue
                    output_path = parts[1]
                    self.export_filtered_dataset(output_path)

                else:
                    print(f"[ERROR] Unknown command: {action}")

            except KeyboardInterrupt:
                print("\n[INFO] Exiting validator")
                break
            except ValueError as e:
                print(f"[ERROR] Invalid argument: {e}")
            except Exception as e:
                print(f"[ERROR] {e}")

        # Cleanup robot connection on exit
        self._disconnect_robot()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate and review VLA training episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive review (default dataset: data/lerobot_episodes/)
    python scripts/validate_vla_episodes.py

    # List all episodes
    python scripts/validate_vla_episodes.py --list

    # Show info for episode 0
    python scripts/validate_vla_episodes.py --info 0

    # Playback episode 0 (video only)
    python scripts/validate_vla_episodes.py --play 0

    # Execute episode 0 on robot hardware
    python scripts/validate_vla_episodes.py --execute 0

    # Execute at half speed (safer for testing)
    python scripts/validate_vla_episodes.py --execute 0 --speed 0.5

    # Dry run (simulate without robot)
    python scripts/validate_vla_episodes.py --execute 0 --dry-run

    # Execute with specific robot port
    python scripts/validate_vla_episodes.py --execute 0 --robot-port /dev/ttyACM0

    # Use custom dataset path
    python scripts/validate_vla_episodes.py --dataset data/my_episodes/

    # Export good episodes
    python scripts/validate_vla_episodes.py --export data/episodes_filtered/
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/lerobot_episodes",
        help="Path to episode dataset directory (default: data/lerobot_episodes)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all episodes and exit"
    )

    parser.add_argument(
        "--info",
        type=int,
        metavar="IDX",
        help="Show info for specific episode and exit"
    )

    parser.add_argument(
        "--play",
        type=int,
        metavar="IDX",
        help="Playback specific episode and exit"
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export good episodes to specified path and exit"
    )

    parser.add_argument(
        "--no-lerobot",
        action="store_true",
        help="Disable LeRobot dataset format (use raw files only)"
    )

    # Robot execution arguments
    parser.add_argument(
        "--execute",
        type=int,
        metavar="IDX",
        help="Execute episode on robot hardware and exit"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for --execute (default: 1.0)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without sending robot commands"
    )

    parser.add_argument(
        "--robot-port",
        type=str,
        default=None,
        help="Robot serial port for --execute (auto-detect if not specified)"
    )

    args = parser.parse_args()

    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"[ERROR] Dataset path not found: {args.dataset}")
        return 1

    # Create validator
    validator = EpisodeValidator(
        dataset_path=args.dataset,
        use_lerobot=not args.no_lerobot
    )

    # Handle command-line actions
    if args.list:
        validator.list_episodes()
        return 0

    if args.info is not None:
        validator.display_episode_info(args.info)
        return 0

    if args.play is not None:
        validator.playback_episode(args.play)
        return 0

    if args.execute is not None:
        success = validator.execute_episode(
            episode_index=args.execute,
            speed_multiplier=args.speed,
            dry_run=args.dry_run,
            robot_port=args.robot_port
        )
        validator._disconnect_robot()
        return 0 if success else 1

    if args.export:
        validator.export_filtered_dataset(args.export)
        return 0

    # Default: run interactive review
    validator.run_interactive_review()

    return 0


if __name__ == "__main__":
    sys.exit(main())
