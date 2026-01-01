#!/usr/bin/env python3
"""
VLA Episode Validation Tool

Review, playback, and filter collected VLA episodes.
Supports both LeRobot dataset format and raw file storage.

Features:
- Episode information display (metadata, statistics)
- Video playback with controls (pause, speed, seek)
- Quality control (mark good/bad, delete bad episodes)
- Export filtered dataset

Usage:
    # Interactive review
    python scripts/validate_vla_episodes.py --dataset data/episodes/

    # Export good episodes only
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --export data/episodes_filtered/

    # List all episodes
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --list
"""

import argparse
import json
import shutil
import time
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
        """Load LeRobot format dataset."""
        try:
            # Local datasets need repo_id + root path
            # repo_id format matches what collect_vla_episodes.py uses
            repo_id = f"local/chess_vla_{self.dataset_path.name}"
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=str(self.dataset_path),
                download_videos=False,
                video_backend="pyav"  # Use pyav instead of torchcodec (FFmpeg issues)
            )

            # LeRobot v0.4: episode info is in meta.episodes
            fps = self.dataset.fps
            ep_dataset = self.dataset.meta.episodes

            for i in range(len(ep_dataset)):
                row = ep_dataset[i]
                ep_idx = row["episode_index"]
                frame_count = row["length"]
                tasks = row["tasks"]

                self.episodes.append({
                    "index": int(ep_idx),
                    "frame_count": frame_count,
                    "fps": fps,
                    "duration": frame_count / fps if fps > 0 else 0,
                    "task": tasks[0] if tasks else "",
                    "format": "lerobot"
                })

        except Exception as e:
            print(f"[ERROR] Failed to load LeRobot dataset: {e}")
            print("[INFO] Falling back to raw file detection")
            self._load_raw_dataset()

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

    def _load_episode_frames(self, episode_index: int) -> Optional[List[Dict]]:
        """
        Load all frames for an episode.

        Args:
            episode_index: Episode index

        Returns:
            List of frame dictionaries with 'global_frame', 'gripper_frame', 'joint_positions'
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

        elif self.use_lerobot and self.dataset is not None:
            # Load from LeRobot dataset (v0.4 API)
            try:
                # Calculate start index by summing lengths of previous episodes
                start_idx = 0
                for ep in self.episodes:
                    if ep["index"] < episode_index:
                        start_idx += ep["frame_count"]
                    elif ep["index"] == episode_index:
                        break

                # Load frames for this episode
                for i in range(episode["frame_count"]):
                    global_idx = start_idx + i

                    # Get frame data from dataset
                    sample = self.dataset[global_idx]

                    # Get task from sample or episode metadata
                    task_val = sample.get("language_instruction", sample.get("task", episode_task))

                    # Handle tensor conversion for images
                    global_cam = sample.get("observation.global_camera")
                    gripper_cam = sample.get("observation.gripper_camera")
                    joint_pos = sample.get("observation.joint_positions", [0.0] * 6)

                    # Convert tensors to numpy if needed
                    if hasattr(global_cam, 'numpy'):
                        global_cam = global_cam.permute(1, 2, 0).numpy()  # CHW -> HWC
                        global_cam = (global_cam * 255).astype(np.uint8)
                        global_cam = cv2.cvtColor(global_cam, cv2.COLOR_RGB2BGR)
                    if hasattr(gripper_cam, 'numpy'):
                        gripper_cam = gripper_cam.permute(1, 2, 0).numpy()
                        gripper_cam = (gripper_cam * 255).astype(np.uint8)
                        gripper_cam = cv2.cvtColor(gripper_cam, cv2.COLOR_RGB2BGR)
                    if hasattr(joint_pos, 'numpy'):
                        joint_pos = joint_pos.numpy()

                    frames.append({
                        "global_frame": global_cam,
                        "gripper_frame": gripper_cam,
                        "joint_positions": joint_pos,
                        "timestamp": i / episode["fps"],
                        "task": task_val
                    })

            except Exception as e:
                print(f"[ERROR] Failed to load LeRobot frames: {e}")
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
            jp_val = jp if isinstance(jp, (int, float)) else 0.0
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

            # Remove from qualities
            del self.episode_qualities[ep_idx]

        # Save updated qualities
        self._save_qualities()

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

    def run_interactive_review(self):
        """Run interactive episode review loop."""
        print(f"\n{'='*60}")
        print("VLA Episode Validator - Interactive Review")
        print(f"{'='*60}\n")

        if len(self.episodes) == 0:
            print("[ERROR] No episodes found in dataset")
            return

        print("Commands:")
        print("  l        - List all episodes")
        print("  i <idx>  - Show episode info")
        print("  p <idx>  - Playback episode")
        print("  g <idx>  - Mark episode as good")
        print("  b <idx>  - Mark episode as bad")
        print("  d        - Delete all bad episodes")
        print("  e <path> - Export good episodes to path")
        print("  q        - Quit")
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

                elif action == 'g' or action == 'good':
                    if len(parts) < 2:
                        print("[ERROR] Usage: g <episode_index>")
                        continue
                    ep_idx = int(parts[1])
                    self.mark_episode(ep_idx, "good")

                elif action == 'b' or action == 'bad':
                    if len(parts) < 2:
                        print("[ERROR] Usage: b <episode_index>")
                        continue
                    ep_idx = int(parts[1])
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


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate and review VLA training episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive review
    python scripts/validate_vla_episodes.py --dataset data/episodes/

    # List all episodes
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --list

    # Show info for episode 0
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --info 0

    # Playback episode 0
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --play 0

    # Export good episodes
    python scripts/validate_vla_episodes.py --dataset data/episodes/ --export data/episodes_filtered/
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/episodes",
        help="Path to episode dataset directory (default: data/episodes)"
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

    if args.export:
        validator.export_filtered_dataset(args.export)
        return 0

    # Default: run interactive review
    validator.run_interactive_review()

    return 0


if __name__ == "__main__":
    sys.exit(main())
