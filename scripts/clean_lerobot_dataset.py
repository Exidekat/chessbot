#!/usr/bin/env python3
"""
Clean LeRobot Dataset - Advanced Preprocessing for VLA Training

Features:
- Angle unwrapping: Fixes +-pi discontinuities in joint angles
- Action smoothing: Gaussian filtering to reduce jitter
- Idle frame filtering: Remove frames where robot isn't moving

Based on best practices from OpenPI, VLA-Cache, and OpenVLA.
See: https://github.com/Physical-Intelligence/openpi
"""

import argparse
import json
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.angle_utils import fix_angle_spikes, smooth_actions


def load_quality_annotations(dataset_path: Path) -> Dict[int, str]:
    """
    Load episode quality annotations from dataset.

    Quality annotations are saved by validate_vla_episodes.py to:
    {dataset_path}/episode_qualities.json

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        Dict mapping episode_index -> quality ("good", "bad", or "unreviewed")
    """
    quality_file = dataset_path / "episode_qualities.json"
    if not quality_file.exists():
        return {}

    try:
        with open(quality_file, 'r') as f:
            loaded = json.load(f)
            return {int(k): v for k, v in loaded.items()}
    except Exception as e:
        print(f"[WARN] Failed to load quality annotations: {e}")
        return {}


class FilterMode(Enum):
    """Filtering mode for removing idle frames."""
    POSITION = "position"      # Original: position-based comparison
    MAGNITUDE = "magnitude"    # New: action magnitude filtering


# Alias for backward compatibility
def unwrap_angles(angles: np.ndarray, threshold: float = 2.5, spike_window: int = 5) -> np.ndarray:
    """Alias for fix_angle_spikes() - backward compatible name."""
    return fix_angle_spikes(angles, threshold=threshold, spike_window=spike_window)


def find_idle_frames_by_magnitude(
    actions: np.ndarray,
    threshold: float = 0.01,
) -> Set[int]:
    """
    Find idle frames based on action change magnitude (delta norm).

    This follows OpenPI's approach: filter frames where the action
    CHANGE is below a threshold, indicating the robot isn't moving.

    For position-based action datasets (like ours), we compute the
    difference between consecutive frames to detect motion.

    For a run of N idle frames, keeps first and last, removes middle frames.

    Args:
        actions: Array of action values (N, num_joints)
        threshold: Minimum action change magnitude to consider "moving" (default: 0.01)

    Returns:
        Set of frame indices to REMOVE
    """
    n_frames = len(actions)
    if n_frames < 3:
        return set()

    # Compute action CHANGE magnitude (L2 norm of differences)
    # For position-based actions, this detects actual motion
    action_diffs = np.diff(actions, axis=0)  # (N-1, num_joints)
    diff_magnitudes = np.linalg.norm(action_diffs, axis=1)  # (N-1,)

    # Pad to match original length (first frame has no prior, so mark as "moving")
    action_magnitudes = np.zeros(n_frames)
    action_magnitudes[1:] = diff_magnitudes  # Frame i's magnitude = change from i-1 to i
    action_magnitudes[0] = threshold + 1  # First frame always considered "moving"

    # Find idle frames (below threshold)
    is_idle = action_magnitudes < threshold

    # Find runs of idle frames and mark middle ones for removal
    frames_to_remove = set()
    run_start = None

    for i in range(n_frames):
        if is_idle[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                # End of idle run - remove middle frames (keep first and last)
                run_end = i - 1  # Last idle frame
                if run_end > run_start:
                    for frame_idx in range(run_start + 1, run_end):
                        frames_to_remove.add(frame_idx)
                run_start = None

    # Handle run that extends to end
    if run_start is not None:
        run_end = n_frames - 1
        if run_end > run_start:
            for frame_idx in range(run_start + 1, run_end):
                frames_to_remove.add(frame_idx)

    return frames_to_remove


def find_static_frames(
    states: np.ndarray,
    actions: np.ndarray,
    tolerance: float = 0.005,
) -> Set[int]:
    """
    Find frames to remove (middle frames of static runs).

    A "static run" is a sequence of consecutive frames where both
    observation.state and action values don't change beyond tolerance.

    For a static run of N frames, we keep first and last, removing N-2 middle frames.

    Args:
        states: Array of observation.state values (N, 6)
        actions: Array of action values (N, 6)
        tolerance: Maximum allowed difference to consider frames "same"

    Returns:
        Set of frame indices to REMOVE
    """
    n_frames = len(states)
    if n_frames < 3:
        return set()

    frames_to_remove = set()

    # Find where state OR action changes beyond tolerance
    state_diffs = np.abs(np.diff(states, axis=0)).max(axis=1)  # (N-1,)
    action_diffs = np.abs(np.diff(actions, axis=0)).max(axis=1)  # (N-1,)

    # A transition is "static" if both state and action don't change
    is_static = (state_diffs < tolerance) & (action_diffs < tolerance)

    # Find runs of static frames
    run_start = None
    for i in range(len(is_static)):
        if is_static[i]:
            if run_start is None:
                run_start = i  # Start of static run (frame i and i+1 are same)
        else:
            if run_start is not None:
                # End of static run
                run_end = i  # Frame run_end is different from run_end-1
                # Static run is from frame run_start to frame run_end (inclusive)
                # We keep run_start and run_end, remove everything in between
                for frame_idx in range(run_start + 1, run_end):
                    frames_to_remove.add(frame_idx)
                run_start = None

    # Handle run that extends to end
    if run_start is not None:
        run_end = n_frames - 1
        for frame_idx in range(run_start + 1, run_end):
            frames_to_remove.add(frame_idx)

    return frames_to_remove


def get_video_frame_count(video_path: Path) -> int:
    """Get number of frames in a video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        # Fallback: count packets
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return int(result.stdout.strip())


def extract_frames_to_video(
    input_video: Path,
    output_video: Path,
    keep_frames: List[int],
    fps: int = 15,
    codec: str = "av1",
) -> bool:
    """
    Extract specific frames from input video and create new video.

    Args:
        input_video: Source video path
        output_video: Destination video path
        keep_frames: List of frame indices to keep (0-indexed)
        fps: Output video FPS
        codec: Video codec (av1, h264)

    Returns:
        True if successful
    """
    if not keep_frames:
        return False

    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Create a select filter for specific frames
    # FFmpeg select filter uses 1-indexed frame numbers with eq(n,X)
    select_expr = "+".join([f"eq(n\\,{f})" for f in keep_frames])

    # Codec settings
    if codec == "av1":
        codec_args = ["-c:v", "libaom-av1", "-crf", "30", "-cpu-used", "8"]
    else:
        codec_args = ["-c:v", "libx264", "-crf", "23", "-preset", "fast"]

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_video),
        "-vf", f"select='{select_expr}',setpts=N/{fps}/TB",
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        str(output_video)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] FFmpeg error: {result.stderr}")
        return False

    return output_video.exists()


def extract_frames_to_png(
    input_video: Path,
    output_dir: Path,
    keep_frames: List[int],
    episode_idx: int,
) -> bool:
    """
    Extract specific frames from video to PNG files.

    This provides fast random access during training by eliminating video
    decoding overhead. Results in ~10-20x larger storage but significantly
    faster data loading (no keyframe seeking).

    Output format matches LeRobot's DEFAULT_IMAGE_PATH:
        images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png

    Args:
        input_video: Source video path
        output_dir: Base output directory for images (e.g., images/observation.images.global)
        keep_frames: List of original frame indices to extract (0-indexed)
        episode_idx: Episode index for output directory structure

    Returns:
        True if successful
    """
    if not keep_frames:
        return False

    # Create episode-specific output directory
    # LeRobot format: episode-XXXXXX (with hyphen, not underscore)
    episode_dir = output_dir / f"episode-{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Extract all frames at once using ffmpeg's select filter (more efficient)
    # Build select expression for the frames we want
    select_expr = "+".join([f"eq(n\\,{f})" for f in keep_frames])

    # Use a temporary approach: extract to temp files with ffmpeg frame numbers,
    # then rename to our sequential numbering
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract selected frames to temp directory
        # ffmpeg will output frame_0001.png, frame_0002.png, etc. for selected frames
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(input_video),
            "-vf", f"select='{select_expr}'",
            "-vsync", "vfr",  # Variable frame rate to handle select filter
            str(tmpdir / "frame_%06d.png")
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] FFmpeg PNG extraction error: {result.stderr}")
            return False

        # Rename extracted frames to sequential numbering (0-indexed)
        # LeRobot format: frame-XXXXXX.png (with hyphen, not underscore)
        extracted_frames = sorted(tmpdir.glob("frame_*.png"))
        if len(extracted_frames) != len(keep_frames):
            print(f"[WARN] Expected {len(keep_frames)} frames, got {len(extracted_frames)}")

        for new_idx, src_path in enumerate(extracted_frames):
            dst_path = episode_dir / f"frame-{new_idx:06d}.png"
            shutil.move(str(src_path), str(dst_path))

    return episode_dir.exists() and len(list(episode_dir.glob("*.png"))) > 0


def clean_episode(
    episode_df: pd.DataFrame,
    mode: FilterMode = FilterMode.MAGNITUDE,
    tolerance: float = 0.01,
    idle_threshold: float = 0.01,
    do_unwrap_angles: bool = True,
    unwrap_threshold: float = 3.0,
    do_smooth_actions: bool = False,
    smooth_sigma: float = 1.0,
) -> Tuple[pd.DataFrame, Set[int], int, int, Dict]:
    """
    Clean a single episode with advanced preprocessing.

    Pipeline order: unwrap -> smooth -> filter

    Args:
        episode_df: DataFrame for one episode
        mode: Filtering mode (POSITION or MAGNITUDE)
        tolerance: Tolerance for position-based filtering
        idle_threshold: Threshold for magnitude-based filtering
        do_unwrap_angles: Whether to fix +-pi angle jumps
        unwrap_threshold: Jump threshold for unwrapping
        do_smooth_actions: Whether to apply Gaussian smoothing
        smooth_sigma: Smoothing sigma

    Returns:
        Tuple of (cleaned_df, removed_frame_indices, original_count, new_count, preprocess_info)
    """
    states = np.array(episode_df["observation.state"].tolist())
    actions = np.array(episode_df["action"].tolist())

    preprocess_info = {
        "unwrap_fixes_state": 0,
        "unwrap_fixes_action": 0,
    }

    # Step 1: Angle unwrapping (fixes discontinuities first)
    if do_unwrap_angles:
        orig_states = states.copy()
        orig_actions = actions.copy()

        states = unwrap_angles(states, threshold=unwrap_threshold)
        actions = unwrap_angles(actions, threshold=unwrap_threshold)

        # Count fixes (significant changes)
        preprocess_info["unwrap_fixes_state"] = int(np.sum(np.abs(states - orig_states) > 0.1))
        preprocess_info["unwrap_fixes_action"] = int(np.sum(np.abs(actions - orig_actions) > 0.1))

    # Step 2: Action smoothing (on continuous data)
    if do_smooth_actions:
        actions = smooth_actions(actions, sigma=smooth_sigma)

    # Step 3: Filter idle frames
    if mode == FilterMode.MAGNITUDE:
        frames_to_remove = find_idle_frames_by_magnitude(actions, idle_threshold)
    else:
        frames_to_remove = find_static_frames(states, actions, tolerance)

    original_count = len(episode_df)

    # Get original frame indices to keep
    all_frame_indices = episode_df["frame_index"].values
    keep_mask = ~np.isin(all_frame_indices, list(frames_to_remove))

    # Update the DataFrame with preprocessed values
    cleaned_df = episode_df[keep_mask].copy()

    # Replace state and action values with preprocessed versions
    cleaned_indices = np.where(keep_mask)[0]
    cleaned_df["observation.state"] = [states[i].tolist() for i in cleaned_indices]
    cleaned_df["action"] = [actions[i].tolist() for i in cleaned_indices]

    new_count = len(cleaned_df)

    return cleaned_df, frames_to_remove, original_count, new_count, preprocess_info


def clean_dataset(
    input_path: str,
    output_path: str,
    mode: FilterMode = FilterMode.MAGNITUDE,
    tolerance: float = 0.01,
    idle_threshold: float = 0.01,
    unwrap_angles: bool = True,
    unwrap_threshold: float = 3.0,
    smooth_actions: bool = False,
    smooth_sigma: float = 1.0,
    output_format: str = "png",
    skip_bad: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Clean a LeRobot dataset with advanced preprocessing for VLA training.

    Pipeline order: unwrap angles -> smooth actions -> filter idle frames

    Args:
        input_path: Path to input LeRobot dataset
        output_path: Path to output cleaned dataset
        mode: Filtering mode (POSITION or MAGNITUDE)
        tolerance: Tolerance for position-based filtering
        idle_threshold: Threshold for magnitude-based filtering
        unwrap_angles: Whether to fix +-pi angle jumps
        unwrap_threshold: Jump threshold for unwrapping
        smooth_actions: Whether to apply Gaussian smoothing
        smooth_sigma: Smoothing sigma
        output_format: Output format - 'video' (AV1 re-encode) or 'png' (frame extraction)
                      PNG format provides ~10-20x faster training (no video decoding)
                      at the cost of ~10-20x more storage
        skip_bad: Skip episodes marked as 'bad' in episode_qualities.json (default: True)
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.exists():
        raise ValueError(f"Input dataset not found: {input_dir}")

    # Load metadata
    info_path = input_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    fps = info.get("fps", 15)

    # Load quality annotations if skip_bad is enabled
    quality_annotations = {}
    bad_episodes = set()
    if skip_bad:
        quality_annotations = load_quality_annotations(input_dir)
        bad_episodes = {idx for idx, quality in quality_annotations.items() if quality == "bad"}

    if verbose:
        print(f"[*] Loading dataset from: {input_dir}")
        print(f"    Original episodes: {info['total_episodes']}")
        print(f"    Original frames: {info['total_frames']}")
        print(f"    Filter mode: {mode.value}")
        if skip_bad and quality_annotations:
            print(f"    Quality annotations: {len(quality_annotations)} episodes annotated")
            if bad_episodes:
                print(f"    Skipping bad episodes: {sorted(bad_episodes)}")
        if mode == FilterMode.MAGNITUDE:
            print(f"    Idle threshold: {idle_threshold}")
        else:
            print(f"    Position tolerance: {tolerance}")
        print(f"    Angle unwrapping: {unwrap_angles}")
        print(f"    Action smoothing: {smooth_actions}")
        print(f"    Output format: {output_format}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each parquet file
    data_dir = input_dir / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

    stats = {
        "original_frames": 0,
        "cleaned_frames": 0,
        "removed_frames": 0,
        "episodes_processed": 0,
        "episodes_skipped": 0,
        "skipped_episode_indices": [],
        "per_episode": [],
    }

    # Track frame mapping per episode for video processing
    episode_frame_maps = {}  # episode_idx -> list of original frame indices to keep
    episode_orig_lengths = {}  # episode_idx -> original frame count (for video offset calc)

    all_cleaned_dfs = []
    global_new_index = 0

    for pq_file in parquet_files:
        if verbose:
            print(f"[*] Processing: {pq_file.relative_to(input_dir)}")

        df = pq.read_table(pq_file).to_pandas()

        # Process each episode in this file
        for ep_idx in df["episode_index"].unique():
            # Skip episodes marked as 'bad' in quality annotations
            if ep_idx in bad_episodes:
                if verbose:
                    print(f"    Episode {ep_idx}: SKIPPED (marked as bad)")
                stats["episodes_skipped"] += 1
                stats["skipped_episode_indices"].append(ep_idx)
                continue

            ep_df = df[df["episode_index"] == ep_idx].copy()

            cleaned_df, removed_frames, orig_count, new_count, preprocess_info = clean_episode(
                ep_df,
                mode=mode,
                tolerance=tolerance,
                idle_threshold=idle_threshold,
                do_unwrap_angles=unwrap_angles,
                unwrap_threshold=unwrap_threshold,
                do_smooth_actions=smooth_actions,
                smooth_sigma=smooth_sigma,
            )

            # Build frame mapping for video
            # removed_frames are row indices (0, 1, 2, ...), same as frame_index values
            orig_frames = ep_df["frame_index"].values
            keep_frames = [f for f in orig_frames if f not in removed_frames]
            episode_frame_maps[ep_idx] = keep_frames
            episode_orig_lengths[ep_idx] = orig_count  # Store original length for offset

            # Update indices for cleaned frames
            cleaned_df = cleaned_df.reset_index(drop=True)
            cleaned_df["frame_index"] = range(len(cleaned_df))
            cleaned_df["index"] = range(global_new_index, global_new_index + len(cleaned_df))

            # Recompute timestamps based on new frame indices
            cleaned_df["timestamp"] = cleaned_df["frame_index"] / fps

            # For PNG output, add image path columns
            # HuggingFace datasets.Image() expects dict with 'path' key
            # Paths are relative to the working directory where training will run
            if output_format == "png":
                for cam_key in ["observation.images.global", "observation.images.gripper"]:
                    # Build paths for each frame in the cleaned episode
                    # Include full path from working directory for HuggingFace datasets compatibility
                    paths = []
                    for frame_idx in range(len(cleaned_df)):
                        path = f"{output_path}/images/{cam_key}/episode-{ep_idx:06d}/frame-{frame_idx:06d}.png"
                        paths.append({"path": path, "bytes": None})
                    cleaned_df[cam_key] = paths

            global_new_index += len(cleaned_df)

            all_cleaned_dfs.append(cleaned_df)

            stats["original_frames"] += orig_count
            stats["cleaned_frames"] += new_count
            stats["removed_frames"] += (orig_count - new_count)
            stats["episodes_processed"] += 1
            stats["unwrap_fixes_state"] = stats.get("unwrap_fixes_state", 0) + preprocess_info["unwrap_fixes_state"]
            stats["unwrap_fixes_action"] = stats.get("unwrap_fixes_action", 0) + preprocess_info["unwrap_fixes_action"]
            stats["per_episode"].append({
                "episode": ep_idx,
                "original": orig_count,
                "cleaned": new_count,
                "removed": orig_count - new_count,
                "reduction_pct": round(100 * (orig_count - new_count) / orig_count, 1) if orig_count > 0 else 0,
                "unwrap_fixes": preprocess_info["unwrap_fixes_state"] + preprocess_info["unwrap_fixes_action"],
            })

            if verbose:
                reduction = 100 * (orig_count - new_count) / orig_count if orig_count > 0 else 0
                unwrap_info = ""
                if preprocess_info["unwrap_fixes_state"] + preprocess_info["unwrap_fixes_action"] > 0:
                    unwrap_info = f" [{preprocess_info['unwrap_fixes_state'] + preprocess_info['unwrap_fixes_action']} unwrap fixes]"
                print(f"    Episode {ep_idx}: {orig_count} -> {new_count} frames ({reduction:.1f}% reduction){unwrap_info}")

    # Combine all cleaned data
    if all_cleaned_dfs:
        combined_df = pd.concat(all_cleaned_dfs, ignore_index=True)
    else:
        print("[X] No data to clean!")
        return stats

    # Write cleaned parquet files (single chunk for simplicity)
    out_data_dir = output_dir / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    # Split into files by episode count (same as input structure)
    episodes_per_file = 5
    unique_episodes = combined_df["episode_index"].unique()

    file_idx = 0
    for i in range(0, len(unique_episodes), episodes_per_file):
        ep_batch = unique_episodes[i:i+episodes_per_file]
        batch_df = combined_df[combined_df["episode_index"].isin(ep_batch)]

        out_file = out_data_dir / f"file-{file_idx:03d}.parquet"
        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        pq.write_table(table, out_file)

        if verbose:
            print(f"[OK] Wrote: {out_file.relative_to(output_dir)} ({len(batch_df)} frames)")

        file_idx += 1

    # Determine which episodes are in which video file (needed for both formats)
    # This requires reading the meta/episodes parquet
    episodes_meta_dir = input_dir / "meta" / "episodes" / "chunk-000"

    ep_to_video = {}  # episode_idx -> video_file_idx
    video_to_eps = {}  # video_file_idx -> list of episode_idx

    for meta_file in sorted(episodes_meta_dir.glob("file-*.parquet")):
        file_idx_str = meta_file.stem.split("-")[-1]
        file_idx_int = int(file_idx_str)

        ep_meta = pq.read_table(meta_file).to_pandas()
        for ep_idx in ep_meta["episode_index"].unique():
            ep_to_video[ep_idx] = file_idx_int
            if file_idx_int not in video_to_eps:
                video_to_eps[file_idx_int] = []
            video_to_eps[file_idx_int].append(ep_idx)

    # Process videos/images based on output format
    for video_key in ["observation.images.global", "observation.images.gripper"]:
        video_in_dir = input_dir / "videos" / video_key / "chunk-000"

        if not video_in_dir.exists():
            continue

        if output_format == "png":
            # PNG output: Extract frames to images/ directory
            image_out_dir = output_dir / "images" / video_key
            image_out_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"[*] Extracting PNG frames: {video_key}")

            # Group episodes by source video file
            video_files = sorted(video_in_dir.glob("file-*.mp4"))

            # Process each input video file
            for vid_file_idx, video_file in enumerate(video_files):
                if vid_file_idx not in video_to_eps:
                    continue

                eps_in_video = video_to_eps[vid_file_idx]

                # Process each episode in this video separately
                frame_offset = 0

                for ep_idx in sorted(eps_in_video):
                    if ep_idx not in episode_frame_maps:
                        # Still need to track offset
                        orig_length = episode_orig_lengths.get(ep_idx, 0)
                        frame_offset += orig_length
                        continue

                    ep_frames = episode_frame_maps[ep_idx]
                    if not ep_frames:
                        orig_length = episode_orig_lengths.get(ep_idx, 0)
                        frame_offset += orig_length
                        continue

                    # Calculate absolute frame indices in the video
                    abs_frames = [f + frame_offset for f in ep_frames]

                    if verbose:
                        print(f"    Episode {ep_idx}: extracting {len(abs_frames)} frames to PNG...")

                    success = extract_frames_to_png(
                        video_file, image_out_dir, abs_frames, ep_idx
                    )

                    if success and verbose:
                        print(f"    [OK] Created: episode_{ep_idx:06d}/")
                    elif not success:
                        print(f"    [WARN] Failed to extract episode {ep_idx}")

                    # Update offset for next episode
                    orig_length = episode_orig_lengths.get(ep_idx, 0)
                    frame_offset += orig_length

        else:
            # Video output: Re-encode to videos/ directory (original behavior)
            video_out_dir = output_dir / "videos" / video_key / "chunk-000"
            video_out_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"[*] Processing videos: {video_key}")

            # Group episodes by source video file
            video_files = sorted(video_in_dir.glob("file-*.mp4"))

            # Process each input video file
            for vid_file_idx, video_file in enumerate(video_files):
                if vid_file_idx not in video_to_eps:
                    continue

                eps_in_video = video_to_eps[vid_file_idx]

                # Collect all frames to keep across episodes in this video
                # Note: frames are sequential across episodes in the video
                all_keep_frames = []
                frame_offset = 0

                for ep_idx in sorted(eps_in_video):
                    if ep_idx in episode_frame_maps:
                        # Offset by previous episodes' frames
                        ep_frames = episode_frame_maps[ep_idx]
                        all_keep_frames.extend([f + frame_offset for f in ep_frames])

                        # Use original episode length for offset (not max of kept frames)
                        orig_length = episode_orig_lengths.get(ep_idx, 0)
                        frame_offset += orig_length

                if not all_keep_frames:
                    continue

                out_video = video_out_dir / f"file-{vid_file_idx:03d}.mp4"

                if verbose:
                    print(f"    {video_file.name}: extracting {len(all_keep_frames)} frames...")

                # Get codec from info
                codec = "av1"  # Default
                for feat_key, feat_val in info.get("features", {}).items():
                    if video_key in feat_key and "info" in feat_val:
                        codec = feat_val["info"].get("video.codec", "av1")
                        break

                success = extract_frames_to_video(
                    video_file, out_video, all_keep_frames, fps=fps, codec=codec
                )

                if success and verbose:
                    print(f"    [OK] Created: {out_video.name}")
                elif not success:
                    print(f"    [WARN] Failed to create: {out_video.name}")

    # Copy and update metadata
    out_meta_dir = output_dir / "meta"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    # Update info.json
    new_info = info.copy()
    new_info["total_frames"] = stats["cleaned_frames"]
    # Update splits
    new_info["splits"] = {"train": f"0:{info['total_episodes']}"}

    # Update dtype for image features based on output format
    if output_format == "png":
        # Change dtype from "video" to "image" for camera features
        for key in new_info.get("features", {}):
            if "images" in key:
                new_info["features"][key]["dtype"] = "image"
                # Remove video-specific info (codec, fps, etc.)
                if "info" in new_info["features"][key]:
                    del new_info["features"][key]["info"]
        # Remove video_path template (LeRobot uses DEFAULT_IMAGE_PATH for images)
        if "video_path" in new_info:
            del new_info["video_path"]
        # Update size info (no video files)
        new_info["video_files_size_in_mb"] = 0
        if verbose:
            print(f"[*] Updated feature dtype to 'image' for PNG output")

    with open(out_meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # Copy tasks.parquet
    if (input_dir / "meta" / "tasks.parquet").exists():
        shutil.copy(input_dir / "meta" / "tasks.parquet", out_meta_dir / "tasks.parquet")

    # Create episodes metadata by copying and updating original
    out_episodes_dir = out_meta_dir / "episodes" / "chunk-000"
    out_episodes_dir.mkdir(parents=True, exist_ok=True)

    # Load and update original episode metadata
    episodes_meta_dir = input_dir / "meta" / "episodes" / "chunk-000"
    all_orig_ep_meta = []
    for meta_file in sorted(episodes_meta_dir.glob("file-*.parquet")):
        ep_meta = pq.read_table(meta_file).to_pandas()
        all_orig_ep_meta.append(ep_meta)

    if all_orig_ep_meta:
        orig_ep_meta = pd.concat(all_orig_ep_meta, ignore_index=True)
    else:
        orig_ep_meta = pd.DataFrame()

    # Build updated episodes metadata preserving original columns
    updated_records = []
    dataset_from = 0  # Running counter for dataset indices

    for ep_stat in stats["per_episode"]:
        ep_idx = ep_stat["episode"]
        new_length = ep_stat["cleaned"]

        # Get original record for this episode
        if len(orig_ep_meta) > 0 and ep_idx in orig_ep_meta["episode_index"].values:
            orig_row = orig_ep_meta[orig_ep_meta["episode_index"] == ep_idx].iloc[0].to_dict()
        else:
            orig_row = {}

        # Find which data file this episode is in (based on our output structure)
        ep_file_idx = ep_idx // episodes_per_file

        # Find which video file this episode belongs to (from original mapping)
        vid_file_idx = ep_to_video.get(ep_idx, 0)

        # Compute video timestamps for this episode in the cleaned video
        # This requires knowing the frame offset within the output video
        video_frame_start = 0
        for prev_ep in sorted(episode_frame_maps.keys()):
            if prev_ep < ep_idx and ep_to_video.get(prev_ep, -1) == vid_file_idx:
                video_frame_start += len(episode_frame_maps.get(prev_ep, []))
            elif prev_ep >= ep_idx:
                break

        from_ts = video_frame_start / fps
        to_ts = (video_frame_start + new_length) / fps

        # Build record with all required columns
        record = {
            "episode_index": ep_idx,
            "tasks": orig_row.get("tasks", [f"Episode {ep_idx}"]),
            "length": new_length,
            "data/chunk_index": 0,
            "data/file_index": ep_file_idx,
            "dataset_from_index": dataset_from,
            "dataset_to_index": dataset_from + new_length,
        }

        # Add video/image metadata for each camera key
        for vid_key in ["observation.images.global", "observation.images.gripper"]:
            if output_format == "png":
                # Image-based metadata: no timestamp info, just chunk/file indices
                record[f"images/{vid_key}/chunk_index"] = 0
                record[f"images/{vid_key}/episode_index"] = ep_idx
            else:
                # Video-based metadata: includes timestamps
                record[f"videos/{vid_key}/chunk_index"] = 0
                record[f"videos/{vid_key}/file_index"] = vid_file_idx
                record[f"videos/{vid_key}/from_timestamp"] = from_ts
                record[f"videos/{vid_key}/to_timestamp"] = to_ts

        # Copy over stats columns from original if present
        for col in orig_row:
            if col.startswith("stats/") and col not in record:
                record[col] = orig_row[col]

        # Add meta/episodes file index (which metadata file this episode is in)
        meta_file_idx = ep_idx // episodes_per_file
        record["meta/episodes/chunk_index"] = 0
        record["meta/episodes/file_index"] = meta_file_idx

        updated_records.append(record)
        dataset_from += new_length

    # Write episodes metadata (split to match output data files)
    for i in range(0, len(updated_records), episodes_per_file):
        batch = updated_records[i:i+episodes_per_file]
        file_idx = i // episodes_per_file

        ep_df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(ep_df, preserve_index=False)
        pq.write_table(table, out_episodes_dir / f"file-{file_idx:03d}.parquet")

    # Recompute stats.json
    recompute_stats(output_dir)

    # Copy norm_stats.json if exists
    if (input_dir / "norm_stats.json").exists():
        shutil.copy(input_dir / "norm_stats.json", output_dir / "norm_stats.json")

    if verbose:
        print()
        print("=" * 50)
        print("[OK] Dataset cleaning complete!")
        print("=" * 50)
        print(f"    Episodes processed: {stats['episodes_processed']}")
        if stats['episodes_skipped'] > 0:
            print(f"    Episodes skipped (bad): {stats['episodes_skipped']} {stats['skipped_episode_indices']}")
        print(f"    Original frames: {stats['original_frames']}")
        print(f"    Cleaned frames: {stats['cleaned_frames']}")
        print(f"    Removed frames: {stats['removed_frames']}")
        reduction = 100 * stats['removed_frames'] / stats['original_frames'] if stats['original_frames'] > 0 else 0
        print(f"    Total reduction: {reduction:.1f}%")
        total_unwrap_fixes = stats.get("unwrap_fixes_state", 0) + stats.get("unwrap_fixes_action", 0)
        if total_unwrap_fixes > 0:
            print(f"    Angle unwrap fixes: {total_unwrap_fixes} (state: {stats.get('unwrap_fixes_state', 0)}, action: {stats.get('unwrap_fixes_action', 0)})")
        print(f"    Output: {output_dir}")

    return stats


def recompute_stats(dataset_path: Path):
    """Recompute stats.json from cleaned data."""
    data_dir = dataset_path / "data"

    all_dfs = []
    for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        df = pq.read_table(pq_file).to_pandas()
        all_dfs.append(df)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute statistics
    stats = {}

    for col in ["observation.state", "action"]:
        if col in combined.columns:
            values = np.array(combined[col].tolist())
            stats[col] = {
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
            }

    with open(dataset_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Clean LeRobot dataset with advanced preprocessing for VLA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: PNG frame extraction (fast training, larger storage)
    # Episodes marked as 'bad' in episode_qualities.json are skipped by default
    python scripts/clean_lerobot_dataset.py -i data/lerobot_episodes -o data/clean_lerobot_episodes

    # Include bad episodes (override default skip behavior)
    python scripts/clean_lerobot_dataset.py -i data/lerobot_episodes -o data/clean_lerobot_episodes --include-bad

    # Use video format instead (compact, slower training)
    python scripts/clean_lerobot_dataset.py -i data/lerobot_episodes -o data/clean_lerobot_episodes --no-png

    # With action smoothing
    python scripts/clean_lerobot_dataset.py --smooth-actions --smooth-sigma 1.0

    # Backward compatible: position-based filtering
    python scripts/clean_lerobot_dataset.py --mode position --tolerance 0.01

    # Dry run to preview changes (shows which episodes would be skipped)
    python scripts/clean_lerobot_dataset.py --dry-run

Based on OpenPI, VLA-Cache, and OpenVLA best practices.
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/lerobot_episodes",
        help="Input LeRobot dataset path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/clean_lerobot_episodes",
        help="Output cleaned dataset path"
    )

    # Filtering mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["position", "magnitude"],
        default="magnitude",
        help="Filtering mode: 'magnitude' (action norm, recommended) or 'position' (original)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=0.01,
        help="Position tolerance for 'position' mode (default: 0.01)"
    )
    parser.add_argument(
        "--idle-threshold",
        type=float,
        default=0.01,
        help="Action magnitude threshold for 'magnitude' mode (default: 0.01)"
    )

    # Angle unwrapping
    parser.add_argument(
        "--unwrap-angles",
        action="store_true",
        default=True,
        help="Fix +-pi jumps in joint angles (default: True)"
    )
    parser.add_argument(
        "--no-unwrap-angles",
        action="store_true",
        help="Disable angle unwrapping"
    )
    parser.add_argument(
        "--unwrap-threshold",
        type=float,
        default=2.5,
        help="Jump threshold for spike/angle detection in radians (default: 2.5)"
    )

    # Action smoothing
    parser.add_argument(
        "--smooth-actions",
        action="store_true",
        help="Apply Gaussian smoothing to actions (after unwrapping)"
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma (default: 1.0)"
    )

    parser.add_argument(
        "--output-format",
        type=str,
        choices=["video", "png"],
        default="png",
        help="Output format: 'video' (AV1 re-encode) or 'png' (frame extraction, default)"
    )
    parser.add_argument(
        "--skip-bad",
        action="store_true",
        default=True,
        help="Skip episodes marked as 'bad' in episode_qualities.json (default: True)"
    )
    parser.add_argument(
        "--include-bad",
        action="store_true",
        help="Include episodes marked as 'bad' (overrides --skip-bad)"
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable PNG frame extraction, use video format instead (shortcut for --output-format video)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without creating output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()

    # Handle --no-unwrap-angles flag
    if args.no_unwrap_angles:
        args.unwrap_angles = False

    # Handle --no-png flag (shortcut for --output-format video)
    if args.no_png:
        args.output_format = "video"

    # Handle --include-bad flag (overrides --skip-bad)
    if args.include_bad:
        args.skip_bad = False

    if args.dry_run:
        print("[DRY RUN] Analyzing dataset without creating output...")
        print(f"    Mode: {args.mode}")
        if args.mode == "magnitude":
            print(f"    Idle threshold: {args.idle_threshold}")
        else:
            print(f"    Position tolerance: {args.tolerance}")
        print(f"    Angle unwrapping: {args.unwrap_angles}")
        print(f"    Action smoothing: {args.smooth_actions}")
        print(f"    Skip bad episodes: {args.skip_bad}")
        print()

        input_dir = Path(args.input)
        data_dir = input_dir / "data"

        # Load quality annotations for dry run
        quality_annotations = {}
        bad_episodes = set()
        if args.skip_bad:
            quality_annotations = load_quality_annotations(input_dir)
            bad_episodes = {idx for idx, quality in quality_annotations.items() if quality == "bad"}
            if quality_annotations:
                print(f"Quality annotations: {len(quality_annotations)} episodes annotated")
                if bad_episodes:
                    print(f"Bad episodes to skip: {sorted(bad_episodes)}")
                print()

        total_orig = 0
        total_keep = 0
        total_unwrap_fixes = 0
        total_skipped = 0

        for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
            df = pq.read_table(pq_file).to_pandas()

            for ep_idx in df["episode_index"].unique():
                # Skip bad episodes in dry run
                if ep_idx in bad_episodes:
                    print(f"Episode {ep_idx}: SKIPPED (marked as bad)")
                    total_skipped += 1
                    continue

                ep_df = df[df["episode_index"] == ep_idx]
                states = np.array(ep_df["observation.state"].tolist())
                actions = np.array(ep_df["action"].tolist())

                # Preview preprocessing
                if args.unwrap_angles:
                    orig_states = states.copy()
                    states = unwrap_angles(states, threshold=args.unwrap_threshold)
                    actions = unwrap_angles(actions, threshold=args.unwrap_threshold)
                    # Count fixes
                    fixes = np.sum(np.abs(states - orig_states) > 0.1)
                    total_unwrap_fixes += fixes

                if args.smooth_actions:
                    actions = smooth_actions(actions, sigma=args.smooth_sigma)

                # Select filtering method
                if args.mode == "magnitude":
                    removed = find_idle_frames_by_magnitude(actions, args.idle_threshold)
                else:
                    removed = find_static_frames(states, actions, args.tolerance)

                orig = len(ep_df)
                keep = orig - len(removed)
                total_orig += orig
                total_keep += keep

                reduction = 100 * len(removed) / orig if orig > 0 else 0
                print(f"Episode {ep_idx}: {orig} -> {keep} frames ({reduction:.1f}% reduction)")

        print()
        print(f"Total: {total_orig} -> {total_keep} frames")
        print(f"Would remove: {total_orig - total_keep} frames ({100*(total_orig-total_keep)/total_orig:.1f}%)")
        if total_skipped > 0:
            print(f"Episodes skipped (bad): {total_skipped}")
        if args.unwrap_angles and total_unwrap_fixes > 0:
            print(f"Angle unwrap fixes: {total_unwrap_fixes} corrections")
        return

    clean_dataset(
        args.input,
        args.output,
        mode=FilterMode(args.mode),
        tolerance=args.tolerance,
        idle_threshold=args.idle_threshold,
        unwrap_angles=args.unwrap_angles,
        unwrap_threshold=args.unwrap_threshold,
        smooth_actions=args.smooth_actions,
        smooth_sigma=args.smooth_sigma,
        output_format=args.output_format,
        skip_bad=args.skip_bad,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
