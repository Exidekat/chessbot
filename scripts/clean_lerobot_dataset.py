#!/usr/bin/env python3
"""
Clean LeRobot Dataset - Remove Consecutive Static Frames

Removes consecutive frames where joint positions don't change (within tolerance).
For each run of N static frames, keeps only the first and last frame.

This reduces dataset size while preserving meaningful motion data.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Set
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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


def clean_episode(
    episode_df: pd.DataFrame,
    tolerance: float,
) -> Tuple[pd.DataFrame, Set[int], int, int]:
    """
    Clean a single episode by removing static middle frames.

    Args:
        episode_df: DataFrame for one episode
        tolerance: Tolerance for considering frames "same"

    Returns:
        Tuple of (cleaned_df, removed_frame_indices, original_count, new_count)
    """
    states = np.array(episode_df["observation.state"].tolist())
    actions = np.array(episode_df["action"].tolist())

    frames_to_remove = find_static_frames(states, actions, tolerance)

    original_count = len(episode_df)

    # Get original frame indices to keep
    all_frame_indices = episode_df["frame_index"].values
    keep_mask = ~np.isin(all_frame_indices, list(frames_to_remove))

    cleaned_df = episode_df[keep_mask].copy()
    new_count = len(cleaned_df)

    return cleaned_df, frames_to_remove, original_count, new_count


def clean_dataset(
    input_path: str,
    output_path: str,
    tolerance: float = 0.005,
    verbose: bool = True,
) -> Dict:
    """
    Clean a LeRobot dataset by removing static frames.

    Args:
        input_path: Path to input LeRobot dataset
        output_path: Path to output cleaned dataset
        tolerance: Tolerance for frame comparison
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

    if verbose:
        print(f"[*] Loading dataset from: {input_dir}")
        print(f"    Original episodes: {info['total_episodes']}")
        print(f"    Original frames: {info['total_frames']}")
        print(f"    Tolerance: {tolerance}")

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
            ep_df = df[df["episode_index"] == ep_idx].copy()

            cleaned_df, removed_frames, orig_count, new_count = clean_episode(
                ep_df, tolerance
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

            global_new_index += len(cleaned_df)

            all_cleaned_dfs.append(cleaned_df)

            stats["original_frames"] += orig_count
            stats["cleaned_frames"] += new_count
            stats["removed_frames"] += (orig_count - new_count)
            stats["episodes_processed"] += 1
            stats["per_episode"].append({
                "episode": ep_idx,
                "original": orig_count,
                "cleaned": new_count,
                "removed": orig_count - new_count,
                "reduction_pct": round(100 * (orig_count - new_count) / orig_count, 1) if orig_count > 0 else 0,
            })

            if verbose:
                reduction = 100 * (orig_count - new_count) / orig_count if orig_count > 0 else 0
                print(f"    Episode {ep_idx}: {orig_count} -> {new_count} frames ({reduction:.1f}% reduction)")

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

    # Process videos
    for video_key in ["observation.images.global", "observation.images.gripper"]:
        video_in_dir = input_dir / "videos" / video_key / "chunk-000"
        video_out_dir = output_dir / "videos" / video_key / "chunk-000"
        video_out_dir.mkdir(parents=True, exist_ok=True)

        if not video_in_dir.exists():
            continue

        if verbose:
            print(f"[*] Processing videos: {video_key}")

        # Group episodes by source video file
        video_files = sorted(video_in_dir.glob("file-*.mp4"))

        # Determine which episodes are in which video file
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

        # Add video metadata for each video key
        for vid_key in ["observation.images.global", "observation.images.gripper"]:
            record[f"videos/{vid_key}/chunk_index"] = 0
            record[f"videos/{vid_key}/file_index"] = vid_file_idx
            record[f"videos/{vid_key}/from_timestamp"] = from_ts
            record[f"videos/{vid_key}/to_timestamp"] = to_ts

        # Copy over stats columns from original if present
        for col in orig_row:
            if col.startswith("stats/") and col not in record:
                record[col] = orig_row[col]

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
        print(f"    Original frames: {stats['original_frames']}")
        print(f"    Cleaned frames: {stats['cleaned_frames']}")
        print(f"    Removed frames: {stats['removed_frames']}")
        reduction = 100 * stats['removed_frames'] / stats['original_frames'] if stats['original_frames'] > 0 else 0
        print(f"    Total reduction: {reduction:.1f}%")
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
        description="Clean LeRobot dataset by removing static frames"
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
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=0.005,
        help="Tolerance for considering frames 'same' (default: 0.005)"
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

    if args.dry_run:
        print("[DRY RUN] Analyzing dataset without creating output...")
        # Just analyze and report
        input_dir = Path(args.input)
        data_dir = input_dir / "data"

        total_orig = 0
        total_keep = 0

        for pq_file in sorted(data_dir.glob("chunk-*/file-*.parquet")):
            df = pq.read_table(pq_file).to_pandas()

            for ep_idx in df["episode_index"].unique():
                ep_df = df[df["episode_index"] == ep_idx]
                states = np.array(ep_df["observation.state"].tolist())
                actions = np.array(ep_df["action"].tolist())

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
        return

    clean_dataset(
        args.input,
        args.output,
        tolerance=args.tolerance,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
