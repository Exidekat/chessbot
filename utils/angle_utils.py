"""
Angle Utilities - Spike Protection and Smoothing for Robot Joint Data

Functions for detecting and fixing angle discontinuities in robot joint data:
- Spikes/glitches: Sudden jumps that return to normal within a few frames
- Wraparound: Angle crossing the -pi/+pi boundary

Used by:
- scripts/clean_lerobot_dataset.py: Preprocessing training data
- scripts/vla_deploy.py: Real-time observation processing during inference

This ensures data consistency between training and inference.
"""

import numpy as np
from typing import Optional


def fix_angle_spikes(
    angles: np.ndarray,
    threshold: float = 2.5,
    spike_window: int = 5,
) -> np.ndarray:
    """
    Fix angle discontinuities: both wraparound and spikes/glitches.

    Handles two types of issues:
    1. Angle wraparound at -pi/+pi boundary (continuous motion through boundary)
    2. Spikes/glitches where value jumps and returns within a few frames

    For spikes, we interpolate over the anomalous frames rather than
    shifting all following frames.

    Args:
        angles: Array of joint angles (N, num_joints) or (N,) or (num_joints,)
        threshold: Jump threshold in radians (default: 2.5)
        spike_window: Max frames to look ahead for spike return (default: 5)

    Returns:
        Corrected angles with continuous trajectories
    """
    # Handle single frame (inference case)
    if angles.ndim == 1:
        # Single frame - nothing to fix, return as-is
        return angles.copy()

    corrected = angles.copy().astype(np.float64)
    n_frames, n_joints = corrected.shape

    if n_frames < 2:
        return corrected

    for j in range(n_joints):
        i = 1
        while i < n_frames:
            diff = corrected[i, j] - corrected[i-1, j]

            if abs(diff) > threshold:
                # Found a large jump - check if it's a spike or wraparound
                pre_spike_val = corrected[i-1, j]

                # Look ahead to see if value returns to pre-spike level
                spike_end = None
                for k in range(i, min(i + spike_window + 1, n_frames)):
                    return_diff = corrected[k, j] - pre_spike_val
                    # Check if we're back near the original value
                    if abs(return_diff) < threshold / 2:
                        # Value returned - this was a spike
                        spike_end = k
                        break

                if spike_end is not None and spike_end > i:
                    # SPIKE DETECTED: interpolate over the anomalous frames
                    # Find the next valid value after the spike
                    post_spike_val = corrected[spike_end, j]

                    # Linear interpolation from pre-spike to post-spike
                    for idx in range(i, spike_end):
                        t = (idx - (i - 1)) / (spike_end - (i - 1))
                        corrected[idx, j] = pre_spike_val + t * (post_spike_val - pre_spike_val)

                    i = spike_end + 1
                else:
                    # TRUE WRAPAROUND: shift all following frames
                    if diff > threshold:
                        corrected[i:, j] -= 2 * np.pi
                    else:
                        corrected[i:, j] += 2 * np.pi
                    i += 1
            else:
                i += 1

    return corrected


class RealTimeSpikeFilter:
    """
    Real-time spike filter for streaming joint data during inference.

    Maintains a history buffer to detect and fix spikes as they occur.
    Uses the same spike detection logic as batch processing for consistency.

    Usage:
        filter = RealTimeSpikeFilter(num_joints=6)

        # In control loop:
        fixed_state = filter.process(current_state)
    """

    def __init__(
        self,
        num_joints: int = 6,
        threshold: float = 2.5,
        history_size: int = 5,
    ):
        """
        Initialize real-time spike filter.

        Args:
            num_joints: Number of robot joints
            threshold: Jump threshold in radians (default: 2.5)
            history_size: Number of frames to keep in history buffer
        """
        self.num_joints = num_joints
        self.threshold = threshold
        self.history_size = history_size

        # History buffer: stores last N valid readings
        self.history = np.zeros((history_size, num_joints), dtype=np.float64)
        self.history_idx = 0
        self.frames_seen = 0

        # Track if we're in a spike for each joint
        self.in_spike = np.zeros(num_joints, dtype=bool)
        self.spike_start_val = np.zeros(num_joints, dtype=np.float64)
        self.spike_frame_count = np.zeros(num_joints, dtype=int)

    def reset(self):
        """Reset filter state."""
        self.history.fill(0)
        self.history_idx = 0
        self.frames_seen = 0
        self.in_spike.fill(False)
        self.spike_start_val.fill(0)
        self.spike_frame_count.fill(0)

    def process(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Process a single frame of joint positions.

        Detects spikes by comparing to recent history and either:
        - Returns the value if it's normal
        - Returns interpolated value if we're in/exiting a spike
        - Starts tracking a spike if a sudden jump is detected

        Args:
            joint_positions: Current joint positions (num_joints,)

        Returns:
            Fixed joint positions (num_joints,)
        """
        current = np.asarray(joint_positions, dtype=np.float64).flatten()

        if len(current) != self.num_joints:
            # Wrong size - just return as-is
            return current

        self.frames_seen += 1

        # First frame - just store and return
        if self.frames_seen == 1:
            self._add_to_history(current)
            return current

        # Get last valid reading
        last_valid = self.history[(self.history_idx - 1) % self.history_size]

        fixed = current.copy()

        for j in range(self.num_joints):
            diff = current[j] - last_valid[j]

            if self.in_spike[j]:
                # We're in a spike - check if value returned to normal
                return_diff = current[j] - self.spike_start_val[j]
                self.spike_frame_count[j] += 1

                if abs(return_diff) < self.threshold / 2:
                    # Spike ended - use current value
                    fixed[j] = current[j]
                    self.in_spike[j] = False
                    self.spike_frame_count[j] = 0
                elif self.spike_frame_count[j] >= self.history_size:
                    # Spike lasted too long - probably not a spike
                    # Accept the new value as valid
                    fixed[j] = current[j]
                    self.in_spike[j] = False
                    self.spike_frame_count[j] = 0
                else:
                    # Still in spike - hold at pre-spike value
                    fixed[j] = self.spike_start_val[j]

            elif abs(diff) > self.threshold:
                # New spike detected
                self.in_spike[j] = True
                self.spike_start_val[j] = last_valid[j]
                self.spike_frame_count[j] = 1
                # Hold at pre-spike value
                fixed[j] = last_valid[j]
            else:
                # Normal reading
                fixed[j] = current[j]

        # Add fixed value to history
        self._add_to_history(fixed)

        return fixed

    def _add_to_history(self, values: np.ndarray):
        """Add values to circular history buffer."""
        self.history[self.history_idx] = values
        self.history_idx = (self.history_idx + 1) % self.history_size


def smooth_actions(
    actions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to action sequences.

    Args:
        actions: Array of actions (N, num_joints)
        sigma: Standard deviation for Gaussian kernel (default: 1.0)

    Returns:
        Smoothed actions
    """
    from scipy.ndimage import gaussian_filter1d

    if actions.ndim == 1:
        return gaussian_filter1d(actions, sigma=sigma, axis=0)

    # Apply smoothing independently to each joint
    smoothed = np.zeros_like(actions)
    for j in range(actions.shape[1]):
        smoothed[:, j] = gaussian_filter1d(actions[:, j], sigma=sigma)

    return smoothed
