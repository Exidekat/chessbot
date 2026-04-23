"""
SO-101 kinematics and board calibration for chessbot.

Replaces the deprecated `claw.bridge.kinematics` module. Provides:

- `SO101Kinematics`: forward/inverse kinematics wrapper around
  `lerobot.model.RobotKinematics` (placo-based) using the SO-101 URDF at
  `data/urdf/so101_new_calib.urdf`.
- `BoardCalibration`: loader for `data/calibration/board_calibration.json`
  with bilinear interpolation for chess squares and IK-based joint angle
  resolution.

All joint angles in this module are degrees in "midpoint convention":
0 deg = servo encoder midpoint (2048 counts). The SO-101 URDF defines
mechanical zero at the servo midpoint, so these values are passed straight
through to placo with no offset.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


URDF_PATH = Path(__file__).resolve().parent.parent / "data" / "urdf" / "so101_new_calib.urdf"
ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]
TARGET_FRAME = "gripper_frame_link"

# Max Jacobian-iteration count for IK; placo's single solve step is a linear
# update, so we loop until the position error converges.
_IK_MAX_ITERS = 30
_IK_POS_TOL_M = 1e-4  # 0.1 mm


class SO101Kinematics:
    """FK/IK for the SO-101 arm (5 revolute joints, gripper excluded)."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        from lerobot.model.kinematics import RobotKinematics

        if not Path(urdf_path).exists():
            raise FileNotFoundError(
                f"SO-101 URDF not found at {urdf_path}. "
                "Run `python scripts/download.py` to fetch it."
            )
        self._rk = RobotKinematics(
            str(urdf_path),
            target_frame_name=TARGET_FRAME,
            joint_names=list(ARM_JOINT_NAMES),
        )

    def _dict_to_array(self, angles_deg: dict) -> np.ndarray:
        return np.array([float(angles_deg[n]) for n in ARM_JOINT_NAMES])

    def _array_to_dict(self, arr: np.ndarray) -> dict:
        return {n: round(float(arr[i]), 3) for i, n in enumerate(ARM_JOINT_NAMES)}

    def forward_kinematics(self, joint_angles_deg: dict) -> np.ndarray:
        """Return [x, y, z] in meters for the given joint angles (degrees)."""
        arr = self._dict_to_array(joint_angles_deg)
        T = self._rk.forward_kinematics(arr)
        return np.asarray(T[0:3, 3], dtype=float)

    def inverse_kinematics(
        self,
        target_xyz: np.ndarray,
        seed_deg: dict,
    ) -> dict:
        """Solve IK for target XYZ (meters); return joint angles dict (degrees).

        Uses position-only IK (orientation weight = 0); appropriate for the
        5-DOF SO-101 which cannot achieve arbitrary orientations anyway.
        Iterates placo until the position error is below tolerance.
        """
        pose = np.eye(4)
        pose[0:3, 3] = np.asarray(target_xyz, dtype=float)

        current = self._dict_to_array(seed_deg)
        last_err = float("inf")
        for _ in range(_IK_MAX_ITERS):
            current = self._rk.inverse_kinematics(
                current, pose, position_weight=1.0, orientation_weight=0.0,
            )
            reached = self._rk.forward_kinematics(current)[0:3, 3]
            err = float(np.linalg.norm(reached - pose[0:3, 3]))
            if err < _IK_POS_TOL_M:
                return self._array_to_dict(current)
            last_err = err

        raise ValueError(
            f"IK did not converge to target {target_xyz.tolist()} "
            f"(final error {last_err * 1000:.1f} mm after {_IK_MAX_ITERS} iters)"
        )


class BoardCalibration:
    """Loader + interpolation + IK for the two-plane chess board calibration.

    JSON schema (see scripts/calibrate_ik.py for writer):
      {
        "corners": {
          "a1": {"bottom": {"xyz_meters": [...], "joint_angles_deg": {...}},
                 "top":    {"xyz_meters": [...], "joint_angles_deg": {...}}},
          "h1": {...}, "a8": {...}, "h8": {...}
        },
        "graveyard": {"xyz_meters": [...], "joint_angles_deg": {...}},
        "computed": {"z_bottom_m": ..., "z_top_m": ..., ...}
      }
    """

    _CORNER_NAMES = ("a1", "h1", "a8", "h8")

    def __init__(self, kin: SO101Kinematics):
        self.kin = kin
        self.corners: dict = {}
        self.graveyard: Optional[dict] = None
        self.computed: Optional[dict] = None

    @classmethod
    def load(cls, path: str, kin: SO101Kinematics) -> "BoardCalibration":
        with open(path) as f:
            data = json.load(f)
        cal = cls(kin)
        cal.corners = data.get("corners", {}) or {}
        cal.graveyard = data.get("graveyard")
        cal.computed = data.get("computed")
        return cal

    @property
    def is_calibrated(self) -> bool:
        if not self.computed:
            return False
        return all(name in self.corners for name in self._CORNER_NAMES)

    @property
    def z_bottom(self) -> float:
        return float(self.computed["z_bottom_m"]) if self.computed else 0.0

    @property
    def z_top(self) -> float:
        return float(self.computed["z_top_m"]) if self.computed else 0.0

    @staticmethod
    def _parse_square(square: str) -> tuple[int, int]:
        if len(square) != 2:
            raise ValueError(f"Invalid square: {square!r}")
        file_idx = ord(square[0].lower()) - ord("a")
        rank_idx = int(square[1]) - 1
        if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
            raise ValueError(f"Invalid square: {square!r}")
        return file_idx, rank_idx

    def _corner_xyz(self, corner: str, plane: str) -> np.ndarray:
        return np.asarray(
            self.corners[corner][plane]["xyz_meters"], dtype=float,
        )

    def square_to_xyz(self, square: str, plane: str) -> np.ndarray:
        """Bilinearly interpolate a chess square to XYZ (meters) on the given plane.

        `plane` is "bottom" or "top". `square` may be "graveyard".
        """
        if square == "graveyard":
            if not self.graveyard:
                raise ValueError("Graveyard not calibrated")
            return np.asarray(self.graveyard["xyz_meters"], dtype=float)

        if not self.is_calibrated:
            raise ValueError("Board calibration incomplete")
        if plane not in ("bottom", "top"):
            raise ValueError(f"Unknown plane: {plane!r}")

        file_idx, rank_idx = self._parse_square(square)
        u = file_idx / 7.0  # a->0, h->1
        v = rank_idx / 7.0  # 1->0, 8->1

        a1 = self._corner_xyz("a1", plane)
        h1 = self._corner_xyz("h1", plane)
        a8 = self._corner_xyz("a8", plane)
        h8 = self._corner_xyz("h8", plane)

        return (
            (1 - u) * (1 - v) * a1
            + u * (1 - v) * h1
            + (1 - u) * v * a8
            + u * v * h8
        )

    def _nearest_corner_seed(self, square: str, plane: str) -> dict:
        """Pick the calibration corner whose XY is closest to the target square
        as a seed for IK. Returns the recorded joint angles dict."""
        target = self.square_to_xyz(square, plane)
        best_name = min(
            self._CORNER_NAMES,
            key=lambda c: np.linalg.norm(self._corner_xyz(c, plane)[:2] - target[:2]),
        )
        return dict(self.corners[best_name][plane]["joint_angles_deg"])

    def get_joint_angles(
        self,
        square: str,
        plane: str,
        seed: Optional[dict] = None,
    ) -> dict:
        """Solve IK for the given board square at the given plane.

        For `square == "graveyard"`, the stored joint angles are returned
        directly (no IK) to preserve the exact recorded pose.
        """
        if square == "graveyard":
            if not self.graveyard:
                raise ValueError("Graveyard not calibrated")
            return {k: float(v) for k, v in self.graveyard["joint_angles_deg"].items()}

        xyz = self.square_to_xyz(square, plane)
        if seed is None:
            seed = self._nearest_corner_seed(square, plane)
        # Only the 5 arm joints are needed; drop any extras (e.g. gripper).
        seed5 = {n: float(seed.get(n, 0.0)) for n in ARM_JOINT_NAMES}
        return self.kin.inverse_kinematics(xyz, seed5)
