"""
Chesster kinematics and board calibration for chessbot.

Mirrors `controls.so101_kinematics` (the deprecated SO-101 module) but for
the Chesster arm: 4 arm joints (wrist_roll removed) + 1 gripper. Provides:

- `ChessterKinematics`: forward/inverse kinematics wrapper around
  `lerobot.model.RobotKinematics` (placo-based) using the Chesster URDF at
  `data/urdf/chesster.urdf`.
- `BoardCalibration`: loader for `data/calibration/board_calibration.json`
  with bilinear interpolation for chess squares and IK-based joint angle
  resolution. Adds an INACTIVE pose alongside HOME so that scripts can
  drive the arm to a torque-safe rest position before disabling torque
  (Chesster's longer links make HOME -> torque-off unsafe).

All joint angles in this module are degrees in "midpoint convention":
0 deg = servo encoder midpoint (2048 counts), matching the URDF's
mechanical zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


URDF_PATH = Path(__file__).resolve().parent.parent / "data" / "urdf" / "chesster.urdf"

# Physical motor wiring on Chesster: joint name at each joint_positions
# index (where index = motor ID - 1). Verified empirically by wiggling
# each motor and watching which physical joint moved. Note the gripper
# is *not* at the end of the array on Chesster -- it sits at motor ID 2,
# between elbow and shoulder-pan. shoulder_lift is the last motor.
JOINT_NAMES_BY_INDEX = [
    "wrist_flex",     # motor ID 1
    "gripper",        # motor ID 2
    "shoulder_pan",   # motor ID 3
    "shoulder_lift",  # motor ID 4
    "elbow_flex",     # motor ID 5
]

# Convenience derivations. ARM_JOINT_NAMES is the 4 arm joints in motor
# order (no gripper); used by placo via joint_names. Order is irrelevant
# to placo since it looks up joints by name in the URDF.
ARM_JOINT_NAMES = [n for n in JOINT_NAMES_BY_INDEX if n != "gripper"]
GRIPPER_INDEX = JOINT_NAMES_BY_INDEX.index("gripper")  # = 1

# Joint name -> motor index in joint_positions. Use this for any lookup
# that needs to map a logical joint to its position in the 5-element
# joint_positions array.
JOINT_TO_MOTOR_INDEX = {n: i for i, n in enumerate(JOINT_NAMES_BY_INDEX)}
TARGET_FRAME = "gripper_frame_link"

_IK_MAX_ITERS = 30
_IK_POS_TOL_M = 1e-4  # 0.1 mm


class ChessterKinematics:
    """FK/IK for the Chesster arm (4 revolute joints, gripper excluded)."""

    def __init__(self, urdf_path: Path = URDF_PATH):
        from lerobot.model.kinematics import RobotKinematics

        if not Path(urdf_path).exists():
            raise FileNotFoundError(
                f"Chesster URDF not found at {urdf_path}."
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
        arr = self._dict_to_array(joint_angles_deg)
        T = self._rk.forward_kinematics(arr)
        return np.asarray(T[0:3, 3], dtype=float)

    def inverse_kinematics(
        self,
        target_xyz: np.ndarray,
        seed_deg: dict,
    ) -> dict:
        """Solve IK for target XYZ (meters); return joint angles dict (degrees).

        Position-only IK (orientation weight = 0). Iterates placo until the
        position error converges to _IK_POS_TOL_M.
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

    JSON schema (writer: scripts/chesster_calibrate_ik.py):
      {
        "corners": {
          "a1": {"bottom": {"xyz_meters": [...], "joint_angles_deg": {...}},
                 "top":    {"xyz_meters": [...], "joint_angles_deg": {...}}},
          "h1": {...}, "a8": {...}, "h8": {...}
        },
        "graveyard": {"xyz_meters": [...], "joint_angles_deg": {...}},
        "gripper_open_rad": float,
        "gripper_closed_rad": float,
        "home_rad":     {joint_name: float, ..., "gripper": float},
        "inactive_rad": {joint_name: float, ..., "gripper": float},
        "computed": {"z_bottom_m": ..., "z_top_m": ..., ...}
      }
    """

    _CORNER_NAMES = ("a1", "h1", "a8", "h8")

    def __init__(self, kin: ChessterKinematics):
        self.kin = kin
        self.corners: dict = {}
        self.graveyard: Optional[dict] = None
        self.computed: Optional[dict] = None
        self.home_rad: Optional[dict] = None
        self.inactive_rad: Optional[dict] = None
        self.gripper_open_rad: Optional[float] = None
        self.gripper_closed_rad: Optional[float] = None

    @classmethod
    def load(cls, path: str, kin: ChessterKinematics) -> "BoardCalibration":
        with open(path) as f:
            data = json.load(f)
        cal = cls(kin)
        cal.corners = data.get("corners", {}) or {}
        cal.graveyard = data.get("graveyard")
        cal.computed = data.get("computed")
        cal.home_rad = data.get("home_rad")
        cal.inactive_rad = data.get("inactive_rad")
        cal.gripper_open_rad = data.get("gripper_open_rad")
        cal.gripper_closed_rad = data.get("gripper_closed_rad")
        return cal

    @property
    def is_calibrated(self) -> bool:
        # Joint-space interpolation only needs the four corners and an
        # INACTIVE pose. The 'computed' block (FK-derived board geometry)
        # is informational; we do not rely on it for motion planning since
        # FK is currently unreliable (motor zeros not aligned to URDF).
        if not all(name in self.corners for name in self._CORNER_NAMES):
            return False
        return self.inactive_rad is not None

    @property
    def z_bottom(self) -> float:
        """FK-derived mean Z of bottom corners. Informational only -- not
        used for motion planning (FK is unreliable until motor zeros are
        calibrated). Returns 0.0 if 'computed' block is missing."""
        if self.computed and "z_bottom_m" in self.computed:
            return float(self.computed["z_bottom_m"])
        return 0.0

    @property
    def z_top(self) -> float:
        """FK-derived mean Z of top corners. Informational only (see z_bottom)."""
        if self.computed and "z_top_m" in self.computed:
            return float(self.computed["z_top_m"])
        return 0.0

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
        if square == "graveyard":
            if not self.graveyard:
                raise ValueError("Graveyard not calibrated")
            return np.asarray(self.graveyard["xyz_meters"], dtype=float)

        if not all(name in self.corners for name in self._CORNER_NAMES):
            raise ValueError("Board corners not calibrated")
        if plane not in ("bottom", "top"):
            raise ValueError(f"Unknown plane: {plane!r}")

        file_idx, rank_idx = self._parse_square(square)
        u = file_idx / 7.0
        v = rank_idx / 7.0

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
        seed: Optional[dict] = None,  # accepted for API compat; unused
    ) -> dict:
        """Resolve joint angles for a board square at a given plane via
        bilinear interpolation across the four recorded corner poses.

        We interpolate directly in joint space rather than calling IK on
        the FK-reported corner XYZs: the Chesster URDF's link lengths are
        correct, but per-motor encoder zero offsets are not yet calibrated
        out, so FK is numerically unreliable. The corner joint angles
        themselves are ground truth (the user physically moved the arm to
        those poses), so interpolating between them produces correct arm
        poses for any square on the board to within ~1 cm at the TCP.

        For 'graveyard' the stored joint angles are returned directly.
        """
        if square == "graveyard":
            if not self.graveyard:
                raise ValueError("Graveyard not calibrated")
            return {k: float(v) for k, v in self.graveyard["joint_angles_deg"].items()}

        if not all(name in self.corners for name in self._CORNER_NAMES):
            raise ValueError("Board corners not calibrated")
        if plane not in ("bottom", "top"):
            raise ValueError(f"Unknown plane: {plane!r}")

        file_idx, rank_idx = self._parse_square(square)
        u = file_idx / 7.0  # a->0, h->1
        v = rank_idx / 7.0  # 1->0, 8->1

        # Fallback for joints that weren't recorded at every corner. The
        # pre-swap calibration only captured motors 1-4 per corner, which
        # under Chesster's true wiring means shoulder_lift (motor 5) is
        # absent from every corner. To keep validation/execute runnable
        # before a full re-calibration, fall back to HOME's value for any
        # missing joint -- effectively holding that joint at its HOME
        # position across the board. Joint-space interp degrades to "lift
        # constant" until the user re-records corners.
        home_fallback_deg = self._home_to_midpoint_deg()

        def joint_at(corner: str, name: str) -> float:
            j = self.corners[corner][plane]["joint_angles_deg"]
            if name in j:
                return float(j[name])
            if name in home_fallback_deg:
                return home_fallback_deg[name]
            raise KeyError(
                f"Corner {corner} {plane} is missing joint '{name}' and no "
                f"HOME fallback is available. Re-run scripts/"
                f"chesster_calibrate_ik.py to record full corner poses."
            )

        result: dict = {}
        for name in ARM_JOINT_NAMES:
            a1 = joint_at("a1", name)
            h1 = joint_at("h1", name)
            a8 = joint_at("a8", name)
            h8 = joint_at("h8", name)
            result[name] = round(
                (1 - u) * (1 - v) * a1
                + u * (1 - v) * h1
                + (1 - u) * v * a8
                + u * v * h8,
                3,
            )
        return result

    def _home_to_midpoint_deg(self) -> dict:
        """Convert HOME pose (stored as raw radians) to midpoint-deg keyed
        by joint name. Used as a fallback for joints missing from corner
        records."""
        if not self.home_rad:
            return {}
        counts_per_rev = 4096
        midpoint = counts_per_rev // 2
        out = {}
        for name, raw_rad in self.home_rad.items():
            counts = int(float(raw_rad) / (2.0 * np.pi / counts_per_rev))
            out[name] = (counts - midpoint) * (360.0 / counts_per_rev)
        return out
