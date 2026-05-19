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
        "grid_extras": {                # optional, enables biquadratic interp
          "mid_de1":  {bottom, top},    # rank-1 edge midpoint (between d1 and e1)
          "mid_de8":  {bottom, top},    # rank-8 edge midpoint
          "mid_a45":  {bottom, top},    # file-a edge midpoint (between a4 and a5)
          "mid_h45":  {bottom, top},    # file-h edge midpoint
          "center":   {bottom, top}     # board center (between d4/d5/e4/e5)
        },
        "graveyard": {"xyz_meters": [...], "joint_angles_deg": {...}},
        "gripper_open_rad": float,
        "gripper_closed_rad": float,
        "home_rad":     {joint_name: float, ..., "gripper": float},
        "inactive_rad": {joint_name: float, ..., "gripper": float},
        "computed": {"z_bottom_m": ..., "z_top_m": ..., ...}
      }

    Interpolation:
      - If `grid_extras` is fully populated, get_joint_angles uses biquadratic
        Lagrange tensor-product interpolation over a 3x3 grid (corners + 5
        midpoints). This captures the joint-space non-linearity that makes
        bilinear-from-corners-alone systematically miss interior squares by
        up to one rank.
      - Otherwise, falls back to bilinear from the 4 corners.
    """

    _CORNER_NAMES = ("a1", "h1", "a8", "h8")

    # 3x3 grid layout for biquadratic interpolation. Grid coordinates
    # (i, j) range over {0, 1, 2} in each axis; each cell maps to a
    # calibration source (either a corner or a `grid_extras` entry) and
    # a normalized (u, v) ∈ {0, 0.5, 1} that matches the Lagrange basis
    # node positions.
    #
    #   (i, j)  →  (kind, name, u_node, v_node)
    _GRID_LAYOUT = [
        # i=0 (u=0, file a)
        [("corner", "a1",      0.0, 0.0),
         ("extras", "mid_a45", 0.0, 0.5),
         ("corner", "a8",      0.0, 1.0)],
        # i=1 (u=0.5, between files d and e)
        [("extras", "mid_de1", 0.5, 0.0),
         ("extras", "center",  0.5, 0.5),
         ("extras", "mid_de8", 0.5, 1.0)],
        # i=2 (u=1, file h)
        [("corner", "h1",      1.0, 0.0),
         ("extras", "mid_h45", 1.0, 0.5),
         ("corner", "h8",      1.0, 1.0)],
    ]

    _GRID_EXTRAS_NAMES = (
        "mid_de1", "mid_de8", "mid_a45", "mid_h45", "center",
    )

    def __init__(self, kin: ChessterKinematics):
        self.kin = kin
        self.corners: dict = {}
        self.grid_extras: dict = {}
        self.graveyard: Optional[dict] = None
        self.computed: Optional[dict] = None
        self.home_rad: Optional[dict] = None
        self.inactive_rad: Optional[dict] = None
        self.gripper_open_rad: Optional[float] = None
        self.gripper_closed_rad: Optional[float] = None
        # IK block (populated by scripts/chesster_fit_ik.py).
        self.ik: Optional[dict] = None

    @classmethod
    def load(cls, path: str, kin: ChessterKinematics) -> "BoardCalibration":
        with open(path) as f:
            data = json.load(f)
        cal = cls(kin)
        cal.corners = data.get("corners", {}) or {}
        cal.grid_extras = data.get("grid_extras", {}) or {}
        cal.graveyard = data.get("graveyard")
        cal.computed = data.get("computed")
        cal.home_rad = data.get("home_rad")
        cal.inactive_rad = data.get("inactive_rad")
        cal.gripper_open_rad = data.get("gripper_open_rad")
        cal.gripper_closed_rad = data.get("gripper_closed_rad")
        cal.ik = data.get("ik")
        return cal

    @property
    def has_ik(self) -> bool:
        """True iff a fitted ik block is available (encoder offsets +
        joint signs + board pose). When True, `get_joint_angles` will
        use closed-form IK by default; pass use_ik=False to force
        interpolation."""
        if not self.ik:
            return False
        return all(k in self.ik for k in (
            "encoder_offsets_deg", "joint_signs", "board_pose_in_robot",
            "square_pitch_m",
        ))

    def has_full_grid(self, plane: str = "bottom") -> bool:
        """True iff all 5 grid_extras midpoints are present at the given plane."""
        for name in self._GRID_EXTRAS_NAMES:
            entry = self.grid_extras.get(name)
            if not entry or plane not in entry:
                return False
        return True

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
        seed: Optional[dict] = None,
        use_ik: Optional[bool] = None,
    ) -> dict:
        """Resolve joint angles for a board square at a given plane via
        biquadratic (9-point) or bilinear (4-corner) interpolation across
        the recorded calibration poses.

        We interpolate directly in joint space rather than calling IK on
        the FK-reported corner XYZs: the Chesster URDF's link lengths are
        correct, but per-motor encoder zero offsets are not yet calibrated
        out, so FK is numerically unreliable. The recorded joint angles
        are ground truth (the user physically moved the arm to those
        poses), so interpolating between them produces correct arm poses.

        Bilinear from 4 corners alone has a known limitation: the arm's
        joint-to-Cartesian map is non-linear (shoulder_lift in particular
        covers ~78° across the board), so the joint-space midpoint does
        NOT correspond to the board midpoint -- interior squares get
        biased by up to ~1 rank. Adding 5 midpoint calibration poses
        (`grid_extras` block) lets us use **piecewise bilinear** on a 3x3
        grid: the target's (u,v) is localized to one of four sub-cells
        (each spanning half the board) and bilinearly interpolated from
        that sub-cell's 4 anchors. This captures non-linearity AND avoids
        the Runge-phenomenon overshoot that pure biquadratic (a single
        polynomial through 9 points) produces on highly non-linear fields.

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

        # Default: use IK if a fitted ik block is available, else interp.
        # `use_ik=False` forces the interp path (e.g., for the
        # --interpolation flag in the execute script).
        if use_ik is None:
            use_ik = self.has_ik
        if use_ik and self.has_ik:
            return self._joint_angles_via_ik(square, plane, seed)

        file_idx, rank_idx = self._parse_square(square)
        u = file_idx / 7.0  # a->0, h->1
        v = rank_idx / 7.0  # 1->0, 8->1

        # Fallback for joints that weren't recorded at every corner.
        home_fallback_deg = self._home_to_midpoint_deg()

        if self.has_full_grid(plane):
            return self._piecewise_bilinear_interp(u, v, plane, home_fallback_deg)
        return self._bilinear_interp(u, v, plane, home_fallback_deg)

    def _joint_angles_via_ik(self, square: str, plane: str,
                             seed: Optional[dict]) -> dict:
        """Closed-form IK from the fitted ik block.

        Maps the requested chess square to a target XYZ in the robot base
        frame (using the fitted board pose + square pitch), runs the
        Chesster 4-DOF IK with the nearest-anchor seed for branch
        selection, then converts URDF angles back to servo angles via
        the fitted signs and offsets.
        """
        import numpy as np
        from controls.chesster_ik_fit import (
            chesster_ik, servo_to_urdf, urdf_to_servo,
        )

        ik = self.ik
        offsets = ik["encoder_offsets_deg"]
        signs = ik["joint_signs"]
        T = np.array(ik["board_pose_in_robot"])
        pitch = ik["square_pitch_m"]
        top_h = ik.get("top_plane_height_m", 0.10)
        # Board-frame correction applied to every IK target. Accounts for
        # any systematic bias between the gripper-tip pose recorded
        # during calibration and the effective grasping point during
        # pickup -- e.g. if the URDF TCP is offset from the centerline
        # between the gripper jaws, or if the user calibrated against a
        # consistent fraction-of-a-square offset. Tunable via
        # scripts/chesster_set_tcp_offset.py.
        tcp_off = np.array(ik.get("tcp_offset_board_m", [0.0, 0.0, 0.0]))

        file_idx, rank_idx = self._parse_square(square)
        z_board = top_h if plane == "top" else 0.0
        p_board = np.array([file_idx * pitch, rank_idx * pitch, z_board]) + tcp_off
        p_robot = T[:3, :3] @ p_board + T[:3, 3]

        # Seed: prefer caller-supplied (last commanded pose, for trajectory
        # continuity). Otherwise use the nearest calibrated anchor's
        # recorded pose, converted to URDF angles.
        if seed is not None:
            seed_urdf = servo_to_urdf(seed, offsets, signs)
        else:
            nearest_servo = self._nearest_anchor_joints(square, plane)
            seed_urdf = servo_to_urdf(nearest_servo, offsets, signs)

        j_urdf = chesster_ik(p_robot, seed_joints_deg=seed_urdf)
        j_servo = urdf_to_servo(j_urdf, offsets, signs)
        return {k: round(float(v), 3) for k, v in j_servo.items()}

    def _nearest_anchor_joints(self, square: str, plane: str) -> dict:
        """Return the joint_angles_deg of the calibrated anchor closest
        to `square` in (file, rank) board coordinates."""
        target_f, target_r = self._parse_square(square)
        anchor_specs = {
            "a1": (0, 0), "h1": (7, 0), "a8": (0, 7), "h8": (7, 7),
            "mid_de1": (3.5, 0), "mid_de8": (3.5, 7),
            "mid_a45": (0, 3.5), "mid_h45": (7, 3.5),
            "center": (3.5, 3.5),
        }
        best, dmin = None, float("inf")
        for name, (fx, fy) in anchor_specs.items():
            src = self.corners.get(name) or self.grid_extras.get(name)
            if not src or plane not in src:
                continue
            d = (fx - target_f) ** 2 + (fy - target_r) ** 2
            if d < dmin:
                dmin, best = d, src[plane]["joint_angles_deg"]
        if best is None:
            raise ValueError(f"No anchors available for plane {plane!r}")
        return best

    def _joint_at_grid(self, i: int, j: int, plane: str, name: str,
                       home_fallback_deg: dict) -> float:
        """Look up joint `name` at grid cell (i, j) for the given plane,
        falling back to HOME if the cell is missing that joint."""
        kind, src_name, _, _ = self._GRID_LAYOUT[i][j]
        bucket = self.corners if kind == "corner" else self.grid_extras
        entry = bucket.get(src_name)
        if entry is None:
            raise KeyError(f"Grid cell ({i},{j}) source {kind}:{src_name} missing")
        plane_entry = entry.get(plane)
        if plane_entry is None:
            raise KeyError(f"Grid cell ({i},{j}) source {src_name} missing plane {plane!r}")
        joints = plane_entry["joint_angles_deg"]
        if name in joints:
            return float(joints[name])
        if name in home_fallback_deg:
            return home_fallback_deg[name]
        raise KeyError(
            f"Grid cell ({i},{j}) [{src_name} {plane}] is missing joint "
            f"'{name}' and no HOME fallback is available. Re-run "
            f"scripts/chesster_calibrate_ik.py to record full poses."
        )

    def _piecewise_bilinear_interp(self, u: float, v: float, plane: str,
                                   home_fallback_deg: dict) -> dict:
        """Piecewise bilinear over the 3x3 grid.

        Locate the 2x2 sub-cell that contains (u, v), then bilinearly
        interpolate using the 4 anchors of that sub-cell. Sub-cells are:

          left  (i=0..1)  |  right (i=1..2)
          lower (j=0..1)  |  upper (j=1..2)

        At sub-cell boundaries (u=0.5 or v=0.5) the interpolation is
        continuous (both adjacent sub-cells agree on the shared anchor),
        so the field is C^0 across the whole board.
        """
        # Pick sub-cell indices in {0, 1} along each axis.
        ci = 0 if u <= 0.5 else 1  # column block
        cj = 0 if v <= 0.5 else 1  # row block

        # Normalize (u, v) to [0, 1] within the sub-cell. Each sub-cell
        # spans 0.5 in (u, v), starting at ci*0.5 / cj*0.5.
        lu = (u - ci * 0.5) / 0.5
        lv = (v - cj * 0.5) / 0.5

        # 4 anchors of this sub-cell at grid indices (ci, cj), (ci+1, cj),
        # (ci, cj+1), (ci+1, cj+1).
        i0, i1 = ci, ci + 1
        j0, j1 = cj, cj + 1

        result: dict = {}
        for name in ARM_JOINT_NAMES:
            v00 = self._joint_at_grid(i0, j0, plane, name, home_fallback_deg)
            v10 = self._joint_at_grid(i1, j0, plane, name, home_fallback_deg)
            v01 = self._joint_at_grid(i0, j1, plane, name, home_fallback_deg)
            v11 = self._joint_at_grid(i1, j1, plane, name, home_fallback_deg)
            result[name] = round(
                (1 - lu) * (1 - lv) * v00
                + lu * (1 - lv) * v10
                + (1 - lu) * lv * v01
                + lu * lv * v11,
                3,
            )
        return result

    def _bilinear_interp(self, u: float, v: float, plane: str,
                         home_fallback_deg: dict) -> dict:
        """Bilinear fallback when grid_extras is missing or incomplete."""
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
