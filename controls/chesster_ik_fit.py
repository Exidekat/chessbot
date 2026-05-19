"""
Chesster IK calibration: fit encoder offsets and board pose, then provide
forward and inverse kinematics in URDF angle space.

Why this exists separately from `chesster_kinematics.py`:
- `chesster_kinematics.ChessterKinematics` wraps placo (a C++ kinematics
  library) for general URDF FK/IK. It's flexible but requires the cb
  conda environment and doesn't know about per-motor encoder zero
  offsets (the difference between servo midpoint and URDF mechanical
  zero).
- This module is pure numpy. It hardcodes Chesster's planar-arm
  geometry (4 joints: pan around Z, then lift/elbow/wrist around Y),
  uses link lengths from the URDF, and fits encoder offsets directly
  from the calibrated 9-anchor board grid. The fitted offsets are
  what makes FK/IK produce physically correct results.

The closed-form 4-DOF IK assumes a "gripper points straight down"
constraint (cumulative URDF tilt = pi). This matches how the user
calibrates each anchor (gripper tip touching board, gripper vertical),
and is the natural orientation for chess piece pickup.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# Chesster URDF link constants (meters). These must match
# `data/urdf/chesster.urdf` joint origins. If you edit the URDF, edit
# these too -- there is no shared source of truth.
Z_BASE_TO_PAN = 0.050     # base_link -> shoulder_pan joint
Z_PAN_TO_LIFT = 0.1524    # shoulder_pan -> shoulder_lift joint
L_UPPER = 0.2032          # shoulder_lift -> elbow_flex joint (upper arm)
L_FORE  = 0.2794          # elbow_flex -> wrist_flex joint (forearm)
L_GRIP  = 0.1778          # wrist_flex -> gripper TCP

# Convenience constants
Z_LIFT_JOINT = Z_BASE_TO_PAN + Z_PAN_TO_LIFT  # 0.2024 m

# Joint names we operate on. Wrist is last in the chain.
_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def chesster_fk(joints_deg: dict) -> np.ndarray:
    """Forward kinematics for the Chesster arm.

    Args:
        joints_deg: dict of URDF joint angles in degrees. Required keys:
            shoulder_pan, shoulder_lift, elbow_flex, wrist_flex.

    Returns:
        TCP position (x, y, z) in robot base frame, meters.
    """
    th_pan   = np.radians(joints_deg["shoulder_pan"])
    th_lift  = np.radians(joints_deg["shoulder_lift"])
    th_elbow = np.radians(joints_deg["elbow_flex"])
    th_wrist = np.radians(joints_deg["wrist_flex"])

    # Planar chain in pan-rotated frame. r is the radial coordinate in
    # the direction of pan rotation (pan-rotated +X). z is vertical.
    r, z = 0.0, Z_LIFT_JOINT
    cumul = th_lift
    r += L_UPPER * np.sin(cumul)
    z += L_UPPER * np.cos(cumul)
    cumul += th_elbow
    r += L_FORE * np.sin(cumul)
    z += L_FORE * np.cos(cumul)
    cumul += th_wrist
    r += L_GRIP * np.sin(cumul)
    z += L_GRIP * np.cos(cumul)

    x = r * np.cos(th_pan)
    y = r * np.sin(th_pan)
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Inverse kinematics (closed-form, gripper-down constraint)
# ---------------------------------------------------------------------------

def _reachable_orient_rad(r_signed: float, z: float,
                          preferred_rad: float = np.pi) -> float:
    """Solve for the gripper cumulative-tilt closest to `preferred_rad`
    such that the wrist joint position is within the arm's reach envelope.
    Returns the chosen tilt in radians.

    NB: `r_signed` is the SIGNED radial coordinate in the pan-rotated
    plane (positive for pan_forward branch, negative for pan_backward
    branch). The valid tilt range depends on the sign of r, so this
    function must be called per pan branch.

    For Chesster's geometry, peripheral targets (e.g., near board corners)
    aren't reachable with the gripper pointing straight down -- the arm
    is too short. We must tilt the gripper back. This function computes
    the smallest deviation from `preferred_rad` that achieves reachability,
    mirroring how the user calibrates those positions (gripper tilted
    back, tip still touching the board).
    """
    a = r_signed
    b = z - Z_LIFT_JOINT
    g = L_GRIP
    R = L_UPPER + L_FORE
    # Constraint: (a - g sin phi)^2 + (b - g cos phi)^2 <= R^2
    # Expand: (a^2 + b^2 + g^2 - R^2) <= 2 g (a sin phi + b cos phi)
    K = a * a + b * b + g * g - R * R
    if K <= 0:
        return preferred_rad
    rho = np.sqrt(a * a + b * b)
    if rho < 1e-9:
        return preferred_rad
    rhs = K / (2.0 * g * rho)
    if rhs > 1.0 + 1e-9:
        raise ValueError(
            f"Target r={r_signed:.3f}m z={z:.3f}m is outside workspace "
            f"for any gripper orientation."
        )
    rhs = float(np.clip(rhs, -1.0, 1.0))
    psi = np.arctan2(b, a)
    # Valid range: phi + psi in [arcsin(rhs), pi - arcsin(rhs)] (mod 2pi).
    base = np.arcsin(rhs)
    lo = base - psi
    hi = (np.pi - base) - psi

    # Pick the value in the (closed) interval [lo, hi] closest to
    # `preferred_rad`, handling 2pi periodicity. Translate the interval
    # to be near `preferred_rad`.
    def wrap(angle, ref):
        return angle - 2 * np.pi * np.round((angle - ref) / (2 * np.pi))

    lo_w = wrap(lo, preferred_rad)
    hi_w = wrap(hi, preferred_rad)
    if lo_w > hi_w:
        lo_w, hi_w = hi_w, lo_w
    return float(np.clip(preferred_rad, lo_w, hi_w))


def _planar_ik(r_signed: float, z: float, phi: float
               ) -> tuple:
    """Closed-form planar IK for given signed radial r and tilt phi.
    Returns (lift_up, lift_down, elbow_up, elbow_down) in radians, or
    raises ValueError if unreachable in this plane."""
    wrist_r = r_signed - L_GRIP * np.sin(phi)
    wrist_z = z - L_GRIP * np.cos(phi)
    dr = wrist_r - 0.0
    dz = wrist_z - Z_LIFT_JOINT
    D = np.sqrt(dr * dr + dz * dz)
    if D > L_UPPER + L_FORE + 1e-6:
        raise ValueError(f"D={D*1000:.1f}mm > max")
    if D < abs(L_UPPER - L_FORE) - 1e-6:
        raise ValueError(f"D={D*1000:.1f}mm < min")
    cos_elbow = (D * D - L_UPPER * L_UPPER - L_FORE * L_FORE) \
                / (2.0 * L_UPPER * L_FORE)
    cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))
    elbow_mag = float(np.arccos(cos_elbow))
    alpha = np.arctan2(dr, dz)
    solutions = []
    for elbow_sign in (+1.0, -1.0):
        th_elbow = elbow_sign * elbow_mag
        phi_off = np.arctan2(
            L_FORE * np.sin(th_elbow),
            L_UPPER + L_FORE * np.cos(th_elbow),
        )
        th_lift = alpha - phi_off
        solutions.append((th_lift, th_elbow))
    return solutions


def _angle_dist(a: float, b: float) -> float:
    """Smallest absolute distance between two angles in radians,
    accounting for 2pi wraparound."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


def chesster_ik(
    target_xyz: np.ndarray,
    seed_joints_deg: Optional[dict] = None,
    gripper_orient_rad: float = np.pi,
    adaptive_orient: bool = True,
) -> dict:
    """Inverse kinematics for the Chesster arm.

    The 4-DOF arm has 4 valid solution branches per target:
      - pan_forward (arm body in front, reaching forward, r > 0), or
        pan_backward (arm body behind, reaching backward, r < 0)
      - elbow_up or elbow_down

    We enumerate all 4 and pick the one closest to `seed_joints_deg`
    in URDF-angle space. Without a seed we fall back to (pan_forward,
    elbow_down), but for Chesster's calibrated setup the user reaches
    over the board with pan_backward, so a seed from the nearest
    anchor is strongly recommended.

    The gripper-orientation constraint (`gripper_orient_rad`, default
    pi = straight down) is honored when the target is fully inside the
    workspace. Near the edge, `adaptive_orient` tilts the gripper just
    enough to keep the target reachable.

    Returns URDF joint angles (degrees) for `shoulder_pan`,
    `shoulder_lift`, `elbow_flex`, `wrist_flex`. Raises ValueError if
    no branch is reachable.
    """
    x, y, z = (float(v) for v in target_xyz)
    r_unsigned = np.sqrt(x * x + y * y)

    # If a seed is provided, prefer its cumulative gripper tilt over the
    # default pi (straight down). The seed has a tilt that the user
    # actually calibrated to at a nearby anchor, so following it produces
    # IK solutions that look similar to recorded anchor poses instead of
    # picking a different (but kinematically valid) branch.
    if seed_joints_deg is not None:
        seed_cumul = np.radians(
            seed_joints_deg.get("shoulder_lift", 0.0)
            + seed_joints_deg.get("elbow_flex", 0.0)
            + seed_joints_deg.get("wrist_flex", 0.0)
        )
        preferred_phi = float(seed_cumul)
    else:
        preferred_phi = float(gripper_orient_rad)

    # Try both pan branches: forward (r > 0) and backward (r < 0,
    # equivalent to pan rotated by pi). The valid gripper-tilt range
    # depends on the sign of r, so compute phi per branch.
    pan_forward = np.arctan2(y, x)
    pan_backward = pan_forward + np.pi
    pan_candidates = [(pan_forward, +r_unsigned), (pan_backward, -r_unsigned)]

    branches = []
    last_err = None
    for th_pan, r_signed in pan_candidates:
        try:
            if adaptive_orient:
                phi = _reachable_orient_rad(
                    r_signed, z, preferred_rad=preferred_phi,
                )
            else:
                phi = preferred_phi
            sols = _planar_ik(r_signed, z, phi)
        except ValueError as e:
            last_err = e
            continue
        for th_lift, th_elbow in sols:
            th_wrist = phi - th_lift - th_elbow
            branches.append((th_pan, th_lift, th_elbow, th_wrist))

    if not branches:
        raise ValueError(
            f"Target {target_xyz} unreachable on any branch: {last_err}"
        )

    def _wrap_to(angle, ref):
        """Add k*2pi to `angle` to bring it nearest to `ref`."""
        return angle - 2 * np.pi * np.round((angle - ref) / (2 * np.pi))

    if seed_joints_deg is not None:
        seed_pan   = float(np.radians(seed_joints_deg.get("shoulder_pan", 0.0)))
        seed_lift  = float(np.radians(seed_joints_deg.get("shoulder_lift", 0.0)))
        seed_elbow = float(np.radians(seed_joints_deg.get("elbow_flex", 0.0)))
        seed_wrist = float(np.radians(seed_joints_deg.get("wrist_flex", 0.0)))

        def cost(branch):
            p, l, e, w = branch
            return (
                _angle_dist(p, seed_pan) ** 2
                + _angle_dist(l, seed_lift) ** 2
                + _angle_dist(e, seed_elbow) ** 2
                + _angle_dist(w, seed_wrist) ** 2
            )

        th_pan, th_lift, th_elbow, th_wrist = min(branches, key=cost)
        # Wrap each output to the nearest representative of the seed angle.
        # FK is 2pi-periodic in each joint, so the wrapped value is exactly
        # equivalent kinematically but stays within the motor's reachable
        # encoder range.
        th_pan   = _wrap_to(th_pan,   seed_pan)
        th_lift  = _wrap_to(th_lift,  seed_lift)
        th_elbow = _wrap_to(th_elbow, seed_elbow)
        th_wrist = _wrap_to(th_wrist, seed_wrist)
    else:
        # No seed: prefer pan_forward + elbow_down (first branch tried).
        th_pan, th_lift, th_elbow, th_wrist = branches[0]

    return {
        "shoulder_pan":  float(np.degrees(th_pan)),
        "shoulder_lift": float(np.degrees(th_lift)),
        "elbow_flex":    float(np.degrees(th_elbow)),
        "wrist_flex":    float(np.degrees(th_wrist)),
    }


# ---------------------------------------------------------------------------
# Offset + board pose fitting
# ---------------------------------------------------------------------------

# Board-frame XY (in square-pitch units) of each anchor.
_ANCHOR_BOARD_XY = {
    "a1":       (0.0, 0.0),
    "h1":       (7.0, 0.0),
    "a8":       (0.0, 7.0),
    "h8":       (7.0, 7.0),
    "mid_de1":  (3.5, 0.0),
    "mid_de8":  (3.5, 7.0),
    "mid_a45":  (0.0, 3.5),
    "mid_h45":  (7.0, 3.5),
    "center":   (3.5, 3.5),
}


def _euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix for intrinsic XYZ Euler angles."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def collect_anchors(cal_data: dict, square_pitch_m: float) -> list:
    """Pull (name, joints_deg, p_board) for every bottom-plane anchor present
    in the calibration JSON."""
    out = []
    corners = cal_data.get("corners", {}) or {}
    extras = cal_data.get("grid_extras", {}) or {}
    for name, (fx, fy) in _ANCHOR_BOARD_XY.items():
        src = corners.get(name) or extras.get(name)
        if src and "bottom" in src:
            joints = src["bottom"]["joint_angles_deg"]
            p_board = np.array([fx * square_pitch_m, fy * square_pitch_m, 0.0])
            out.append((name, joints, p_board))
    return out


def _normalize_offset_deg(v: float) -> float:
    """Wrap an offset into (-180, 180] degrees (FK is 360-periodic)."""
    v = ((v + 180.0) % 360.0) - 180.0
    return float(v)


def _fit_with_signs(anchors: list, signs: dict,
                    n_restarts: int = 6, rng: Optional[np.random.Generator] = None
                    ) -> tuple:
    """Run least-squares fit assuming the given per-joint servo->URDF sign
    mapping. Tries multiple random restarts to escape local minima, since
    the offset-space is multi-modal (FK is 360-periodic per joint).

    Returns (rms_mm, offsets_dict (normalized to (-180, 180]), T_4x4, result).
    """
    from scipy.optimize import least_squares

    if rng is None:
        rng = np.random.default_rng(0)

    def residuals(params):
        offs = {n: params[i] for i, n in enumerate(_JOINT_NAMES)}
        tx, ty, tz = params[4:7]
        R = _euler_xyz_matrix(params[7], params[8], params[9])
        t = np.array([tx, ty, tz])
        res = []
        for _name, joints, p_board in anchors:
            corrected = {n: signs[n] * joints[n] + offs[n] for n in offs}
            p_fk = chesster_fk(corrected)
            p_expected = R @ p_board + t
            res.extend(p_fk - p_expected)
        return np.asarray(res)

    best = None
    for trial in range(n_restarts):
        x0 = np.zeros(10)
        # Reasonable initial board pose: in front of the robot, slightly
        # elevated, no rotation. Random perturbations on subsequent
        # restarts escape local minima.
        x0[5] = -0.30
        x0[6] = 0.05
        if trial > 0:
            # Random offsets in [-180, 180], random pose perturbations.
            x0[0:4] = rng.uniform(-180, 180, size=4)
            x0[4:7] += rng.uniform(-0.05, 0.05, size=3)
            x0[7:10] = rng.uniform(-np.pi / 4, np.pi / 4, size=3)

        result = least_squares(residuals, x0, method="lm", max_nfev=20000)
        rms = float(np.sqrt(np.mean(result.fun ** 2)) * 1000.0)
        # Normalize offsets to (-180, 180].
        normalized_offsets = {
            n: _normalize_offset_deg(float(result.x[i]))
            for i, n in enumerate(_JOINT_NAMES)
        }
        # Score with mild canonical-form preference among (near-)tied RMSes.
        score = rms + 1e-4 * sum(abs(v) for v in normalized_offsets.values())
        if best is None or score < best[0]:
            T = np.eye(4)
            T[:3, :3] = _euler_xyz_matrix(result.x[7], result.x[8], result.x[9])
            T[:3, 3] = result.x[4:7]
            best = (score, rms, normalized_offsets, T, result)

    _, rms, offsets, T, result = best
    return rms, offsets, T, result


def fit_ik_calibration(
    cal_data: dict,
    square_pitch_m: float = 0.04445,
    joint_signs: Optional[dict] = None,
) -> dict:
    """Fit encoder offsets and board pose from the bottom-plane anchors.

    Per-joint servo-to-URDF sign mappings are auto-discovered (16 LM fits,
    one per +-1 combination), since the Feetech servos' encoder direction
    depends on how each motor is mounted and there's no a priori guarantee
    it matches the URDF axis direction. The combination producing the
    lowest RMS residual wins.

    Args:
        cal_data: parsed `board_calibration.json` dict.
        square_pitch_m: distance between adjacent square centers
            (default 0.04445 m = 1.75 in, for a 14-inch board).
        joint_signs: optional sign override per joint (+1 or -1). If
            provided, skips the sweep and uses just this combination.

    Returns:
        dict with `encoder_offsets_deg`, `joint_signs`,
        `board_pose_in_robot` (4x4 list of lists), `square_pitch_m`,
        per-anchor residuals, rms/max residuals, and convergence stats.
    """
    anchors = collect_anchors(cal_data, square_pitch_m)
    if len(anchors) < 6:
        raise ValueError(
            f"Need >=6 anchors; got {len(anchors)}. Re-run calibration first."
        )

    # Enumerate sign combinations (or use the supplied one).
    if joint_signs is not None:
        sign_combos = [joint_signs]
    else:
        sign_combos = []
        for s in range(16):
            sign_combos.append({
                _JOINT_NAMES[i]: (1 if (s >> i) & 1 == 0 else -1)
                for i in range(4)
            })

    best = None
    for signs in sign_combos:
        rms, offsets, T, result = _fit_with_signs(anchors, signs)
        # Tie-breaker for equivalent-RMS solutions: prefer smaller sum of
        # absolute offsets (canonical representation). Treat fits within
        # 0.5 mm RMS of each other as a tie.
        score = rms + 1e-4 * sum(abs(v) for v in offsets.values())
        if best is None or score < best[0]:
            best = (score, rms, offsets, T, result, signs)
    _, rms, offsets_deg, T, result, signs = best

    # Per-anchor residuals at the best fit.
    per_anchor = []
    R = T[:3, :3]
    t = T[:3, 3]
    for name, joints, p_board in anchors:
        corrected = {n: signs[n] * joints[n] + offsets_deg[n] for n in offsets_deg}
        p_fk = chesster_fk(corrected)
        p_expected = R @ p_board + t
        err = p_fk - p_expected
        per_anchor.append({
            "name": name,
            "residual_mm": float(np.linalg.norm(err) * 1000.0),
            "residual_xyz_mm": [float(e * 1000.0) for e in err],
        })

    return {
        "encoder_offsets_deg": offsets_deg,
        "joint_signs": signs,
        "board_pose_in_robot": T.tolist(),
        "square_pitch_m": float(square_pitch_m),
        "anchors": per_anchor,
        "rms_residual_mm": rms,
        "max_residual_mm": float(max(a["residual_mm"] for a in per_anchor)),
        "n_anchors": len(anchors),
        "success": bool(result.success),
        "cost": float(result.cost),
    }


def compute_top_height(cal_data: dict, encoder_offsets_deg: dict,
                       joint_signs: dict) -> float:
    """Average vertical distance from each anchor's bottom FK to its top FK
    (with offsets and signs applied). Used as the uniform 'transit plane'
    height for over-board approach moves."""
    deltas = []
    for src_dict in [cal_data.get("corners", {}) or {},
                     cal_data.get("grid_extras", {}) or {}]:
        for _name, entry in src_dict.items():
            if not entry or "bottom" not in entry or "top" not in entry:
                continue
            b = {n: joint_signs[n] * entry["bottom"]["joint_angles_deg"][n]
                    + encoder_offsets_deg[n]
                 for n in encoder_offsets_deg}
            t = {n: joint_signs[n] * entry["top"]["joint_angles_deg"][n]
                    + encoder_offsets_deg[n]
                 for n in encoder_offsets_deg}
            deltas.append(chesster_fk(t)[2] - chesster_fk(b)[2])
    if not deltas:
        return 0.10
    return float(np.mean(deltas))


# ---------------------------------------------------------------------------
# Convenience: end-to-end servo -> URDF -> FK pipeline
# ---------------------------------------------------------------------------

def servo_to_urdf(joints_servo_deg: dict, encoder_offsets_deg: dict,
                  joint_signs: dict) -> dict:
    """Convert servo-frame joint angles (midpoint convention degrees) to
    URDF-frame angles using fitted signs and offsets."""
    return {
        n: joint_signs[n] * joints_servo_deg[n] + encoder_offsets_deg[n]
        for n in encoder_offsets_deg
    }


def urdf_to_servo(joints_urdf_deg: dict, encoder_offsets_deg: dict,
                  joint_signs: dict) -> dict:
    """Inverse of servo_to_urdf: convert URDF angles back to servo angles
    suitable for sending to motors."""
    return {
        n: (joints_urdf_deg[n] - encoder_offsets_deg[n]) / joint_signs[n]
        for n in encoder_offsets_deg
    }
