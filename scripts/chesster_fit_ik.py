#!/usr/bin/env python3
"""
Chesster IK Calibration Fit

Reads `data/calibration/board_calibration.json` (which must already
contain 4 corners + 5 grid_extras at the bottom plane, recorded via
`chesster_calibrate_ik.py` and `chesster_calibrate_ik.py --grid-extras`),
solves for the per-motor encoder offsets, joint-sign conventions, and
board pose in the robot base frame, then writes the result back into
the same JSON under a top-level `ik` block.

After this script runs, `chesster_execute_ik_move_cv_assisted.py` will
use closed-form IK by default. Pass `--interpolation` to fall back to
the previous joint-space interp.

Usage:
    python scripts/chesster_fit_ik.py
    python scripts/chesster_fit_ik.py --board-size-inches 14
    python scripts/chesster_fit_ik.py --output data/calibration/board_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from controls.chesster_ik_fit import (
    fit_ik_calibration, compute_top_height, chesster_fk, chesster_ik,
    urdf_to_servo, servo_to_urdf,
)


def default_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "calibration" / "board_calibration.json")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Chesster IK encoder offsets and board pose from "
                    "the 9-anchor board calibration."
    )
    parser.add_argument("--cal-path", type=str, default=None,
                        help="Path to board_calibration.json "
                             "(default: data/calibration/board_calibration.json).")
    parser.add_argument("--board-size-inches", type=float, default=14.0,
                        help="Playing-area edge length in inches "
                             "(8 squares wide). Default: 14.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write the fitted ik block back to a separate "
                             "JSON path (default: in place).")
    args = parser.parse_args()

    cal_path = args.cal_path or default_path()
    output_path = args.output or cal_path

    if not Path(cal_path).exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return 1

    with open(cal_path) as f:
        cal = json.load(f)

    # Square pitch: total board edge / 8 squares.
    pitch_m = (args.board_size_inches * 0.0254) / 8.0

    print("=" * 60)
    print("Chesster IK Fit")
    print("=" * 60)
    print(f"Input:  {cal_path}")
    print(f"Output: {output_path}")
    print(f"Board edge: {args.board_size_inches} in -> "
          f"square pitch = {pitch_m*1000:.2f} mm")
    print()

    try:
        result = fit_ik_calibration(cal, square_pitch_m=pitch_m)
    except Exception as e:
        print(f"[X] Fit failed: {e}", file=sys.stderr)
        return 1

    print(f"Solve converged: {result['success']}, cost={result['cost']:.6e}")
    print(f"RMS residual: {result['rms_residual_mm']:.2f} mm")
    print(f"Max residual: {result['max_residual_mm']:.2f} mm")
    print()
    print("Per-anchor residuals (mm):")
    print(f"  {'anchor':<12s} {'err':>6s}  {'x':>6s} {'y':>6s} {'z':>6s}")
    for a in result["anchors"]:
        x, y, z = a["residual_xyz_mm"]
        print(f"  {a['name']:<12s} {a['residual_mm']:>6.2f}  "
              f"{x:>6.2f} {y:>6.2f} {z:>6.2f}")
    print()
    print(f"Joint signs (servo -> URDF direction):")
    for n, s in result["joint_signs"].items():
        print(f"  {n:>15s}: {s:+d}")
    print()
    print(f"Encoder offsets (deg, applied AFTER sign):")
    for n, v in result["encoder_offsets_deg"].items():
        print(f"  {n:>15s}: {v:+7.2f}")
    print()

    T = np.array(result["board_pose_in_robot"])
    print(f"Board origin in robot frame (m): "
          f"[{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}]")
    print(f"Board normal (world-frame z-col of R): "
          f"[{T[0,2]:+.4f}, {T[1,2]:+.4f}, {T[2,2]:+.4f}]")
    if abs(T[2, 2]) < 0.95:
        print("  [WARN] Board normal is >18 deg from vertical -- check that "
              "the board is level and corners were recorded at the same "
              "physical height.")
    print()

    top_h = compute_top_height(cal, result["encoder_offsets_deg"],
                                result["joint_signs"])
    print(f"Transit (top) plane height above bottom: {top_h*1000:.1f} mm")
    print()

    # Sanity: round-trip IK on each anchor with seed = recorded angles.
    # The predicted servo should be close to recorded if the fit is good.
    print("IK round-trip vs recorded (max joint delta, degrees):")
    sq_map = {"a1":(0,0),"h1":(7,0),"a8":(0,7),"h8":(7,7),
              "mid_de1":(3.5,0),"mid_de8":(3.5,7),
              "mid_a45":(0,3.5),"mid_h45":(7,3.5),"center":(3.5,3.5)}
    JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
    max_overall = 0.0
    for name, (fx, fy) in sq_map.items():
        src = (cal.get("corners", {}).get(name)
               or cal.get("grid_extras", {}).get(name))
        if not src or "bottom" not in src:
            continue
        rec_servo = src["bottom"]["joint_angles_deg"]
        seed_urdf = servo_to_urdf(rec_servo, result["encoder_offsets_deg"],
                                  result["joint_signs"])
        p_board = np.array([fx * pitch_m, fy * pitch_m, 0.0])
        p_robot = T[:3, :3] @ p_board + T[:3, 3]
        try:
            j_urdf = chesster_ik(p_robot, seed_joints_deg=seed_urdf)
        except Exception as e:
            print(f"  {name}: IK failed: {e}")
            continue
        j_servo = urdf_to_servo(j_urdf, result["encoder_offsets_deg"],
                                result["joint_signs"])
        max_d = max(abs(j_servo[k] - rec_servo[k]) for k in JOINTS)
        max_overall = max(max_overall, max_d)
        print(f"  {name:<10s} max_delta={max_d:5.2f}")
    print(f"  worst-case joint delta: {max_overall:.2f} deg")
    print()

    # Write back. Keep the rest of the file intact.
    # Preserve any existing tcp_offset_board_m so re-running the fit
    # doesn't clobber a manually-tuned bias correction.
    prev_tcp = (cal.get("ik") or {}).get("tcp_offset_board_m") or [0.0, 0.0, 0.0]
    cal["ik"] = {
        "version": 1,
        "square_pitch_m": result["square_pitch_m"],
        "joint_signs": result["joint_signs"],
        "encoder_offsets_deg": result["encoder_offsets_deg"],
        "board_pose_in_robot": result["board_pose_in_robot"],
        "top_plane_height_m": top_h,
        "tcp_offset_board_m": prev_tcp,
        "fit": {
            "rms_residual_mm": result["rms_residual_mm"],
            "max_residual_mm": result["max_residual_mm"],
            "n_anchors": result["n_anchors"],
            "cost": result["cost"],
        },
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")
    tmp.rename(out)
    print(f"[OK] Wrote ik block to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
