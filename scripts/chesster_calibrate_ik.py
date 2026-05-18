#!/usr/bin/env python3
"""
Chesster IK Calibration Script

Interactive CLI that walks the user through recording calibration data
for the Chesster arm (4 arm joints + gripper, longer links than SO-101):

  - 4 board corners x 2 planes (bottom/top) = 8 corner positions
  - 1 graveyard position (optional via --skip-graveyard)
  - Gripper OPEN reading
  - Gripper CLOSED reading
  - HOME pose (working rest)
  - INACTIVE pose (torque-safe rest -- Chesster's longer links make
    HOME-position torque release unsafe; INACTIVE must be a pose where
    disabling torque will not drop the arm over the board)

Reads joint positions directly from the Feetech servos over serial,
and computes forward kinematics via `controls.chesster_kinematics`
(placo-based, backed by data/urdf/chesster.urdf).

Usage:
    # Standard calibration
    python scripts/chesster_calibrate_ik.py

    # Skip graveyard recording
    python scripts/chesster_calibrate_ik.py --skip-graveyard

    # Custom serial port and output path
    python scripts/chesster_calibrate_ik.py --port /dev/ttyACM1 --output data/cal.json
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.chesster_arm import ChessterArm
from controls.chesster_kinematics import (
    ChessterKinematics, ARM_JOINT_NAMES, JOINT_NAMES_BY_INDEX,
    JOINT_TO_MOTOR_INDEX,
)


CORNERS = ["a1", "h1", "a8", "h8"]


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    print("[X] No /dev/ttyACM* devices found", file=sys.stderr)
    sys.exit(1)


_KIN: ChessterKinematics | None = None


def compute_fk(joint_angles_deg: dict) -> list:
    """Compute forward kinematics. Returns [x, y, z] in meters."""
    global _KIN
    if _KIN is None:
        _KIN = ChessterKinematics()
    return _KIN.forward_kinematics(joint_angles_deg).tolist()


def read_positions(arm: ChessterArm) -> dict:
    """Read current arm joint positions (4 arm joints) via direct serial.

    Returns dict of joint_name -> angle in degrees using the midpoint
    convention (2048 counts = 0 deg) matching ChessterKinematics/URDF.
    """
    COUNTS_PER_REV = 4096
    ENCODER_MIDPOINT = COUNTS_PER_REV // 2

    state = arm.get_state()
    positions = {}
    # Iterate the 4 arm joints by name and look up each one's motor index.
    # On Chesster the arm joints are at motor indices [0, 2, 3, 4] (motor 2
    # is the gripper), so a naive `for i, name in enumerate(ARM_JOINT_NAMES)`
    # would read the wrong motors.
    for name in ARM_JOINT_NAMES:
        idx = JOINT_TO_MOTOR_INDEX[name]
        raw_rad = float(state.joint_positions[idx])
        counts = int(raw_rad / (2.0 * np.pi / COUNTS_PER_REV))
        deg = (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)
        positions[name] = round(deg, 2)
    return positions


def record_position(arm: ChessterArm, label: str) -> dict:
    joints = read_positions(arm)
    xyz = compute_fk(joints)
    print(f"  Joints: {json.dumps({k: round(v, 2) for k, v in joints.items()})}")
    print(f"  XYZ (m): [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    return {"joint_angles_deg": joints, "xyz_meters": xyz}


def default_output_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "calibration" / "board_calibration.json")


def _capture_pose(arm: ChessterArm, n_motors: int) -> dict:
    """Read all motor positions as a {joint_name: radians} dict including
    gripper. Names follow JOINT_NAMES_BY_INDEX so each saved value matches
    the physically connected joint."""
    state = arm.get_state()
    return {
        JOINT_NAMES_BY_INDEX[i]: round(float(state.joint_positions[i]), 4)
        for i in range(n_motors)
    }


def main():
    parser = argparse.ArgumentParser(
        description="IK calibration for Chesster arm (4 arm joints + gripper)"
    )
    parser.add_argument("--port", type=str, default="auto",
                        help="Serial port (default: auto-detect /dev/ttyACM*)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output calibration JSON path")
    parser.add_argument("--skip-graveyard", action="store_true",
                        help="Skip graveyard position recording")
    parser.add_argument("--gripper-only", action="store_true",
                        help="Only recalibrate gripper open/closed + home + inactive")
    parser.add_argument("--top-only", action="store_true",
                        help="Only re-record the 4 corner TOP positions; "
                             "leaves bottoms, graveyard, gripper, home, inactive "
                             "untouched. Use after raising the transit height.")
    parser.add_argument("--home-only", action="store_true",
                        help="Only re-record the HOME pose; leaves corners, "
                             "graveyard, gripper, and INACTIVE untouched.")
    args = parser.parse_args()

    exclusive = sum([args.gripper_only, args.top_only, args.home_only])
    if exclusive > 1:
        print("[X] --gripper-only, --top-only, --home-only are mutually exclusive",
              file=sys.stderr)
        return 1

    output_path = args.output or default_output_path()
    port = resolve_port(args.port)

    print("=" * 60)
    print("Chesster IK Calibration")
    print("=" * 60)
    print(f"Serial: {port}")
    print(f"Output: {output_path}")
    print(f"Corners: {', '.join(CORNERS)}")
    print(f"Positions per corner: 2 (bottom + top)")
    grave_note = "" if args.skip_graveyard else " + 1 graveyard"
    print(f"Total positions: {len(CORNERS) * 2}{grave_note}")
    print()

    print(f"  Connecting directly to {port}...")
    arm = ChessterArm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        return 1
    print(f"[OK] Direct serial connection to {port}")
    print()

    if args.gripper_only or args.top_only or args.home_only:
        if not Path(output_path).exists():
            print(f"[X] No existing calibration at {output_path}", file=sys.stderr)
            arm.disconnect()
            return 1
        with open(output_path) as f:
            calibration = json.load(f)
        print(f"[OK] Loaded existing calibration (v{calibration.get('version')})")
        if args.top_only:
            for c in CORNERS:
                if c not in calibration.get("corners", {}):
                    print(f"[X] Existing calibration is missing corner {c}; "
                          "run a full calibration first.", file=sys.stderr)
                    arm.disconnect()
                    return 1
    else:
        calibration = {
            "version": 4,
            "arm": "chesster",
            "type": "two_plane_ik_explicit",
            "corners": {},
            "graveyard": None,
            "computed": None,
        }

    print("-" * 60)
    if args.gripper_only:
        print("GRIPPER-ONLY MODE: gripper, home, and inactive only.")
    elif args.home_only:
        print("HOME-ONLY MODE: re-recording HOME pose only.")
        print("  HOME must be a high-transit pose: arm extended VERTICALLY")
        print("  with both shoulder_lift near midpoint AND elbow_flex near")
        print("  midpoint (forearm continuing straight up, not folded back).")
        print("  A folded-back HOME makes the forearm too low for safe")
        print("  joint-space transit toward over-board poses.")
    elif args.top_only:
        print("TOP-ONLY MODE: re-recording the 4 corner TOP positions only.")
        print("  Aim for ~10-15 cm above the board surface. Use the shoulder")
        print("  LIFT joint to gain height -- wrist tilt alone gives ~3-6 cm.")
    else:
        print("INSTRUCTIONS:")
        print("  For each corner you will position the gripper twice:")
        print("  1. BOTTOM: gripper tip touching board surface at the corner")
        print("  2. TOP: gripper raised ~10-15 cm above the corner (use lift")
        print("          joint, not just wrist tilt -- wrist alone is too short)")
    print()
    print("  The arm is torque-free -- move it by hand.")
    print("  Press Enter after positioning to record each point.")
    print("-" * 60)
    print()

    def cleanup():
        arm.disconnect()

    try:
        if args.home_only:
            print(f"{'=' * 60}")
            print("HOME POSITION")
            print(f"{'=' * 60}")
            print("  HOME = high transit waypoint between INACTIVE and over-board")
            print("  motion. Extend the arm VERTICALLY: shoulder_lift near")
            print("  midpoint AND elbow_flex near midpoint (forearm straight up,")
            print("  not folded back). The forearm position at HOME determines")
            print("  whether joint-space sweeps to corners stay safely high.")
            input("  Move arm to HOME pose. Press Enter...")
            calibration["home_rad"] = _capture_pose(arm, arm.NUM_MOTORS)
            print(f"  [OK] Home (rad): {json.dumps(calibration['home_rad'])}")
            print()
            cleanup()
        elif args.top_only:
            for corner in CORNERS:
                print(f"{'=' * 60}")
                print(f"CORNER: {corner} (TOP only)")
                print(f"{'=' * 60}")
                bottom = calibration["corners"][corner]["bottom"]
                bot_joints = bottom["joint_angles_deg"]
                print(f"  (existing bottom joints: {json.dumps(bot_joints)})")

                input(f"  Raise gripper to {corner} TOP (~10-15 cm above board). "
                      "Press Enter...")
                print(f"  Recording {corner} top...")
                top = record_position(arm, f"{corner} top")

                lift_delta = abs(top["joint_angles_deg"]["shoulder_lift"]
                                 - bot_joints["shoulder_lift"])
                if lift_delta < 3.0:
                    print(f"  [WARN] shoulder_lift moved only {lift_delta:.1f}deg "
                          "between bottom and top. Real transit height needs "
                          "10+ deg of lift motion. Wrist tilt alone won't clear "
                          "tall pieces. Consider re-recording.")
                    redo = input(f"  Re-record {corner} TOP? [y/N] ").strip().lower()
                    if redo == "y":
                        input(f"  Raise gripper to {corner} TOP. Press Enter...")
                        top = record_position(arm, f"{corner} top")

                calibration["corners"][corner]["top"] = top
                print()
        elif not args.gripper_only:
            for corner in CORNERS:
                print(f"{'=' * 60}")
                print(f"CORNER: {corner}")
                print(f"{'=' * 60}")

                input(f"  Position gripper at {corner} BOTTOM. Press Enter...")
                print(f"  Recording {corner} bottom...")
                bottom = record_position(arm, f"{corner} bottom")

                input(f"  Raise gripper to {corner} TOP (~10-15 cm above board). "
                      "Press Enter...")
                print(f"  Recording {corner} top...")
                top = record_position(arm, f"{corner} top")

                # Validate using joint-space deltas (FK is unreliable until
                # motor zeros are calibrated to the URDF).
                lift_delta = abs(top["joint_angles_deg"]["shoulder_lift"]
                                 - bottom["joint_angles_deg"]["shoulder_lift"])
                if lift_delta < 3.0:
                    print(f"  [WARN] shoulder_lift moved only {lift_delta:.1f}deg "
                          "between bottom and top -- transit height likely "
                          "insufficient for tall pieces. Aim for 10+ deg of lift.")
                    redo = input(f"  Re-record {corner} TOP? [y/N] ").strip().lower()
                    if redo == "y":
                        input(f"  Raise gripper to {corner} TOP. Press Enter...")
                        top = record_position(arm, f"{corner} top")

                calibration["corners"][corner] = {"bottom": bottom, "top": top}
                print(f"  [OK] {corner}: lift_delta={lift_delta:.1f}deg")
                print()

            if not args.skip_graveyard:
                print(f"{'=' * 60}")
                print("GRAVEYARD")
                print(f"{'=' * 60}")
                input("  Position gripper at graveyard. Press Enter...")
                print("  Recording graveyard...")
                graveyard = record_position(arm, "graveyard")
                calibration["graveyard"] = graveyard
                print(f"  [OK] Graveyard: XYZ = {graveyard['xyz_meters']}")
                print()

        # Gripper / HOME / INACTIVE are skipped in --top-only and
        # --home-only modes (already handled or not relevant).
        if not args.top_only and not args.home_only:
            # Gripper open
            print(f"{'=' * 60}")
            print("GRIPPER OPEN")
            print(f"{'=' * 60}")
            input("  Open the gripper fully by hand. Press Enter...")
            state = arm.get_state()
            gripper_open_rad = round(float(state.joint_positions[arm.GRIPPER_INDEX]), 4)
            print(f"  [OK] Gripper open: {gripper_open_rad:.4f} rad")
            calibration["gripper_open_rad"] = gripper_open_rad
            print()

            # Gripper closed
            print(f"{'=' * 60}")
            print("GRIPPER CLOSED")
            print(f"{'=' * 60}")
            input("  Close the gripper on a chess piece. Press Enter...")
            state = arm.get_state()
            gripper_closed_rad = round(float(state.joint_positions[arm.GRIPPER_INDEX]), 4)
            print(f"  [OK] Gripper closed: {gripper_closed_rad:.4f} rad")
            calibration["gripper_closed_rad"] = gripper_closed_rad
            print()

            # HOME
            print(f"{'=' * 60}")
            print("HOME POSITION")
            print(f"{'=' * 60}")
            print("  HOME = high transit waypoint between INACTIVE and over-board")
            print("  motion. Extend the arm VERTICALLY: shoulder_lift near")
            print("  midpoint AND elbow_flex near midpoint (forearm straight up,")
            print("  not folded back).")
            input("  Move arm to HOME pose. Press Enter...")
            calibration["home_rad"] = _capture_pose(arm, arm.NUM_MOTORS)
            print(f"  [OK] Home (rad): {json.dumps(calibration['home_rad'])}")
            print()

            # INACTIVE -- the safety-critical addition for Chesster
            print(f"{'=' * 60}")
            print("INACTIVE POSITION (torque-safe rest)")
            print(f"{'=' * 60}")
            print("  INACTIVE = pose where disabling torque will NOT drop the arm")
            print("  over the board. Typically rest the arm folded against itself,")
            print("  on the chassis, or in a cradle. Execute scripts target this")
            print("  pose before releasing torque at shutdown.")
            input("  Move arm to INACTIVE pose. Press Enter...")
            calibration["inactive_rad"] = _capture_pose(arm, arm.NUM_MOTORS)
            print(f"  [OK] Inactive (rad): {json.dumps(calibration['inactive_rad'])}")
            print()

        cleanup()

        # Compute summary. Skipped in gripper-only and home-only modes
        # since the corner data didn't change; in top-only mode we
        # resummarize to show the new transit deltas.
        # Motion planning uses joint-space bilinear interpolation across
        # the 4 recorded corners; the FK-derived XYZ block below is
        # informational only (and currently unreliable until per-motor
        # encoder zeros are aligned with the URDF mechanical zero).
        if not args.gripper_only and not args.home_only:
            print(f"{'=' * 60}")
            print("CALIBRATION SUMMARY (joint-space interpolation)")
            print(f"{'=' * 60}")

            # Joint-space transit deltas per joint per corner (degrees).
            print("  Bottom->top joint deltas (deg) per corner:")
            for c in CORNERS:
                bot = calibration["corners"][c]["bottom"]["joint_angles_deg"]
                top = calibration["corners"][c]["top"]["joint_angles_deg"]
                deltas = {k: round(top[k] - bot[k], 2) for k in bot}
                print(f"    {c}: {json.dumps(deltas)}")

            # FK-derived geometry (informational). Will be inconsistent
            # until motor zeros are calibrated to URDF; this is expected
            # right now and is not a calibration failure.
            bottom_z = [calibration["corners"][c]["bottom"]["xyz_meters"][2]
                        for c in CORNERS]
            z_bottom = sum(bottom_z) / len(bottom_z)
            top_z = [calibration["corners"][c]["top"]["xyz_meters"][2]
                     for c in CORNERS]
            z_top = sum(top_z) / len(top_z)
            transit_height_mm = (z_top - z_bottom) * 1000

            def dist_2d(p1, p2):
                return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

            a1_xy = calibration["corners"]["a1"]["bottom"]["xyz_meters"][:2]
            a8_xy = calibration["corners"]["a8"]["bottom"]["xyz_meters"][:2]
            h1_xy = calibration["corners"]["h1"]["bottom"]["xyz_meters"][:2]
            h8_xy = calibration["corners"]["h8"]["bottom"]["xyz_meters"][:2]

            board_width = (dist_2d(a1_xy, h1_xy) + dist_2d(a8_xy, h8_xy)) / 2
            board_depth = (dist_2d(a1_xy, a8_xy) + dist_2d(h1_xy, h8_xy)) / 2

            calibration["computed"] = {
                "z_bottom_m": round(z_bottom, 4),
                "z_top_m": round(z_top, 4),
                "transit_height_mm": round(transit_height_mm, 1),
                "board_width_m": round(board_width, 4),
                "board_depth_m": round(board_depth, 4),
                "note": "FK-derived; informational only. Motion uses joint-space interpolation.",
            }

            print()
            print("  FK-derived geometry (informational, not used for motion):")
            print(f"    z_bottom={z_bottom*1000:.1f}mm  z_top={z_top*1000:.1f}mm")
            print(f"    board(FK)={board_width*1000:.1f}x{board_depth*1000:.1f}mm")

        print()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(calibration, f, indent=2)
            f.write("\n")
        tmp.rename(out)

        print(f"[OK] Calibration saved to {output_path}")
        print()
        print("=" * 60)
        print("[OK] Calibration complete!")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n\n[X] Cancelled by user")
        cleanup()
        return 1
    except Exception as e:
        print(f"\n[X] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        try:
            cleanup()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
