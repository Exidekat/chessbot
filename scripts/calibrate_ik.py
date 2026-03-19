#!/usr/bin/env python3
"""
Two-Plane IK Calibration Script

Interactive CLI that walks the user through recording 8 positions for
the SO-101 chess arm: each of the 4 board corners at both the bottom
(grasp) plane and the top (transit) plane.

Reads joint positions directly from the Feetech servos over serial
(same approach as create_so100_config.py). If the bridge holds the
serial port, the script disconnects it first via POST /robot/disconnect
and reconnects when done.

Uses the bridge only for FK computation and calibration reload.

Usage:
    # Standard calibration (8 corner positions + graveyard)
    python scripts/calibrate_ik.py

    # Skip graveyard recording
    python scripts/calibrate_ik.py --skip-graveyard

    # Custom bridge URL, serial port, and output path
    python scripts/calibrate_ik.py --bridge http://localhost:8420 --port /dev/ttyACM1 --output data/cal.json
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for chessbot module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.so100_arm import SO100Arm

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install: pip install requests",
          file=sys.stderr)
    sys.exit(1)


DEFAULT_BRIDGE = "http://localhost:8420"
CORNERS = ["a1", "h1", "a8", "h8"]

# Arm joint names matching kinematics.py (excludes gripper)
ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]


def resolve_port(port: str) -> str:
    """Resolve serial port, supporting 'auto' to scan /dev/ttyACM*."""
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    print("[X] No /dev/ttyACM* devices found", file=sys.stderr)
    sys.exit(1)


def check_health(bridge: str) -> bool:
    try:
        r = requests.get(f"{bridge}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def disconnect_bridge_robot(bridge: str) -> bool:
    """Tell the bridge to release the serial port."""
    try:
        r = requests.post(f"{bridge}/robot/disconnect", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  [OK] {data.get('message', 'Bridge robot disconnected')}")
            return True
        print(f"  [WARN] /robot/disconnect returned {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"  [WARN] Could not disconnect bridge robot: {e}")
        return False


def reconnect_bridge_robot(bridge: str) -> bool:
    """Tell the bridge to reconnect to the serial port."""
    try:
        r = requests.post(f"{bridge}/robot/reconnect", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("success"):
                print(f"  [OK] {data.get('message', 'Bridge robot reconnected')}")
                return True
            print(f"  [WARN] Reconnect failed: {data.get('message')}")
            return False
        print(f"  [WARN] /robot/reconnect returned {r.status_code}")
        return False
    except Exception as e:
        print(f"  [WARN] Could not reconnect bridge robot: {e}")
        return False


def compute_fk(bridge: str, joint_angles_deg: dict) -> list:
    """Compute forward kinematics via bridge. Returns [x, y, z] in meters."""
    r = requests.post(
        f"{bridge}/kinematics/fk",
        json={"joint_angles_deg": joint_angles_deg},
        timeout=5,
    )
    return r.json()["xyz_meters"]


def read_positions(arm: SO100Arm) -> dict:
    """Read current arm joint positions via direct serial.

    Returns dict of joint_name -> angle in degrees using the midpoint
    convention (2048 counts = 0 deg) matching SO101Kinematics/URDF.
    SO100Arm returns raw radians (0 counts = 0 rad), so we convert
    through encoder counts to apply the midpoint offset.
    """
    COUNTS_PER_REV = 4096
    ENCODER_MIDPOINT = COUNTS_PER_REV // 2

    state = arm.get_state()
    positions = {}
    for i, name in enumerate(ARM_JOINT_NAMES):
        raw_rad = float(state.joint_positions[i])
        counts = int(raw_rad / (2.0 * np.pi / COUNTS_PER_REV))
        deg = (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)
        positions[name] = round(deg, 2)
    return positions


def record_position(arm: SO100Arm, bridge: str, label: str) -> dict:
    """Read current arm position (direct serial) and compute FK (bridge).

    Returns dict with joint_angles_deg and xyz_meters.
    """
    joints = read_positions(arm)
    xyz = compute_fk(bridge, joints)
    print(f"  Joints: {json.dumps({k: round(v, 2) for k, v in joints.items()})}")
    print(f"  XYZ (m): [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
    return {"joint_angles_deg": joints, "xyz_meters": xyz}


def default_output_path() -> str:
    """Determine default calibration output path."""
    agent_name = os.environ.get("AGENT_NAME", "")
    claw_dir = Path(__file__).resolve().parent.parent.parent.parent / "claw"
    if agent_name:
        ws = claw_dir / "agents" / agent_name / "workspace"
    else:
        chessbot_ws = claw_dir / "agents" / "chessbot" / "workspace"
        if chessbot_ws.exists():
            ws = chessbot_ws
        else:
            return str(Path("data") / "calibration" / "board_calibration.json")
    return str(ws / "calibration" / "board_calibration.json")


def main():
    parser = argparse.ArgumentParser(
        description="Two-plane IK calibration for SO-101 chess arm"
    )
    parser.add_argument(
        "--bridge", type=str, default=DEFAULT_BRIDGE,
        help=f"Bridge URL (default: {DEFAULT_BRIDGE})"
    )
    parser.add_argument(
        "--port", type=str, default="auto",
        help="Serial port (default: auto-detect /dev/ttyACM*)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output calibration JSON path (auto-detected if not specified)"
    )
    parser.add_argument(
        "--skip-graveyard", action="store_true",
        help="Skip graveyard position recording"
    )
    parser.add_argument(
        "--no-reload", action="store_true",
        help="Skip auto-reloading bridge calibration after saving"
    )
    parser.add_argument(
        "--gripper-only", action="store_true",
        help="Only recalibrate gripper open/closed and home position (skip corners/graveyard)"
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path()
    port = resolve_port(args.port)

    print("=" * 60)
    print("Two-Plane IK Calibration")
    print("=" * 60)
    print(f"Bridge: {args.bridge}")
    print(f"Serial: {port}")
    print(f"Output: {output_path}")
    print(f"Corners: {', '.join(CORNERS)}")
    print(f"Positions per corner: 2 (bottom + top)")
    print(f"Total positions: {len(CORNERS) * 2}{'' if args.skip_graveyard else ' + 1 graveyard'}")
    print()

    # Pre-flight: bridge health (needed for FK)
    if not check_health(args.bridge):
        print(f"[X] Bridge not healthy at {args.bridge}", file=sys.stderr)
        return 1
    print("[OK] Bridge is healthy")

    # Disconnect bridge robot so we can open serial directly
    print("  Disconnecting bridge robot to free serial port...")
    disconnect_bridge_robot(args.bridge)
    time.sleep(0.2)  # Let serial port be released

    # Open direct serial connection
    print(f"  Connecting directly to {port}...")
    arm = SO100Arm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        print("  Reconnecting bridge robot...")
        reconnect_bridge_robot(args.bridge)
        return 1
    print(f"[OK] Direct serial connection to {port}")
    print()

    # Load existing calibration or start fresh
    if args.gripper_only:
        # Load existing file and merge gripper/home into it
        if not Path(output_path).exists():
            print(f"[X] No existing calibration at {output_path}", file=sys.stderr)
            print("    Run full calibration first (without --gripper-only)")
            arm.disconnect()
            reconnect_bridge_robot(args.bridge)
            return 1
        with open(output_path) as f:
            calibration = json.load(f)
        print(f"[OK] Loaded existing calibration (v{calibration.get('version')})")
    else:
        calibration = {
            "version": 3,
            "type": "two_plane_ik_explicit",
            "corners": {},
            "graveyard": None,
            "computed": None,
        }

    print("-" * 60)
    if args.gripper_only:
        print("GRIPPER-ONLY MODE: Recording gripper open, closed, and home.")
    else:
        print("INSTRUCTIONS:")
        print("  For each corner you will position the gripper twice:")
        print("  1. BOTTOM: gripper tip touching the board surface at the corner")
        print("  2. TOP: gripper raised to safe transit height above that corner")
    print()
    print("  The arm is torque-free -- move it by hand.")
    print("  Press Enter after positioning to record each point.")
    print("-" * 60)
    print()

    def cleanup():
        """Release serial and reconnect bridge."""
        arm.disconnect()
        time.sleep(0.2)
        print("  Reconnecting bridge robot...")
        reconnect_bridge_robot(args.bridge)

    try:
        if not args.gripper_only:
          for corner in CORNERS:
            print(f"{'=' * 60}")
            print(f"CORNER: {corner}")
            print(f"{'=' * 60}")

            input(f"  Position gripper at {corner} BOTTOM (board surface). Press Enter...")

            print(f"  Recording {corner} bottom...")
            bottom = record_position(arm, args.bridge, f"{corner} bottom")

            input(f"  Raise gripper to {corner} TOP (transit height). Press Enter...")

            print(f"  Recording {corner} top...")
            top = record_position(arm, args.bridge, f"{corner} top")

            delta_z = (top["xyz_meters"][2] - bottom["xyz_meters"][2]) * 1000

            # Warn and offer re-record if top is suspiciously close to bottom
            if abs(delta_z) < 20:
                print(f"  [WARN] Delta Z is only {delta_z:.1f}mm (expected ~100-150mm).")
                print(f"         Did you move the arm up between recordings?")
                redo = input(f"  Re-record {corner} TOP? [y/N] ").strip().lower()
                if redo == "y":
                    input(f"  Raise gripper to {corner} TOP (transit height). Press Enter...")
                    print(f"  Re-recording {corner} top...")
                    top = record_position(arm, args.bridge, f"{corner} top")
                    delta_z = (top["xyz_meters"][2] - bottom["xyz_meters"][2]) * 1000

            calibration["corners"][corner] = {
                "bottom": bottom,
                "top": top,
            }

            print(f"  [OK] {corner}: bottom Z={bottom['xyz_meters'][2]*1000:.1f}mm, "
                  f"top Z={top['xyz_meters'][2]*1000:.1f}mm, "
                  f"delta={delta_z:.1f}mm")
            print()

        # --- Graveyard ---
        if not args.skip_graveyard and not args.gripper_only:
            print(f"{'=' * 60}")
            print("GRAVEYARD")
            print(f"{'=' * 60}")

            input("  Position gripper at graveyard (off-board capture area). Press Enter...")

            print("  Recording graveyard...")
            graveyard = record_position(arm, args.bridge, "graveyard")
            calibration["graveyard"] = graveyard
            print(f"  [OK] Graveyard: XYZ = {graveyard['xyz_meters']}")
            print()

        # --- Gripper open ---
        print(f"{'=' * 60}")
        print("GRIPPER OPEN")
        print(f"{'=' * 60}")

        input("  Open the gripper fully by hand. Press Enter...")
        state = arm.get_state()
        gripper_open_rad = round(float(state.joint_positions[5]), 4)
        print(f"  [OK] Gripper open: {gripper_open_rad:.4f} rad")
        calibration["gripper_open_rad"] = gripper_open_rad
        print()

        # --- Gripper closed ---
        print(f"{'=' * 60}")
        print("GRIPPER CLOSED")
        print(f"{'=' * 60}")

        input("  Close the gripper on a chess piece. Press Enter...")
        state = arm.get_state()
        gripper_closed_rad = round(float(state.joint_positions[5]), 4)
        print(f"  [OK] Gripper closed: {gripper_closed_rad:.4f} rad")
        calibration["gripper_closed_rad"] = gripper_closed_rad
        print()

        # --- Home position ---
        print(f"{'=' * 60}")
        print("HOME POSITION")
        print(f"{'=' * 60}")

        input("  Move arm to a safe home/rest position. Press Enter...")
        state = arm.get_state()
        home_rad = [round(float(state.joint_positions[i]), 4) for i in range(6)]
        home_names = ARM_JOINT_NAMES + ["gripper"]
        home_dict = {name: home_rad[i] for i, name in enumerate(home_names)}
        print(f"  [OK] Home (rad): {json.dumps(home_dict)}")
        calibration["home_rad"] = home_dict
        print()

        # Release serial and reconnect bridge before computing/saving
        cleanup()

        # --- Compute calibration (skip if gripper-only, already has computed) ---
        if not args.gripper_only:
            print(f"{'=' * 60}")
            print("COMPUTING CALIBRATION")
            print(f"{'=' * 60}")

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

            near_edge = dist_2d(a1_xy, h1_xy)
            far_edge = dist_2d(a8_xy, h8_xy)
            left_edge = dist_2d(a1_xy, a8_xy)
            right_edge = dist_2d(h1_xy, h8_xy)

            board_width = (near_edge + far_edge) / 2
            board_depth = (left_edge + right_edge) / 2

            calibration["computed"] = {
                "z_bottom_m": round(z_bottom, 4),
                "z_top_m": round(z_top, 4),
                "transit_height_mm": round(transit_height_mm, 1),
                "board_width_m": round(board_width, 4),
                "board_depth_m": round(board_depth, 4),
            }

            z_bottom_spread = (max(bottom_z) - min(bottom_z)) * 1000
            z_top_spread = (max(top_z) - min(top_z)) * 1000

            print(f"  z_bottom: {z_bottom*1000:.1f}mm (spread: {z_bottom_spread:.1f}mm)")
            print(f"  z_top:    {z_top*1000:.1f}mm (spread: {z_top_spread:.1f}mm)")
            print(f"  Transit height: {transit_height_mm:.1f}mm")
            print(f"  Board width:  {board_width*1000:.1f}mm")
            print(f"  Board depth:  {board_depth*1000:.1f}mm")

            if z_bottom_spread > 30:
                print(f"  [WARN] Bottom Z spread {z_bottom_spread:.1f}mm > 30mm -- board may not be level")
            if z_top_spread > 30:
                print(f"  [WARN] Top Z spread {z_top_spread:.1f}mm > 30mm -- transit plane uneven")
            if transit_height_mm < 50:
                print(f"  [WARN] Transit height {transit_height_mm:.1f}mm seems too low (< 50mm)")
            if transit_height_mm > 200:
                print(f"  [WARN] Transit height {transit_height_mm:.1f}mm seems too high (> 200mm)")

        print()

        # --- Save ---
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(calibration, f, indent=2)
            f.write("\n")
        tmp.rename(out)

        print(f"[OK] Calibration saved to {output_path}")

        # --- Reload bridge calibration ---
        if not args.no_reload:
            print("  Reloading bridge calibration...")
            try:
                r = requests.post(f"{args.bridge}/calibration/reload", timeout=5)
                if r.status_code == 200 and r.json().get("success"):
                    print("  [OK] Bridge calibration reloaded")
                else:
                    print(f"  [WARN] Reload failed: {r.text}")
            except Exception as e:
                print(f"  [WARN] Could not reload: {e}")

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
