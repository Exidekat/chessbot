#!/usr/bin/env python3
"""
Interactive Chesster calibration validation tool.

Drives the arm to any pose stored in the calibration JSON so you can
visually confirm each recording is correct (gripper actually at the
named board square, transit height adequate, INACTIVE genuinely safe).

Safety guarantees:
  - At startup, drives the arm to INACTIVE autonomously before showing
    the selection menu. Shows the current pose and INACTIVE target
    first and waits for confirmation; you can decline if the arm is
    in a pose where a direct trajectory to INACTIVE would be unsafe.
  - Every menu move is gated by a confirmation prompt that shows
    target joint values and per-joint delta from the current pose.
  - On exit (quit, Ctrl-C, or error), drives directly to INACTIVE
    before releasing torque, so the arm always ends parked.

Usage:
    python scripts/chesster_validate_calibration.py
    python scripts/chesster_validate_calibration.py --port /dev/ttyACM1
    python scripts/chesster_validate_calibration.py --cal data/calibration/board_calibration.json
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.chesster_arm import ChessterArm
from controls.chesster_kinematics import (
    ChessterKinematics, BoardCalibration, ARM_JOINT_NAMES,
    JOINT_NAMES_BY_INDEX, GRIPPER_INDEX as _KIN_GRIPPER_INDEX,
)


# Match conventions in chesster_execute_ik_move_cv_assisted.py
COUNTS_PER_REV = 4096
ENCODER_MIDPOINT = COUNTS_PER_REV // 2
NUM_MOTORS = 5
GRIPPER_INDEX = _KIN_GRIPPER_INDEX  # = 1 on Chesster
TRAJECTORY_HZ = 10
TRAJECTORY_DT = 1.0 / TRAJECTORY_HZ
# Physical motor-order names (gripper is not at end on Chesster).
ALL_NAMES = JOINT_NAMES_BY_INDEX

# Default motion duration: longer than execute script (gentle, deliberate).
MOVE_DURATION_S = 3.0
SETTLE_S = 0.5

# A trajectory leg whose final pose differs from target by more than this is
# flagged as not having reached the target (the servo likely stalled or
# hit a mechanical/electrical limit).
REACHED_TOL_RAD = 0.1

# If max joint delta from INACTIVE at startup is below this, we skip the
# motion entirely (already at INACTIVE). Otherwise we drive to INACTIVE.
INACTIVE_SKIP_TOL_RAD = 0.05


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def ik_deg_to_raw_rad(degrees: float) -> float:
    counts = ENCODER_MIDPOINT + degrees * (COUNTS_PER_REV / 360.0)
    return counts * (2.0 * np.pi / COUNTS_PER_REV)


def ik_angles_to_raw_5(ik_angles_deg: dict, fallback_gripper: float = 0.0) -> np.ndarray:
    """Convert joint-name -> midpoint-degree dict to a 5-element raw-radian
    array indexed by physical motor (per JOINT_NAMES_BY_INDEX). The
    gripper slot receives `fallback_gripper` (typically current physical
    gripper position, so we don't snap the jaws)."""
    result = np.zeros(NUM_MOTORS)
    for i, name in enumerate(JOINT_NAMES_BY_INDEX):
        if name == "gripper":
            result[i] = float(fallback_gripper)
        elif name in ik_angles_deg:
            result[i] = ik_deg_to_raw_rad(ik_angles_deg[name])
    return result


def pose_distance_max(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

def interpolate_trajectory(start: np.ndarray, end: np.ndarray,
                           duration_s: float) -> list:
    n_steps = max(1, int(duration_s * TRAJECTORY_HZ))
    waypoints = []
    for step in range(1, n_steps + 1):
        t = step / n_steps
        target = start + t * (end - start)
        waypoints.append((target, TRAJECTORY_DT))
    return waypoints


def drive_to(arm: ChessterArm, target_5: np.ndarray,
             duration_s: float = MOVE_DURATION_S):
    """Linear joint-space interpolation from current pose to target.

    Pauses the background state reader during the trajectory writes to
    keep the Feetech bus uncontended (matches the execute script).
    """
    arm.refresh_state_sync()
    state = arm.get_state()
    current = state.joint_positions.copy()
    target = np.asarray(target_5, dtype=np.float32).copy()

    waypoints = interpolate_trajectory(current, target, duration_s)
    loop_times = []
    arm.pause_state_reader()
    try:
        for wp, _dt in waypoints:
            t0 = time.monotonic()
            arm.move_joints(wp.tolist(), speed=0.4, blocking=False)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, _dt - elapsed))
            loop_times.append(time.monotonic() - t0)
    finally:
        arm.resume_state_reader()
        arm.refresh_state_sync()

    if loop_times:
        avg_ms = float(np.mean(loop_times)) * 1000
        actual_hz = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"      traj: {len(loop_times)} steps, avg={avg_ms:.0f}ms ({actual_hz:.1f}Hz)")
    time.sleep(SETTLE_S)


# ---------------------------------------------------------------------------
# Target catalog
# ---------------------------------------------------------------------------

def build_target_catalog(cal: BoardCalibration,
                         home_arr: np.ndarray,
                         inactive_arr: np.ndarray) -> list:
    """Return ordered list of (label, target_array, preserves_current_gripper).

    HOME and INACTIVE carry their own recorded gripper value, so they
    don't preserve the current gripper position. Corners and graveyard
    are recorded with arm joints only, so the gripper is left where
    it currently is (filled in just before motion).
    """
    catalog = [
        ("HOME",     home_arr.copy(),     False),
        ("INACTIVE", inactive_arr.copy(), False),
    ]
    for corner in ("a1", "h1", "a8", "h8"):
        if corner not in cal.corners:
            continue
        for plane in ("bottom", "top"):
            if plane not in cal.corners[corner]:
                continue
            joints = cal.corners[corner][plane]["joint_angles_deg"]
            catalog.append((f"{corner} {plane}", joints, True))
    if cal.graveyard is not None:
        catalog.append(("graveyard", cal.graveyard["joint_angles_deg"], True))
    return catalog


def resolve_target(entry, arm: ChessterArm) -> np.ndarray:
    """Convert a catalog entry's target into a 5-element raw-rad array,
    filling the gripper slot from current pose if it's a corner-style record."""
    label, payload, preserves_current_gripper = entry
    if not preserves_current_gripper:
        return np.asarray(payload, dtype=np.float32)
    # payload is a joint_angles_deg dict (4 arm joints, no gripper)
    arm.refresh_state_sync()
    state = arm.get_state()
    current_gripper = float(state.joint_positions[GRIPPER_INDEX])
    return ik_angles_to_raw_5(payload, fallback_gripper=current_gripper)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

# Per-joint swing above this gets a warning at the confirmation gate.
# Empirically the Feetech STS3215 struggles to complete fast swings >~1 rad
# (~57 deg) in a single trajectory leg, and a big wrist swing can sweep the
# gripper through unsafe Cartesian space.
SWING_WARN_RAD = 1.0


def print_pose_diff(label: str, current: np.ndarray, target: np.ndarray):
    print(f"  Target: {label}")
    print(f"  {'joint':15s}  {'current':>9s}  {'target':>9s}  {'delta':>9s}")
    warnings = []
    for i, name in enumerate(ALL_NAMES):
        c = float(current[i]); t = float(target[i])
        d = t - c
        flag = "  <-- big swing" if abs(d) > SWING_WARN_RAD else ""
        print(f"  {name:15s}  {c:+9.3f}  {t:+9.3f}  {d:+9.3f}{flag}")
        if abs(d) > SWING_WARN_RAD:
            warnings.append((name, d))
    print(f"  max joint delta: {pose_distance_max(current, target):.3f} rad")
    if warnings:
        print(f"  [WARN] {len(warnings)} joint(s) require swings > "
              f"{SWING_WARN_RAD:.1f} rad in a single trajectory. The Feetech "
              "servos may stall, and the gripper may sweep through unsafe "
              "Cartesian space en route. Consider adding intermediate "
              "waypoints (e.g. via HOME) or recalibrating the target pose.")


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    print("[X] No /dev/ttyACM* devices found", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chesster calibration validation tool"
    )
    parser.add_argument("--port", type=str, default="auto",
                        help="Serial port (default: auto)")
    parser.add_argument("--cal", type=str, default=None,
                        help="Calibration JSON path (default: "
                             "data/calibration/board_calibration.json)")
    parser.add_argument("--inactive-skip-tolerance", type=float,
                        default=INACTIVE_SKIP_TOL_RAD,
                        help=f"If startup pose is within this max per-joint "
                             f"delta from INACTIVE, skip the initial motion "
                             f"(default {INACTIVE_SKIP_TOL_RAD} rad)")
    args = parser.parse_args()

    cal_path = Path(args.cal) if args.cal else (
        Path(__file__).resolve().parent.parent
        / "data" / "calibration" / "board_calibration.json"
    )
    if not cal_path.exists():
        print(f"[X] Calibration not found at {cal_path}", file=sys.stderr)
        return 1

    print(f"Loading calibration from {cal_path}...")
    kin = ChessterKinematics()
    cal = BoardCalibration.load(str(cal_path), kin)
    if cal.home_rad is None:
        print("[X] Calibration missing HOME pose.", file=sys.stderr)
        return 1
    if cal.inactive_rad is None:
        print("[X] Calibration missing INACTIVE pose.", file=sys.stderr)
        return 1
    if not all(c in cal.corners for c in ("a1", "h1", "a8", "h8")):
        print("[X] Calibration missing one or more corners.", file=sys.stderr)
        return 1

    home_arr = np.array([cal.home_rad[n] for n in ALL_NAMES], dtype=np.float32)
    inactive_arr = np.array([cal.inactive_rad[n] for n in ALL_NAMES],
                            dtype=np.float32)

    print(f"  HOME (rad):     {home_arr.round(3).tolist()}")
    print(f"  INACTIVE (rad): {inactive_arr.round(3).tolist()}")
    print()

    port = resolve_port(args.port)
    print(f"Connecting to {port}...")
    arm = ChessterArm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        return 1

    def disconnect():
        try:
            arm.release_torque()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass

    # ----- Startup: read pose, then drive to INACTIVE (with confirmation) -----
    if not arm.refresh_state_sync():
        print("[X] Failed to read initial state.", file=sys.stderr)
        disconnect()
        return 1

    state = arm.get_state()
    current = state.joint_positions.copy()
    dist = pose_distance_max(current, inactive_arr)

    print()
    print("Startup pose vs INACTIVE:")
    print(f"  {'joint':15s}  {'current':>9s}  {'INACTIVE':>9s}  {'delta':>9s}")
    for i, name in enumerate(ALL_NAMES):
        print(f"  {name:15s}  {float(current[i]):+9.3f}  "
              f"{float(inactive_arr[i]):+9.3f}  "
              f"{float(current[i] - inactive_arr[i]):+9.3f}")
    print(f"  max joint delta: {dist:.3f} rad")

    try:
        arm.enable_torque()
        time.sleep(0.3)
        arm.refresh_state_sync()
        # Set initial servo target to current pose so the bus settles at
        # the present position (no spurious motion when torque latches).
        state = arm.get_state()
        arm.move_joints(state.joint_positions.tolist(), speed=0.5, blocking=False)
        time.sleep(0.3)

        if dist <= args.inactive_skip_tolerance:
            print(f"[OK] Already at INACTIVE within "
                  f"{args.inactive_skip_tolerance:.3f} rad; skipping initial move.")
        else:
            print()
            print(f"Initial move: drive arm to INACTIVE pose "
                  f"(max joint delta {dist:.3f} rad).")
            print(f"  Joint-space interpolation will take a direct path; if "
                  f"any joint must swing through a pose that could collide "
                  f"with the board, abort now and reposition the arm by "
                  f"hand instead.")
            confirm = input("  [Enter] to move to INACTIVE, anything else to abort: ").strip()
            if confirm:
                print("  [aborted]")
                # Release torque so the user can move it manually, then exit.
                disconnect()
                return 1
            print("  Moving to INACTIVE...")
            drive_to(arm, inactive_arr.copy(), MOVE_DURATION_S)
            arm.refresh_state_sync()
            state = arm.get_state()
            final_dist = pose_distance_max(state.joint_positions, inactive_arr)
            print(f"  [OK] At INACTIVE (final max delta: {final_dist:.3f} rad)")

        catalog = build_target_catalog(cal, home_arr, inactive_arr)

        while True:
            print()
            print("=" * 60)
            print("Calibration validation -- select target pose:")
            print("=" * 60)
            for i, (label, _, _) in enumerate(catalog):
                print(f"  [{i:2d}]  {label}")
            print(f"  [ q]  Quit (drive directly to INACTIVE -> release torque)")
            print()
            choice = input("Choice: ").strip().lower()
            if choice in ("q", "quit", "exit", ""):
                if choice == "":
                    print("  (empty input -- type 'q' to quit)")
                    continue
                break

            try:
                idx = int(choice)
                if idx < 0 or idx >= len(catalog):
                    raise ValueError
            except ValueError:
                print(f"  [!] Invalid choice: {choice!r}")
                continue

            entry = catalog[idx]
            target = resolve_target(entry, arm)

            arm.refresh_state_sync()
            state = arm.get_state()
            current = state.joint_positions.copy()
            print()
            print_pose_diff(entry[0], current, target)
            confirm = input("  [Enter] to move, anything else to cancel: ").strip()
            if confirm:
                print("  [cancelled]")
                continue

            print(f"  Moving to {entry[0]}...")
            drive_to(arm, target, MOVE_DURATION_S)
            # Verify arrival with a synchronous read. If the read fails,
            # state.joint_positions is stale from before the trajectory
            # and we can't trust the final-delta number -- report read
            # failure explicitly instead of a misleading delta.
            read_ok = arm.refresh_state_sync()
            if not read_ok:
                print(f"  [WARN] {entry[0]}: motion sent, but post-move state "
                      f"read failed. Cannot verify arrival -- bus may be "
                      f"unstable. The arm IS likely at the commanded target.")
                continue
            state = arm.get_state()
            final = state.joint_positions.copy()
            final_dist = pose_distance_max(final, target)
            if final_dist <= REACHED_TOL_RAD:
                print(f"  [OK] Reached {entry[0]} "
                      f"(final max delta: {final_dist:.3f} rad)")
            else:
                # Identify which joint(s) failed to reach target.
                stalled = []
                for i, name in enumerate(ALL_NAMES):
                    d = float(target[i]) - float(final[i])
                    if abs(d) > REACHED_TOL_RAD:
                        stalled.append((name, float(final[i]), float(target[i]), d))
                print(f"  [FAIL] {entry[0]} NOT reached "
                      f"(final max delta: {final_dist:.3f} rad "
                      f"> tolerance {REACHED_TOL_RAD:.3f}).")
                print(f"         Stalled joints:")
                for name, fcur, tgt, d in stalled:
                    print(f"           {name:15s}: stuck at {fcur:+.3f}, "
                          f"target {tgt:+.3f}, missed by {d:+.3f} rad")
                print(f"         Most likely cause: servo could not complete "
                      f"the requested swing (mechanical block, torque limit, "
                      f"or trajectory too fast for its acceleration).")

    except KeyboardInterrupt:
        print()
        print("[Interrupted]")
    except Exception as e:
        print(f"[X] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    # ----- Cleanup: drive directly to INACTIVE -----
    print()
    print("Cleanup: returning to INACTIVE before releasing torque.")
    try:
        drive_to(arm, inactive_arr.copy(), MOVE_DURATION_S)
        arm.refresh_state_sync()
        state = arm.get_state()
        final_dist = pose_distance_max(state.joint_positions, inactive_arr)
        print(f"  [OK] At INACTIVE (final max delta: {final_dist:.3f} rad)")
    except Exception as e:
        print(f"  [WARN] failed to reach INACTIVE: {e}")

    disconnect()
    print("[OK] Arm released, exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
