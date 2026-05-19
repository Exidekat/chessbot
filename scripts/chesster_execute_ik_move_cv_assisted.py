#!/usr/bin/env python3
"""
Execute a chess move using IK pick-and-place on the Chesster arm.

Full pipeline: camera capture -> board detection -> Stockfish ->
move decomposition -> IK-driven pick-and-place via ChessterArm.

The Chesster arm has 4 arm joints + 1 gripper (wrist_roll removed).
Longer links than the SO-101 mean it is NOT safe to disable torque
from HOME -- the arm would drop over the board. This script REQUIRES
an INACTIVE pose in the calibration file: cleanup() drives the arm to
INACTIVE before releasing torque.

For now this script talks to ChessterArm directly (no RobotController
20Hz stability loop -- that's hardcoded to 6 motors and out of scope
to port here). Trajectory interpolation happens in-process at 10Hz.

Usage:
    python scripts/chesster_execute_ik_move_cv_assisted.py
    python scripts/chesster_execute_ik_move_cv_assisted.py --turn black --rotation right
    python scripts/chesster_execute_ik_move_cv_assisted.py --dry-run
    python scripts/chesster_execute_ik_move_cv_assisted.py --verify
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


# Encoder constants
COUNTS_PER_REV = 4096
ENCODER_MIDPOINT = COUNTS_PER_REV // 2

# Trajectory interpolation rate
TRAJECTORY_HZ = 10
TRAJECTORY_DT = 1.0 / TRAJECTORY_HZ

NUM_MOTORS = 5
GRIPPER_INDEX = _KIN_GRIPPER_INDEX  # = 1 on Chesster (motor ID 2 is the gripper)

# Debug pause between motion steps (seconds)
STEP_PAUSE_S = 1.0

# Manual mode: when True, pause before each motion leg and wait for ENTER.
# Set from --manual CLI flag in main(). Sentinel for aborts.
MANUAL_MODE = False


class _ManualAbort(Exception):
    """Raised when the user types 'a' at a manual-mode prompt."""


def _manual_gate(label: str):
    """If MANUAL_MODE, print the action and wait for ENTER to proceed.
    Type 'a' to abort (raises _ManualAbort, which main() catches and
    routes through the normal cleanup -> INACTIVE path)."""
    if not MANUAL_MODE:
        return
    response = input(f"      [manual] next: {label}  -- [Enter] to run, "
                     f"'a' to abort: ").strip().lower()
    if response in ("a", "abort", "q", "quit"):
        raise _ManualAbort(label)

# Defaults (overridden by calibration if available)
GRIPPER_OPEN_RAD = ENCODER_MIDPOINT * (2.0 * np.pi / COUNTS_PER_REV)
GRIPPER_CLOSED_RAD = (ENCODER_MIDPOINT - 1400) * (2.0 * np.pi / COUNTS_PER_REV)
HOME_RAD: np.ndarray | None = None      # 5-element raw-radian array
INACTIVE_RAD: np.ndarray | None = None  # 5-element raw-radian array

# Motion durations (seconds)
MOVE_LATERAL_S = 2.0
MOVE_VERTICAL_S = 1.5


# ---------------------------------------------------------------------------
# Coordinate conversions: IK midpoint-deg <-> raw-rad (servo native)
# ---------------------------------------------------------------------------

def ik_deg_to_raw_rad(degrees: float) -> float:
    counts = ENCODER_MIDPOINT + degrees * (COUNTS_PER_REV / 360.0)
    return counts * (2.0 * np.pi / COUNTS_PER_REV)


def raw_rad_to_ik_deg(radians: float) -> float:
    counts = radians / (2.0 * np.pi / COUNTS_PER_REV)
    return (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)


def ik_angles_to_raw_5(ik_angles_deg: dict) -> np.ndarray:
    """Convert IK joint angles dict (joint name -> midpoint-degree) to a
    5-element raw-radian array indexed by physical motor (per
    JOINT_NAMES_BY_INDEX). The gripper slot is left at 0 and should be
    overwritten with the current physical gripper position by the caller
    (execute_trajectory preserves it from the live state)."""
    result = np.zeros(NUM_MOTORS)
    for i, name in enumerate(JOINT_NAMES_BY_INDEX):
        if name == "gripper":
            continue  # caller fills in the gripper slot
        if name in ik_angles_deg:
            result[i] = ik_deg_to_raw_rad(ik_angles_deg[name])
    return result


def read_arm_ik_deg(arm: ChessterArm) -> dict:
    state = arm.get_state()
    return {
        name: round(raw_rad_to_ik_deg(float(state.joint_positions[i])), 2)
        for i, name in enumerate(ARM_JOINT_NAMES)
    }


# ---------------------------------------------------------------------------
# Trajectory execution
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


def execute_trajectory(arm: ChessterArm, target_5: np.ndarray,
                       duration_s: float):
    """Linear interpolation from current pose to target over duration_s.

    Pauses the background state reader for the duration of the trajectory
    to keep the Feetech bus uncontended (concurrent reads + 10 Hz writes
    cause burst read failures and phantom disconnects). Resumes after.

    Preserves the gripper joint (index 4) at its current value -- gripper
    motion is handled by execute_gripper().
    """
    state = arm.get_state()
    current = state.joint_positions.copy()
    target_5[GRIPPER_INDEX] = current[GRIPPER_INDEX]

    waypoints = interpolate_trajectory(current, target_5, duration_s)
    loop_times = []
    arm.pause_state_reader()
    try:
        for wp, dt in waypoints:
            t0 = time.monotonic()
            arm.move_joints(wp.tolist(), speed=0.5, blocking=False)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, dt - elapsed))
            loop_times.append(time.monotonic() - t0)
    finally:
        arm.resume_state_reader()
        arm.refresh_state_sync()  # populate fresh state for next leg

    if loop_times:
        avg_ms = float(np.mean(loop_times)) * 1000
        max_ms = float(np.max(loop_times)) * 1000
        actual_hz = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"      traj: {len(loop_times)} steps, "
              f"avg={avg_ms:.0f}ms max={max_ms:.0f}ms ({actual_hz:.1f}Hz)")

    time.sleep(0.3)  # Settle


def execute_gripper(arm: ChessterArm, open_: bool):
    target_rad = GRIPPER_OPEN_RAD if open_ else GRIPPER_CLOSED_RAD
    label = "open" if open_ else "close"

    state = arm.get_state()
    current = state.joint_positions.copy()
    current_gripper = float(current[GRIPPER_INDEX])
    delta = abs(target_rad - current_gripper)
    print(f"      gripper {label}: {current_gripper:.3f} -> {target_rad:.3f} rad "
          f"(delta={delta:.3f})")

    target = current.copy()
    target[GRIPPER_INDEX] = target_rad
    waypoints = interpolate_trajectory(current, target, 1.5)
    arm.pause_state_reader()
    try:
        for wp, dt in waypoints:
            t0 = time.monotonic()
            arm.move_joints(wp.tolist(), speed=0.4, blocking=False)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, dt - elapsed))
    finally:
        arm.resume_state_reader()
        arm.refresh_state_sync()
    time.sleep(0.5)  # Settle


def go_home(arm: ChessterArm):
    if HOME_RAD is None:
        return
    execute_trajectory(arm, HOME_RAD.copy(), 2.5)


def go_inactive(arm: ChessterArm):
    """Drive arm to INACTIVE pose. Required before torque release on Chesster."""
    if INACTIVE_RAD is None:
        raise RuntimeError("INACTIVE pose not loaded -- refusing to release torque")
    execute_trajectory(arm, INACTIVE_RAD.copy(), 3.0)


# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------

def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    print("[X] No /dev/ttyACM* devices found", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Calibration + IK
# ---------------------------------------------------------------------------

def load_calibration():
    """Load BoardCalibration and gripper/home/inactive from calibration file."""
    global GRIPPER_OPEN_RAD, GRIPPER_CLOSED_RAD, HOME_RAD, INACTIVE_RAD

    cal_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "calibration" / "board_calibration.json"
    )

    if not cal_path.exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return None, None

    kin = ChessterKinematics()
    cal = BoardCalibration.load(str(cal_path), kin)

    # INACTIVE is mandatory on Chesster
    if cal.inactive_rad is None:
        print("[X] Calibration is missing INACTIVE pose.", file=sys.stderr)
        print("    Re-run scripts/chesster_calibrate_ik.py to record INACTIVE.",
              file=sys.stderr)
        return None, None

    if not cal.is_calibrated:
        print("[X] Calibration not finalized (need 4 corners + INACTIVE pose)",
              file=sys.stderr)
        return None, None

    if cal.gripper_open_rad is not None and cal.gripper_closed_rad is not None:
        GRIPPER_OPEN_RAD = cal.gripper_open_rad
        GRIPPER_CLOSED_RAD = cal.gripper_closed_rad
        print(f"  Gripper open:   {GRIPPER_OPEN_RAD:.4f} rad")
        print(f"  Gripper closed: {GRIPPER_CLOSED_RAD:.4f} rad")

    # Build 5-element arrays in physical motor order, not "ARM_JOINT_NAMES +
    # gripper" -- the Chesster gripper is at motor index 1, not at the end.
    all_names = JOINT_NAMES_BY_INDEX
    if cal.home_rad is not None:
        HOME_RAD = np.array([cal.home_rad[n] for n in all_names])
        print(f"  Home position loaded ({len(HOME_RAD)} joints)")
    INACTIVE_RAD = np.array([cal.inactive_rad[n] for n in all_names])
    print(f"  Inactive position loaded ({len(INACTIVE_RAD)} joints)")

    mode = ("closed-form IK" if cal.has_ik
            else "joint-space interpolation (no IK block fitted)")
    print(f"[OK] Loaded calibration (default motion mode: {mode})")
    return kin, cal


# ---------------------------------------------------------------------------
# CV detection + move computation
# ---------------------------------------------------------------------------

def detect_and_compute(args) -> dict:
    from guidance.board_detector import BoardDetector
    from guidance.move_calculator import MoveCalculator
    from guidance.move_decomposer import decompose_move
    from utils.camera_helpers import (
        capture_720p_yuyv, capture_4k_downscale,
        get_default_global_camera, DEFAULT_USE_YUYV,
    )
    import chess

    device = args.global_camera or get_default_global_camera()
    if not device:
        return {"success": False, "error": "No camera found"}

    image_path = Path("data/vlm_prompt_capture.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    use_yuyv = DEFAULT_USE_YUYV
    if args.mjpeg:
        use_yuyv = False
    elif args.yuyv:
        use_yuyv = True

    print(f"[Capture] Capturing from {device} ({'YUYV' if use_yuyv else 'MJPEG'})...")
    ok = (capture_720p_yuyv(device, image_path) if use_yuyv
          else capture_4k_downscale(device, image_path))
    if not ok:
        return {"success": False, "error": "Camera capture failed"}

    print("[Detect] Detecting board state...")
    detector = BoardDetector(camera_position=args.rotation,
                             use_corner_rgb=True, use_piece_rgb=True)

    turn = "w" if args.turn == "white" else "b"
    try:
        fen, transformed = detector.detect_board_state(
            str(image_path), corner_conf=args.corner_conf,
            debug=True, turn=turn,
        )
    except Exception as e:
        return {"success": False, "error": f"Detection failed: {e}"}

    print(f"[OK] FEN: {fen}")
    try:
        board = chess.Board(fen)
    except ValueError:
        return {"success": False, "error": f"Invalid FEN: {fen}"}
    print(f"\n{board}\n")

    print(f"[Engine] Computing best move (Stockfish, {args.time}s)...")
    try:
        calculator = MoveCalculator(engine_path=args.engine)
        move = calculator.calculate_best_move(board, time_limit=args.time)
    except FileNotFoundError:
        return {"success": False, "error": f"Engine not found: {args.engine}", "fen": fen}
    if not move:
        return {"success": False, "error": "No legal moves available", "fen": fen}

    san = board.san(move)
    print(f"[OK] Best move: {san} ({move.uci()})")
    stages = decompose_move(board, move)

    return {
        "success": True, "fen": fen,
        "best_move": {"uci": move.uci(), "san": san},
        "stages": stages, "board_ascii": str(board),
    }


# ---------------------------------------------------------------------------
# Two-plane pick-and-place execution
# ---------------------------------------------------------------------------

def _read_current_pan_urdf_rad(arm: ChessterArm, cal: BoardCalibration) -> float:
    """Read the live shoulder_pan servo position and convert it to URDF
    radians (after sign + offset). Used by CV-assist to project the
    gripper-cam image axes back onto the board frame.
    """
    from controls.chesster_kinematics import JOINT_TO_MOTOR_INDEX
    state = arm.get_state()
    pan_idx = JOINT_TO_MOTOR_INDEX["shoulder_pan"]
    raw_rad = float(state.joint_positions[pan_idx])
    pan_servo_deg = raw_rad_to_ik_deg(raw_rad)
    sign = cal.ik["joint_signs"]["shoulder_pan"]
    off = cal.ik["encoder_offsets_deg"]["shoulder_pan"]
    pan_urdf_deg = sign * pan_servo_deg + off
    return float(np.radians(pan_urdf_deg))


def execute_stages(arm: ChessterArm, cal: BoardCalibration, stages: list,
                   use_ik: bool = True, gripper_cam=None) -> dict:
    seed = None
    results = []

    cv_enabled = gripper_cam is not None and cal.has_cv_assist and use_ik
    if gripper_cam is not None and not cv_enabled:
        # Camera was opened but we can't refine -- explain why.
        reason = ("calibration lacks cv_assist block" if not cal.has_cv_assist
                  else "running in interpolation mode (cv-assist requires IK)")
        print(f"    [WARN] CV-assist disabled at runtime: {reason}.")

    def angles(sq, plane, seed):
        return cal.get_joint_angles(sq, plane, seed, use_ik=use_ik)

    for i, stage in enumerate(stages):
        pick_sq = stage.get("pickup_square") or "graveyard"
        place_sq = stage.get("place_square") or "graveyard"
        print(f"  Stage {i+1}: {pick_sq} -> {place_sq}")

        try:
            try:
                pick_top = angles(pick_sq, "top", seed)
            except ValueError as ik_err:
                xyz = cal.square_to_xyz(pick_sq, "top")
                print(f"    [WARN] Top-plane IK failed for {pick_sq}: {ik_err}")
                print(f"           target={xyz.round(4).tolist()}, falling back to bottom")
                pick_top = angles(pick_sq, "bottom", seed)
            pick_bottom = angles(pick_sq, "bottom", pick_top)

            try:
                place_top = angles(place_sq, "top", pick_top)
            except ValueError as ik_err:
                xyz = cal.square_to_xyz(place_sq, "top")
                print(f"    [WARN] Top-plane IK failed for {place_sq}: {ik_err}")
                print(f"           target={xyz.round(4).tolist()}, falling back to bottom")
                place_top = angles(place_sq, "bottom", pick_top)
            place_bottom = angles(place_sq, "bottom", place_top)
        except ValueError as e:
            print(f"    [X] IK failed: {e}")
            results.append({"stage": i, "success": False, "error": str(e)})
            return {
                "success": False,
                "stages_completed": i, "stages_total": len(stages),
                "error": str(e), "stage_results": results,
            }

        pick_top_5 = ik_angles_to_raw_5(pick_top)
        pick_bot_5 = ik_angles_to_raw_5(pick_bottom)
        place_top_5 = ik_angles_to_raw_5(place_top)
        place_bot_5 = ik_angles_to_raw_5(place_bottom)

        # CV-assist refine helpers. Update pick_bot_5 / place_bot_5 in place
        # so subsequent step lambdas (which call `.copy()` at trajectory
        # time) pick up the refined target.
        def _refine_into(target_arr, square, label, current_seed):
            from controls.chesster_cv_assist import cv_assist_refine
            pan_now = _read_current_pan_urdf_rad(arm, cal)
            debug_path = Path("data/cv_assist_debug") / f"stage{i+1}_{label}.png"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            refined, info = cv_assist_refine(
                cal, gripper_cam, square,
                bottom_plane="bottom",
                seed_servo=current_seed,
                current_pan_rad=pan_now,
                debug_frame_path=str(debug_path),
            )
            if refined is None:
                if info.get("delta_mm") is not None:
                    print(f"    [WARN] {label} CV-refine skipped: "
                          f"{info['reason']} (delta={info['delta_mm']:.1f}mm)")
                else:
                    print(f"    [WARN] {label} CV-refine skipped: "
                          f"{info['reason']}")
                return
            dx, dy = info["board_delta_m"]
            print(f"    [CV ] {label} centroid uv={info['centroid_uv']}  "
                  f"board_delta=({dx*1000:+.1f}, {dy*1000:+.1f}) mm  "
                  f"|d|={info['delta_mm']:.1f} mm")
            refined_5 = ik_angles_to_raw_5(refined)
            # Preserve gripper slot (refined dict has no gripper key).
            refined_5[1] = target_arr[1]
            target_arr[:] = refined_5

        cv_pick = (lambda: _refine_into(pick_bot_5, pick_sq, "pick", pick_top))
        cv_place = (lambda: _refine_into(place_bot_5, place_sq, "place", pick_top))

        steps = [
            ("move top(pick)",    lambda: execute_trajectory(arm, pick_top_5.copy(), MOVE_LATERAL_S)),
        ]
        if cv_enabled:
            steps.append(("cv refine pick", cv_pick))
        steps.extend([
            ("gripper open",      lambda: execute_gripper(arm, True)),
            ("descend pick",      lambda: execute_trajectory(arm, pick_bot_5.copy(), MOVE_VERTICAL_S)),
            ("gripper close",     lambda: execute_gripper(arm, False)),
            ("ascend pick",       lambda: execute_trajectory(arm, pick_top_5.copy(), MOVE_VERTICAL_S)),
            ("move top(place)",   lambda: execute_trajectory(arm, place_top_5.copy(), MOVE_LATERAL_S)),
        ])
        if cv_enabled:
            steps.append(("cv refine place", cv_place))
        steps.extend([
            ("descend place",     lambda: execute_trajectory(arm, place_bot_5.copy(), MOVE_VERTICAL_S)),
            ("gripper open",      lambda: execute_gripper(arm, True)),
            ("travel up",         lambda: execute_trajectory(arm, place_top_5.copy(), MOVE_VERTICAL_S)),
            ("home",              lambda: go_home(arm)),
        ])

        step_results = []
        for label, action in steps:
            try:
                _manual_gate(label)
                print(f"    [>>] {label}")
                action()
                step_results.append({"action": label, "success": True})
            except _ManualAbort:
                # Propagate to main() so cleanup runs.
                raise
            except Exception as e:
                print(f"    [X] {label}: {e}")
                step_results.append({"action": label, "success": False, "error": str(e)})
            time.sleep(STEP_PAUSE_S)

        seed = None
        results.append({"stage": i, "success": True, "steps": step_results})

    return {
        "success": True,
        "stages_completed": len(stages),
        "stages_total": len(stages),
        "stage_results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chesster: CV detection + IK pick-and-place"
    )
    parser.add_argument("--global-camera", type=str, default=None)
    parser.add_argument("--engine", type=str, default="stockfish")
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--rotation", type=str, default="right",
                        choices=["left", "right", "top", "bottom"])
    parser.add_argument("--turn", type=str, default="black",
                        choices=["white", "black"])
    parser.add_argument("--yuyv", action="store_true")
    parser.add_argument("--mjpeg", action="store_true")
    parser.add_argument("--corner-conf", type=float, default=0.005)
    parser.add_argument("--port", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--manual", action="store_true",
                        help="Pause before every motion leg (preflight, each "
                             "stage step, cleanup) and require ENTER to "
                             "proceed. Type 'a' to abort cleanly (drives to "
                             "INACTIVE via HOME then exits).")
    parser.add_argument("--interpolation", action="store_true",
                        help="Use piecewise-bilinear joint-space interpolation "
                             "across the calibrated 3x3 anchor grid instead "
                             "of the fitted IK. Useful as a fallback if the "
                             "IK block in the calibration is missing or you "
                             "want to compare behaviours.")
    parser.add_argument("--cv-assist", action="store_true",
                        help="After arriving at pick-top and place-top, "
                             "capture the gripper camera, detect the piece "
                             "centroid via OpenCV, and shift the descend "
                             "target by the implied board-frame XY delta. "
                             "Requires a `cv_assist` block in the "
                             "calibration JSON -- run "
                             "scripts/chesster_calibrate_cv_assist.py once "
                             "to populate it.")
    parser.add_argument("--cv-assist-camera-id", type=int, default=1,
                        help="Gripper-camera device id (default 1)")
    parser.add_argument("--cv-assist-camera-resolution", type=int, nargs=2,
                        default=[640, 480], metavar=("W", "H"))
    args = parser.parse_args()

    global MANUAL_MODE
    MANUAL_MODE = bool(args.manual)
    if MANUAL_MODE:
        print("[--manual] Manual mode active: every motion leg requires ENTER.")
        print("[--manual] Type 'a' at any prompt to abort cleanly via HOME -> INACTIVE.")

    # CV pipeline
    detection = detect_and_compute(args)
    if not detection.get("success"):
        print(f"[X] Detection failed: {detection.get('error')}", file=sys.stderr)
        if args.json:
            print(json.dumps(detection, indent=2))
        return 1

    stages = detection["stages"]
    best_move = detection["best_move"]

    print(f"\n[Move] {best_move['san']} ({best_move['uci']})")
    print(f"[Stages] {len(stages)}")
    for i, s in enumerate(stages):
        pickup = s.get("pickup_square") or "graveyard"
        place = s.get("place_square") or "graveyard"
        print(f"  [{i+1}] {pickup} -> {place}: {s['vlm_prompt']}")

    if args.dry_run:
        print("\n--- Dry run complete ---")
        if args.json:
            print(json.dumps(detection, indent=2))
        return 0

    # Calibration (requires INACTIVE)
    kin, cal = load_calibration()
    if cal is None:
        return 1

    port = resolve_port(args.port)
    print(f"[Setup] Connecting to {port}...")
    arm = ChessterArm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        return 1

    def cleanup(retreat: bool):
        """Optionally retreat via HOME -> INACTIVE, then release torque + disconnect.

        Joint-space interpolation can sweep through unsafe Cartesian space
        between two distant poses (e.g. over-board -> folded). Routing
        through HOME first keeps every leg short and predictable since
        HOME is calibrated as a known-safe above-the-board pose.

        In --manual mode each cleanup leg is gated by ENTER as well.
        Manual aborts during cleanup itself are silently swallowed and
        the torque is released regardless -- safer to let the user
        finalize by hand than to leave the arm stuck.
        """
        try:
            if retreat:
                if HOME_RAD is not None:
                    print("[Cleanup] Driving arm to HOME pose...")
                    try:
                        _manual_gate("cleanup HOME")
                        go_home(arm)
                    except _ManualAbort:
                        print("[Cleanup] manual abort during HOME leg.")
                    except Exception as e:
                        print(f"[Cleanup] WARN: failed to reach HOME: {e}")
                if INACTIVE_RAD is not None:
                    print("[Cleanup] Driving arm to INACTIVE pose...")
                    try:
                        _manual_gate("cleanup INACTIVE")
                        go_inactive(arm)
                    except _ManualAbort:
                        print("[Cleanup] manual abort during INACTIVE leg.")
                    except Exception as e:
                        print(f"[Cleanup] WARN: failed to reach INACTIVE: {e}")
        finally:
            arm.release_torque()
            arm.disconnect()
            print("  [OK] Arm released")

    try:
        arm.enable_torque()

        # Synchronously read current pose before the background state
        # thread has had time to populate self.state.joint_positions
        # (otherwise it's still np.zeros and we'd send the arm to all-zero
        # joint targets -- caught by the limit check but pointlessly noisy).
        if not arm.refresh_state_sync():
            print("[X] Initial state read failed; aborting before motion.",
                  file=sys.stderr)
            cleanup(retreat=False)  # no retreat -- we don't know where the arm is
            return 1

        state = arm.get_state()
        arm.move_joints(state.joint_positions.tolist(), speed=0.5, blocking=False)
        time.sleep(0.2)

        # Preflight: target INACTIVE first as the canonical starting state,
        # then HOME as the high transit waypoint, then begin stages.
        # INACTIVE first ensures we always start from a known parked pose
        # regardless of where the arm physically is at script start; HOME
        # is the safe transit point between INACTIVE and active board
        # motion.
        if INACTIVE_RAD is not None:
            print("[Preflight] Driving arm to INACTIVE pose...")
            try:
                _manual_gate("preflight INACTIVE")
                go_inactive(arm)
            except _ManualAbort:
                raise
            except Exception as e:
                print(f"[Preflight] WARN: failed to reach INACTIVE: {e}")
                print("[Preflight] Aborting to avoid unsafe motion.")
                cleanup(retreat=True)
                return 1

        if HOME_RAD is not None:
            print("[Preflight] Driving arm to HOME pose before stages...")
            try:
                _manual_gate("preflight HOME")
                go_home(arm)
            except _ManualAbort:
                raise
            except Exception as e:
                print(f"[Preflight] WARN: failed to reach HOME: {e}")
                print("[Preflight] Aborting to avoid unsafe motion.")
                cleanup(retreat=True)
                return 1
        else:
            print("[Preflight] WARN: no HOME pose loaded -- motion may be unsafe.")

        # Default to IK; flip to interpolation if user passed --interpolation
        # OR if no fitted ik block is in the calibration.
        use_ik = (not args.interpolation) and cal.has_ik

        # Open the gripper camera if --cv-assist; tolerate failures by
        # falling back to unrefined IK.
        gripper_cam = None
        if args.cv_assist:
            if not cal.has_cv_assist:
                print("[CV  ] WARN: --cv-assist requested but calibration has no "
                      "`cv_assist` block. Run scripts/chesster_calibrate_cv_assist.py "
                      "first. Proceeding without CV-assist.")
            elif not use_ik:
                print("[CV  ] WARN: --cv-assist requires IK mode; --interpolation "
                      "is set or no IK block. Proceeding without CV-assist.")
            else:
                from cameras.gripper_camera import GripperCamera
                gripper_cam = GripperCamera(
                    camera_id=args.cv_assist_camera_id,
                    resolution=tuple(args.cv_assist_camera_resolution),
                )
                if gripper_cam.start():
                    from controls.chesster_cv_assist import probe_camera
                    if probe_camera(gripper_cam, args.cv_assist_camera_id):
                        print(f"[CV  ] Gripper camera open on id="
                              f"{args.cv_assist_camera_id} "
                              f"@ {tuple(args.cv_assist_camera_resolution)}")
                    else:
                        print("[CV  ] WARN: gripper camera produced no frames; "
                              "proceeding without CV-assist.")
                        try: gripper_cam.stop()
                        except Exception: pass
                        gripper_cam = None
                else:
                    print("[CV  ] WARN: gripper camera failed to open; "
                          "proceeding without CV-assist.")
                    gripper_cam = None

        cv_mode_str = " +CV" if gripper_cam is not None else ""
        print(f"\n[Execute] {best_move['san']} via ChessterArm on {port} "
              f"(mode={'IK' if use_ik else 'interpolation'}{cv_mode_str})...")
        start = time.time()
        try:
            result = execute_stages(arm, cal, stages, use_ik=use_ik,
                                    gripper_cam=gripper_cam)
        finally:
            if gripper_cam is not None:
                try:
                    gripper_cam.stop()
                except Exception as e:
                    print(f"[CV  ] WARN: gripper camera stop raised: {e}")
        elapsed = time.time() - start

        success = result.get("success", False)
        status = "[OK] SUCCESS" if success else "[X] FAILED"
        print(f"\n{status} in {int(elapsed*1000)}ms")
        print(f"Stages completed: {result.get('stages_completed')}/{result.get('stages_total')}")

        if not success:
            print(f"Error: {result.get('error')}", file=sys.stderr)

        if result.get("stage_results"):
            for sr in result["stage_results"]:
                stage_status = "[OK]" if sr["success"] else "[X]"
                print(f"  Stage {sr['stage']}: {stage_status}")

        # Always retreat to INACTIVE before releasing torque on Chesster
        cleanup(retreat=True)

        if args.verify and success:
            print("\n[Verify] Verifying board state after move...")
            verify_result = detect_and_compute(args)
            if verify_result.get("success"):
                print(f"[OK] Post-move FEN: {verify_result['fen']}")
            else:
                print(f"[WARN] Verification capture failed: {verify_result.get('error')}")

        if args.json:
            combined = {"detection": detection, "execution": result}
            print(json.dumps(combined, indent=2))

        return 0 if success else 1

    except _ManualAbort as ab:
        print(f"\n[--manual] User aborted at: {ab}")
        cleanup(retreat=True)
        return 1
    except KeyboardInterrupt:
        print("\n\n[X] Cancelled by user")
        # On Ctrl-C, still try to retreat to INACTIVE before torque release
        cleanup(retreat=True)
        return 1
    except Exception as e:
        print(f"\n[X] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        try:
            # Best-effort retreat even on error
            cleanup(retreat=True)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
