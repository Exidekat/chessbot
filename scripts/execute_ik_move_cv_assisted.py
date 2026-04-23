#!/usr/bin/env python3
"""
Execute a chess move using IK pick-and-place -- CV-assisted.

Uses the full vision pipeline (camera capture -> board detection -> Stockfish ->
move decomposition) to determine the best move, then executes it physically
using RobotController (stability system + 20Hz control loop) with 10Hz
trajectory interpolation for smooth two-plane IK pick-and-place.

Usage:
    python scripts/execute_ik_move_cv_assisted.py
    python scripts/execute_ik_move_cv_assisted.py --turn black --rotation right
    python scripts/execute_ik_move_cv_assisted.py --dry-run
    python scripts/execute_ik_move_cv_assisted.py --mjpeg
    python scripts/execute_ik_move_cv_assisted.py --verify
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
from controls.robot_controller import (
    RobotController, JointConfig, load_joint_configs_for_port,
)


# Arm joint names matching kinematics (excludes gripper)
ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

# Encoder constants
COUNTS_PER_REV = 4096
ENCODER_MIDPOINT = COUNTS_PER_REV // 2

# Trajectory interpolation rate.
# The optimized control loop runs at ~20Hz (position writes only),
# so 10Hz trajectory feed gives 2 control iterations per waypoint.
TRAJECTORY_HZ = 10
TRAJECTORY_DT = 1.0 / TRAJECTORY_HZ


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------
# IK (kinematics.py) uses degrees in "midpoint convention": 0 deg = 2048 counts
# RobotController uses radians in "raw convention": 0 rad = 0 counts
# Conversion: midpoint_deg -> counts -> raw_rad

def ik_deg_to_raw_rad(degrees: float) -> float:
    """Convert IK midpoint-convention degrees to RobotController raw radians."""
    counts = ENCODER_MIDPOINT + degrees * (COUNTS_PER_REV / 360.0)
    return counts * (2.0 * np.pi / COUNTS_PER_REV)


def raw_rad_to_ik_deg(radians: float) -> float:
    """Convert RobotController raw radians to IK midpoint-convention degrees."""
    counts = radians / (2.0 * np.pi / COUNTS_PER_REV)
    return (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)


def ik_angles_to_raw_6(ik_angles_deg: dict) -> np.ndarray:
    """Convert IK joint angles dict (5 arm joints, degrees) to a 6-element
    raw-radian array for RobotController (index 5 = gripper, unchanged)."""
    result = np.zeros(6)
    for i, name in enumerate(ARM_JOINT_NAMES):
        result[i] = ik_deg_to_raw_rad(ik_angles_deg.get(name, 0.0))
    # result[5] = gripper, left at 0 (set separately)
    return result


def read_arm_ik_deg(robot: RobotController) -> dict:
    """Read current arm positions as IK-convention degrees."""
    state = robot.arm.get_state()
    result = {}
    for i, name in enumerate(ARM_JOINT_NAMES):
        result[name] = round(raw_rad_to_ik_deg(float(state.joint_positions[i])), 2)
    return result


# ---------------------------------------------------------------------------
# Trajectory execution via RobotController
# ---------------------------------------------------------------------------

# Defaults (overridden by calibration if available)
GRIPPER_OPEN_RAD = ENCODER_MIDPOINT * (2.0 * np.pi / COUNTS_PER_REV)
GRIPPER_CLOSED_RAD = (ENCODER_MIDPOINT - 1400) * (2.0 * np.pi / COUNTS_PER_REV)
HOME_RAD = None  # 6-element raw-radian array, loaded from calibration

# Debug pause between motion steps (seconds)
STEP_PAUSE_S = 1.0


def interpolate_trajectory(start: np.ndarray, end: np.ndarray,
                           duration_s: float) -> list:
    """Generate interpolated waypoints at TRAJECTORY_HZ between start and end.

    Returns list of (target_6dof, dt) tuples. Each target is a 6-element
    raw-radian array. dt is the time to wait after setting this target.
    """
    n_steps = max(1, int(duration_s * TRAJECTORY_HZ))
    waypoints = []
    for step in range(1, n_steps + 1):
        t = step / n_steps  # 0..1
        target = start + t * (end - start)
        waypoints.append((target, TRAJECTORY_DT))
    return waypoints


def execute_trajectory(robot: RobotController, target_6: np.ndarray,
                       duration_s: float):
    """Move arm from current position to target over duration_s at 10Hz.

    Reads current position, generates linear interpolation, and feeds
    intermediate targets to RobotController's 20Hz control loop.
    """
    state = robot.arm.get_state()
    current = state.joint_positions.copy()

    # Preserve gripper position during arm moves
    target_6[5] = current[5]

    waypoints = interpolate_trajectory(current, target_6, duration_s)

    loop_times = []
    for wp, dt in waypoints:
        t0 = time.monotonic()
        robot.set_target_positions(wp)
        elapsed = time.monotonic() - t0
        sleep_remaining = max(0, dt - elapsed)
        time.sleep(sleep_remaining)
        loop_times.append(time.monotonic() - t0)

    # Report trajectory timing
    if loop_times:
        avg_ms = np.mean(loop_times) * 1000
        max_ms = np.max(loop_times) * 1000
        actual_hz = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"      traj: {len(loop_times)} steps, "
              f"avg={avg_ms:.0f}ms max={max_ms:.0f}ms ({actual_hz:.1f}Hz)")

    # Settle: wait for control loop to converge
    time.sleep(0.3)


def execute_gripper(robot: RobotController, open_: bool):
    """Open or close the gripper via trajectory.

    Only changes joint index 5 (gripper); arm joints hold their current
    control-loop targets to prevent jitter.
    """
    target_rad = GRIPPER_OPEN_RAD if open_ else GRIPPER_CLOSED_RAD
    label = "open" if open_ else "close"

    # Read actual gripper position for delta reporting
    state = robot.arm.get_state()
    current_gripper = float(state.joint_positions[5])
    delta = abs(target_rad - current_gripper)

    print(f"      gripper {label}: {current_gripper:.3f} -> {target_rad:.3f} rad "
          f"(delta={delta:.3f})")

    # Update only the gripper target, keep arm targets as-is in the control loop
    targets = robot.target_positions.copy()
    targets[5] = target_rad

    # Interpolate over 1.5s for the large range this gripper servo covers
    waypoints = interpolate_trajectory(robot.target_positions.copy(), targets, 1.5)

    loop_times = []
    for wp, dt in waypoints:
        t0 = time.monotonic()
        robot.set_target_positions(wp)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, dt - elapsed))
        loop_times.append(time.monotonic() - t0)

    if loop_times:
        avg_ms = np.mean(loop_times) * 1000
        print(f"      traj: {len(loop_times)} steps, avg={avg_ms:.0f}ms "
              f"({1000.0/avg_ms:.1f}Hz)")

    time.sleep(0.5)  # Settle


def go_home(robot: RobotController):
    """Move arm to calibrated home position."""
    if HOME_RAD is None:
        return  # No home position calibrated
    execute_trajectory(robot, HOME_RAD.copy(), 2.5)


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
    """Load BoardCalibration and gripper/home settings from calibration file."""
    global GRIPPER_OPEN_RAD, GRIPPER_CLOSED_RAD, HOME_RAD

    from controls.kinematics import BoardCalibration, SO101Kinematics

    cal_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "calibration" / "board_calibration.json"
    )

    if not cal_path.exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return None, None

    kin = SO101Kinematics()
    cal = BoardCalibration.load(str(cal_path), kin)

    if not cal.is_calibrated:
        print("[X] Calibration not finalized", file=sys.stderr)
        return None, None

    # Load gripper and home from raw calibration JSON
    with open(cal_path) as f:
        raw_cal = json.load(f)

    if "gripper_open_rad" in raw_cal and "gripper_closed_rad" in raw_cal:
        GRIPPER_OPEN_RAD = raw_cal["gripper_open_rad"]
        GRIPPER_CLOSED_RAD = raw_cal["gripper_closed_rad"]
        print(f"  Gripper open:  {GRIPPER_OPEN_RAD:.4f} rad")
        print(f"  Gripper closed: {GRIPPER_CLOSED_RAD:.4f} rad")
    if "home_rad" in raw_cal:
        home = raw_cal["home_rad"]
        all_names = ARM_JOINT_NAMES + ["gripper"]
        HOME_RAD = np.array([home[n] for n in all_names])
        print(f"  Home position loaded ({len(HOME_RAD)} joints)")

    print(f"[OK] Loaded calibration: z_bottom={cal.z_bottom*1000:.1f}mm, "
          f"z_top={cal.z_top*1000:.1f}mm")
    return kin, cal


# ---------------------------------------------------------------------------
# CV detection + move computation
# ---------------------------------------------------------------------------

def detect_and_compute(args) -> dict:
    """Run the full CV pipeline: capture -> detect -> engine -> decompose."""
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
    ok = capture_720p_yuyv(device, image_path) if use_yuyv else capture_4k_downscale(device, image_path)
    if not ok:
        return {"success": False, "error": "Camera capture failed"}

    print("[Detect] Detecting board state...")
    detector = BoardDetector(camera_position=args.rotation, use_corner_rgb=True, use_piece_rgb=True)

    turn = "w" if args.turn == "white" else "b"
    try:
        fen, transformed = detector.detect_board_state(
            str(image_path), corner_conf=args.corner_conf, debug=True, turn=turn,
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

# Duration for each motion step (seconds)
MOVE_LATERAL_S = 2.0   # top-plane lateral moves
MOVE_VERTICAL_S = 1.5  # ascend/descend between planes


def execute_stages(robot: RobotController, cal, stages: list) -> dict:
    """Execute chess move stages using RobotController + trajectory interpolation.

    Each stage is a pick-and-place sequence:
     1. Move to top plane over pickup square
     2. Open gripper
     3. Descend to bottom plane (grasp)
     4. Close gripper
     5. Ascend to top plane
     6. Move to top plane over place square
     7. Descend to bottom plane
     8. Open gripper
     9. Ascend to top plane (travel up after place)
    10. Return to home position

    A 1-second debug pause is inserted between every step.
    """
    seed = None
    results = []

    for i, stage in enumerate(stages):
        pick_sq = stage.get("pickup_square") or "graveyard"
        place_sq = stage.get("place_square") or "graveyard"
        print(f"  Stage {i+1}: {pick_sq} -> {place_sq}")

        try:
            try:
                pick_top = cal.get_joint_angles(pick_sq, "top", seed)
            except ValueError as ik_err:
                xyz = cal.square_to_xyz(pick_sq, "top")
                print(f"    [WARN] Top-plane IK failed for {pick_sq}: {ik_err}")
                print(f"           target={xyz.round(4).tolist()}, falling back to bottom")
                pick_top = cal.get_joint_angles(pick_sq, "bottom", seed)
            pick_bottom = cal.get_joint_angles(pick_sq, "bottom", pick_top)

            try:
                place_top = cal.get_joint_angles(place_sq, "top", pick_top)
            except ValueError as ik_err:
                xyz = cal.square_to_xyz(place_sq, "top")
                print(f"    [WARN] Top-plane IK failed for {place_sq}: {ik_err}")
                print(f"           target={xyz.round(4).tolist()}, falling back to bottom")
                place_top = cal.get_joint_angles(place_sq, "bottom", pick_top)
            place_bottom = cal.get_joint_angles(place_sq, "bottom", place_top)
        except ValueError as e:
            print(f"    [X] IK failed: {e}")
            results.append({"stage": i, "success": False, "error": str(e)})
            return {"success": False, "stages_completed": i, "stages_total": len(stages),
                    "error": str(e), "stage_results": results}

        # Convert IK degrees to raw radians for RobotController
        pick_top_6 = ik_angles_to_raw_6(pick_top)
        pick_bot_6 = ik_angles_to_raw_6(pick_bottom)
        place_top_6 = ik_angles_to_raw_6(place_top)
        place_bot_6 = ik_angles_to_raw_6(place_bottom)

        # Execute step sequence with 1s debug pause between each
        steps = [
            ("move top(pick)",    lambda: execute_trajectory(robot, pick_top_6.copy(), MOVE_LATERAL_S)),
            ("gripper open",      lambda: execute_gripper(robot, True)),
            ("descend pick",      lambda: execute_trajectory(robot, pick_bot_6.copy(), MOVE_VERTICAL_S)),
            ("gripper close",     lambda: execute_gripper(robot, False)),
            ("ascend pick",       lambda: execute_trajectory(robot, pick_top_6.copy(), MOVE_VERTICAL_S)),
            ("move top(place)",   lambda: execute_trajectory(robot, place_top_6.copy(), MOVE_LATERAL_S)),
            ("descend place",     lambda: execute_trajectory(robot, place_bot_6.copy(), MOVE_VERTICAL_S)),
            ("gripper open",      lambda: execute_gripper(robot, True)),
            ("travel up",         lambda: execute_trajectory(robot, place_top_6.copy(), MOVE_VERTICAL_S)),
            ("home",              lambda: go_home(robot)),
        ]

        step_results = []
        for label, action in steps:
            try:
                print(f"    [>>] {label}")
                action()
                step_results.append({"action": label, "success": True})
            except Exception as e:
                print(f"    [X] {label}: {e}")
                step_results.append({"action": label, "success": False, "error": str(e)})
            # Debug pause between steps
            time.sleep(STEP_PAUSE_S)

        # Reset seed to None so next stage uses nearest calibration corner,
        # not the home position (which is far from any board square).
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
        description="Execute a chess move via CV detection + IK pick-and-place"
    )
    parser.add_argument("--global-camera", type=str, default=None,
                        help="Camera device (auto-detects WBC-0E01)")
    parser.add_argument("--engine", type=str, default="stockfish",
                        help="Stockfish path (default: stockfish)")
    parser.add_argument("--time", type=float, default=1.0,
                        help="Engine time limit in seconds (default: 1.0)")
    parser.add_argument("--rotation", type=str, default="right",
                        choices=["left", "right", "top", "bottom"],
                        help="Camera rotation (default: right)")
    parser.add_argument("--turn", type=str, default="black",
                        choices=["white", "black"],
                        help="Whose turn (default: black)")
    parser.add_argument("--yuyv", action="store_true",
                        help="Force native 720p YUYV capture")
    parser.add_argument("--mjpeg", action="store_true",
                        help="Use 4K MJPEG -> 720p downscale")
    parser.add_argument("--corner-conf", type=float, default=0.005,
                        help="Corner detection threshold (default: 0.005)")
    parser.add_argument("--port", type=str, default="auto",
                        help="Serial port (default: auto-detect /dev/ttyACM*)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect + compute only, don't execute physically")
    parser.add_argument("--verify", action="store_true",
                        help="Re-capture board after move to verify")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON result to stdout")
    args = parser.parse_args()

    # Step 1: CV detection + move computation
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

    # Step 2: Load calibration
    kin, cal = load_calibration()
    if cal is None:
        return 1

    # Step 3: Connect RobotController
    port = resolve_port(args.port)
    print(f"[Setup] Connecting to {port}...")

    # Load joint configs for this port (3-tier fallback: port-specific > generic > defaults)
    configs, config_source = load_joint_configs_for_port(port)
    robot = RobotController(port, configs, config_source, enable_stability_system=True)

    if not robot.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        return 1

    def cleanup():
        robot.stop_control_loop()
        robot.release_torque()
        robot.disconnect()
        print("  [OK] Arm released")

    try:
        # Enable torque and start the 20Hz stability control loop
        robot.enable_torque()
        robot.start_control_loop()

        # Set initial target to current position (prevent jump on start)
        state = robot.arm.get_state()
        robot.set_target_positions(state.joint_positions.copy())
        time.sleep(0.2)  # Let control loop stabilize

        # Step 4: Execute
        print(f"\n[Execute] {best_move['san']} via RobotController on {port}...")
        start = time.time()
        result = execute_stages(robot, cal, stages)
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

        cleanup()

        # Step 5: Optional verification
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
