#!/usr/bin/env python3
"""
Execute a chess move using IK pick-and-place -- CV-assisted.

Uses the full vision pipeline (camera capture -> board detection -> Stockfish ->
move decomposition) to determine the best move, then executes it physically
using direct serial control of the SO-100 arm via two-plane IK.

Usage:
    python scripts/execute_ik_move_cv_assisted.py
    python scripts/execute_ik_move_cv_assisted.py --turn black --rotation right
    python scripts/execute_ik_move_cv_assisted.py --dry-run
    python scripts/execute_ik_move_cv_assisted.py --mjpeg
    python scripts/execute_ik_move_cv_assisted.py --bridge http://localhost:8420
    python scripts/execute_ik_move_cv_assisted.py --verify
"""

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

# Add parent directory to path for chessbot module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.so100_arm import SO100Arm
from controls.robot_controller import RobotController

try:
    import requests
except ImportError:
    print("ERROR: requests library required. Install: pip install requests",
          file=sys.stderr)
    sys.exit(1)


DEFAULT_BRIDGE = "http://localhost:8420"

# Arm joint names matching kinematics (excludes gripper)
ARM_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll",
]

# All joints including gripper, indexed 0-5
ALL_JOINT_NAMES = ARM_JOINT_NAMES + ["gripper"]

# Motor IDs by joint name (1-indexed to match Feetech protocol)
JOINT_MOTOR_IDS = {
    "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
    "wrist_flex": 4, "wrist_roll": 5, "gripper": 6,
}

# Encoder constants (from so101.py)
COUNTS_PER_REV = 4096
ENCODER_MIDPOINT = COUNTS_PER_REV // 2
COUNTS_PER_DEG = COUNTS_PER_REV / 360.0

# Per-joint motor parameters -- read from RobotController class constants
# so they stay in sync with tele-op and other control code.
def _load_joint_params():
    """Build joint params dict from RobotController class constants."""
    params = {}
    for idx, name in enumerate(ALL_JOINT_NAMES):
        params[name] = {
            "speed": getattr(RobotController, f"JOINT_{idx}_SPEED_LIMIT", 1000),
            "accel": getattr(RobotController, f"JOINT_{idx}_ACCEL", 0),
            "torque": getattr(RobotController, f"JOINT_{idx}_TORQUE", 100),
            "deadband": getattr(RobotController, f"JOINT_{idx}_DEADBAND", 0.0),
            "smoothing": getattr(RobotController, f"JOINT_{idx}_SMOOTHING_ALPHA", 1.0),
        }
    return params

JOINT_PARAMS = _load_joint_params()

# Gripper positions (encoder counts, from calibration)
GRIPPER_OPEN_COUNTS = ENCODER_MIDPOINT
GRIPPER_CLOSED_COUNTS = ENCODER_MIDPOINT - 1400

# Register addresses
REG_GOAL_POSITION = 0x2A
REG_SPEED = 0x2E
REG_ACCEL = 0x29
REG_TORQUE_LIMIT = 0x30
REG_TORQUE_ENABLE = 0x28


def _deg_to_counts(degrees: float) -> int:
    """Convert degrees (IK output) to encoder counts. 0 deg = midpoint."""
    counts = int(ENCODER_MIDPOINT + degrees * COUNTS_PER_DEG)
    return max(0, min(COUNTS_PER_REV - 1, counts))


def _counts_to_deg(counts: int) -> float:
    """Convert encoder counts to degrees."""
    return (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)


def _checksum(data):
    return (~sum(data)) & 0xFF


# ---------------------------------------------------------------------------
# Direct arm control helpers
# ---------------------------------------------------------------------------

def write_register(arm: SO100Arm, motor_id: int, reg: int, value: int, nbytes: int = 2):
    """Write a value to a motor register."""
    if nbytes == 1:
        data = [motor_id, 0x04, arm.INSTR_WRITE, reg, value & 0xFF]
    else:
        data = [motor_id, 0x05, arm.INSTR_WRITE, reg, value & 0xFF, (value >> 8) & 0xFF]
    cs = arm._calculate_checksum(data)
    arm.serial.write(bytes([*arm.HEADER] + data + [cs]))
    time.sleep(0.003)


def enable_torque_all(arm: SO100Arm):
    """Enable torque on all motors with per-joint params."""
    for joint, mid in JOINT_MOTOR_IDS.items():
        params = JOINT_PARAMS[joint]
        write_register(arm, mid, REG_TORQUE_ENABLE, 1, nbytes=1)
        write_register(arm, mid, REG_SPEED, params["speed"])
        write_register(arm, mid, REG_ACCEL, params["accel"], nbytes=1)
        write_register(arm, mid, REG_TORQUE_LIMIT, params["torque"])


def disable_torque_all(arm: SO100Arm):
    """Disable torque on all motors."""
    for mid in JOINT_MOTOR_IDS.values():
        write_register(arm, mid, REG_TORQUE_LIMIT, 0)
        write_register(arm, mid, REG_TORQUE_ENABLE, 0, nbytes=1)


def move_arm(arm: SO100Arm, joint_angles_deg: dict, duration_ms: int = 1000):
    """Move arm joints to target angles (degrees).

    Applies the RobotController stability system:
    - Per-joint speed limits, acceleration, torque limits
    - Deadband: skips move if error is within deadband (prevents oscillation)
    - Smoothing: low-pass filter on target (alpha=1.0 means no smoothing)
    """
    # Duration -> speed conversion (from SO101Controller)
    base_speed = int(1500 * (500.0 / max(duration_ms, 100)))
    base_speed = max(50, min(1500, base_speed))

    # Read current positions for deadband check
    current = read_arm_deg(arm)

    for joint, target_deg in joint_angles_deg.items():
        mid = JOINT_MOTOR_IDS.get(joint)
        if mid is None:
            continue
        params = JOINT_PARAMS[joint]
        speed = min(base_speed, params["speed"])

        # Deadband check: skip if error is within deadband
        deadband_deg = math.degrees(params["deadband"]) if params["deadband"] > 0 else 0
        current_deg = current.get(joint, 0.0)
        if deadband_deg > 0 and abs(target_deg - current_deg) < deadband_deg:
            # Inside deadband -- disable torque on this joint to stop oscillation
            write_register(arm, mid, REG_TORQUE_ENABLE, 0, nbytes=1)
            continue

        # Outside deadband -- ensure torque is enabled if deadband is active
        if deadband_deg > 0:
            write_register(arm, mid, REG_TORQUE_ENABLE, 1, nbytes=1)

        # Apply smoothing (low-pass filter)
        alpha = params["smoothing"]
        if alpha < 1.0:
            # Blend target with current: smoothed = alpha*target + (1-alpha)*current
            target_deg = alpha * target_deg + (1.0 - alpha) * current_deg

        counts = _deg_to_counts(target_deg)
        write_register(arm, mid, REG_SPEED, speed)
        write_register(arm, mid, REG_ACCEL, params["accel"], nbytes=1)
        write_register(arm, mid, REG_TORQUE_LIMIT, params["torque"])
        write_register(arm, mid, REG_GOAL_POSITION, counts)

    # Wait for motion
    time.sleep(duration_ms / 1000.0 + 0.2)


def gripper_open(arm: SO100Arm):
    """Open gripper."""
    write_register(arm, JOINT_MOTOR_IDS["gripper"], REG_GOAL_POSITION, GRIPPER_OPEN_COUNTS)
    time.sleep(0.5)


def gripper_close(arm: SO100Arm):
    """Close gripper."""
    write_register(arm, JOINT_MOTOR_IDS["gripper"], REG_GOAL_POSITION, GRIPPER_CLOSED_COUNTS)
    time.sleep(0.5)


def read_arm_deg(arm: SO100Arm) -> dict:
    """Read current arm joint angles in degrees (kinematics convention).

    SO100Arm returns radians where 0 counts = 0 rad (raw encoder).
    Kinematics uses degrees where 2048 counts = 0 deg (midpoint convention).
    Convert: counts -> _counts_to_deg (accounts for midpoint offset).
    """
    state = arm.get_state()
    result = {}
    for i, name in enumerate(ARM_JOINT_NAMES):
        # Convert raw radians back to counts, then to kinematics degrees
        raw_rad = float(state.joint_positions[i])
        counts = int(raw_rad / (2 * np.pi / COUNTS_PER_REV))
        result[name] = round(_counts_to_deg(counts), 2)
    return result


# ---------------------------------------------------------------------------
# Bridge helpers (FK, calibration load, disconnect/reconnect)
# ---------------------------------------------------------------------------

def check_health(bridge: str) -> bool:
    try:
        return requests.get(f"{bridge}/health", timeout=5).status_code == 200
    except Exception:
        return False


def disconnect_bridge_robot(bridge: str):
    try:
        r = requests.post(f"{bridge}/robot/disconnect", timeout=5)
        if r.status_code == 200:
            print(f"  [OK] {r.json().get('message', 'Bridge robot disconnected')}")
    except Exception:
        pass  # Bridge may not have robot connected


def reconnect_bridge_robot(bridge: str):
    try:
        r = requests.post(f"{bridge}/robot/reconnect", timeout=10)
        if r.status_code == 200 and r.json().get("success"):
            print(f"  [OK] {r.json().get('message', 'Bridge robot reconnected')}")
    except Exception:
        pass


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

def load_calibration(bridge: str):
    """Load BoardCalibration from the bridge's calibration path."""
    # Import from gen-int claw
    claw_dir = Path(__file__).resolve().parent.parent.parent.parent / "claw"
    sys.path.insert(0, str(claw_dir.parent))

    from claw.bridge.kinematics import BoardCalibration, SO101Kinematics

    # Determine calibration file path (same logic as bridge)
    import os
    agent_name = os.environ.get("AGENT_NAME", "")
    if agent_name:
        ws = claw_dir / "agents" / agent_name / "workspace"
    else:
        ws = claw_dir / "agents" / "chessbot" / "workspace"
    cal_path = ws / "calibration" / "board_calibration.json"

    if not cal_path.exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return None, None

    kin = SO101Kinematics()
    cal = BoardCalibration.load(str(cal_path), kin)

    if not cal.is_calibrated:
        print("[X] Calibration not finalized", file=sys.stderr)
        return None, None

    print(f"[OK] Loaded calibration: z_bottom={cal.z_bottom*1000:.1f}mm, "
          f"z_top={cal.z_top*1000:.1f}mm")
    return kin, cal


# ---------------------------------------------------------------------------
# CV detection + move computation (unchanged from original)
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

def execute_stages(arm: SO100Arm, cal, kin, stages: list) -> dict:
    """Execute chess move stages using direct serial + two-plane IK.

    Each stage is a pick-and-place sequence:
    1. Move to top plane over pickup square
    2. Open gripper
    3. Descend to bottom plane (grasp)
    4. Close gripper
    5. Ascend to top plane
    6. Move to top plane over place square
    7. Descend to bottom plane
    8. Open gripper
    9. Ascend to top plane
    """
    seed = read_arm_deg(arm)
    results = []

    for i, stage in enumerate(stages):
        pick_sq = stage.get("pickup_square") or "graveyard"
        place_sq = stage.get("place_square") or "graveyard"
        print(f"  Stage {i+1}: {pick_sq} -> {place_sq}")

        try:
            # IK solve 4 waypoints with seed chaining
            pick_top = cal.get_joint_angles(pick_sq, "top", seed)
            pick_bottom = cal.get_joint_angles(pick_sq, "bottom", pick_top)
            place_top = cal.get_joint_angles(place_sq, "top", pick_top)
            place_bottom = cal.get_joint_angles(place_sq, "bottom", place_top)
        except ValueError as e:
            print(f"    [X] IK failed: {e}")
            results.append({"stage": i, "success": False, "error": str(e)})
            return {"success": False, "stages_completed": i, "stages_total": len(stages),
                    "error": str(e), "stage_results": results}

        # Execute sequence
        steps = [
            ("move top(pick)",    pick_top,     1000),
            ("gripper open",      None,         0),
            ("descend pick",      pick_bottom,  800),
            ("gripper close",     None,         0),
            ("ascend pick",       pick_top,     800),
            ("move top(place)",   place_top,    1000),
            ("descend place",     place_bottom, 800),
            ("gripper open",      None,         0),
            ("ascend place",      place_top,    800),
        ]

        step_results = []
        for label, angles, dur in steps:
            try:
                if "gripper open" in label:
                    print(f"    [>>] {label}")
                    gripper_open(arm)
                elif "gripper close" in label:
                    print(f"    [>>] {label}")
                    gripper_close(arm)
                else:
                    print(f"    [>>] {label}")
                    move_arm(arm, angles, dur)
                step_results.append({"action": label, "success": True})
            except Exception as e:
                print(f"    [X] {label}: {e}")
                step_results.append({"action": label, "success": False, "error": str(e)})

        seed = place_top  # Chain seed to next stage
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
    parser.add_argument("--bridge", type=str, default=DEFAULT_BRIDGE,
                        help=f"Bridge URL (default: {DEFAULT_BRIDGE})")
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

    # Step 2: Load calibration (direct, no bridge)
    kin, cal = load_calibration(args.bridge)
    if cal is None:
        return 1

    # Step 3: Get serial port and connect directly
    port = resolve_port(args.port)

    print(f"[Setup] Disconnecting bridge robot, connecting to {port}...")
    disconnect_bridge_robot(args.bridge)
    time.sleep(0.2)

    arm = SO100Arm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        reconnect_bridge_robot(args.bridge)
        return 1

    def cleanup():
        disable_torque_all(arm)
        arm.disconnect()
        time.sleep(0.2)
        print("  Reconnecting bridge robot...")
        reconnect_bridge_robot(args.bridge)

    try:
        # Enable torque with per-joint params
        enable_torque_all(arm)

        # Step 4: Execute
        print(f"\n[Execute] {best_move['san']} via direct serial on {port}...")
        start = time.time()
        result = execute_stages(arm, cal, kin, stages)
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

        # Release arm and reconnect bridge
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
