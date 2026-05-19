#!/usr/bin/env python3
"""
Chesster CV-assist Calibration

One-time setup for gripper-camera-based fine pickup refinement.

What it does:
  1. Drives the Chesster arm to pick-top above a chosen calibration
     square (default e4) using the existing fitted IK.
  2. Captures a frame from the gripper camera; finds the piece centroid
     via the same OpenCV heuristic used at runtime.
  3. Asks the user to move the piece by exactly +1 square along the file
     axis (e.g. e4 -> f4), captures again, and computes the pixel delta.
  4. Derives `mm_per_pixel` and `board_x_in_image_rad` (the angle the
     board's +file axis makes in the gripper-cam image at this pose).
  5. Optionally repeats along the rank axis for a sanity check.
  6. Writes the result back into data/calibration/board_calibration.json
     under a top-level `cv_assist` block. The chesster execute script
     picks it up when run with --cv-assist.

Usage:
  python scripts/chesster_calibrate_cv_assist.py
  python scripts/chesster_calibrate_cv_assist.py --square e4 --next-file-square f4
  python scripts/chesster_calibrate_cv_assist.py --camera-id 1
  python scripts/chesster_calibrate_cv_assist.py --skip-rank-check

After running, verify with:
  python scripts/chesster_execute_ik_move_cv_assisted.py --cv-assist --manual
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from controls.chesster_arm import ChessterArm
from controls.chesster_kinematics import (
    ChessterKinematics, BoardCalibration, JOINT_NAMES_BY_INDEX,
    JOINT_TO_MOTOR_INDEX,
)
from controls.chesster_cv_assist import (
    capture_frame, detect_piece_centroid, probe_camera,
)
from cameras.gripper_camera import GripperCamera


COUNTS_PER_REV = 4096
ENCODER_MIDPOINT = COUNTS_PER_REV // 2
TRAJ_HZ = 10
MOVE_LATERAL_S = 2.0


# ---------------------------------------------------------------------------
# Small helpers (mirroring chesster_execute_ik_move_cv_assisted.py conventions)
# ---------------------------------------------------------------------------

def default_cal_path() -> str:
    return str(Path(__file__).resolve().parent.parent
               / "data" / "calibration" / "board_calibration.json")


def resolve_port(port: str) -> str:
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if not candidates:
        print("[X] No /dev/ttyACM* devices found", file=sys.stderr)
        sys.exit(1)
    return candidates[0]


def ik_deg_to_raw_rad(degrees: float) -> float:
    counts = ENCODER_MIDPOINT + degrees * (COUNTS_PER_REV / 360.0)
    return counts * (2.0 * np.pi / COUNTS_PER_REV)


def ik_angles_to_raw_5(ik_angles_deg: dict) -> np.ndarray:
    """Convert IK joint-angle dict (midpoint-deg) to a 5-element raw-radian
    array indexed by physical motor. Gripper slot left at 0 (caller fills)."""
    result = np.zeros(5)
    for i, name in enumerate(JOINT_NAMES_BY_INDEX):
        if name == "gripper":
            continue
        if name in ik_angles_deg:
            result[i] = ik_deg_to_raw_rad(ik_angles_deg[name])
    return result


def linear_trajectory(start: np.ndarray, end: np.ndarray,
                      duration_s: float) -> list:
    n = max(1, int(duration_s * TRAJ_HZ))
    return [start + (end - start) * (i + 1) / n for i in range(n)]


def drive_to_joints(arm: ChessterArm, target_5: np.ndarray,
                    duration_s: float) -> bool:
    state = arm.get_state()
    current = state.joint_positions.copy()
    # Preserve current gripper position in the trajectory.
    target = target_5.copy()
    target[1] = current[1]
    waypoints = linear_trajectory(current, target, duration_s)
    arm.pause_state_reader()
    try:
        dt = 1.0 / TRAJ_HZ
        for wp in waypoints:
            arm.move_joints(wp.tolist())
            time.sleep(dt)
        ok = arm.refresh_state_sync()
    finally:
        arm.resume_state_reader()
    return bool(ok)


def drive_to_inactive(arm: ChessterArm, cal: BoardCalibration,
                      duration_s: float = 2.5) -> bool:
    """Drive the arm to the calibrated INACTIVE pose. Returns True on
    successful trajectory completion. Safety-critical on Chesster:
    INACTIVE is the only pose where torque release will NOT drop the
    arm onto the board / chassis."""
    if cal.inactive_rad is None:
        print("[X] No INACTIVE pose in calibration; cannot park safely.",
              file=sys.stderr)
        return False
    target = np.array([cal.inactive_rad[n] for n in JOINT_NAMES_BY_INDEX])
    return drive_to_joints(arm, target, duration_s)


# ---------------------------------------------------------------------------
# Calibration measurement
# ---------------------------------------------------------------------------

def derive_axis_calibration(centroid_a, centroid_b, square_pitch_m: float):
    """Given two centroids representing a one-square displacement along a
    board axis (a -> b), return (mm_per_pixel, axis_angle_in_image_rad)."""
    du = centroid_b[0] - centroid_a[0]
    dv = centroid_b[1] - centroid_a[1]
    pix = float(np.hypot(du, dv))
    if pix < 1e-6:
        raise ValueError("Centroids are identical; piece did not move.")
    mm_per_px = (square_pitch_m * 1000.0) / pix
    # Right-handed image coordinates flip v. Angle (CCW from image +u)
    # that the board axis makes:
    angle = float(np.arctan2(-dv, du))
    return mm_per_px, angle, pix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=str, default="auto",
                   help="Serial port for the arm (default: autodetect ttyACM*)")
    p.add_argument("--camera-id", type=int, default=1,
                   help="Gripper-camera device id (default 1)")
    p.add_argument("--camera-resolution", type=int, nargs=2, default=[640, 480],
                   metavar=("W", "H"), help="Gripper-camera resolution")
    p.add_argument("--cal-path", type=str, default=None,
                   help="Path to board_calibration.json")
    p.add_argument("--square", type=str, default="e4",
                   help="Calibration starting square (default e4). Pick a "
                        "centred square so the +file move stays on the board.")
    p.add_argument("--next-file-square", type=str, default=None,
                   help="Target after the +1 file displacement (default: "
                        "compute as `square` shifted one file toward h).")
    p.add_argument("--next-rank-square", type=str, default=None,
                   help="Target after the +1 rank displacement (default: "
                        "compute as `square` shifted one rank toward 8).")
    p.add_argument("--skip-rank-check", action="store_true",
                   help="Skip the second-axis sanity check.")
    p.add_argument("--roi-px", type=int, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="Centred ROI for centroid detection (default: full frame)")
    p.add_argument("--threshold-mode", choices=["otsu", "adaptive"], default="otsu")
    p.add_argument("--max-correction-mm", type=float, default=25.0,
                   help="Runtime safety cap on CV-assist correction magnitude "
                        "(stored in the calibration block, default 25 mm).")
    p.add_argument("--debug-dir", type=str, default="data/cv_assist_debug",
                   help="Directory to save captured frames + overlays.")
    p.add_argument("--no-autodrive", action="store_true",
                   help="Skip auto-driving to pick-top; assume the arm is "
                        "already positioned. The current pan reading is used "
                        "as calibration_pan_rad.")
    args = p.parse_args()

    cal_path = args.cal_path or default_cal_path()
    if not Path(cal_path).exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return 1

    # Derive auto next-file / next-rank squares if not provided.
    sq = args.square.lower()
    if len(sq) != 2 or sq[0] not in "abcdefgh" or sq[1] not in "12345678":
        print(f"[X] --square must be a chess square (got {args.square!r})",
              file=sys.stderr)
        return 1
    file_idx = ord(sq[0]) - ord("a")
    rank_idx = int(sq[1]) - 1
    next_file = args.next_file_square
    if not next_file:
        if file_idx >= 7:
            print("[X] Cannot auto-derive --next-file-square from file h; "
                  "pass --next-file-square explicitly.", file=sys.stderr)
            return 1
        next_file = chr(ord("a") + file_idx + 1) + str(rank_idx + 1)
    next_rank = args.next_rank_square
    if not next_rank:
        if rank_idx >= 7:
            print("[X] Cannot auto-derive --next-rank-square from rank 8; "
                  "pass --next-rank-square explicitly.", file=sys.stderr)
            return 1
        next_rank = sq[0] + str(rank_idx + 2)

    print("=" * 60)
    print("Chesster CV-assist Calibration")
    print("=" * 60)
    print(f"Calibration file: {cal_path}")
    print(f"Calibration square (starting): {sq}")
    print(f"Next-file square (+1 file):   {next_file}")
    print(f"Next-rank square (+1 rank):   {next_rank}")
    print()

    # Load calibration (needs IK block for auto-drive).
    try:
        kin = ChessterKinematics()
    except Exception as e:
        # Calibration script does not strictly require placo if we skip
        # auto-drive; otherwise IK target computation needs the URDF FK.
        if not args.no_autodrive:
            print(f"[X] Failed to init ChessterKinematics (placo): {e}",
                  file=sys.stderr)
            print("    Re-run with --no-autodrive after manually moving the "
                  "arm to pick-top above the calibration square.",
                  file=sys.stderr)
            return 1
        kin = None
    cal = BoardCalibration.load(cal_path, kin)
    if not cal.has_ik:
        print("[X] Calibration has no `ik` block. Run "
              "scripts/chesster_fit_ik.py first.", file=sys.stderr)
        return 1

    # Open arm + camera.
    port = resolve_port(args.port)
    print(f"[Arm] Connecting to {port}...")
    arm = ChessterArm(port=port, baudrate=1000000)
    if not arm.connect():
        print(f"[X] Failed to connect to arm on {port}", file=sys.stderr)
        return 1
    if not arm.enable_torque():
        print("[X] Failed to enable torque", file=sys.stderr)
        arm.disconnect()
        return 1
    if not arm.refresh_state_sync():
        print("[X] Failed to read initial state", file=sys.stderr)
        arm.release_torque(); arm.disconnect()
        return 1
    print("[Arm] Torqued and ready.")

    print(f"[Cam] Opening gripper camera id={args.camera_id} @ "
          f"{args.camera_resolution}...")
    cam = GripperCamera(camera_id=args.camera_id,
                        resolution=tuple(args.camera_resolution))
    if not cam.start():
        print("[X] Failed to open gripper camera", file=sys.stderr)
        arm.release_torque(); arm.disconnect()
        return 1
    if not probe_camera(cam, args.camera_id):
        print("[X] Gripper camera produced no frames; aborting before motion.",
              file=sys.stderr)
        try: cam.stop()
        except Exception: pass
        arm.release_torque(); arm.disconnect()
        return 1
    print("[Cam] Verified frame capture.")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    detect_cfg = {
        "roi_px": tuple(args.roi_px) if args.roi_px else None,
        "threshold_mode": args.threshold_mode,
        "invert": "auto",
        "blur_kernel": 5,
        "morph_kernel": 5,
        "min_area_px": 200,
        "max_area_px": 80000,
    }

    centroids = {}
    try:
        # 1) Drive to pick-top above the calibration square.
        if not args.no_autodrive:
            try:
                pick_top = cal.get_joint_angles(sq, "top")
            except Exception as e:
                print(f"[X] IK failed for {sq} top: {e}", file=sys.stderr)
                return 1
            target_5 = ik_angles_to_raw_5(pick_top)
            print(f"[Move] Driving to pick-top above {sq}...")
            input("       Press Enter to begin motion (or Ctrl-C to abort): ")
            if not drive_to_joints(arm, target_5, MOVE_LATERAL_S):
                print("[WARN] post-move state read failed; continuing.")
        else:
            print("[Skip] --no-autodrive set; assuming arm already at pick-top.")

        # 2) Record calibration_pan_rad from live state.
        if not arm.refresh_state_sync():
            print("[X] Could not read shoulder_pan", file=sys.stderr)
            return 1
        state = arm.get_state()
        pan_idx = JOINT_TO_MOTOR_INDEX["shoulder_pan"]
        # joint_positions is raw radians (servo native). Convert to URDF
        # angle (after sign + offset) so it matches what chesster_ik uses.
        ik = cal.ik
        pan_offset_deg = ik["encoder_offsets_deg"]["shoulder_pan"]
        pan_sign = ik["joint_signs"]["shoulder_pan"]
        raw_rad = float(state.joint_positions[pan_idx])
        # raw_rad -> midpoint-deg servo
        counts = int(raw_rad / (2.0 * np.pi / COUNTS_PER_REV))
        pan_servo_deg = (counts - ENCODER_MIDPOINT) * (360.0 / COUNTS_PER_REV)
        pan_urdf_deg = pan_sign * pan_servo_deg + pan_offset_deg
        pan_urdf_rad = float(np.radians(pan_urdf_deg))
        print(f"[Pan ] servo={pan_servo_deg:.2f}deg  -> "
              f"urdf={pan_urdf_deg:.2f}deg ({pan_urdf_rad:+.4f} rad)")

        # 3) Capture centroid at each of three positions: sq, next_file, next_rank.
        def measure(label, save_name):
            input(f"\n[Setup] Place a chess piece at the CENTER of {label} "
                  "(remove any other pieces from view). Press Enter when ready: ")
            frame = capture_frame(cam)
            if frame is None:
                raise RuntimeError("No frame from gripper camera")
            centroid = detect_piece_centroid(frame, detect_cfg)
            out_path = debug_dir / f"{save_name}.png"
            overlay = frame.copy()
            if centroid is not None:
                cv2.drawMarker(overlay, (int(centroid[0]), int(centroid[1])),
                               (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                # Draw image center for reference
                h, w = frame.shape[:2]
                cv2.drawMarker(overlay, (w // 2, h // 2),
                               (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
            cv2.imwrite(str(out_path), overlay)
            if centroid is None:
                print(f"[X] No centroid detected. See {out_path} for the frame.")
                raise RuntimeError(f"No centroid for {label}")
            print(f"[OK] {label}: centroid u={centroid[0]:.1f} v={centroid[1]:.1f}  "
                  f"(overlay -> {out_path})")
            return centroid

        c0 = measure(f"{sq} (calibration square)", "01_center")
        c_file = measure(f"{next_file} (one square along file toward h)", "02_file")
        if not args.skip_rank_check:
            c_rank = measure(f"{next_rank} (one square along rank toward 8)",
                             "03_rank")
        else:
            c_rank = None

        # 4) Compute mm_per_pixel and angle.
        pitch_m = float(ik["square_pitch_m"])
        mm_per_px_file, angle_file_rad, pix_file = derive_axis_calibration(
            c0, c_file, pitch_m,
        )
        print()
        print(f"File-axis sample: pixel_delta={pix_file:.1f} px  "
              f"-> mm/px={mm_per_px_file:.3f}  "
              f"angle={np.degrees(angle_file_rad):+.2f}deg")
        results = {"mm_per_pixel": mm_per_px_file,
                   "board_x_in_image_rad": angle_file_rad}

        if c_rank is not None:
            mm_per_px_rank, angle_rank_rad, pix_rank = derive_axis_calibration(
                c0, c_rank, pitch_m,
            )
            print(f"Rank-axis sample: pixel_delta={pix_rank:.1f} px  "
                  f"-> mm/px={mm_per_px_rank:.3f}  "
                  f"angle={np.degrees(angle_rank_rad):+.2f}deg")
            # Sanity: rank should be roughly perpendicular (90 deg CCW from file).
            angle_diff = (angle_rank_rad - angle_file_rad + np.pi) % (2 * np.pi) - np.pi
            print(f"Rank vs file angle diff: {np.degrees(angle_diff):+.2f}deg "
                  f"(expected ~+90deg)")
            diff_mm_per_px = abs(mm_per_px_file - mm_per_px_rank)
            if diff_mm_per_px > 0.2 * max(mm_per_px_file, mm_per_px_rank):
                print(f"[WARN] mm/px disagree across axes by "
                      f"{diff_mm_per_px:.3f} mm/px (>20%); the gripper camera "
                      f"may not be perpendicular to the board, or the piece "
                      f"placements were not on square centres.")
            # Average for the final scalar.
            results["mm_per_pixel"] = (mm_per_px_file + mm_per_px_rank) / 2.0

        # 5) Build and write the cv_assist block.
        block = {
            "version": 1,
            "mm_per_pixel": float(results["mm_per_pixel"]),
            "board_x_in_image_rad": float(angle_file_rad),
            "calibration_pan_rad": float(pan_urdf_rad),
            "calibration_square": sq,
            "next_file_square": next_file,
            "next_rank_square": next_rank if c_rank is not None else None,
            "roi_px": list(args.roi_px) if args.roi_px else None,
            "threshold_mode": args.threshold_mode,
            "invert": "auto",
            "blur_kernel": 5,
            "morph_kernel": 5,
            "min_area_px": 200,
            "max_area_px": 80000,
            "max_correction_mm": float(args.max_correction_mm),
        }
        print()
        print("cv_assist block to write:")
        print(json.dumps(block, indent=2))

        # Merge into JSON (preserve other top-level keys).
        with open(cal_path) as f:
            cal_json = json.load(f)
        cal_json["cv_assist"] = block
        tmp = Path(cal_path).with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cal_json, f, indent=2)
            f.write("\n")
        tmp.rename(cal_path)
        print(f"[OK] Wrote cv_assist block to {cal_path}")
        return 0

    except RuntimeError as e:
        print(f"[X] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[X] Cancelled by user.")
        return 1
    finally:
        # SAFETY-CRITICAL on Chesster: the arm cannot have torque released
        # from arbitrary poses (longer links than SO-101 mean it would drop
        # over the board). Always retreat to INACTIVE before disabling
        # motors. Skip the retreat only if we never enabled torque or the
        # arm is no longer connected.
        try:
            cam.stop()
        except Exception:
            pass
        try:
            if arm.connected:
                print("[Cleanup] Driving arm to INACTIVE before releasing torque...")
                ok = drive_to_inactive(arm, cal)
                if not ok:
                    print("[Cleanup] WARN: INACTIVE retreat reported failure. "
                          "Brace the arm before torque release.")
        except Exception as e:
            print(f"[Cleanup] WARN: failed during INACTIVE retreat: {e}")
        try:
            arm.release_torque()
            arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
