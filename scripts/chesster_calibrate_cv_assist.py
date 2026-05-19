#!/usr/bin/env python3
"""
Chesster CV-assist Calibration

One-time setup for gripper-camera-based fine pickup refinement.

What it does:
  1. Drives the Chesster arm to pick-top above a chosen calibration
     square (default e4) using the existing fitted IK.
  2. Captures a frame from the gripper camera; finds the piece centroid
     via the same OpenCV heuristic used at runtime.
  3. Asks the user to move the piece by N squares along the file axis
     (default N=2, i.e. e4 -> g4, to keep the piece on the same-colour
     square); captures and computes the pixel delta. N is inferred from
     --square / --next-file-square; must be a pure-file move.
  4. Repeats along the rank axis (default e4 -> e6) for the perpendicular
     calibration; the script refuses to write the block if the two
     pixel axes aren't perpendicular within +/-20 deg.
  5. Derives `mm_per_pixel` and `board_x_in_image_rad` (the angle the
     board's +file axis makes in the gripper-cam image at this pose),
     scaling mm_per_pixel by the actual displacement in squares.
  6. Writes the result back into data/calibration/board_calibration.json
     under a top-level `cv_assist` block. The chesster execute script
     picks it up when run with --cv-assist.

Usage:
  # Default: --square e4 plus two-square same-colour displacements
  # (e4 -> g4 along file, e4 -> e6 along rank). The "+2 squares" default
  # keeps the piece on the same-colour square at both ends of each move,
  # which is critical for thresholding -- a piece on a dark square gives
  # very low local contrast and gets merged into the square by both OTSU
  # and adaptive thresholding.
  python scripts/chesster_calibrate_cv_assist.py

  # Custom single-square displacement (the original behaviour). Only
  # useful if your gripper-cam has high local contrast on dark squares
  # and you don't run into the piece-merged-with-dark-square failure.
  python scripts/chesster_calibrate_cv_assist.py \
      --square e4 --next-file-square f4 --next-rank-square e5

  # Adaptive threshold + centred ROI, recommended for cluttered fields
  # of view (gripper hardware visible in frame, half-board in shot, etc.)
  python scripts/chesster_calibrate_cv_assist.py \
      --threshold-mode adaptive --roi-px 200 200

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
    _default_cfg_with,
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

def _all_candidate_contours(frame: np.ndarray, cfg: dict) -> list:
    """Re-run the same threshold pipeline as detect_piece_centroid and
    return ALL contours that survive the area filter (sorted by distance
    from frame centre, then area descending). Used to let the user pick
    when auto-selection lands on the wrong contour."""
    cfg = _default_cfg_with(cfg)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    roi_px = cfg.get("roi_px")
    if roi_px:
        rw, rh = int(roi_px[0]), int(roi_px[1])
        x0 = max(0, (w - rw) // 2); y0 = max(0, (h - rh) // 2)
        x1 = min(w, x0 + rw); y1 = min(h, y0 + rh)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    k = int(cfg["blur_kernel"])
    if k > 1:
        if k % 2 == 0: k += 1
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    if cfg["threshold_mode"] == "otsu":
        _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        b = int(cfg["adaptive_block"])
        if b % 2 == 0: b += 1
        th = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, b, int(cfg["adaptive_c"]))
    inv = cfg["invert"]
    if inv is True or (inv == "auto" and th.size and th.mean() > 127):
        th = 255 - th
    mk = int(cfg["morph_kernel"])
    if mk > 1:
        th = cv2.morphologyEx(
            th, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk)),
        )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_a, max_a = float(cfg["min_area_px"]), float(cfg["max_area_px"])
    out = []
    cx_f, cy_f = w / 2.0, h / 2.0
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cu = m["m10"]/m["m00"] + x0
        cv_ = m["m01"]/m["m00"] + y0
        # Sample underlying brightness in a 30x30 patch (helps the user
        # distinguish piece (dark) from gripper hardware (lighter).
        patch = frame[max(0, int(cv_)-15):int(cv_)+15,
                       max(0, int(cu)-15):int(cu)+15]
        b = float(patch.mean()) if patch.size else 0.0
        d = ((cu - cx_f)**2 + (cv_ - cy_f)**2) ** 0.5
        out.append((a, cu, cv_, b, d))
    out.sort(key=lambda r: (r[4], -r[0]))  # distance asc, area desc
    return out


def prompt_user_centroid(frame: np.ndarray, cfg: dict, label: str,
                         debug_path: Path) -> tuple:
    """Detect and ask the user to confirm the centroid. Returns (u, v).

    Prints the top contour candidates with area, centroid, underlying
    patch brightness, and distance from frame centre. The user can:
      - press Enter to accept the auto-pick (top-of-list)
      - type a candidate index (e.g. "1") to pick a different contour
      - type "u,v" (e.g. "349,301") to enter pixel coords directly
      - type "r" to retry (skip the prompt and recapture later)
    Saves an overlay PNG marking ALL candidates so the user can confirm
    visually (scp the file, view it).
    """
    candidates = _all_candidate_contours(frame, cfg)
    h, w = frame.shape[:2]

    # Annotated overlay with EVERY candidate labelled, plus a coordinate
    # grid so the user can read pixel coords by eye without a measuring
    # tool. Grid lines every 50 px with labels at the top and left edges.
    overlay = frame.copy()
    # Light grid
    grid_color = (60, 60, 60)
    for gx in range(0, w, 50):
        cv2.line(overlay, (gx, 0), (gx, h), grid_color, 1)
        cv2.putText(overlay, str(gx), (gx + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    for gy in range(0, h, 50):
        cv2.line(overlay, (0, gy), (w, gy), grid_color, 1)
        cv2.putText(overlay, str(gy), (2, gy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    # Image centre as a red diagonal cross
    cv2.drawMarker(overlay, (w // 2, h // 2), (0, 0, 255),
                    cv2.MARKER_TILTED_CROSS, 30, 2)
    for idx, (a, cu, cv_, b, d) in enumerate(candidates):
        color = (0, 255, 0) if idx == 0 else (0, 200, 255)
        cv2.drawMarker(overlay, (int(cu), int(cv_)), color,
                        cv2.MARKER_CROSS, 22, 2)
        cv2.putText(overlay, f"#{idx} a={int(a)}", (int(cu)+10, int(cv_)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(overlay, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(debug_path), overlay)

    print(f"  [{label}] overlay -> {debug_path}")
    if not candidates:
        print("  [WARN] No contours passed the area filter "
              f"({cfg.get('min_area_px',200)}-{cfg.get('max_area_px',80000)} px). "
              "Try a different --threshold-mode, --min-area-px or "
              "--max-area-px, or enter pixel coords manually.")
    else:
        print(f"  Candidates (sorted by distance-from-centre):")
        print(f"    {'idx':<4s} {'area_px':<10s} {'centroid':<18s} "
              f"{'bg_bright':<10s} {'dist_centre'}")
        for idx, (a, cu, cv_, b, d) in enumerate(candidates):
            print(f"    {idx:<4d} {a:<10.0f} ({cu:6.1f},{cv_:6.1f})   "
                  f"{b:<10.1f} {d:.1f}")

    while True:
        resp = input(
            "  Press Enter to accept #0, type a candidate index "
            "(0,1,2,...), or type pixel coords 'u,v' to override: "
        ).strip()
        if resp == "" and candidates:
            a, cu, cv_, b, d = candidates[0]
            return (float(cu), float(cv_))
        if resp.isdigit():
            i = int(resp)
            if 0 <= i < len(candidates):
                a, cu, cv_, b, d = candidates[i]
                return (float(cu), float(cv_))
            print(f"  [X] Index {i} out of range; please retry.")
            continue
        if "," in resp:
            try:
                u_str, v_str = resp.split(",", 1)
                u = float(u_str.strip()); v = float(v_str.strip())
                if not (0 <= u < w and 0 <= v < h):
                    print(f"  [X] ({u},{v}) outside frame {w}x{h}; please retry.")
                    continue
                return (u, v)
            except ValueError:
                print("  [X] could not parse 'u,v'; please retry.")
                continue
        print("  [X] unrecognised input; press Enter, or type an index, "
              "or 'u,v' coords.")


def derive_axis_calibration(centroid_a, centroid_b, square_pitch_m: float,
                            n_squares: float = 1.0):
    """Given two centroids that represent a known board-frame displacement
    of `n_squares` square-pitches along an axis (file or rank), return
    (mm_per_pixel, axis_angle_in_image_rad, pixel_distance).

    n_squares > 1 is encouraged because adjacent-square displacements
    keep the piece on alternating-colour squares (chessboard parity),
    which defeats simple thresholding. Two-square displacements along
    file/rank keep the piece on a SAME-colour square at both ends.
    """
    du = centroid_b[0] - centroid_a[0]
    dv = centroid_b[1] - centroid_a[1]
    pix = float(np.hypot(du, dv))
    if pix < 1e-6:
        raise ValueError("Centroids are identical; piece did not move.")
    mm_per_px = (n_squares * square_pitch_m * 1000.0) / pix
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
    p.add_argument("--camera-id", type=int, default=0,
                   help="Gripper-camera device id (default 0, the Innomaker "
                        "U20CAM capture endpoint = /dev/video0; id 1 is the "
                        "matching metadata endpoint and won't return live "
                        "frames). Try v4l2-ctl --list-devices to confirm.")
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
    p.add_argument("--min-area-px", type=int, default=200,
                   help="Lower bound on detected piece contour area (px^2). "
                        "Lower this if the piece's contour is small at this "
                        "camera distance (e.g. small pieces or distant pose).")
    p.add_argument("--max-area-px", type=int, default=8000,
                   help="Upper bound on detected piece contour area (px^2). "
                        "Tune below the area of any STATIC feature in view "
                        "(gripper hardware, full board half, etc.) so the "
                        "detector can't latch onto it. Inspect "
                        "data/cv_assist_debug/0[123]_*.png to see candidate "
                        "areas in the printed list. Default 8000 excludes "
                        "the typical gripper-jaw blob seen on this bench.")
    p.add_argument("--max-correction-mm", type=float, default=25.0,
                   help="Runtime safety cap on CV-assist correction magnitude "
                        "(stored in the calibration block, default 25 mm).")
    p.add_argument("--allow-bad-geometry", action="store_true",
                   help="Skip the rank-vs-file perpendicularity check (a "
                        "calibration where file and rank pixel axes are far "
                        "from perpendicular implies a bad capture; the "
                        "script aborts without writing the block unless "
                        "this flag is passed).")
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
    # Default to TWO squares away (e.g. e4 -> g4 and e4 -> e6) so the
    # piece stays on the same-colour square -- alternating-colour squares
    # defeat global / adaptive thresholding when the piece sits on a
    # dark square and has little local contrast.
    DEFAULT_SQ_STEP = 2
    sq = args.square.lower()
    if len(sq) != 2 or sq[0] not in "abcdefgh" or sq[1] not in "12345678":
        print(f"[X] --square must be a chess square (got {args.square!r})",
              file=sys.stderr)
        return 1
    file_idx = ord(sq[0]) - ord("a")
    rank_idx = int(sq[1]) - 1

    def parse_sq(name):
        s = name.lower()
        if len(s) != 2 or s[0] not in "abcdefgh" or s[1] not in "12345678":
            raise ValueError(f"Invalid square {name!r}")
        return ord(s[0]) - ord("a"), int(s[1]) - 1

    next_file = args.next_file_square
    if not next_file:
        nf = file_idx + DEFAULT_SQ_STEP
        if nf > 7:
            nf = file_idx - DEFAULT_SQ_STEP
            if nf < 0:
                print("[X] Cannot auto-derive --next-file-square: --square "
                      "too close to file edge for the default 2-square step.",
                      file=sys.stderr)
                return 1
        next_file = chr(ord("a") + nf) + str(rank_idx + 1)
    next_rank = args.next_rank_square
    if not next_rank:
        nr = rank_idx + DEFAULT_SQ_STEP
        if nr > 7:
            nr = rank_idx - DEFAULT_SQ_STEP
            if nr < 0:
                print("[X] Cannot auto-derive --next-rank-square: --square "
                      "too close to rank edge for the default 2-square step.",
                      file=sys.stderr)
                return 1
        next_rank = sq[0] + str(nr + 1)

    # Validate the moves are pure file / pure rank.
    try:
        nf_file, nf_rank = parse_sq(next_file)
        nr_file, nr_rank = parse_sq(next_rank)
    except ValueError as e:
        print(f"[X] {e}", file=sys.stderr); return 1
    if nf_rank != rank_idx:
        print(f"[X] --next-file-square {next_file!r} is on rank "
              f"{nf_rank+1}, not the same rank as --square {sq!r} "
              f"(rank {rank_idx+1}). The file calibration move must be "
              f"PURE file (rank unchanged).", file=sys.stderr)
        return 1
    if nr_file != file_idx:
        print(f"[X] --next-rank-square {next_rank!r} is on file "
              f"{chr(ord('a')+nr_file)}, not the same file as --square "
              f"{sq!r} (file {sq[0]}). The rank calibration move must "
              f"be PURE rank (file unchanged).", file=sys.stderr)
        return 1
    file_n_squares = abs(nf_file - file_idx)
    rank_n_squares = abs(nr_rank - rank_idx)
    if file_n_squares < 1 or rank_n_squares < 1:
        print(f"[X] file/rank displacements must be >=1 square "
              f"(got file={file_n_squares}, rank={rank_n_squares}).",
              file=sys.stderr)
        return 1

    # Warn if the displacement keeps the piece on alternating-colour
    # squares. With (file+rank) parity, a 1-square move flips colour;
    # a 2-square move preserves it.
    def square_colour(fi, ri):
        return "dark" if (fi + ri) % 2 == 0 else "light"
    sq_colour = square_colour(file_idx, rank_idx)
    nf_colour = square_colour(nf_file, nf_rank)
    nr_colour = square_colour(nr_file, nr_rank)
    if nf_colour != sq_colour or nr_colour != sq_colour:
        print(f"[WARN] Some calibration squares are different colours "
              f"({sq}={sq_colour}, {next_file}={nf_colour}, "
              f"{next_rank}={nr_colour}). Adaptive/OTSU thresholding may "
              f"fail to separate piece from dark squares. Prefer a "
              f"2-square displacement on same-colour squares.")

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
        "min_area_px": int(args.min_area_px),
        "max_area_px": int(args.max_area_px),
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
        # The user picks the right candidate (or enters pixel coords) per
        # capture -- auto-detection can lock onto static features like
        # gripper hardware visible in the FOV.
        def measure(label, save_name):
            input(f"\n[Setup] Place a chess piece at the CENTER of {label} "
                  "(remove any other pieces from view). Press Enter when ready: ")
            frame = capture_frame(cam)
            if frame is None:
                raise RuntimeError("No frame from gripper camera")
            out_path = debug_dir / f"{save_name}.png"
            centroid = prompt_user_centroid(frame, detect_cfg, label, out_path)
            print(f"  [OK] {label}: chosen centroid u={centroid[0]:.1f} "
                  f"v={centroid[1]:.1f}")
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
            c0, c_file, pitch_m, n_squares=file_n_squares,
        )
        print()
        print(f"File-axis sample ({file_n_squares} square(s) displacement): "
              f"pixel_delta={pix_file:.1f} px  "
              f"-> mm/px={mm_per_px_file:.3f}  "
              f"angle={np.degrees(angle_file_rad):+.2f}deg")
        results = {"mm_per_pixel": mm_per_px_file,
                   "board_x_in_image_rad": angle_file_rad}

        if c_rank is not None:
            mm_per_px_rank, angle_rank_rad, pix_rank = derive_axis_calibration(
                c0, c_rank, pitch_m, n_squares=rank_n_squares,
            )
            print(f"Rank-axis sample ({rank_n_squares} square(s) displacement): "
                  f"pixel_delta={pix_rank:.1f} px  "
                  f"-> mm/px={mm_per_px_rank:.3f}  "
                  f"angle={np.degrees(angle_rank_rad):+.2f}deg")
            # Sanity: rank should be roughly perpendicular (90 deg CCW from file).
            angle_diff = (angle_rank_rad - angle_file_rad + np.pi) % (2 * np.pi) - np.pi
            angle_diff_deg = np.degrees(angle_diff)
            print(f"Rank vs file angle diff: {angle_diff_deg:+.2f}deg "
                  f"(expected ~+90deg)")
            diff_mm_per_px = abs(mm_per_px_file - mm_per_px_rank)
            if diff_mm_per_px > 0.2 * max(mm_per_px_file, mm_per_px_rank):
                print(f"[WARN] mm/px disagree across axes by "
                      f"{diff_mm_per_px:.3f} mm/px (>20%); the gripper camera "
                      f"may not be perpendicular to the board, or the piece "
                      f"placements were not on square centres.")
            # Hard gate: if the file and rank pixel axes are far from
            # perpendicular, something is wrong with the capture (most
            # likely the centroid latched onto a non-piece feature in one
            # of the three frames, or the piece was placed at the wrong
            # square). Refuse to write the cv_assist block.
            if abs(abs(angle_diff_deg) - 90.0) > 20.0:
                msg = (f"Rank vs file angle diff is "
                       f"{angle_diff_deg:+.2f}deg, expected ~+90deg "
                       f"(tolerance +-20deg). The two pixel deltas are "
                       f"nearly parallel, which means the piece did NOT "
                       f"move in two independent directions across the "
                       f"three captures.\n"
                       f"  Look at data/cv_assist_debug/0[123]_*.png: "
                       f"is the green crosshair on the actual piece in "
                       f"each frame, and do the three frames clearly show "
                       f"the piece at three different board positions?\n"
                       f"  If yes, the camera optics likely need a richer "
                       f"mapping than a single mm/px scalar -- raise an "
                       f"issue. If no, re-run after fixing centroid "
                       f"detection (try --threshold-mode adaptive, lower "
                       f"--min-area-px, or smaller --roi-px to crop "
                       f"out non-board content).")
                if args.allow_bad_geometry:
                    print(f"[WARN] {msg}")
                    print("[WARN] --allow-bad-geometry set; writing anyway.")
                else:
                    print(f"[X] {msg}", file=sys.stderr)
                    print("[X] Refusing to write cv_assist block. "
                          "Pass --allow-bad-geometry to override.",
                          file=sys.stderr)
                    return 1
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
            "min_area_px": int(args.min_area_px),
            "max_area_px": int(args.max_area_px),
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
