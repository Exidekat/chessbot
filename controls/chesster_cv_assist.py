"""
Chesster CV-assist: gripper-camera-based fine-pickup refinement.

Given the IK system reliably brings the gripper to within ~15-20 mm of a
piece at pick-top, this module closes the residual gap by:
  1. Capturing a frame from the arm-mounted gripper camera.
  2. Detecting the piece centroid in the image via a simple OpenCV
     contour heuristic (no model required).
  3. Translating the pixel offset to a board-frame XY delta using a
     once-calibrated mm/pixel scalar and the angle that the board's
     +file axis makes in the gripper-cam image.
  4. Re-running the closed-form IK with the shifted target so the
     subsequent descend lands on the piece.

The calibration parameters live in the `cv_assist` block of
`data/calibration/board_calibration.json`, written once by
`scripts/chesster_calibrate_cv_assist.py`.

Coordinate conventions:
  - Image coordinates (u, v): pixels with u increasing rightwards
    and v increasing downwards (OpenCV standard).
  - "Right-handed image": (du, -dv) so the y axis points UP in image,
    matching the convention of board-frame xy.
  - `board_x_in_image_rad`: at calibration_pan_rad, the angle (CCW from
    image +u) that the board's +file direction makes in image. As the
    arm pans the camera rotates with the gripper, so the current angle
    is `board_x_in_image_rad - (current_pan_rad - calibration_pan_rad)`.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from controls.chesster_ik_fit import (
    chesster_ik, servo_to_urdf, urdf_to_servo,
)


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_frame(cam, warmup_frames: int = 5, settle_s: float = 0.04,
                  max_wait_s: float = 1.0) -> Optional[np.ndarray]:
    """Pull one stable frame from a GripperCamera-like object.

    Waits up to `max_wait_s` for the first available frame, then drops
    `warmup_frames` more to let auto-exposure / white-balance settle.
    Returns the last frame or None if the camera never produced one.
    """
    deadline = time.time() + max_wait_s
    last = None
    while time.time() < deadline:
        f = cam.get_frame()
        if f is not None:
            last = f
            break
        time.sleep(0.02)
    if last is None:
        return None
    for _ in range(warmup_frames):
        time.sleep(settle_s)
        f = cam.get_frame()
        if f is not None:
            last = f
    return last


# ---------------------------------------------------------------------------
# Centroid detection
# ---------------------------------------------------------------------------

def _default_cfg_with(cfg: dict) -> dict:
    """Merge cfg over the module defaults so callers can override piecewise."""
    out = {
        "roi_px": None,             # None -> use full frame
        "threshold_mode": "otsu",   # or "adaptive"
        "invert": "auto",           # True / False / "auto" (foreground = white)
        "blur_kernel": 5,
        "morph_kernel": 5,
        "min_area_px": 200,
        "max_area_px": 80000,
        "adaptive_block": 51,
        "adaptive_c": 5,
    }
    out.update(cfg or {})
    return out


def detect_piece_centroid(frame: np.ndarray, cfg: dict
                          ) -> Optional[Tuple[float, float]]:
    """Find the centroid of the most likely piece in the gripper-cam image.

    Strategy: centre-crop to a configurable ROI (so neighbouring squares
    don't confuse the threshold), greyscale + blur, threshold (OTSU or
    adaptive), morphological close to consolidate the piece blob, then
    pick the contour whose centroid is closest to the FRAME centre
    (since IK has already aimed roughly at the target).

    Returns the centroid (u, v) in ORIGINAL frame coordinates, or None
    if no qualifying contour was found.
    """
    cfg = _default_cfg_with(cfg)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    roi_px = cfg["roi_px"]
    if roi_px:
        rw, rh = int(roi_px[0]), int(roi_px[1])
        x0 = max(0, (w - rw) // 2)
        y0 = max(0, (h - rh) // 2)
        x1 = min(w, x0 + rw)
        y1 = min(h, y0 + rh)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    k = int(cfg["blur_kernel"])
    if k > 1:
        if k % 2 == 0:
            k += 1
        roi = cv2.GaussianBlur(roi, (k, k), 0)

    if cfg["threshold_mode"] == "otsu":
        _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block = int(cfg["adaptive_block"])
        if block % 2 == 0:
            block += 1
        th = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, int(cfg["adaptive_c"]),
        )

    invert = cfg["invert"]
    if invert is True:
        th = 255 - th
    elif invert == "auto":
        # Foreground (piece) should be the smaller-area class -- one piece
        # on a square is small relative to the background. If white pixels
        # dominate, invert so the piece becomes the white blob.
        if th.size and th.mean() > 127:
            th = 255 - th

    mk = int(cfg["morph_kernel"])
    if mk > 1:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kern)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_a = float(cfg["min_area_px"])
    max_a = float(cfg["max_area_px"])
    cx_frame = w / 2.0
    cy_frame = h / 2.0
    candidates = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cu = m["m10"] / m["m00"] + x0
        cv_ = m["m01"] / m["m00"] + y0
        d2 = (cu - cx_frame) ** 2 + (cv_ - cy_frame) ** 2
        candidates.append((d2, cu, cv_, a))

    if not candidates:
        return None

    candidates.sort()  # closest-to-centre first
    _, cu, cv_, _ = candidates[0]
    return (float(cu), float(cv_))


# ---------------------------------------------------------------------------
# Pixel -> board-frame conversion
# ---------------------------------------------------------------------------

def pixel_offset_to_board_delta(
    centroid_uv: Tuple[float, float],
    frame_shape: Tuple[int, int],
    cv_cfg: dict,
    current_pan_rad: float,
) -> Tuple[float, float]:
    """Convert a pixel centroid into a (dx, dy) board-frame XY in meters.

    Required cv_cfg keys: `mm_per_pixel`, `board_x_in_image_rad`,
    `calibration_pan_rad`.
    """
    h, w = frame_shape[:2]
    du = centroid_uv[0] - w / 2.0
    dv = centroid_uv[1] - h / 2.0

    mm_per_px = float(cv_cfg["mm_per_pixel"])
    dx_img_m = du * mm_per_px * 1e-3
    dy_img_m = (-dv) * mm_per_px * 1e-3  # flip v to right-handed image y

    angle_calib = float(cv_cfg["board_x_in_image_rad"])
    pan_calib = float(cv_cfg["calibration_pan_rad"])
    # As pan rotates the gripper, the camera image rotates with it, so
    # board+x appears at a smaller CCW angle in image at higher pan.
    angle_now = angle_calib - (current_pan_rad - pan_calib)

    c = float(np.cos(angle_now))
    s = float(np.sin(angle_now))
    # Inverse rotation: from image axes to board axes.
    dx_board = dx_img_m * c + dy_img_m * s
    dy_board = -dx_img_m * s + dy_img_m * c
    return float(dx_board), float(dy_board)


# ---------------------------------------------------------------------------
# Refined IK
# ---------------------------------------------------------------------------

def compute_refined_servo_angles(
    cal,
    square: str,
    plane: str,
    board_delta_m: Tuple[float, float],
    seed_servo: Optional[dict] = None,
) -> dict:
    """Mirror `BoardCalibration._joint_angles_via_ik` but inject an extra
    board-frame XY offset into the IK target before solving."""
    ik = cal.ik
    if ik is None:
        raise ValueError("Calibration has no fitted ik block")
    offsets = ik["encoder_offsets_deg"]
    signs = ik["joint_signs"]
    T = np.array(ik["board_pose_in_robot"])
    pitch = float(ik["square_pitch_m"])
    top_h = float(ik.get("top_plane_height_m", 0.10))
    tcp_off = np.array(ik.get("tcp_offset_board_m", [0.0, 0.0, 0.0]))

    file_idx, rank_idx = cal._parse_square(square)
    z_board = top_h if plane == "top" else 0.0
    p_board = np.array([
        file_idx * pitch + board_delta_m[0],
        rank_idx * pitch + board_delta_m[1],
        z_board,
    ]) + tcp_off
    p_robot = T[:3, :3] @ p_board + T[:3, 3]

    # Mirror BoardCalibration._joint_angles_via_ik: if no caller seed,
    # fall back to the nearest-anchor recorded pose so the IK picks
    # Chesster's backward-reaching branch instead of pan_forward default.
    if seed_servo is None:
        seed_servo = cal._nearest_anchor_joints(square, plane)
    seed_urdf = servo_to_urdf(seed_servo, offsets, signs)
    j_urdf = chesster_ik(p_robot, seed_joints_deg=seed_urdf)
    j_servo = urdf_to_servo(j_urdf, offsets, signs)
    return {k: round(float(v), 3) for k, v in j_servo.items()}


# ---------------------------------------------------------------------------
# Top-level refine entry
# ---------------------------------------------------------------------------

def cv_assist_refine(
    cal,
    cam,
    square: str,
    bottom_plane: str = "bottom",
    seed_servo: Optional[dict] = None,
    current_pan_rad: float = 0.0,
    debug_frame_path: Optional[str] = None,
) -> Tuple[Optional[dict], dict]:
    """Run the full capture + detect + refine pipeline for one approach.

    Returns:
        (refined_servo_dict, info_dict)
            refined_servo_dict: refined servo-frame joint angles for
                (square, bottom_plane), or None on any failure / out-of-bounds.
            info_dict: diagnostic info -- always populated. Keys:
                'reason'        : str (failure reason, if any)
                'centroid_uv'   : (u, v) detected or None
                'board_delta_m' : (dx, dy) board-frame delta or None
                'delta_mm'      : magnitude in mm if computed
                'frame_shape'   : (H, W, C) if a frame was captured
    """
    cv_cfg = getattr(cal, "cv_assist", None)
    info = {"reason": None, "centroid_uv": None, "board_delta_m": None,
            "delta_mm": None, "frame_shape": None}

    if not cv_cfg:
        info["reason"] = "no cv_assist block in calibration"
        return None, info

    frame = capture_frame(cam)
    if frame is None:
        info["reason"] = "no frame from gripper camera"
        return None, info
    info["frame_shape"] = frame.shape

    if debug_frame_path:
        try:
            cv2.imwrite(str(debug_frame_path), frame)
        except Exception:
            pass

    centroid = detect_piece_centroid(frame, cv_cfg)
    info["centroid_uv"] = centroid
    if centroid is None:
        info["reason"] = "no piece centroid detected"
        return None, info

    dx, dy = pixel_offset_to_board_delta(
        centroid, frame.shape, cv_cfg, current_pan_rad,
    )
    info["board_delta_m"] = (dx, dy)
    delta_mm = float(np.hypot(dx, dy) * 1000.0)
    info["delta_mm"] = delta_mm

    max_corr_mm = float(cv_cfg.get("max_correction_mm", 25.0))
    if delta_mm > max_corr_mm:
        info["reason"] = (
            f"correction {delta_mm:.1f}mm exceeds max {max_corr_mm:.1f}mm"
        )
        return None, info

    try:
        refined = compute_refined_servo_angles(
            cal, square, bottom_plane, (dx, dy), seed_servo,
        )
    except Exception as e:
        info["reason"] = f"re-IK failed: {e}"
        return None, info

    return refined, info
