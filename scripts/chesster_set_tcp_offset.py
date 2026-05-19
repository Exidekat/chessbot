#!/usr/bin/env python3
"""
Chesster TCP Offset Calibration

Sets `ik.tcp_offset_board_m` in `data/calibration/board_calibration.json`
to compensate for a systematic bias between where the calibration data
implies the gripper TCP is and where the bot actually grasps a piece.

After running the IK fit, you may observe the arm landing consistently
off-target by a fraction of a square -- e.g. asking for f5 but landing
between f5 and f6. That happens when the URDF TCP and the effective
gripping point don't coincide, or when the gripper-tip placement during
calibration didn't match the square center. This tool lets you correct
for that bias with one shifted-target setting in board-frame meters.

You can specify the offset two ways:

1) Directly, in millimetres (board-frame x, y, z):
       python scripts/chesster_set_tcp_offset.py --dx-mm 0 --dy-mm -22 --dz-mm 0

   `+y` is the direction from rank 1 toward rank 8. If asked-for f5 lands at
   f5/f6 boundary (i.e. ~22 mm too far toward rank 8), set --dy-mm -22.

2) From observed (asked square, observed square) pairs. The tool averages
   the per-pair board-frame correction:
       python scripts/chesster_set_tcp_offset.py \\
            --observation "asked=f5 actual=f5_5" \\
            --observation "asked=e5 actual=e5_5"

   where `actual=f5_5` means halfway between f5 and f6 along rank, and
   files can use `f_g` for halfway between f and g.

3) Reset to zero:
       python scripts/chesster_set_tcp_offset.py --reset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def default_path() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    return str(repo_root / "data" / "calibration" / "board_calibration.json")


def parse_actual(text: str) -> tuple:
    """Parse 'actual=...' values like 'f5', 'f5_6', 'f_g_5', 'f_g_5_6'.

    Files: 'a'..'h' -> 0..7. Fractional file uses 'X_Y' for halfway
    between files X and Y (e.g. 'd_e' -> 3.5).
    Ranks: '1'..'8' -> 0..7. Fractional rank uses 'N_M' (e.g. '5_6' -> 4.5).
    """
    text = text.strip().lower()
    # Split at the first digit: everything before is the file part,
    # everything from there is the rank part.
    split = re.search(r"\d", text)
    if not split:
        raise ValueError(f"No rank in actual position: {text!r}")
    file_part = text[:split.start()].rstrip("_")
    rank_part = text[split.start():]

    # File part: 'f' or 'f_g'.
    files = [f for f in file_part.split("_") if f]
    if not files or not all(re.match(r"^[a-h]$", f) for f in files):
        raise ValueError(f"Bad file part {file_part!r} in {text!r}")
    file_idx = sum(ord(f) - ord("a") for f in files) / len(files)

    # Rank part: '5' or '5_6'.
    ranks = [r for r in rank_part.split("_") if r]
    if not ranks or not all(re.match(r"^[1-8]$", r) for r in ranks):
        raise ValueError(f"Bad rank part {rank_part!r} in {text!r}")
    rank_idx = sum(int(r) - 1 for r in ranks) / len(ranks)

    return (float(file_idx), float(rank_idx))


def parse_asked(text: str) -> tuple:
    text = text.strip().lower()
    m = re.match(r"^([a-h])(\d)$", text)
    if not m:
        raise ValueError(f"Cannot parse asked square: {text!r} (expected e.g. f5)")
    return (ord(m.group(1)) - ord("a"), int(m.group(2)) - 1)


def parse_observation(obs: str) -> tuple:
    """Parse 'asked=f5 actual=f5_5' -> ((asked_file, asked_rank), (actual_file, actual_rank))."""
    parts = dict(p.split("=", 1) for p in obs.split() if "=" in p)
    if "asked" not in parts or "actual" not in parts:
        raise ValueError(
            f"Observation must include 'asked=XX' and 'actual=YY': {obs!r}"
        )
    return parse_asked(parts["asked"]), parse_actual(parts["actual"])


def main():
    p = argparse.ArgumentParser(
        description="Set tcp_offset_board_m in the IK calibration block.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cal-path", default=None,
                   help="Path to board_calibration.json")
    p.add_argument("--dx-mm", type=float, default=None,
                   help="Direct x offset in board-frame millimetres "
                        "(along file axis: +x toward h).")
    p.add_argument("--dy-mm", type=float, default=None,
                   help="Direct y offset in board-frame millimetres "
                        "(along rank axis: +y toward rank 8).")
    p.add_argument("--dz-mm", type=float, default=None,
                   help="Direct z offset in board-frame millimetres "
                        "(out of board plane: +z upward).")
    p.add_argument("--observation", action="append", default=[],
                   help="Pair 'asked=XX actual=YY' (repeatable).")
    p.add_argument("--reset", action="store_true",
                   help="Set tcp_offset_board_m to [0, 0, 0].")
    args = p.parse_args()

    cal_path = args.cal_path or default_path()
    if not Path(cal_path).exists():
        print(f"[X] Calibration file not found: {cal_path}", file=sys.stderr)
        return 1

    with open(cal_path) as f:
        cal = json.load(f)

    if "ik" not in cal:
        print("[X] Calibration has no 'ik' block. Run "
              "scripts/chesster_fit_ik.py first.", file=sys.stderr)
        return 1
    pitch = float(cal["ik"]["square_pitch_m"])

    prev = cal["ik"].get("tcp_offset_board_m") or [0.0, 0.0, 0.0]
    print(f"Current tcp_offset_board_m: "
          f"[{prev[0]*1000:+.1f}, {prev[1]*1000:+.1f}, {prev[2]*1000:+.1f}] mm")

    if args.reset:
        new = [0.0, 0.0, 0.0]
    elif args.observation:
        # Average board-frame deltas. For each observation, the gripper
        # landed at `actual` when we asked for `asked` -- so the systematic
        # error is (actual - asked) in board-frame square units. To
        # compensate, we OFFSET the target by -(actual - asked) so the
        # eventual landing matches the asked square.
        dx_units, dy_units = 0.0, 0.0
        for obs in args.observation:
            (af, ar), (xf, xr) = parse_observation(obs)
            dx_units += (xf - af)
            dy_units += (xr - ar)
        n = len(args.observation)
        avg_dx = dx_units / n
        avg_dy = dy_units / n
        # Correction is opposite direction in board-frame metres.
        new = [
            float(prev[0] - avg_dx * pitch),
            float(prev[1] - avg_dy * pitch),
            float(prev[2]),
        ]
        print(f"\nObservations: {n}")
        print(f"  Average asked->actual delta (square units): "
              f"({avg_dx:+.3f}, {avg_dy:+.3f})")
        print(f"  Correction (subtracted from target): "
              f"({avg_dx*pitch*1000:+.1f}, {avg_dy*pitch*1000:+.1f}) mm")
    else:
        # Direct millimetre overrides; defaults preserve any unspecified axis.
        new = list(prev)
        if args.dx_mm is not None:
            new[0] = args.dx_mm / 1000.0
        if args.dy_mm is not None:
            new[1] = args.dy_mm / 1000.0
        if args.dz_mm is not None:
            new[2] = args.dz_mm / 1000.0
        if new == prev:
            print("\nNo change requested. Pass --dx-mm / --dy-mm / --dz-mm, "
                  "--observation, or --reset.")
            return 0

    print(f"New tcp_offset_board_m:    "
          f"[{new[0]*1000:+.1f}, {new[1]*1000:+.1f}, {new[2]*1000:+.1f}] mm")

    cal["ik"]["tcp_offset_board_m"] = new

    tmp = Path(cal_path).with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(cal, f, indent=2)
        f.write("\n")
    tmp.rename(cal_path)
    print(f"[OK] Wrote tcp_offset_board_m to {cal_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
