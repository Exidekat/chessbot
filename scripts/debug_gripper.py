#!/usr/bin/env python3
"""
Gripper Debug Script

Diagnoses gripper motor issues by directly commanding motor 6 over serial.

Usage:
    python scripts/debug_gripper.py
    python scripts/debug_gripper.py --port /dev/ttyACM1
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from controls.so100_arm import SO100Arm

GRIPPER_MOTOR_ID = 6
COUNTS_PER_REV = 4096

# Register addresses
REG_GOAL_POSITION = 0x2A
REG_SPEED = 0x2E
REG_ACCEL = 0x29
REG_TORQUE_LIMIT = 0x30
REG_TORQUE_ENABLE = 0x28
REG_POSITION = 0x38


def write_reg(arm, motor_id, reg, value, nbytes=2):
    data = [motor_id, 0x04 if nbytes == 1 else 0x05, arm.INSTR_WRITE, reg]
    if nbytes == 1:
        data.append(value & 0xFF)
    else:
        data += [value & 0xFF, (value >> 8) & 0xFF]
    cs = arm._calculate_checksum(data)
    arm.serial.write(bytes([*arm.HEADER] + data + [cs]))
    time.sleep(0.005)


def read_gripper_counts(arm):
    """Read gripper position using SO100Arm's proven method."""
    pos = arm._read_motor_position(GRIPPER_MOTOR_ID)
    return pos


def resolve_port(port):
    if port != "auto":
        return port
    candidates = sorted(glob.glob("/dev/ttyACM*"))
    if candidates:
        return candidates[0]
    print("[X] No /dev/ttyACM* found")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Gripper debug script")
    parser.add_argument("--port", default="auto", help="Serial port")
    args = parser.parse_args()

    port = resolve_port(args.port)
    print(f"Connecting to {port}...")
    arm = SO100Arm(port=port, baudrate=1000000)
    if not arm.connect():
        print("[X] Failed to connect")
        return 1

    try:
        # 1. Read all positions using SO100Arm's own method
        print(f"\n[1] Reading all motor positions (SO100Arm):")
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll", "gripper"]
        for mid in arm.MOTOR_IDS:
            pos = arm._read_motor_position(mid)
            rad = pos * (2 * np.pi / COUNTS_PER_REV) if pos is not None else None
            name = names[mid - 1]
            if pos is not None:
                print(f"    Motor {mid} ({name:15s}): {pos:4d} counts  ({rad:.4f} rad)")
            else:
                print(f"    Motor {mid} ({name:15s}): READ FAILED")

        gripper_start = read_gripper_counts(arm)
        print(f"\n    Gripper starting position: {gripper_start} counts")
        if gripper_start is None:
            print("[X] Cannot read gripper position, aborting")
            return 1

        # 2. Enable torque on gripper only
        print(f"\n[2] Enabling torque on gripper (motor {GRIPPER_MOTOR_ID})...")
        write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_ENABLE, 1, nbytes=1)
        print("    [OK] Torque enabled")

        # 3. Test small moves in BOTH directions
        print(f"\n[3] Testing small moves from {gripper_start}:")
        write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_LIMIT, 300)
        write_reg(arm, GRIPPER_MOTOR_ID, REG_SPEED, 300)
        write_reg(arm, GRIPPER_MOTOR_ID, REG_ACCEL, 50, nbytes=1)

        for delta in [+100, -100, +200, -200]:
            target = gripper_start + delta
            if target < 0 or target >= COUNTS_PER_REV:
                print(f"    delta={delta:+4d}: SKIP (target={target} out of range)")
                continue
            write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, target)
            time.sleep(0.5)
            actual = read_gripper_counts(arm)
            if actual is not None:
                real_delta = actual - gripper_start
                print(f"    delta={delta:+4d}: target={target:4d}  actual={actual:4d}  "
                      f"real_delta={real_delta:+4d}")
            else:
                print(f"    delta={delta:+4d}: target={target:4d}  actual=READ_FAILED")

            # Return to start
            write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, gripper_start)
            time.sleep(0.5)

        # 4. Determine open direction by testing +100 vs -100
        print(f"\n[4] Determining open/close directions:")
        # Test positive direction
        write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, gripper_start + 100)
        time.sleep(0.5)
        input("    Motor moved +100 counts. Is the gripper MORE OPEN or MORE CLOSED? (open/closed): ")
        write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, gripper_start)
        time.sleep(0.5)

        # 5. Step to find open limit (positive direction)
        print(f"\n[5] Stepping toward open (positive direction):")
        write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_LIMIT, 300)
        write_reg(arm, GRIPPER_MOTOR_ID, REG_SPEED, 200)

        step = 50
        max_target = min(gripper_start + 1000, COUNTS_PER_REV - 10)
        current_target = gripper_start
        last_actual = gripper_start
        stall_count = 0

        while current_target + step <= max_target:
            current_target += step
            write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, current_target)
            time.sleep(0.3)
            actual = read_gripper_counts(arm)
            if actual is not None:
                moved = abs(actual - last_actual)
                total = actual - gripper_start
                print(f"    target={current_target:4d}  actual={actual:4d}  "
                      f"step_moved={moved:3d}  total={total:+4d}")
                if moved < 5:
                    stall_count += 1
                    if stall_count >= 3:
                        print(f"    [STALL] at {actual} counts")
                        break
                else:
                    stall_count = 0
                last_actual = actual
            else:
                print(f"    target={current_target:4d}  READ_FAILED")
                break

        open_limit = last_actual

        # Return to start
        write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, gripper_start)
        time.sleep(1.0)

        # 6. Step to find closed limit (negative direction)
        print(f"\n[6] Stepping toward closed (negative direction):")
        current_target = gripper_start
        last_actual = gripper_start
        stall_count = 0
        min_target = max(gripper_start - 1000, 10)

        while current_target - step >= min_target:
            current_target -= step
            write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, current_target)
            time.sleep(0.3)
            actual = read_gripper_counts(arm)
            if actual is not None:
                moved = abs(actual - last_actual)
                total = actual - gripper_start
                print(f"    target={current_target:4d}  actual={actual:4d}  "
                      f"step_moved={moved:3d}  total={total:+4d}")
                if moved < 5:
                    stall_count += 1
                    if stall_count >= 3:
                        print(f"    [STALL] at {actual} counts")
                        break
                else:
                    stall_count = 0
                last_actual = actual
            else:
                print(f"    target={current_target:4d}  READ_FAILED")
                break

        closed_limit = last_actual

        # Return to start
        write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, gripper_start)
        time.sleep(1.0)

        # 7. Summary
        print(f"\n{'=' * 60}")
        print(f"GRIPPER SUMMARY:")
        print(f"  Start:  {gripper_start:4d} counts ({gripper_start * 2*np.pi/COUNTS_PER_REV:.4f} rad)")
        print(f"  Open:   {open_limit:4d} counts ({open_limit * 2*np.pi/COUNTS_PER_REV:.4f} rad)")
        print(f"  Closed: {closed_limit:4d} counts ({closed_limit * 2*np.pi/COUNTS_PER_REV:.4f} rad)")
        print(f"  Range:  {abs(open_limit - closed_limit)} counts")
        print(f"{'=' * 60}")

        # 8. Full cycle test
        print(f"\n[8] Full cycle (open -> closed -> open):")
        write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_LIMIT, 300)
        write_reg(arm, GRIPPER_MOTOR_ID, REG_SPEED, 300)

        for label, tc in [("OPEN", open_limit), ("CLOSED", closed_limit), ("OPEN", open_limit)]:
            write_reg(arm, GRIPPER_MOTOR_ID, REG_GOAL_POSITION, tc)
            time.sleep(1.5)
            actual = read_gripper_counts(arm)
            if actual is not None:
                error = abs(actual - tc)
                print(f"    -> {label:6s}: target={tc:4d}  actual={actual:4d}  error={error}")
            else:
                print(f"    -> {label:6s}: target={tc:4d}  actual=READ_FAILED")

    except KeyboardInterrupt:
        print("\n[X] Cancelled")
    finally:
        try:
            write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_LIMIT, 0)
            write_reg(arm, GRIPPER_MOTOR_ID, REG_TORQUE_ENABLE, 0, nbytes=1)
            print("\n[OK] Gripper torque released")
        except Exception:
            pass
        arm.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
