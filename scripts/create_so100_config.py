#!/usr/bin/env python3
"""
SO-100 Configuration Calibration Script

Records min, max, and home positions for each of the 6 SO-100 motors.
Outputs calibration data to CSV file in data/ directory.

Usage:
    python scripts/create_so100_config.py [--port /dev/ttyACM0]

Process:
    1. Connects to SO-100
    2. For each joint (0-5):
       - User manually moves joint to MIN position, presses ENTER
       - User manually moves joint to MAX position, presses ENTER
       - User manually moves joint to HOME position, presses ENTER
    3. Saves calibration data to data/so100_config.csv
"""

import argparse
import sys
import time
import csv
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controls.so100_arm import SO100Arm
import numpy as np


def calibrate_joint(arm: SO100Arm, joint_idx: int) -> dict:
    """
    Calibrate a single joint by recording min, max, and home positions.

    Args:
        arm: Connected SO100Arm instance
        joint_idx: Joint index (0-5)

    Returns:
        dict: Calibration data with keys 'min', 'max', 'home' (all in radians)
    """
    print("\n" + "=" * 60)
    print(f"Calibrating Joint {joint_idx}")
    print("=" * 60)

    calibration = {}

    # Get MIN position
    print(f"\nManually move Joint {joint_idx} to its MINIMUM safe position.")
    print("Press ENTER when ready...")
    input()

    state = arm.get_state()
    min_pos = state.joint_positions[joint_idx]
    calibration['min'] = min_pos
    print(f"[OK] MIN position recorded: {min_pos:.4f} rad ({np.degrees(min_pos):.2f} deg)")

    # Get MAX position
    print(f"\nManually move Joint {joint_idx} to its MAXIMUM safe position.")
    print("Press ENTER when ready...")
    input()

    state = arm.get_state()
    max_pos = state.joint_positions[joint_idx]
    calibration['max'] = max_pos
    print(f"[OK] MAX position recorded: {max_pos:.4f} rad ({np.degrees(max_pos):.2f} deg)")

    # Swap if max < min
    if max_pos < min_pos:
        print(f"[WARN] MAX < MIN detected, swapping positions...")
        calibration['min'], calibration['max'] = max_pos, min_pos
        min_pos, max_pos = calibration['min'], calibration['max']
        print(f"[OK] Corrected: MIN={min_pos:.4f} rad, MAX={max_pos:.4f} rad")

    # Get HOME position
    print(f"\nManually move Joint {joint_idx} to its HOME (neutral) position.")
    print("This should be a safe, repeatable reference position.")
    print("Press ENTER when ready...")
    input()

    state = arm.get_state()
    home_pos = state.joint_positions[joint_idx]
    calibration['home'] = home_pos
    print(f"[OK] HOME position recorded: {home_pos:.4f} rad ({np.degrees(home_pos):.2f} deg)")

    # Calculate range
    range_rad = max_pos - min_pos
    range_deg = np.degrees(range_rad)
    print(f"\nRange of motion: {range_rad:.4f} rad ({range_deg:.2f} deg)")

    # Validate range
    MIN_RANGE_DEG = 5.0  # Minimum reasonable range
    if abs(range_rad) < 1e-6:  # Essentially zero
        print(f"[ERROR] Range is zero! MIN and MAX positions are identical.")
        print(f"        This joint was not moved between MIN and MAX recording.")
        print(f"        Please recalibrate this joint.")
        raise ValueError(f"Joint {joint_idx}: Invalid calibration - zero range")
    elif range_deg < MIN_RANGE_DEG:
        print(f"[WARN] Range is very small ({range_deg:.2f} deg)!")
        print(f"       Expected at least {MIN_RANGE_DEG:.1f} degrees.")
        print(f"       Consider recalibrating with a larger range.")

    # Validate home is within range
    if not (min_pos <= home_pos <= max_pos):
        print(f"[WARN] HOME position ({home_pos:.4f}) is outside MIN-MAX range!")
        print(f"       MIN={min_pos:.4f}, MAX={max_pos:.4f}")

    return calibration


def get_port_name(port: str) -> str:
    """
    Extract port name from device path for use in filenames.

    Args:
        port: Device path (e.g., /dev/ttyACM0)

    Returns:
        str: Port name (e.g., ttyACM0)
    """
    return port.split('/')[-1]


def get_default_config_path(port: str) -> Path:
    """
    Get default config file path for a given port.

    Args:
        port: Device path (e.g., /dev/ttyACM0)

    Returns:
        Path: Config file path (e.g., data/so100_config_ttyACM0.csv)
    """
    port_name = get_port_name(port)
    return Path(f"data/so100_config_{port_name}.csv")


def save_calibration(calibrations: list, output_path: Path, port: str):
    """
    Save calibration data to CSV file.

    Args:
        calibrations: List of calibration dicts (one per joint)
        output_path: Path to output CSV file
        port: Device port that was calibrated (e.g., /dev/ttyACM0)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header (includes port column)
        writer.writerow(['joint', 'port', 'min_rad', 'max_rad', 'home_rad', 'min_deg', 'max_deg', 'home_deg', 'range_deg'])

        # Data rows
        for i, cal in enumerate(calibrations):
            min_rad = cal['min']
            max_rad = cal['max']
            home_rad = cal['home']
            min_deg = np.degrees(min_rad)
            max_deg = np.degrees(max_rad)
            home_deg = np.degrees(home_rad)
            range_deg = abs(max_deg - min_deg)

            writer.writerow([
                i,
                port,
                f"{min_rad:.6f}",
                f"{max_rad:.6f}",
                f"{home_rad:.6f}",
                f"{min_deg:.2f}",
                f"{max_deg:.2f}",
                f"{home_deg:.2f}",
                f"{range_deg:.2f}"
            ])

    print(f"\n[OK] Calibration data saved to: {output_path}")
    print(f"     Port: {port}")


def main():
    parser = argparse.ArgumentParser(
        description="SO-100 calibration script - record min/max/home positions"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="SO-100 serial port (default: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data/so100_config_<port>.csv)"
    )

    args = parser.parse_args()

    # Use port-based default path if output not specified
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = get_default_config_path(args.port)

    print("=" * 60)
    print("SO-100 Configuration Calibration")
    print("=" * 60)
    print()
    print("This script will guide you through calibrating each joint.")
    print("You will manually position each joint at MIN, MAX, and HOME positions.")
    print()
    print("IMPORTANT:")
    print("  - Ensure the robot arm is powered and can move freely")
    print("  - Manually move joints by hand (servos should be compliant)")
    print("  - Avoid collisions and extreme positions")
    print("  - HOME should be a safe, repeatable reference position")
    print()
    input("Press ENTER to begin calibration...")

    # Connect to SO-100
    print("\n" + "=" * 60)
    print("Connecting to SO-100...")
    print("=" * 60)

    arm = SO100Arm(port=args.port, baudrate=1000000)

    if not arm.connect():
        print("[X] Failed to connect to SO-100")
        print(f"    Port: {args.port}")
        print("    Check that:")
        print("      - SO-100 is powered on")
        print("      - Serial port is correct")
        print("      - User has serial permissions (dialout group)")
        return 1

    print("[OK] Connected to SO-100")

    # Display current positions
    state = arm.get_state()
    print("\nCurrent joint positions:")
    for i, pos in enumerate(state.joint_positions):
        print(f"  Joint {i}: {pos:.4f} rad ({np.degrees(pos):.2f} deg)")

    # Calibrate each joint
    calibrations = []

    for joint_idx in range(6):
        try:
            cal = calibrate_joint(arm, joint_idx)
            calibrations.append(cal)
        except KeyboardInterrupt:
            print("\n\n[X] Calibration cancelled by user")
            arm.disconnect()
            return 1

    # Save calibration data
    print("\n" + "=" * 60)
    print("Calibration Complete")
    print("=" * 60)

    save_calibration(calibrations, output_path, args.port)

    # Display summary
    print("\nCalibration Summary:")
    print("-" * 60)
    for i, cal in enumerate(calibrations):
        range_deg = abs(np.degrees(cal['max']) - np.degrees(cal['min']))
        print(f"Joint {i}: MIN={np.degrees(cal['min']):6.1f}° "
              f"MAX={np.degrees(cal['max']):6.1f}° "
              f"HOME={np.degrees(cal['home']):6.1f}° "
              f"RANGE={range_deg:6.1f}°")
    print("-" * 60)

    # Disconnect
    arm.disconnect()
    print("\n[OK] Calibration complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
