# Controls Module

SO-100 robotic arm control with Feetech STS3215 smart servos. Provides low-level servo communication and high-level control with joint stability features.

## Overview

The controls module provides direct serial control for SO-100 robot arms equipped with 6 Feetech STS3215 servos. Unlike ROS-based systems, this module communicates directly via the Feetech serial protocol at 1 Mbps.

**Hardware**: SO-100 arm with 6x Feetech STS3215 smart servos (motor IDs 1-6)

## Components

### `so100_arm.py`

Low-level Feetech protocol implementation:

```python
from controls.so100_arm import SO100Arm

arm = SO100Arm(port="/dev/ttyACM0", baudrate=1000000)
arm.connect()

# Read current positions (all 6 motors)
positions = arm.get_joint_positions()  # radians [6]

# Get complete state
state = arm.get_state()  # SO100State: positions, timestamp, is_moving, error_code

arm.disconnect()
```

**Features:**
- Feetech serial protocol with checksum validation
- Threaded state updates at 20 Hz
- Position reading/writing for all 6 motors
- Position resolution: 4096 counts/revolution (0-4095 encoder values)

### `robot_controller.py`

High-level control with stability system and configuration management:

```python
from controls.robot_controller import RobotController, load_joint_configs_for_port

# Load port-specific configuration
configs, source = load_joint_configs_for_port("/dev/ttyACM0")
robot = RobotController("/dev/ttyACM0", configs, source)

# Connect and start control
robot.connect()
robot.enable_torque()
robot.set_home_targets()
robot.start_control_loop()  # 20 Hz background thread

# Set target positions (stability system handles smoothing)
robot.set_target_positions(np.array([...]))  # 6 radians

# Cleanup
robot.stop_control_loop()
robot.release_torque()
robot.disconnect()
```

## Joint Stability System

The base joints (J0/J1) have mechanical compliance and inertia that causes oscillation with naive position control. The stability system mitigates this through per-joint parameters:

### Per-Joint Parameters

Each joint has 5 tunable parameters:

| Parameter | Description | J0 (Base) | J1 (Shoulder) | J2-J5 |
|-----------|-------------|-----------|---------------|-------|
| DEADBAND | Error threshold to disable torque (rad) | 0.025 | 0.0 | 0.0 |
| SMOOTHING_ALPHA | Low-pass filter (1.0=none, 0.05=heavy) | 1.0 | 1.0 | 1.0 |
| SPEED_LIMIT | Max speed (steps/sec, 0-1500) | 1000 | 500 | 1000 |
| ACCEL | Acceleration limit (0=max, 1-254=limited) | 0 | 50 | 0-100 |
| TORQUE | Torque limit (0-1000 = 0-100%) | 50 | 300 | 100-200 |

**Key behaviors:**
- **Deadband**: When position error < deadband, torque is disabled to prevent oscillation
- **J1 has no deadband** because it must fight gravity (would fall if torque disabled)
- **Smoothing**: Low-pass filter on target positions (currently disabled: alpha=1.0)
- **Torque limiting**: Prevents excessive force; J0 uses only 5%, J1 uses 30%

### Feetech STS3215 Registers

Key registers used by the control loop:

| Register | Address | Bytes | Description |
|----------|---------|-------|-------------|
| Torque Enable | 0x28 | 1 | 0=off, 1=on |
| Acceleration | 0x29 | 1 | 0=max, 1-254=limited |
| Goal Position | 0x2A | 2 | Target position (0-4095) |
| Speed Limit | 0x2E | 2 | Max speed (steps/sec) |
| Torque Limit | 0x30 | 2 | Runtime torque limit (0-1000) |
| Current Position | 0x38 | 2 | Read-only encoder value |

## Configuration System

Three-tier configuration fallback:

1. **Port-specific**: `data/so100_config_ttyACM0.csv` (calibrated for specific robot)
2. **Generic**: `data/so100_config.csv` (shared across robots)
3. **Defaults**: Full range [0, 2pi] with home at pi (midpoint)

**Safety**: Only robots with port-specific configs should have torque enabled.

### Config File Format

CSV with columns: `joint,min_rad,max_rad,home_rad`

```csv
joint,min_rad,max_rad,home_rad
0,0.52,5.76,3.14
1,1.05,5.23,3.14
2,0.52,5.76,3.14
3,0.52,5.76,3.14
4,0.52,5.76,3.14
5,1.57,4.71,3.14
```

### Creating Calibration

```bash
# Interactive calibration tool (saves to data/so100_config_ttyACM0.csv)
python scripts/create_so100_config.py --port /dev/ttyACM0
```

## Safety Features

### Stuck-Joint Detection

Monitors position error over time. If a joint (e.g., gripper holding object) cannot reach target:

1. Error exceeds threshold (0.2 rad / 11 deg) for too long (0.5s)
2. Safety triggers: torque disabled, target clamped to current position
3. Auto-recovery when error stays within bounds for 0.5s

```python
# Configure which joints have safety trigger
SAFETY_ENABLED_JOINTS = [5]  # Gripper only

# Manual reset after intervention
robot.reset_safety_trigger(joint_idx=5)
```

### Torque Release

Critical operation with retry logic and warnings if failed:

```python
robot.release_torque()  # Retries 3 times, warns if unsuccessful
```

## Usage with Tele-op

The primary interface is via `scripts/tele_op.py`:

```bash
# Interactive teleoperation with menu
python scripts/tele_op.py

# Test mode (hold at home for 10 seconds)
python scripts/tele_op.py --test
```

**Menu options:**
1. Exit - Clean disconnect
2. Tele-op Leader/Follower - Mirror positions between two arms
3. Adjust Home Positions - Calibrate and save to config

## Utility Functions

```python
from controls.robot_controller import scan_so100_ports, load_joint_configs_for_port

# Find connected SO-100 robots
ports = scan_so100_ports()  # Checks /dev/ttyACM0-9

# Load configs with fallback
configs, source = load_joint_configs_for_port("/dev/ttyACM0")
# source: "port-specific", "generic", or "defaults"
```

## Troubleshooting

### Robot Not Found

```bash
# List available ports
ls /dev/ttyACM* /dev/ttyUSB*

# Check permissions (add user to dialout group)
sudo usermod -a -G dialout $USER
# Log out and back in
```

### Oscillation Issues

If joints oscillate around target:
- Increase DEADBAND for that joint
- Decrease SMOOTHING_ALPHA (more filtering)
- Decrease SPEED_LIMIT
- Decrease TORQUE limit

### Torque Not Releasing

If `release_torque()` fails:
- Power cycle the robot
- Check serial connection
- Verify correct port

## Hardware Reference

**SO-100 Arm:**
- 6 DOF: Base, Shoulder, Elbow, Wrist Pitch, Wrist Roll, Gripper
- Motor IDs: 1-6 (corresponds to joints 0-5 in code)
- Serial: 1 Mbps baud rate

**Feetech STS3215 Servo:**
- 14-bit encoder (4096 positions per revolution)
- Position range: 0-4095 counts = 0 to 2pi radians
- Serial protocol with [0xFF, 0xFF] header and checksum
