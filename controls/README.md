# Controls Module

Robot arm control and movement primitives using ROS integration. This module handles low-level robot communication, movement execution, safety systems, and calibration.

## Overview

The controls module provides direct robot arm control, replacing the deprecated network-based architecture. All communication with the robot happens through direct function calls and ROS messages - no TCP/UDP networking.

**Status**: Skeleton structure created, implementation pending hardware integration.

## Planned Components

### `robot_arm.py`
ROS integration for robot arm control:
```python
from controls import RobotArm

robot = RobotArm(config)
robot.initialize()

# Direct position control
robot.move_to_position(x, y, z, orientation)

# Joint control
robot.move_joints(joint_angles)

# Gripper control
robot.open_gripper()
robot.close_gripper()

# Status
status = robot.get_status()  # position, gripper_state, ready, errors
```

**ROS Topics:**
- `/robot_arm/command` - Movement commands
- `/robot_arm/status` - Robot state feedback
- `/robot_arm/joint_states` - Current joint positions

### `movement.py`
High-level movement primitives for chess:
```python
from controls import MovementPrimitives

primitives = MovementPrimitives(robot_arm, coordinate_mapper)

# Chess-specific movements
primitives.pickup_piece(square="e4", piece_type="P")
primitives.place_piece(square="e5")
primitives.move_to_graveyard(captured_piece="p")

# Coordinated sequences
primitives.execute_capture(from_sq="e4", to_sq="d5")
primitives.execute_castling(king_move, rook_move)
```

**Movement Types:**
- **Normal Move**: pickup → place
- **Capture**: pickup opponent → graveyard → pickup own → place
- **Castling**: king move → rook move (executed sequentially)
- **En Passant**: pickup captured pawn → graveyard → pickup own → place

### `calibration.py`
Camera-robot calibration system:
```python
from controls import CalibrationSystem

calibrator = CalibrationSystem(robot, camera_manager)

# Camera calibration
calibrator.calibrate_camera_intrinsics()
calibrator.calibrate_hand_eye()

# Workspace calibration
calibrator.calibrate_board_corners()
calibrator.calibrate_graveyard_position()

# Save/load calibration
calibrator.save("calibration/current.yaml")
calibrator.load("calibration/current.yaml")
```

**Calibration Data:**
- Camera intrinsics (focal length, distortion)
- Hand-eye transformation (camera to gripper)
- Board corner positions in robot coordinates
- Graveyard position
- Z-heights for pickup/travel/place

### `safety.py`
Safety monitoring and collision avoidance:
```python
from controls import SafetySystem

safety = SafetySystem(robot, camera_manager)

# Workspace limits
safety.set_workspace_bounds(x_min, x_max, y_min, y_max, z_min, z_max)

# Collision detection
safety.enable_collision_detection()
safety.add_obstacle(position, radius)  # e.g., board edges, pieces

# Emergency stop
safety.emergency_stop()

# Status
if safety.is_safe_to_move(target_position):
    robot.move_to_position(target_position)
```

**Safety Features:**
- Workspace boundary enforcement
- Velocity/acceleration limits
- Collision detection
- Emergency stop capability
- Movement validation before execution

## ROS Setup

### Installation
```bash
# Install ROS (Ubuntu)
sudo apt-get install ros-noetic-desktop-full

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS Package Structure
```
controls/
├── __init__.py
├── robot_arm.py         # ROS interface
├── movement.py          # Movement primitives
├── calibration.py       # Calibration system
├── safety.py            # Safety monitoring
└── README.md
```

## Integration with Other Modules

### With Guidance
```python
from guidance import BoardDetector, MoveCalculator
from controls import RobotArm, MovementPrimitives

# Detect board state
detector = BoardDetector()
fen, _ = detector.detect_board_state("board.png")

# Calculate best move
calculator = MoveCalculator()
board = chess.Board(fen)
move = calculator.calculate_best_move(board)

# Execute with robot
robot = RobotArm(config)
primitives = MovementPrimitives(robot, coord_mapper)
primitives.execute_move(move, board)
```

### With Cameras
```python
from cameras import CameraManager
from controls import RobotArm

cameras = CameraManager(config)
cameras.start()

robot = RobotArm(config)

# Use real-time feedback
while moving:
    gripper_view = cameras.get_gripper_frame()
    # Adjust based on visual feedback
```

### With VLA (Future)
```python
from vla import VLAModel
from controls import RobotArm

# VLA directly outputs actions
vla = VLAModel()
robot = RobotArm(config)

observation = get_observation()  # camera frames + state
action = vla.predict(observation)
robot.execute_action(action)
```

## Configuration

Example `config.yaml`:
```yaml
robot:
  type: "ur5"  # or "franka", "kuka", etc.
  ip_address: "192.168.1.100"
  ros_namespace: "/robot_arm"

  workspace:
    x_min: 0.2
    x_max: 0.8
    y_min: -0.3
    y_max: 0.3
    z_min: 0.0
    z_max: 0.5

  speeds:
    approach: 0.1  # m/s
    travel: 0.3
    retreat: 0.1

  gripper:
    type: "robotiq_2f"
    open_position: 0.08
    closed_position: 0.0
    force_limit: 20  # N

calibration:
  board_corners:
    tl: [0.3, 0.2, 0.0]
    tr: [0.7, 0.2, 0.0]
    bl: [0.3, -0.2, 0.0]
    br: [0.7, -0.2, 0.0]

  graveyard: [0.85, 0.0, 0.0]

  z_heights:
    travel: 0.15
    approach: 0.05
    pickup: 0.01
```

## Development Status

- [x] Directory structure created
- [x] `__init__.py` with planned exports
- [ ] `robot_arm.py` - ROS interface
- [ ] `movement.py` - Movement primitives
- [ ] `calibration.py` - Calibration system
- [ ] `safety.py` - Safety monitoring
- [ ] Hardware integration testing

## Next Steps

1. Implement `robot_arm.py` with specific robot model
2. Set up ROS environment and test basic movements
3. Implement calibration workflow
4. Integrate with cameras for visual servoing
5. Test complete move execution pipeline
