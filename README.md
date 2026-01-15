# ChessBot - Unified Chess Robot Control System

A modular chess robot system with YOLO-based computer vision, multi-camera management, SO-100 robot arm control, and VLA integration for end-to-end learning.

## Overview

ChessBot provides a complete software stack for autonomous chess-playing robots:

- **Guidance Module**: YOLO-based board detection and symbolic move calculation
- **Camera Module**: Multi-camera stream management (global + gripper + overlay)
- **Controls Module**: SO-100 robot arm control with Feetech STS3215 servos
- **VLA Module**: Vision-Language-Action model integration with episode collection

**Status**: Guidance, cameras, and controls modules functional. VLA has episode collection and deployment scripts.

## Quick Start

### Installation

```bash
# Clone repository with submodules
git clone --recurse-submodules <repository-url>
cd chessbot

# Install dependencies (to ltx conda environment)
conda activate ltx
pip install -r requirements.txt

# Download YOLO models
python scripts/download.py
```

### Run Demo

```bash
# Detect board and calculate best move (auto-detects camera)
python scripts/best_move_demo.py --debug

# Run with specific camera
python scripts/best_move_demo.py --debug --global-camera /dev/video0
```

See [scripts/USAGE.md](scripts/USAGE.md) for comprehensive command reference.

## Architecture

### Modular Design

```
chessbot/
├── guidance/              # YOLO-based chess vision
│   ├── board_detector.py      # Two-stage YOLO detection pipeline
│   ├── move_calculator.py     # Stockfish UCI integration
│   ├── move_interpreter.py    # Move -> robot action decomposition
│   ├── move_decomposer.py     # VLM prompt generation with color conditioning
│   ├── coordinate_mapper.py   # Chess square -> pixel mapping
│   ├── highlight_renderer.py  # Color-coded overlay rendering
│   ├── guidance_system.py     # High-level orchestrator
│   └── training/              # YOLO training tools
│
├── cameras/               # Multi-camera management
│   ├── global_camera.py       # Overhead camera (1280x720)
│   ├── gripper_camera.py      # Arm-mounted camera (224x224)
│   ├── overlay_generator.py   # Flag-based overlay loading
│   ├── virtual_camera.py      # v4l2loopback output for VLA
│   └── camera_manager.py      # Unified interface
│
├── controls/              # SO-100 robot arm control
│   ├── so100_arm.py           # Low-level Feetech servo protocol
│   └── robot_controller.py    # High-level control with stability system
│
├── vla/                   # Vision-Language-Action integration
│   ├── vla_deploy.py          # Deploy pi0.5 for inference
│   ├── vla_load_model.py      # Load pi0.5 and tokenizer
│   └── verify_openpi.py       # OpenPI installation verification
│
├── utils/                 # Shared utilities
│   ├── state_cache.py         # Thread-safe JSON state cache
│   ├── camera_helpers.py      # Camera discovery and selection
│   └── keyboard_input.py      # Non-blocking terminal input
│
├── viz/                   # Web-based visualization
│   ├── api.py                 # FastAPI + WebSocket server
│   └── site/                  # React frontend
│
├── scripts/               # Execution scripts
│   ├── USAGE.md               # Command reference
│   ├── best_move_demo.py      # Main demo script
│   ├── tele_op.py             # Teleoperation interface
│   ├── collect_vla_episodes.py # VLA training data collection
│   └── ...                    # See scripts/USAGE.md
│
└── configs/               # Configuration management
    ├── config_schema.py       # Pydantic schema
    └── current.yaml           # Active configuration
```

### Module Communication

```
Camera Streams -> Guidance System -> Overlay Generator
                       |
                 Board State + Best Move
                       |
              Controls (Robot Execution)
                       |
                  VLA (End-to-end)
```

## Usage Examples

### Guidance System

```python
from guidance import GuidanceSystem

guidance = GuidanceSystem()

# Detect board, calculate best move, update cache
fen, best_move, actions = guidance.detect_and_calculate(
    "board.png",
    update_cache=True,
    robot_plays_white=True,
    debug=True
)

# Generate overlay with action highlights
guidance.generate_overlay_from_cache()
```

### Robot Control (SO-100)

```python
from controls.robot_controller import RobotController, load_joint_configs_for_port

# Load port-specific configuration
configs, source = load_joint_configs_for_port("/dev/ttyACM0")
robot = RobotController("/dev/ttyACM0", configs, source)

# Connect and control
robot.connect()
robot.enable_torque()
robot.set_home_targets()
robot.start_control_loop()

# Set positions
robot.set_target_positions(np.array([...]))  # 6 joint radians

# Cleanup
robot.release_torque()
robot.disconnect()
```

### Camera System

```python
from cameras import CameraManager

config = {
    'global_camera_id': 0,
    'gripper_camera_id': 1,
    'overlay_path': 'data/guidance_overlay.png',
    'overlay_flag_path': 'data/overlay_ready.flag'
}

cameras = CameraManager(config)
cameras.start()

global_frame = cameras.get_global_frame()
gripper_frame = cameras.get_gripper_frame()
overlay = cameras.get_overlay_frame()

cameras.stop()
```

## Configuration

Example `configs/current.yaml`:

```yaml
guidance:
  corner_model: "data/best_corners.pt"
  piece_model: "data/best_transformed_detection.pt"
  engine_path: "stockfish"

cameras:
  global_camera_id: 0
  global_resolution: [1280, 720]
  gripper_camera_id: 1
  gripper_resolution: [224, 224]

paths:
  state_cache: "data/state_cache.json"
  overlay_image: "data/guidance_overlay.png"
```

## Development Status

### Completed

**Guidance Module:**
- Board detection with two-stage YOLO pipeline
- Stockfish UCI integration for move calculation
- Move decomposition into robot actions
- VLM prompt generation with color conditioning
- Overlay rendering with action highlights

**Camera Module:**
- Threaded global camera (1280x720)
- Threaded gripper camera (224x224)
- Flag-based overlay loading
- Virtual camera output via v4l2loopback

**Controls Module:**
- SO-100 arm with Feetech STS3215 protocol
- Joint stability system (deadband, smoothing, speed/torque limits)
- Port-specific configuration loading
- Teleoperation interface

**VLA Module:**
- Episode collection with LeRobot format
- Episode validation and export
- pi0.5 model loading and deployment

**Utils Module:**
- Thread-safe JSON state cache
- Camera discovery helpers
- Non-blocking keyboard input

### In Progress

- VLA fine-tuning pipeline
- Full autonomous chess game execution

## Hardware

### Cameras
- **Global Camera**: WBC-0E01 USB camera
  - Board detection: 4K MJPEG (3840x2160) downscaled to 720p for best quality
  - Real-time VLA: Native 720p MJPEG at 30fps for low latency
- **Gripper Camera**: eMeet C950 USB camera (640x480 capture, resized to 224x224 for VLA)

### Robot
- **Arm**: SO-100 with 6x Feetech STS3215 smart servos
- **Communication**: Direct serial at 1 Mbps (no ROS required)

### Compute
- **Vision**: CPU (YOLO on PyTorch CPU backend)
- **VLA**: GPU with >8GB VRAM for inference
- **Memory**: 8GB+ RAM

## Module Documentation

Each module has detailed documentation:

- [guidance/README.md](guidance/README.md) - YOLO vision, training tools
- [cameras/README.md](cameras/README.md) - Multi-camera management
- [controls/README.md](controls/README.md) - SO-100 arm control, stability system
- [vla/README.md](vla/README.md) - pi0 integration, episode collection
- [scripts/USAGE.md](scripts/USAGE.md) - Command reference

## Troubleshooting

### Models Not Found
```bash
python scripts/download.py
```

### Camera Not Opening
```python
# List available cameras
from utils.camera_helpers import get_available_cameras
print(get_available_cameras())
```

### Robot Not Found
```bash
# List serial ports
ls /dev/ttyACM* /dev/ttyUSB*

# Add user to dialout group
sudo usermod -a -G dialout $USER
```

### Import Errors
```bash
# Ensure ltx environment
conda activate ltx
pip install -r requirements.txt
```

## Credits

Based on [shainisan/real-life-chess-vision](https://github.com/shainisan/real-life-chess-vision).

## License

See LICENSE file for details.
