# ChessBot - Unified Chess Robot Control System

A modular chess robot system with YOLO-based computer vision, multi-camera management, ROS robot control, and future VLA integration.

## Overview

ChessBot provides a complete software stack for autonomous chess-playing robots:
- **Guidance Module**: YOLO-based board detection and symbolic move calculation
- **Camera Module**: Multi-camera stream management (2 real-time + 1 static overlay)
- **Controls Module**: ROS robot arm control with safety systems (skeleton)
- **VLA Module**: Future Vision-Language-Action model integration (skeleton)

**Status**: Guidance and cameras modules complete and tested. Controls and VLA are skeletal awaiting hardware integration.

## Quick Start

### Installation

```bash
# Clone repository
git clone --recurse-submodules <repository-url>
cd chessbot

# Install dependencies
pip install -r requirements.txt

# Download YOLO models
python download.py
```

### Run Demo

```bash
# Detect board state and calculate best move
python best_move_demo.py --image data/chessboardv2.png --debug

# Or use the old interface (deprecated)
python main.py --image data/chessboardv2.png --debug
```

## Architecture

### Modular Design

```
chessbot/
├── guidance/              # YOLO-based chess vision
│   ├── board_detector.py      # Board state detection
│   ├── move_calculator.py     # Stockfish integration
│   ├── training/              # YOLO training tools
│   └── README.md
│
├── cameras/               # Multi-camera management
│   ├── global_camera.py       # Overhead camera (real-time)
│   ├── gripper_camera.py      # Arm-mounted camera (real-time)
│   ├── overlay_generator.py   # Guidance overlay (on-demand)
│   ├── camera_manager.py      # Unified interface
│   └── README.md
│
├── controls/              # ROS robot control (skeleton)
│   ├── robot_arm.py           # ROS integration (TODO)
│   ├── movement.py            # Movement primitives (TODO)
│   ├── calibration.py         # Calibration system (TODO)
│   ├── safety.py              # Safety monitoring (TODO)
│   └── README.md
│
├── vla/                   # Future VLA integration (skeleton)
│   ├── model/                 # Pi0 wrapper (TODO)
│   ├── data_collection/       # Episode recording (TODO)
│   ├── training/              # VLA training (TODO)
│   └── README.md
│
├── best_move_demo.py      # Demo using new guidance module
└── README.md              # This file
```

**Old Files (Deprecated)**:
- `board_state.py` - Use `guidance/board_detector.py` instead
- `robot_interface.py` - Refactored into `guidance/` and `controls/`
- `tools/` - Moved to `guidance/training/`

### Module Communication

```
Camera Streams → Guidance System → Overlay Generator
                      ↓
                Board State + Best Move
                      ↓
              Controls (Robot Execution)
                      ↓
                  VLA (Future)
```

## Usage

### Guidance System

```python
from guidance import BoardDetector, MoveCalculator
import chess

# Detect board state
detector = BoardDetector()
fen, transformed = detector.detect_board_state(
    "board.png",
    debug=True  # Saves visualization at each stage
)

# Calculate best move
calculator = MoveCalculator(engine_path="stockfish")
board = chess.Board(fen)
best_move = calculator.calculate_best_move(board, time_limit=1.0)

print(f"Best move: {best_move.uci()}")
```

**Debug Visualizations**:
- `chessboard_raw_corners.png` - All corner detections
- `chessboard_corners.png` - Selected 4 corners
- `chessboard_transformed.png` - Perspective-corrected board
- `chessboard_detections.png` - Piece detections with confidence
- `chessboard_grid.png` - 8x8 grid overlay

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

# Get real-time streams
global_frame = cameras.get_global_frame()
gripper_frame = cameras.get_gripper_frame()

# Get static overlay (only loads when guidance signals via flag)
overlay = cameras.get_overlay_frame()

cameras.stop()
```

### Training YOLO Models

```bash
# Collect data from board photos
python -m guidance.training.data_collector \
    --source board_photos/ \
    --output dataset/

# Train YOLO model
python -m guidance.training.train_yolo \
    --data dataset/data.yaml \
    --epochs 150

# Evaluate model
python -m guidance.training.evaluate_yolo \
    --model runs/train/weights/best.pt \
    --data dataset/

# Analyze specific board
python -m guidance.training.analyze_board \
    --image data/test_board.png
```

## Configuration

Example `config.yaml`:

```yaml
# Control mode
control_mode: 'guidance'  # or 'vla' (future)

# Guidance (YOLO-based)
guidance:
  corner_model: "data/best_cornres.pt"
  piece_model: "data/best_transformed_detection.pt"
  engine_path: "stockfish"

# Cameras
cameras:
  global_camera_id: 0
  global_resolution: [1280, 720]
  gripper_camera_id: 1
  gripper_resolution: [640, 480]
  overlay_path: "data/guidance_overlay.png"
  overlay_flag_path: "data/overlay_ready.flag"

# Robot (ROS)
robot:
  type: "ur5"
  ros_namespace: "/robot_arm"
  workspace:
    x_min: 0.2
    x_max: 0.8
    y_min: -0.3
    y_max: 0.3
    z_min: 0.0
    z_max: 0.5
```

## Development Status

### ✅ Completed

**Guidance Module**:
- [x] `board_detector.py` - Full board detection pipeline with visualizations
- [x] `move_calculator.py` - Stockfish UCI integration
- [x] Training tools moved to `guidance/training/`
- [x] `best_move_demo.py` - Working demo script

**Camera Module**:
- [x] `global_camera.py` - Threaded overhead camera capture
- [x] `gripper_camera.py` - Threaded gripper camera capture
- [x] `overlay_generator.py` - Flag-based overlay management
- [x] `camera_manager.py` - Unified stream interface

**Documentation**:
- [x] Module READMEs for all 4 modules
- [x] Root README (this file)

### 🚧 In Progress

**Guidance Module**:
- [ ] `coordinate_mapper.py` - Chess square to pixel mapping
- [ ] `move_interpreter.py` - Move type detection
- [ ] `highlight_renderer.py` - Visual overlay generation
- [ ] `guidance_system.py` - Unified orchestrator

### ⏳ To Do

**Controls Module** (requires hardware):
- [ ] `robot_arm.py` - ROS integration
- [ ] `movement.py` - Movement primitives
- [ ] `calibration.py` - Camera/robot calibration
- [ ] `safety.py` - Safety monitoring

**VLA Module** (future):
- [ ] `pi0_wrapper.py` - HuggingFace Pi0 integration
- [ ] `episode_recorder.py` - Data collection
- [ ] `train_vla.py` - VLA training pipeline

## Module Documentation

Each module has comprehensive documentation:

- **[guidance/README.md](guidance/README.md)** - YOLO vision system, training tools, performance notes
- **[cameras/README.md](cameras/README.md)** - Multi-camera management, stream specifications
- **[controls/README.md](controls/README.md)** - ROS integration, movement primitives, calibration
- **[vla/README.md](vla/README.md)** - Future VLA architecture, Pi0 integration, data collection

## Performance

### Current Baseline

Testing with `chessboardv2.png` (starting position):
- **Detection Rate**: 27/32 pieces (84%)
- **Classification Accuracy**: 9/32 correct (28%)
- **Common Issues**: Pawn misclassification, missed pieces

### Improvement Strategy

1. Collect 100+ board photos from your setup
2. Use `guidance.training.data_collector` to extract labeled pieces
3. Train new models with `guidance.training.train_yolo` (150+ epochs)
4. Evaluate and iterate

## Hardware Requirements

### Cameras
- **Global Camera**: USB camera for overhead view (1280x720+)
- **Gripper Camera**: USB camera on robot arm (640x480+)
- **Mounting**: Stable overhead mount, arm-mounted bracket

### Robot
- **Arm**: ROS-compatible robot arm (UR5, Franka, etc.)
- **Gripper**: 2-finger gripper or suction gripper
- **Workspace**: ~60cm x 60cm chessboard area

### Compute
- **Vision**: CPU (YOLO on CPU with PyTorch 2.7.1+cpu)
- **VLA**: GPU recommended for Pi0 inference (future)
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ for episode data (VLA)

## Dependencies

### Python Packages
```
ultralytics>=8.0.0
opencv-python
numpy
pillow
shapely
python-chess
```

### System
- Python 3.8+
- (Optional) Stockfish for move calculation
- (Future) ROS Noetic for robot control
- (Future) CUDA for VLA training

## Migration from Old Code

If you were using the old `board_state.py` directly:

```python
# OLD
from board_state import BoardState
detector = BoardState()
fen = detector.snapshot("board.png", output_format="fen")
move = detector.bestmove()

# NEW
from guidance import BoardDetector, MoveCalculator
import chess

detector = BoardDetector()
fen, _ = detector.detect_board_state("board.png")

calculator = MoveCalculator()
board = chess.Board(fen)
move = calculator.calculate_best_move(board)
```

For YOLO training:
```bash
# OLD
python tools/train.py --dataset dataset/

# NEW
python -m guidance.training.train_yolo --dataset dataset/
```

## Troubleshooting

### Models Not Found
```bash
python download.py
```

### Camera Not Opening
```python
# List available cameras
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

### Low Detection Accuracy
- Collect more training data from your specific setup
- Ensure good lighting and clear view of board
- Check that all corners are visible (not obscured)
- Adjust confidence thresholds in detection

### Import Errors
Make sure to run from project root:
```bash
cd chessbot
python best_move_demo.py
```

## Credits

Based on [shainisan/real-life-chess-vision](https://github.com/shainisan/real-life-chess-vision).

## License

See LICENSE file for details.
