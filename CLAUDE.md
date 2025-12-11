# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessBot is a modular chess robot system with YOLO-based computer vision, multi-camera management, ROS robot control (skeleton), and future VLA integration. The system uses a two-stage YOLO pipeline for board detection and piece classification, with symbolic guidance for move calculation.

**Status**: Guidance and camera modules complete. Controls and VLA are skeletal awaiting hardware integration.

## Environment Setup

- **Python Environment**: Always use the 'ltx' conda environment for testing
- **Package Installation**: Install packages to 'ltx' environment, never 'base'
- If import errors occur, update requirements.txt and request user to install to ltx environment

## Common Commands

**NOTE**: All execution scripts are located in `scripts/`. See `scripts/USAGE.md` for comprehensive usage examples.

### Demo and Testing
```bash
# Run full pipeline: capture YUYV 720p + detect + calculate move
python scripts/best_move_demo.py --debug

# Run with specific camera
python scripts/best_move_demo.py --debug --device /dev/video0

# Generate overlay from state cache
python scripts/generate_overlay.py

# Check cache status
python scripts/generate_overlay.py --status

# Download YOLO models (required on first run)
python scripts/download.py

# Create default config file
python scripts/create_config.py
```

### YOLO Training Pipeline
```bash
# Collect training data from board photos
python -m guidance.training.data_collector --source board_photos/ --output dataset/

# Train YOLO model
python -m guidance.training.train_yolo --data dataset/data.yaml --epochs 150

# Evaluate model
python -m guidance.training.evaluate_yolo --model runs/train/weights/best.pt --data dataset/

# Analyze specific board
python -m guidance.training.analyze_board --image data/test_board.png
```

### Visualization Tool
```bash
# Development mode (React HMR + FastAPI reload)
python scripts/start_viz_tool.py --dev

# Production mode (bundled React)
python scripts/start_viz_tool.py

# Build React app only
python scripts/start_viz_tool.py --build-only
```

## Architecture

### Module Structure

The codebase uses a modular architecture with clear separation:

```
chessbot/
├── guidance/           # YOLO vision + symbolic move calculation
│   ├── board_detector.py       # Corner detection → perspective transform → piece detection
│   ├── move_calculator.py      # Stockfish UCI integration
│   ├── move_interpreter.py     # Chess move → robot actions
│   ├── coordinate_mapper.py    # Chess squares → pixel coordinates
│   ├── highlight_renderer.py   # Color-coded overlay rendering
│   ├── guidance_system.py      # High-level orchestrator
│   └── training/               # YOLO training tools
│
├── cameras/            # Multi-camera stream management
│   ├── global_camera.py        # Overhead camera (threaded capture)
│   ├── gripper_camera.py       # Arm-mounted camera (threaded capture)
│   ├── overlay_generator.py    # Flag-based overlay loading (resource-efficient)
│   └── camera_manager.py       # Unified interface
│
├── controls/           # ROS robot control (SKELETON - awaiting hardware)
├── vla/                # Future VLA integration (SKELETON)
├── viz/                # Web-based visualization tool
│   ├── api.py                  # FastAPI server with WebSocket
│   ├── stream_manager.py       # JPEG frame encoding
│   ├── file_watcher.py         # Filesystem monitoring
│   └── site/                   # React SPA
│
├── utils/              # Shared utilities
│   └── state_cache.py          # Thread-safe JSON state cache
│
└── configs/            # Configuration management
    ├── config_schema.py        # Pydantic schema
    └── current.yaml            # Active configuration
```

### Data Flow

1. **Detection Phase**: Camera → BoardDetector → FEN + transformed image
2. **Planning Phase**: FEN → MoveCalculator (Stockfish) → Best move → MoveInterpreter → Action sequence
3. **State Cache Update**: GuidanceSystem updates state_cache.json with game state, robot state, and actions
4. **Overlay Generation**: HighlightRenderer renders color-coded highlights → guidance_overlay.png + flag file
5. **Camera Display**: OverlayGenerator detects flag → loads overlay → cameras display to user/VLA

### State Cache System

The system uses a centralized JSON state cache (`data/state_cache.json`) for multi-source updates:

**Structure**:
```json
{
  "game_state": {
    "fen": "...",
    "turn": "white",
    "transformed_board_path": "data/chessboard_transformed.png"
  },
  "robot_state": {
    "is_robot_turn": true,
    "holding_piece": false,
    "current_move": "e2e4",
    "action_sequence": [
      {"type": "pickup", "square": "e2", "piece": "P", "status": "pending"},
      {"type": "place", "square": "e4", "piece": "P", "status": "pending"}
    ],
    "action_index": 0,
    "actions_remaining": 2
  },
  "guidance_state": {
    "best_move": "e2e4",
    "best_move_san": "e4"
  },
  "metadata": {
    "last_updated_by": "guidance|robot|vla|user",
    "timestamp": 1234567890
  }
}
```

**Usage Pattern**:
```python
from utils import StateCache

cache = StateCache("data/state_cache.json")

# Read
current_action = cache.get_current_action()
is_robot_turn = cache.get("robot_state.is_robot_turn")

# Update (thread-safe, multi-source)
cache.update({"robot_state": {"holding_piece": True}}, source="robot")

# Robot helpers
cache.advance_action()  # Move to next action
cache.set_action_status(0, "complete")
```

### Overlay Generation Workflow

The system uses a **flag-based signaling protocol** for resource-efficient overlay updates:

1. Guidance system generates overlay → saves `guidance_overlay.png` → touches `overlay_ready.flag`
2. OverlayGenerator checks flag file modification time (O(1) operation)
3. If updated, loads PNG from disk
4. Cameras display overlay without continuous polling

This avoids wasted resources compared to polling-based approaches.

### Color Scheme for Action Highlights

- 🟢 **Green**: Pickup piece (origin square)
- 🔵 **Blue**: Place piece (destination square)
- 🔴 **Red**: Capture opponent's piece (capture square)
- 🟠 **Orange**: Graveyard placement (off-board)

## YOLO Models

### Required Models
- **Corner Detection**: `data/best_corners.pt` (detects 4 board corners for perspective correction)
- **Piece Detection**: `data/best_transformed_detection.pt` (classifies 12 piece types after transform)

Download with `python download.py` on first run.

### Detection Pipeline

1. **Corner Detection** (YOLO stage 1)
   - Input: Raw board image
   - Output: 4 corners (TL, TR, BR, BL)
   - Params: `corner_conf=0.1`, `min_corner_distance=50.0`

2. **Perspective Transform**
   - Uses detected corners to warp board to orthogonal view
   - Adds 10% top margin for tall pieces (kings, queens)
   - Preprocessing: LAB color space, CLAHE, mild sharpening

3. **Piece Detection** (YOLO stage 2)
   - Input: Transformed board image
   - Output: Piece bounding boxes + class (0-11: b-bishop to W-Rook)
   - Params: `base_conf=0.35`, `pawn_threshold=0.45` (higher to reduce misclassification)

4. **Grid Calculation**
   - Divides transformed board into 8x8 grid
   - Maps detected pieces to chess squares
   - Generates FEN notation

### Debug Visualizations

When `debug=True` is passed to `BoardDetector.detect_board_state()`, these files are saved:

- `chessboard_raw_corners.png` - All corner detections with confidence scores
- `chessboard_corners.png` - Selected 4 corners labeled TL/TR/BR/BL
- `chessboard_transformed.png` - Perspective-corrected board (used for coordinate mapping)
- `chessboard_detections.png` - Piece detections with confidence-based colors
- `chessboard_grid.png` - 8x8 grid overlay
- `guidance_overlay.png` - Color-coded action highlights (if overlay generation called)

## Configuration System

Configuration is managed via YAML files validated by Pydantic schemas.

**Active Config**: `configs/current.yaml`

**Key Settings**:
```yaml
guidance:
  robot_plays_white: false          # Which side robot controls
  engine_path: "stockfish"          # UCI engine path
  engine_time_limit: 1.0            # Seconds per move calculation
  corner_confidence: 0.1            # Corner detection threshold
  min_corner_distance: 50.0         # Min pixels between corners

cameras:
  global_camera_id: 0               # Overhead camera device ID
  gripper_camera_id: 1              # Gripper camera device ID
  global_resolution: [1280, 720]    # Overhead camera resolution
  gripper_resolution: [640, 480]    # Gripper camera resolution

models:
  corner_model: "data/best_corners.pt"
  piece_model: "data/best_transformed_detection.pt"

paths:
  state_cache: "data/state_cache.json"
  overlay_image: "data/guidance_overlay.png"
  overlay_flag: "data/overlay_ready.flag"
  data_dir: "data"

visualization:
  host: "0.0.0.0"
  port: 8000
```

## Module Integration Examples

### Guidance System (High-Level API)

```python
from guidance import GuidanceSystem

guidance = GuidanceSystem()

# Detect board, calculate move, update cache
fen, best_move, actions = guidance.detect_and_calculate(
    "board.png",
    update_cache=True,
    robot_plays_white=True,
    debug=True  # Saves all visualizations
)

# Generate overlay for current action
guidance.generate_overlay_from_cache()

# Robot execution loop
while actions_remaining > 0:
    current_action = cache.get_current_action()

    # Execute action with robot...

    # Mark complete and advance
    guidance.update_robot_action_status(index, "complete", holding_piece=True)
    guidance.advance_to_next_action()

    # Regenerate overlay for next action
    guidance.generate_overlay_from_cache()
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

# Get frames
global_frame = cameras.get_global_frame()        # Real-time overhead
gripper_frame = cameras.get_gripper_frame()      # Real-time gripper
overlay = cameras.get_overlay_frame()            # Static overlay (flag-based)

cameras.stop()
```

### Visualization Tool

The viz tool provides a web-based interface with:
- Real-time camera feeds (global, gripper, overlay)
- State cache monitoring
- WebSocket updates on file changes
- React frontend with auto-refresh

Access at `http://localhost:8000` (production) or `http://localhost:3000` (dev mode).

## Performance Notes

**Current Detection Baseline** (on `chessboardv2.png`):
- Detection rate: 27/32 pieces (84%)
- Classification accuracy: 9/32 correct (28%)
- Common issues: Pawn misclassification, missed pieces

**To Improve Accuracy**:
1. Collect 100+ board photos from YOUR specific setup (lighting, board, pieces)
2. Extract labeled pieces: `python -m guidance.training.data_collector --source photos/ --output dataset/`
3. Train new models: `python -m guidance.training.train_yolo --data dataset/data.yaml --epochs 150`
4. Evaluate and iterate

The system is designed to run on **CPU** (no GPU required for inference), using PyTorch with CPU backend.

## Important File Locations

- **YOLO Models**: `data/best_corners.pt`, `data/best_transformed_detection.pt`
- **State Cache**: `data/state_cache.json` (created on first run)
- **Overlay Files**: `data/guidance_overlay.png`, `data/overlay_ready.flag`
- **Debug Visualizations**: `chessboard_*.png` (root directory)
- **Config**: `configs/current.yaml`

## Deprecated Files

The following files are deprecated and should NOT be used:
- `board_state.py` → Use `guidance/board_detector.py`
- `robot_interface.py` → Refactored into `guidance/` and `controls/`
- `tools/` directory → Moved to `guidance/training/`
- `main.py` → Use `best_move_demo.py` instead

## Hardware Requirements (Future)

**For Full System Operation**:
- Overhead USB camera (1280x720+)
- Gripper-mounted USB camera (640x480+)
- ROS-compatible robot arm (UR5, Franka, etc.)
- 2-finger or suction gripper
- Stockfish chess engine (install via package manager)

**Current Development** (guidance + cameras only):
- CPU (no GPU required)
- 8GB+ RAM
- Optional: USB cameras for live testing
- Optional: Stockfish for move calculation

## Development Workflow

1. **Adding Features**: Always update the README in the same directory or parent directory
2. **Documentation**: Only create unified READMEs (no scattered docs)
3. **Testing**: Run demos with `--debug` flag to generate visualizations
4. **Configuration**: Use `create_config.py` to generate configs, edit `configs/current.yaml`

## Common Troubleshooting

**Models Not Found**:
```bash
python download.py
```

**Import Errors**:
- Check requirements.txt has the package
- Install to ltx environment: `conda activate ltx && pip install -r requirements.txt`

**Low Detection Accuracy**:
- Use `--debug` flag to inspect visualizations
- Adjust `--corner-conf` and `--min-corner-dist` parameters
- Collect training data from YOUR setup and retrain models

**Camera Not Opening**:
```python
# List available cameras
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

**Overlay Not Updating**:
- Check flag file exists and is writable
- Verify overlay PNG path in config
- Manually regenerate: `python generate_overlay.py`
- Lets move our execution scripts to a scripts/ directory. Include a markdown file called USAGE.md that will follow the format of:

# Comment describing the use of a script for a task
python {args for the script execution}

This means we should have exactly 2 lines per script usage. Some scripts may have more than one usage.
- Cached data, results, and intermediate files should be output to data/ or results/ by default. USAGE examples should reflect this standard.