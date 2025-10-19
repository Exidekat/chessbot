# Migration Guide - Unified Control System

This guide explains the migration from the old network-based architecture to the new unified control system.

---

## What Changed

### Old Architecture (DEPRECATED)
```
board_state.py          # Monolithic board detection + move calculation
robot_interface.py      # Network-based robot control
robot_demo.py           # Testing with network simulation
tools/                  # Scattered training tools
```

### New Architecture
```
guidance/               # Chess logic (YOLO-based)
├── board_detector.py
├── move_calculator.py
├── move_interpreter.py
├── highlight_renderer.py
├── coordinate_mapper.py
├── guidance_system.py
└── training/           # YOLO training tools (moved from tools/)

controls/               # ROS robot control
├── robot_arm.py
├── movement.py
├── calibration.py
└── safety.py

vla/                    # Future VLA integration (skeleton)
├── model/
├── data_collection/
└── training/

cameras/                # 3-camera management
├── camera_manager.py
├── global_camera.py
└── overlay_generator.py

main.py                 # Unified control script
config.yaml            # System configuration
```

---

## Migration Mapping

### board_state.py → Multiple Files

| Old (board_state.py) | New Location | Purpose |
|---------------------|--------------|---------|
| `BoardState.__init__()` | `guidance/board_detector.py` | Board detection |
| `BoardState.snapshot()` | `guidance/board_detector.py` | Detect board state |
| `BoardState.bestmove()` | `guidance/move_calculator.py` | Calculate best move |
| `BoardState.current()` | `guidance/board_detector.py` | Get current board |
| Corner detection | `guidance/board_detector.py` | Detect corners |
| Piece detection | `guidance/board_detector.py` | Detect pieces |
| Grid calculation | `guidance/coordinate_mapper.py` | Map coordinates |

### robot_interface.py → Multiple Files

| Old (robot_interface.py) | New Location | Purpose |
|-------------------------|--------------|---------|
| `MoveInterpreter` | `guidance/move_interpreter.py` | Parse chess moves |
| `HighlightRenderer` | `guidance/highlight_renderer.py` | Visual overlays |
| `CoordinateMapper` | `guidance/coordinate_mapper.py` | Square to coords |
| `MoveExecutor` | `controls/movement.py` | Movement primitives |
| `RobotCommunicator` | DELETED | No more networking |

### tools/ → guidance/training/

| Old (tools/) | New Location | Purpose |
|--------------|--------------|---------|
| `data_collector.py` | `guidance/training/data_collector.py` | Collect YOLO data |
| `train.py` | `guidance/training/train_yolo.py` | Train YOLO |
| `evaluate.py` | `guidance/training/evaluate_yolo.py` | Evaluate YOLO |
| `analyze_test_board.py` | `guidance/training/analyze_board.py` | Board analysis |

---

## Code Migration Examples

### Old: Using board_state.py
```python
from board_state import BoardState

detector = BoardState()
detector.snapshot("board.png")
move = detector.bestmove()
```

### New: Using guidance module
```python
from guidance import GuidanceSystem

guidance = GuidanceSystem(config)
frame = camera.get_global_frame()
overlay, action = guidance.process_frame(frame)
```

---

### Old: YOLO Training
```python
# From project root
python tools/train.py --dataset dataset/ --epochs 150
```

### New: YOLO Training
```python
# From project root
python -m guidance.training.train_yolo --dataset dataset/ --epochs 150
```

---

## Deprecated Files

**DO NOT USE - Will be removed in future versions:**

- ❌ `board_state.py` - Use `guidance/` module instead
- ❌ `robot_interface.py` - Use `controls/` and `guidance/` instead
- ❌ `robot_demo.py` - Use `main.py` with manual mode
- ❌ `robot_server_example.py` - No more network communication
- ❌ `tools/data_collector.py` - Use `guidance/training/data_collector.py`
- ❌ `tools/train.py` - Use `guidance/training/train_yolo.py`
- ❌ `tools/evaluate.py` - Use `guidance/training/evaluate_yolo.py`

**These files are kept temporarily for reference but marked DEPRECATED.**

---

## New Features

### 1. Three Camera Streams
```python
from cameras import CameraManager

cameras = CameraManager(config)

# Get frames from all 3 cameras
global_frame = cameras.get_global_frame()      # Overhead view
gripper_frame = cameras.get_gripper_frame()    # Arm-mounted view
overlay = cameras.generate_overlay(global_frame, highlights)  # Guidance view
```

### 2. ROS Integration
```python
from controls import RobotArm

robot = RobotArm(config)
robot.move_to_position(x, y, z)
robot.pick_piece(square="e4", piece="P")
```

### 3. Unified Main Script
```bash
# Auto mode (continuous play)
python main.py --mode auto

# Manual mode (triggered moves)
python main.py --mode manual
```

### 4. VLA Data Collection (Future)
```python
from vla.data_collection import EpisodeRecorder

recorder = EpisodeRecorder()
recorder.start_episode(metadata={'move': 'e2e4'})

# During robot execution
recorder.record_timestep(global_frame, gripper_frame, overlay, robot_state, action)

recorder.end_episode(success=True)
```

---

## Configuration

### Old: Hardcoded in Python
```python
detector = BoardState(
    corner_model_path="data/best_cornres.pt",
    piece_model_path="data/best_transformed_detection.pt"
)
```

### New: YAML Configuration
```yaml
# config.yaml
control_mode: 'guidance'  # or 'vla' (future)

cameras:
  global_camera_id: 0
  gripper_camera_id: 1

guidance:
  corner_model: "data/best_cornres.pt"
  piece_model: "data/best_transformed_detection.pt"
  engine_path: "stockfish"
```

---

## Step-by-Step Migration

### Step 1: Install Dependencies (if needed)
```bash
# ROS dependencies
sudo apt-get install ros-noetic-desktop-full

# Python dependencies (same as before)
pip install -r requirements.txt
```

### Step 2: Update Your Code

**If you were using BoardState directly:**
```python
# OLD
from board_state import BoardState
detector = BoardState()

# NEW - Option 1: Use guidance module
from guidance import GuidanceSystem
guidance = GuidanceSystem(config)

# NEW - Option 2: Use unified main script
# Just run: python main.py
```

**If you were training YOLO models:**
```python
# OLD
python tools/train.py --dataset dataset/

# NEW
python -m guidance.training.train_yolo --dataset dataset/
```

### Step 3: Configure System
```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit for your setup
nano config.yaml
```

### Step 4: Run System
```bash
# Start with manual mode
python main.py --mode manual

# Switch to auto mode when ready
python main.py --mode auto
```

---

## Directory Structure Reference

```
chessbot/
├── controls/                    # NEW - Robot control (ROS)
├── guidance/                    # NEW - Chess logic (YOLO)
│   └── training/               # MOVED from tools/
├── vla/                        # NEW - Future VLA integration
├── cameras/                    # NEW - 3-camera management
├── utils/                      # NEW - Shared utilities
├── main.py                     # NEW - Unified control
├── config.yaml                 # NEW - Configuration
│
├── board_state.py              # DEPRECATED
├── robot_interface.py          # DEPRECATED
├── robot_demo.py               # DEPRECATED
├── tools/                      # DEPRECATED (moved to guidance/training/)
│
└── data/                       # Unchanged - model files
```

---

## FAQ

**Q: Can I still use board_state.py?**
A: Yes, temporarily. It's marked DEPRECATED and will be removed in a future version. Migrate to `guidance/` module.

**Q: Do I need ROS installed?**
A: Not immediately. The system is designed for ROS but can run without it initially for testing.

**Q: What happened to network communication?**
A: Removed! The new system uses direct function calls and ROS messages instead of TCP/UDP networking.

**Q: Can I use the old training scripts?**
A: They're moved to `guidance/training/`. Use the new paths or module imports.

**Q: What is the VLA module?**
A: A skeleton for future Vision-Language-Action model integration using Pi0. Currently not functional - use guidance module.

**Q: How do I switch from guidance to VLA?**
A: Future feature. When VLA is ready, just change `config.yaml`: `control_mode: 'vla'`

---

## Need Help?

- **Architecture Overview:** See `docs/ARCHITECTURE.md`
- **ROS Setup:** See `docs/ROS_INTEGRATION.md`
- **Camera Calibration:** See `docs/CAMERA_CALIBRATION.md`
- **YOLO Training:** See `docs/GUIDANCE_TRAINING.md`
- **VLA Future:** See `docs/VLA_FUTURE.md`

---

**Migration Status:** ✅ Structure created, in progress...

Last Updated: 2025-01
