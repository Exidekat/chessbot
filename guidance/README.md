# Guidance Module

Symbolic chess logic using YOLO-based computer vision. This module handles board detection, piece classification, move calculation, and provides the foundation for visual guidance overlays.

## Overview

The guidance system uses a two-stage YOLO pipeline:
1. **Corner Detection** - Identifies chessboard corners for perspective correction
2. **Piece Detection** - Classifies and locates all 12 piece types

This approach provides explainable, debuggable chess vision that can train on small datasets and run efficiently on CPU.

## Components

### Core Modules

#### `board_detector.py`
Complete board state detection pipeline:
- Corner detection with confidence filtering
- Perspective transform with top margin for tall pieces
- Gentle preprocessing (LAB color space, CLAHE, mild sharpening)
- Piece detection with class-specific thresholds (higher for pawns to reduce misclassification)
- Grid calculation and piece-to-square matching
- FEN generation
- Comprehensive visualization at every stage

**Usage:**
```python
from guidance.board_detector import BoardDetector

detector = BoardDetector()
fen, transformed_image = detector.detect_board_state(
    "board.png",
    corner_conf=0.1,
    min_corner_distance=50.0,
    debug=True  # Saves visualizations
)
```

**Debug Visualizations:**
- `chessboard_raw_corners.png` - All corner detections with confidence
- `chessboard_corners.png` - Selected 4 corners with labels
- `chessboard_transformed.png` - Perspective-corrected board
- `chessboard_detections.png` - Piece detections with confidence colors
- `chessboard_grid.png` - 8x8 grid overlay

#### `move_calculator.py`
UCI chess engine integration for move calculation:
```python
from guidance.move_calculator import MoveCalculator
import chess

calculator = MoveCalculator(engine_path="stockfish")
board = chess.Board(fen)
best_move = calculator.calculate_best_move(board, time_limit=1.0)
```

#### `move_interpreter.py`
Chess move decomposition into atomic robot actions:
```python
from guidance.move_interpreter import MoveInterpreter
import chess

board = chess.Board()
move = chess.Move.from_uci("e2e4")

# Get move type
move_type = MoveInterpreter.get_move_type(move, board)

# Decompose into actions
actions = MoveInterpreter.decompose_move(move, board)
# Normal move: [{"type": "pickup", "square": "e2", ...},
#              {"type": "place", "square": "e4", ...}]

# Capture: [pickup opponent, graveyard, pickup own, place]
# Castling: [king pickup, king place, rook pickup, rook place]
```

#### `coordinate_mapper.py`
Map chess squares to pixel coordinates on transformed board:
```python
from guidance.coordinate_mapper import CoordinateMapper
from PIL import Image

mapper = CoordinateMapper(board_detector)
transformed = Image.open("chessboard_transformed.png")

# Square to pixels
x, y = mapper.square_to_pixels("e4", transformed)

# Get square bounds
x1, y1, x2, y2 = mapper.get_square_bounds("e4", transformed)

# Graveyard location
gx, gy = mapper.get_graveyard_coords(transformed)
```

#### `highlight_renderer.py`
Visual overlay rendering with color-coded highlights:
```python
from guidance.highlight_renderer import HighlightRenderer

renderer = HighlightRenderer(alpha=0.5)

# Render from cache
overlay = renderer.render_from_cache(transformed_image, cache)

# Render action sequence
overlay = renderer.render_action_sequence(
    transformed_image,
    action_sequence,
    current_index=0
)
```

**Color Scheme:**
- 🟢 Green: Pickup piece
- 🔵 Blue: Place piece
- 🔴 Red: Capture (pickup opponent's piece)
- 🟠 Orange: Graveyard placement

#### `guidance_system.py`
High-level orchestrator coordinating all components:
```python
from guidance import GuidanceSystem

guidance = GuidanceSystem(config={
    'corner_model': 'data/best_corners.pt',
    'piece_model': 'data/best_transformed_detection.pt',
    'engine_path': 'stockfish',
    'cache_path': 'data/state_cache.json'
})

# Detect board, calculate move, update cache
fen, best_move, actions = guidance.detect_and_calculate(
    "board.png",
    update_cache=True,
    robot_plays_white=True
)

# Generate overlay from cache
guidance.generate_overlay_from_cache()

# Update robot state
guidance.advance_to_next_action()
guidance.update_robot_action_status(0, "complete", holding_piece=True)
```

### Training Tools (`guidance/training/`)

#### `data_collector.py`
Extract labeled piece images from board photos:
```bash
python -m guidance.training.data_collector \
    --source board_photos/ \
    --output dataset/ \
    --min-conf 0.5
```

#### `train_yolo.py`
Train YOLO models with chess-optimized hyperparameters:
```bash
python -m guidance.training.train_yolo \
    --data dataset/data.yaml \
    --epochs 150 \
    --batch 16
```

**Optimizations:**
- Lower learning rate (0.001) for fine details
- Higher classification weight (0.5)
- Saturation augmentation (0.7) for wood tint variations
- Class-balanced sampling for rare pieces

#### `evaluate_yolo.py`
Comprehensive model evaluation:
```bash
python -m guidance.training.evaluate_yolo \
    --model runs/train/weights/best.pt \
    --data dataset/
```

Generates:
- Per-class precision/recall
- Confusion matrix
- Common misclassification patterns

#### `analyze_board.py`
Compare detections against known starting position:
```bash
python -m guidance.training.analyze_board \
    --image data/test_board.png
```

## YOLO Models

### Required Models
- **Corner Detection**: `data/best_corners.pt`
- **Piece Detection**: `data/best_transformed_detection.pt`

Download with:
```bash
python download.py
```

### Class Mapping
```
0-5: Black pieces (bishop, king, knight, pawn, queen, rook)
6-11: White pieces (Bishop, King, Knight, Pawn, Queen, Rook)
```

### Detection Parameters

**Corner Detection:**
- Default confidence: 0.1
- Min distance: 50.0 pixels
- Expects exactly 4 corners

**Piece Detection:**
- Base confidence: 0.35
- Pawn threshold: 0.45 (higher to reduce misclassification)
- Test-time augmentation: Enabled
- Preprocessing: Enabled by default

## Performance Notes

**Current Baseline (chessboardv2.png):**
- Detection rate: 27/32 pieces (84%)
- Accuracy: 9/32 correct (28%)
- Common issues: Misclassification (especially pawns vs other pieces)

**To Improve:**
1. Collect 100+ board photos from your setup
2. Run `data_collector.py` to extract labeled pieces
3. Train new models with `train_yolo.py` (150+ epochs)
4. Evaluate with `evaluate_yolo.py` and iterate

## State Cache System

The guidance system uses a JSON-based state cache for multi-source updates:

### State Cache (`utils/state_cache.py`)
```python
from utils import StateCache

cache = StateCache("data/state_cache.json")

# Read
state = cache.get()
is_robot_turn = cache.get("robot_state.is_robot_turn")

# Update (from guidance, robot, VLA, or user)
cache.update({
    "robot_state": {"holding_piece": True}
}, source="robot")

# Robot-specific helpers
cache.advance_action()  # Move to next action
cache.set_action_status(0, "complete")
current_action = cache.get_current_action()
```

### Cache Structure
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
    "last_updated_by": "guidance",
    "timestamp": 1234567890
  }
}
```

## Overlay Generation Workflow

### 1. Detect and Calculate
```bash
python -c "from guidance import GuidanceSystem; \
    GuidanceSystem().detect_and_calculate('board.png')"
```
- Detects board state
- Calculates best move
- Decomposes into actions
- Updates state cache

### 2. Generate Overlay
```bash
python generate_overlay.py
```
- Reads state cache
- Renders current action highlight
- Saves to `data/guidance_overlay.png`
- Signals cameras via flag file

### 3. Robot Updates Cache
```python
from utils import StateCache

cache = StateCache()
cache.update({"robot_state": {"holding_piece": True}}, source="robot")
cache.advance_action()
```

### 4. Regenerate Overlay for Next Action
```bash
python generate_overlay.py
```

## Integration

The guidance module integrates with:
- **State Cache** (`utils/state_cache.py`) - Shared state across all systems
- **Cameras** (`cameras/overlay_generator.py`) - Flag-based overlay loading
- **Robot Control** (future) - Action execution and status updates
- **VLA** (future) - Alternative control mode
