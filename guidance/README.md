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
- **Corner Detection**: `data/best_cornres.pt`
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

## Integration

The guidance module is used by:
- `best_move_demo.py` - Standalone demo script
- `cameras/overlay_generator.py` - Generates visual overlays
- Future: `guidance/guidance_system.py` - Full orchestrator

## Future Enhancements

- [ ] `coordinate_mapper.py` - Chess square to pixel/physical coordinates
- [ ] `move_interpreter.py` - Move type detection and decomposition
- [ ] `highlight_renderer.py` - Visual overlay generation
- [ ] `guidance_system.py` - Unified orchestrator

These will be implemented as robot control integration proceeds.
