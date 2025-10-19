# Chess Piece Detection Improvements

This document describes the improvements made to the YOLO piece detection system to address:
1. **Missed detections** - pieces not being detected at all
2. **Pawn misclassification** - pieces incorrectly classified as pawns

---

## Phase 1: Inference-Time Improvements (IMPLEMENTED)

### 1.1 Preprocessing Pipeline

**File:** `board_state.py` - `preprocess_for_detection()` method

**Features:**
- LAB color space conversion for better color/lighting separation
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
- Selective color channel normalization to reduce chess set tint variations
- Mild sharpening (70%/30% blend) to enhance piece edges
- Configurable via `use_preprocessing` parameter in `detect_pieces()`

**Usage:**
```python
detector = BoardState()
detections, boxes = detector.detect_pieces(image, use_preprocessing=True)  # Default
```

**Status:** ✅ Implemented (needs refinement with better test images)

---

### 1.2 Detection Parameter Adjustments

**Changes in `detect_pieces()` method:**
- Confidence threshold: 0.5 → **0.35** (catch more pieces)
- IoU threshold: **0.5** for NMS
- Test-time augmentation: **Enabled**
- Image size: **640px** (consistent)
- Image quality: **95%** JPEG (preserve details)

**Status:** ✅ Implemented

---

### 1.3 Post-Processing Filters

**File:** `board_state.py` - `post_process_detections()` method

**Features:**
- Class-specific confidence thresholds:
  - Pawns (classes 3, 9): **0.45** threshold
  - Other pieces: **0.35** threshold
- Reduces false pawn classifications by requiring higher confidence

**Status:** ✅ Implemented

---

## Phase 2: Data Collection & Training Pipeline (IMPLEMENTED)

### 2.1 Data Collection Tool

**File:** `tools/data_collector.py`

**Features:**
- Extracts piece crops from board images
- Saves in YOLO format (images + labels)
- Tracks per-class statistics
- Configurable confidence filtering

**Usage:**
```bash
# Collect data from board images
python tools/data_collector.py \
    --source path/to/board/images/ \
    --output dataset/ \
    --min-conf 0.5
```

**Output Structure:**
```
dataset/
├── images/
│   ├── board_0000.jpg
│   ├── board_0001.jpg
│   └── ...
└── labels/
    ├── board_0000.txt
    ├── board_0001.txt
    └── ...
```

**Status:** ✅ Implemented

---

### 2.2 Training Pipeline

**File:** `tools/train.py`

**Features:**
- YOLOv8 training with optimized hyperparameters
- Custom augmentation pipeline for chess pieces:
  - **Color augmentation:** HSV jittering (saturation ±70%, brightness ±40%)
  - **Geometric:** Rotation ±15°, scale 0.7-1.3x, horizontal flip
  - **Advanced:** Mosaic, mixup, copy-paste
- Higher classification loss weight (0.5 vs 0.05 box loss)
- AdamW optimizer with learning rate 0.001

**Usage:**
```bash
# Train YOLOv8m model for 150 epochs
python tools/train.py \
    --dataset dataset/ \
    --epochs 150 \
    --batch 16 \
    --model m

# Available models: n (nano), s (small), m (medium), l (large), x (xlarge)
```

**Key Hyperparameters:**
- `box_loss`: 0.05 (localization)
- `cls_loss`: **0.5** (classification - HIGHER for better accuracy)
- `hsv_s`: **0.7** (saturation variation for different wood tints)
- `lr0`: 0.001 (lower learning rate for fine details)

**Status:** ✅ Implemented

---

### 2.3 Evaluation Framework

**File:** `tools/evaluate.py`

**Features:**
- Per-class precision, recall, F1-score
- Confusion matrix visualization
- Detection accuracy metrics
- Identifies common misclassification patterns

**Usage:**
```bash
# Evaluate trained model
python tools/evaluate.py \
    --model runs/train/chess_pieces/weights/best.pt \
    --data dataset/ \
    --output evaluation_results/
```

**Outputs:**
- Console output with per-class metrics
- Confusion matrix heatmap (`confusion_matrix.png`)
- Overall precision/recall/F1 scores

**Status:** ✅ Implemented

---

## Complete Workflow

### Step 1: Collect Training Data
```bash
# Place board images in a directory
mkdir board_images/
# ... add your chess board photos ...

# Extract pieces
python tools/data_collector.py \
    --source board_images/ \
    --output dataset/ \
    --min-conf 0.5
```

**Target:** 200+ pieces per class (2400+ total)

---

### Step 2: Train Model
```bash
# Train on collected data
python tools/train.py \
    --dataset dataset/ \
    --epochs 150 \
    --batch 16 \
    --model m
```

**Training Time:**
- CPU: ~6-12 hours (150 epochs)
- GPU (CUDA): ~1-2 hours
- Cloud/Colab: Recommended for faster training

**Output:** `runs/train/chess_pieces/weights/best.pt`

---

### Step 3: Evaluate Model
```bash
# Evaluate performance
python tools/evaluate.py \
    --model runs/train/chess_pieces/weights/best.pt \
    --data dataset/ \
    --output evaluation_results/
```

---

### Step 4: Deploy New Model
```bash
# Replace the existing piece detection model
cp runs/train/chess_pieces/weights/best.pt data/best_transformed_detection.pt

# Test with main script
python main.py --image data/chessboard.png --debug
```

---

## Expected Improvements

### Phase 1 (Inference-Time)
- **Detection Recall:** +15-25% (fewer missed pieces)
- **Pawn Misclassification:** -20-30% (higher confidence threshold)
- **Overall Accuracy:** +10-15%

### Phase 2 (Retraining)
- **Detection Recall:** +40-60% (better model training)
- **Classification Accuracy:** +30-50% (color normalization, better augmentation)
- **Pawn Errors:** -50-70% (focused loss weighting)

### Combined
- **Near production-ready performance** with proper dataset (2000+ pieces)

---

## Troubleshooting

### Issue: Preprocessing reduces detections
**Solution:** Adjust preprocessing parameters in `preprocess_for_detection()`:
- Reduce CLAHE `clipLimit` (currently 1.5, try 1.2)
- Increase tile size (currently 16x16, try 24x24)
- Reduce sharpening blend (currently 30%, try 20%)

### Issue: Still getting pawn misclassifications
**Solutions:**
1. Increase pawn confidence threshold in `post_process_detections()` (try 0.5 or 0.55)
2. Retrain with higher `cls_loss` weight (try 0.7 or 1.0)
3. Collect more non-pawn training examples

### Issue: Training too slow on CPU
**Solutions:**
1. Use Google Colab with free GPU
2. Reduce batch size (try --batch 8)
3. Use smaller model (try --model s or n)
4. Reduce epochs (minimum 100 recommended)

---

## Configuration Reference

### Preprocessing Toggle
To disable preprocessing for testing:
```python
# In board_state.py, line ~810
detections, boxes = self.detect_pieces(transformed_image, use_preprocessing=False)
```

### Confidence Thresholds
Edit `post_process_detections()` in `board_state.py`:
```python
class_thresholds = {
    3: 0.45,  # black pawn - adjust this
    9: 0.45,  # white pawn - adjust this
    # ... other classes at 0.35
}
```

### YOLO Detection Confidence
Edit `detect_pieces()` in `board_state.py`:
```python
results = self.piece_model.predict(
    source=temp_path,
    conf=0.35,  # Adjust this (lower = more detections, more false positives)
    ...
)
```

---

## Next Steps

1. **Collect diverse dataset:**
   - Multiple chess sets (different wood tints, styles)
   - Various lighting conditions
   - Different camera angles
   - Target: 200+ pieces per class

2. **Train initial model:**
   - Start with YOLOv8m
   - 150 epochs minimum
   - Monitor validation metrics

3. **Evaluate and iterate:**
   - Analyze confusion matrix
   - Identify problematic classes
   - Collect more data for weak classes
   - Retrain with adjusted hyperparameters

4. **Fine-tune preprocessing:**
   - Test with real-world images
   - Adjust based on performance
   - Consider A/B testing with/without preprocessing

---

## Files Modified/Created

### Modified
- `board_state.py` - Added preprocessing pipeline, detection improvements, post-processing

### Created
- `tools/data_collector.py` - Data collection utility
- `tools/train.py` - YOLO training pipeline
- `tools/evaluate.py` - Evaluation framework
- `IMPROVEMENTS_README.md` - This documentation

---

## References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- CLAHE Algorithm: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- LAB Color Space: https://en.wikipedia.org/wiki/CIELAB_color_space
