# Incremental Corner Detection Training Workflow

## Problem
The corner detection model overfits to the initial 20 training images and fails on new board positions, lighting conditions, or camera angles.

## Solution: Incremental Dataset Expansion

Add new difficult/failing cases to your training dataset iteratively.

## Workflow

### Step 1: Capture New Problem Cases (5-10 minutes)

When the model fails to detect corners correctly, capture those specific cases:

```bash
# Capture 10 new photos where detection currently fails
python scripts/collect_corner_training_photos.py \
    --device /dev/video0 \
    --count 10 \
    --output data/training/board_photos
```

**What to capture:**
- Different lighting conditions (darker, brighter, harsh shadows)
- Different camera angles
- Board positions where corners are missed
- Board positions where false corners are detected

### Step 2: Label Only New Images (5-10 minutes)

The labeling script automatically skips already-labeled images:

```bash
# Labels ONLY new images, preserves existing labels
python scripts/label_corners.py \
    --input data/training/board_photos \
    --output data/training/corner_dataset
```

Output will show:
```
Total images: 30
Already labeled: 20
Newly labeled: 10
Final dataset size: 30
```

### Step 3: Retrain with Expanded Dataset (5-10 minutes)

```bash
python scripts/finetune_corners.py \
    --data data/training/corner_dataset/data.yaml \
    --epochs 100 \
    --imgsz 640
```

The model now trains on 30 images instead of 20.

### Step 4: Test and Iterate

```bash
python scripts/best_move_demo.py --debug --no-bestmove --device /dev/video0
```

Check `data/chessboard_raw_corners.png` and `data/chessboard_corners.png`:
- Are all 4 corners detected?
- Are there false corner detections?

**If still failing:**
- Repeat steps 1-4 with more problem cases
- Target: 50-100 diverse training images for robust performance

## Dataset Growth Strategy

**Initial dataset (20 images):**
- Basic board positions
- Good lighting
- Straight-on camera angle

**After 1st expansion (30 images):**
- Add darker/brighter lighting
- Add slight angle variations

**After 2nd expansion (40-50 images):**
- Add edge cases (very dark, very bright)
- Add board at different positions in frame
- Add partial occlusions

**Target (50-100 images):**
- Comprehensive coverage of all conditions
- Model becomes very robust

## Key Advantages

1. **Preserves existing work** - Already-labeled images are skipped
2. **Targeted improvement** - Add specific failing cases
3. **Fast iteration** - 10 new images + labeling + training = ~20-30 minutes total
4. **Incremental cost** - Don't need to relabel everything

## Data.yaml Auto-Update

The labeling script automatically updates `split:` in data.yaml:
- 20 images → split: 0.1 (18 train, 2 val)
- 30 images → split: 0.1 (27 train, 3 val)
- 50 images → split: 0.1 (45 train, 5 val)

Validation set grows proportionally as you add more data.

## Expected Results

**Starting point (20 images):**
- Works well on original 20 positions
- Fails on new positions/lighting

**After 30 images:**
- Handles moderate lighting variation
- Handles slight angle variation

**After 50 images:**
- Robust to most lighting conditions
- Handles various camera angles
- Rare false positives

**After 100 images:**
- Production-ready robustness
- Handles extreme cases
- Very low false positive rate

## Date Created
2025-12-12

## Status
Active - workflow ready for iterative improvement
