# Corner Detection Fine-tuning Guide

## What This Guide Is For

This guide helps you fine-tune the **CORNER DETECTION** model specifically (not piece detection).

Use this when:
- The system detects only 2-3 corners instead of 4
- Corner detection confidence is low
- The board corners are clearly visible but not detected

The corner detection model is currently failing because it was trained on different chessboards. Fine-tuning it on **your specific board** will dramatically improve accuracy.

## Why Fine-tune?

The pre-trained model doesn't recognize your board's characteristics:
- Different wood grain/texture
- Different board size/proportions
- Different lighting conditions
- Different camera angle
- Different corner appearance

Fine-tuning adapts the model to YOUR specific setup in just 20-30 minutes.

## Quick Start (5-Step Process)

### Step 1: Collect Training Photos (5-10 minutes)

Capture 20+ photos of your chessboard from various positions:

```bash
python scripts/collect_corner_training_photos.py --device /dev/video0 --count 20
```

**Tips for good training data:**
- Move the board slightly between photos (different positions, angles)
- Vary lighting if possible (overhead light on/off, etc.)
- Keep all 4 corners clearly visible in every photo
- Include boards with pieces AND empty boards
- Press SPACE to capture each photo, Q to finish early

### Step 2: Label Corners (10-15 minutes)

Interactively label the 4 corners in each photo:

```bash
python scripts/label_corners.py --input data/training/board_photos
```

**Labeling instructions:**
- Click corners in order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**
- Press R to reset if you make a mistake
- Press ENTER to save and move to next image
- Press Q to quit early

This creates a YOLO dataset at `data/training/corner_dataset/`

### Step 3: Fine-tune Model (15-30 minutes)

Train the model on your labeled data:

```bash
python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml
```

**What happens:**
- Loads the base model (`data/best_corners.pt`)
- Fine-tunes on your labeled photos (50 epochs by default)
- Saves best model to `data/training/runs/corner_finetune/weights/best.pt`
- Shows training progress and metrics

**Time estimate:**
- 20 photos: ~15 minutes on CPU
- 50 photos: ~30 minutes on CPU
- With GPU: 3-5 minutes

### Step 4: Deploy Fine-tuned Model

Replace the original model with your fine-tuned version:

```bash
# Backup original
mv data/best_corners.pt data/best_corners_original.pt

# Use fine-tuned model
cp data/training/runs/corner_finetune/weights/best.pt data/best_corners.pt
```

### Step 5: Test!

Run the full pipeline with your new model:

```bash
python scripts/best_move_demo.py --debug
```

You should now see **4 corners detected reliably**!

## Expected Results

**Before fine-tuning:**
- 2-3 corners detected (FAILS)
- Low confidence scores
- Misses your board's corners

**After fine-tuning:**
- 4 corners detected consistently (SUCCESS)
- High confidence scores (>0.5)
- Robust to slight board movement

## Advanced Options

### Collect More Photos

More photos = better accuracy (diminishing returns after ~50 photos):

```bash
python scripts/collect_corner_training_photos.py --device /dev/video0 --count 50
```

### Adjust Training Parameters

For better results with larger datasets:

```bash
# More epochs (better convergence, slower)
python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml --epochs 100

# Larger image size (more detail, slower)
python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml --imgsz 1280
```

### Use GPU (if available)

Edit `scripts/finetune_corners.py` and change:
```python
device='cpu'  # Change to device=0 for GPU
```

Training will be 10-20x faster on GPU.

## Troubleshooting

### "Not enough training photos"

You need at least 10 labeled photos. Capture more:
```bash
python scripts/collect_corner_training_photos.py --device /dev/video0 --count 30
```

### "Model still not detecting corners"

1. Check labeled data quality:
   - Open `data/training/corner_dataset/images/` and verify photos are clear
   - Open `data/training/corner_dataset/labels/` and check label files have 4 lines each

2. Increase training epochs:
   ```bash
   python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml --epochs 100
   ```

3. Collect more diverse training data (different lighting, angles)

### "Training is very slow"

This is normal on CPU. Options:
- Reduce batch size (already set to 8)
- Reduce epochs (try `--epochs 25`)
- Use GPU if available

## How It Works

The fine-tuning process uses **transfer learning**:

1. **Base model** (trained on generic corner detections) provides general corner detection knowledge
2. **Fine-tuning** adapts the model to YOUR specific board by:
   - Learning your board's texture patterns
   - Learning your lighting conditions
   - Learning your camera's perspective
3. **Result**: Model becomes an expert at detecting corners on YOUR setup

The fine-tuned model retains general corner detection ability while being highly optimized for your board.

## Next Steps After Fine-tuning

Once corner detection works reliably, you may want to fine-tune piece detection:

1. Collect photos with pieces on the board
2. Label pieces (12 classes: black/white × pawn/knight/bishop/rook/queen/king)
3. Fine-tune `data/best_transformed_detection.pt`

This is optional - the current piece detection model works reasonably well (28% accuracy can be improved to 80%+ with fine-tuning).
