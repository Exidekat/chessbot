# Piece Detection Fine-tuning Guide

## What This Guide Is For

This guide helps you fine-tune the **PIECE DETECTION** model specifically (not corner detection).

Use this when:
- Pieces are not detected correctly (low accuracy)
- Wrong piece types are identified (pawn vs bishop, etc.)
- Pieces are missed entirely despite being visible
- Your pieces look different from the training data (color, shape, size)
- **CRITICAL**: After pipeline changes that affect when detection runs (e.g., before vs after perspective transform)

The piece detection model was trained on generic chess pieces. Fine-tuning it on **your specific pieces** will dramatically improve accuracy.

## Current Issues (2025-12-12)

**Current Problems**:
- Pawn/bishop confusion
- Low classification accuracy (28%)
- Duplicate detections

**Root Causes**:
1. **Model trained on different images than your specific setup** - Generic training data doesn't match your pieces, lighting, and camera
2. **Duplicate detections** - NMS threshold too lenient (FIXED: increased from 0.5 to 0.7)

**Solution**: Fine-tune model on YOUR specific board photos. You have 30 existing photos in `data/training/board_photos/` - use them to retrain. See sections below.

## Why Fine-tune?

The pre-trained model doesn't recognize your pieces' characteristics:
- Different piece designs (Staunton vs decorative vs minimalist)
- Different colors (brown vs black, ivory vs white)
- Different sizes and proportions
- Different materials (wood vs plastic vs metal)
- Different surface finishes (glossy vs matte)

Fine-tuning adapts the model to YOUR specific piece set in 1-2 hours.

## Quick Start (5-Step Process)

### Step 1: Collect Training Photos (15-20 minutes)

Capture 30+ photos of your chessboard with **various piece positions**:

```bash
python scripts/collect_piece_training_photos.py --device /dev/video0 --count 30
```

**IMPORTANT - Vary board positions between photos:**
- Include ALL piece types in different photos:
  - Pawns, Knights, Bishops, Rooks, Queens, Kings
  - Both white AND black pieces
- Try different game positions:
  - Opening positions (e.g., e4, d4, Nf3)
  - Mid-game positions (pieces scattered across board)
  - Endgame positions (few pieces remaining)
  - Edge/corner positions (test boundary detection)
- Vary lighting if possible
- Keep all 4 corners visible (needed for perspective transform)

**Why this matters:**
The model needs to see each piece type in various positions and contexts to learn to recognize them reliably. 30 photos with 10-20 pieces each = 300-600 piece examples.

### Step 2: Label Pieces (30-60 minutes)

Interactively draw bounding boxes around each piece:

```bash
python scripts/label_pieces.py --input data/training/piece_photos
```

**Labeling instructions:**
- Draw a box around EACH piece in the photo
- Press keyboard shortcuts to select piece class BEFORE drawing:
  - **Black pieces** (lowercase): b=bishop, k=king, n=knight, p=pawn, q=queen, r=rook
  - **White pieces** (uppercase): B=Bishop, K=King, N=Knight, P=Pawn, Q=Queen, R=Rook
- Press u to undo last box
- Press ENTER to save and move to next image
- Press s to skip image, q to quit

**Tips:**
- Draw tight boxes around pieces (include whole piece, not extra board)
- Be accurate with piece classes (don't mislabel pawns as bishops, etc.)
- It's okay to skip photos if pieces are unclear/blurry
- Take breaks - accuracy is more important than speed

This creates a YOLO dataset at `data/training/piece_dataset/`

### Step 3: Fine-tune Model (30-60 minutes)

Train the model on your labeled data:

```bash
python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml
```

**What happens:**
- Loads the base model (`data/best_transformed_detection.pt`)
- Fine-tunes on your labeled photos (100 epochs by default)
- Saves best model to `data/training/runs/piece_finetune/weights/best.pt`
- Shows training progress, metrics, and plots

**Time estimate:**
- 30 photos: ~30 minutes on CPU
- 50 photos: ~60 minutes on CPU
- With GPU: 5-10 minutes

The script uses data augmentation to artificially expand your dataset (rotation, brightness, scale variations).

### Step 4: Deploy Fine-tuned Model

Replace the original model with your fine-tuned version:

```bash
# Backup original
mv data/best_transformed_detection.pt data/best_transformed_detection_original.pt

# Use fine-tuned model
cp data/training/runs/piece_finetune/weights/best.pt data/best_transformed_detection.pt
```

### Step 5: Test!

Run the full pipeline with your new model:

```bash
python scripts/best_move_demo.py --debug
```

Check the debug visualizations:
- `chessboard_detections.png` - Should show all pieces with correct labels
- `chessboard_grid.png` - Should show pieces correctly mapped to squares

## Expected Results

**Before fine-tuning:**
- 27/32 pieces detected (84% detection rate)
- 9/32 correct classification (28% accuracy)
- Frequent pawn/bishop confusion
- Missed pieces

**After fine-tuning (typical):**
- 30-32/32 pieces detected (95%+ detection rate)
- 28-31/32 correct classification (85-95% accuracy)
- Rare misclassifications
- Robust to board movement

## Advanced Options

### Collect More Photos for Better Accuracy

More photos = better accuracy (diminishing returns after ~100 photos):

```bash
python scripts/collect_piece_training_photos.py --device /dev/video0 --count 50
```

Recommended for competitive/production use: 50-100 photos

### Adjust Training Parameters

For better results with larger datasets:

```bash
# More epochs (better convergence, slower)
python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml --epochs 150

# Larger image size (more detail, slower)
python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml --imgsz 1280
```

### Use GPU (if available)

Edit `scripts/finetune_pieces.py` and change:
```python
device='cpu'  # Change to device=0 for GPU
```

Training will be 10-20x faster on GPU.

## Troubleshooting

### "Not enough training photos"

You need at least 15 labeled photos with good piece variety. Capture more:
```bash
python scripts/collect_piece_training_photos.py --device /dev/video0 --count 50
```

### "Model still misclassifies pieces"

1. **Check class balance** - Ensure you have examples of ALL 12 piece types:
   - Open labeled dataset and verify you labeled pawns, knights, bishops, rooks, queens, kings
   - Both black AND white versions

2. **Check labeling accuracy**:
   - Review `data/training/piece_dataset/labels/*.txt`
   - Each line format: `class_id x_center y_center width height`
   - class_id should be 0-11 (see mapping below)

3. **Increase training epochs**:
   ```bash
   python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml --epochs 150
   ```

4. **Collect more diverse training data**:
   - More photos with different piece arrangements
   - Include edge cases (pieces at board edges, clustered pieces)

### "Specific piece type always wrong (e.g., knights)"

This means you don't have enough examples of that piece type:
- Collect 5-10 more photos that specifically include knights in various positions
- Label them carefully
- Re-train

### "Training is very slow"

This is normal on CPU. Options:
- Reduce batch size in `finetune_pieces.py` (change `batch=8` to `batch=4`)
- Reduce epochs (try `--epochs 50`)
- Use GPU if available

### "Pieces detected in empty squares"

This is a false positive issue:
- Increase detection confidence threshold in `guidance/board_detector.py`
- Change `base_conf` parameter (e.g., 0.35 → 0.45)
- Collect more photos with empty boards

## Piece Class Mapping

The model uses 12 classes (0-11):

```
Black pieces (lowercase labels):
  0: b-bishop
  1: b-king
  2: b-knight
  3: b-pawn
  4: b-queen
  5: b-rook

White pieces (uppercase labels):
  6: W-bishop
  7: W-king
  8: W-knight
  9: W-pawn
 10: W-queen
 11: W-rook
```

When labeling, use the keyboard shortcuts:
- Black: b, k, n, p, q, r (lowercase)
- White: B, K, N, P, Q, R (uppercase)

## How It Works

The fine-tuning process uses **transfer learning**:

1. **Base model** (trained on generic chess pieces) provides general piece detection knowledge
2. **Fine-tuning** adapts the model to YOUR specific pieces by:
   - Learning your pieces' color patterns
   - Learning your pieces' shapes and sizes
   - Learning how they appear under your lighting
   - Learning their appearance from your camera's perspective
3. **Data augmentation** artificially expands your dataset:
   - Brightness variations (simulate lighting changes)
   - Small rotations (simulate board alignment variations)
   - Scale variations (simulate camera distance changes)
4. **Result**: Model becomes an expert at detecting YOUR pieces

The fine-tuned model retains general piece detection ability while being highly optimized for your setup.

## Dataset Quality Tips

**Good training data:**
- Clear, sharp photos (1280x720 YUYV format)
- All piece types represented multiple times
- Variety of board positions
- Tight, accurate bounding boxes
- Correct class labels

**Bad training data:**
- Blurry photos
- Only a few piece types (e.g., just pawns)
- Same board position repeated
- Loose bounding boxes (too much empty space)
- Mislabeled pieces

**Rule of thumb:**
Quality > Quantity. 30 well-labeled diverse photos beats 100 repetitive or mislabeled photos.

## Monitoring Training Progress

During training, watch for:
- **Loss decreasing** (metrics/mAP50, metrics/mAP50-95 increasing)
- **Validation mAP > 0.8** means excellent accuracy
- **Early stopping** if loss plateaus (patience=15 epochs)

Training plots are saved to `data/training/runs/piece_finetune/`:
- `results.png` - Training metrics over time
- `confusion_matrix.png` - Which pieces get confused
- `labels.jpg` - Distribution of labeled data

## Using Your Existing Training Data (30 Board Photos)

If you already have board photos in `data/training/board_photos/`, you can retrain without collecting more:

**Regenerate dataset from your existing board photos:**

```bash
python -m guidance.training.data_collector \
    --source data/training/board_photos/ \
    --output data/training/piece_dataset_updated/ \
    --min-conf 0.5
```

**Train new model on YOUR specific pieces:**

```bash
python -m guidance.training.train_yolo \
    --dataset data/training/piece_dataset_updated/ \
    --epochs 200 \
    --batch 16 \
    --model m
```

**Deploy:**

```bash
cp runs/train/chess_pieces/weights/best.pt data/best_transformed_detection.pt
```

This will fine-tune the model to recognize YOUR specific pieces, lighting, and camera setup, dramatically improving accuracy.

## Quick Detection Parameter Tuning

For immediate (but limited) improvements without retraining:

**1. Increase NMS threshold (reduce duplicates)** - ALREADY DONE
   - Changed `iou=0.5` to `iou=0.7` in `guidance/board_detector.py:438`

**2. Increase base confidence (reduce false positives):**

Edit `guidance/board_detector.py:437`:
```python
conf=0.50,  # UP from 0.35
```

**3. Adjust class-specific thresholds (reduce pawn/bishop confusion):**

Edit `guidance/board_detector.py:453-468`:
```python
class_thresholds = {
    0: 0.45,  # black bishop (UP from 0.35)
    1: 0.35,  # black king
    2: 0.35,  # black knight
    3: 0.60,  # black pawn (UP from 0.45)
    4: 0.35,  # black queen
    5: 0.35,  # black rook
    6: 0.45,  # white bishop (UP from 0.35)
    7: 0.35,  # white king
    8: 0.35,  # white knight
    9: 0.60,  # white pawn (UP from 0.45)
    10: 0.35, # white queen
    11: 0.35, # white rook
}
```

These are temporary fixes. Retraining on your data is the proper solution for high accuracy.

## Next Steps After Fine-tuning

Once piece detection works reliably:

1. **Test on real games** - Run full pipeline and verify FEN notation is correct
2. **Iterate if needed** - If specific positions fail, add those to training data
3. **Integrate with robot** - Use the guidance overlay for robot control
4. **Deploy to production** - Your model is now customized for your setup!

## Comparison with Corner Fine-tuning

**Corner fine-tuning:**
- Simpler (1 class: corner)
- Faster labeling (4 points per image)
- Fewer photos needed (20+)
- Faster training (50 epochs)

**Piece fine-tuning:**
- More complex (12 classes: 6 piece types × 2 colors)
- Slower labeling (10-20 boxes per image)
- More photos needed (30+)
- Longer training (100 epochs)

Both are important for full pipeline accuracy!
