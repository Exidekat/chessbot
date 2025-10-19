# Quick Start Guide - YOLO Piece Detection Improvements

**You are here:** ✅ Phase 1 Complete, Ready for Phase 2 (Data Collection)

---

## What We've Accomplished

### ✅ Implemented (Ready to Use)

1. **Inference Improvements** (board_state.py)
   - Preprocessing pipeline (LAB color normalization, CLAHE, sharpening)
   - Lower confidence threshold (0.35)
   - Test-time augmentation
   - Class-specific confidence filtering

2. **Training Tools** (tools/)
   - `data_collector.py` - Extract & label pieces from boards
   - `train.py` - Train YOLO with optimized hyperparameters
   - `evaluate.py` - Detailed performance metrics
   - `analyze_test_board.py` - Compare against known positions

3. **Baseline Analysis** (BASELINE_ANALYSIS.md)
   - Current accuracy: **28%** (needs retraining!)
   - Identified queen overclassification
   - Found row 7 pawn detection failure
   - Set improvement targets: 80%+ accuracy

---

## Current Test Results (chessboardv2.png)

**Expected:** 32 pieces (standard starting position)
**Detected:** 31 pieces
**Correct:** 9 pieces (28% accuracy) ❌

**Major Issues:**
- Queens: Everything gets classified as a queen!
- Row 7 Pawns: 6/8 black pawns missing
- Bishops: Confused with pawns

**Conclusion:** Model desperately needs retraining!

---

## Quick Start: Improve Your Model

### Step 1: Collect Board Images (This Weekend)

Take photos of chess boards:
- **100+ starting position boards** (critical!)
- Different chess sets (wood colors, styles)
- Various lighting (natural, indoor, shadows)
- Different angles (straight-on, slight tilt)
- Use your phone camera or webcam

**Requirements:**
- All 4 corners visible
- Pieces clearly visible
- Good lighting
- 800x800px minimum (bigger is better)

Save to: `board_images/`

---

### Step 2: Extract & Label Pieces (5 minutes)

```bash
# Extract pieces from your board photos
python tools/data_collector.py \
    --source board_images/ \
    --output dataset/ \
    --min-conf 0.5
```

**Output:** `dataset/images/` and `dataset/labels/` in YOLO format

**Target:** 200+ pieces per class (2400+ total)
- If you only get 500 pieces, collect more boards!
- Starting positions give you 32 pieces each (100 boards = 3200 pieces)

---

### Step 3: Train New Model (Cloud/Colab - 1-2 hours)

```bash
# On Google Colab or local GPU
python tools/train.py \
    --dataset dataset/ \
    --epochs 150 \
    --batch 16 \
    --model m
```

**Time:**
- GPU: 1-2 hours
- CPU: 6-12 hours (not recommended)
- **Recommended:** Use Google Colab free GPU

**Output:** `runs/train/chess_pieces/weights/best.pt`

---

### Step 4: Evaluate New Model (1 minute)

```bash
# Check if it's better!
python tools/evaluate.py \
    --model runs/train/chess_pieces/weights/best.pt \
    --data dataset/ \
    --output evaluation_results/
```

**Look for:**
- Precision > 80%
- Recall > 90%
- F1-Score > 85%
- Low queen false positives

---

### Step 5: Deploy & Test (30 seconds)

```bash
# Replace old model with new one
cp runs/train/chess_pieces/weights/best.pt data/best_transformed_detection.pt

# Test on your starting position board
python tools/analyze_test_board.py --image data/chessboardv2.png
```

**Success Criteria:**
- Accuracy > 80% (vs current 28%)
- All 32 pieces detected
- No queen spam!

---

## Google Colab Setup (Recommended)

1. **Upload your code to Google Drive**
   ```bash
   # On your computer
   zip -r chessbot.zip chessbot/
   # Upload chessbot.zip to Google Drive
   ```

2. **Create new Colab notebook:**
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Extract code
   !unzip /content/drive/MyDrive/chessbot.zip
   %cd chessbot

   # Install dependencies
   !pip install -r requirements.txt

   # Train model (will use free GPU automatically)
   !python tools/train.py --dataset dataset/ --epochs 150 --model m

   # Download trained model
   from google.colab import files
   files.download('runs/train/chess_pieces/weights/best.pt')
   ```

3. **Copy best.pt back to your computer**

---

## Testing Your Current Setup

Before collecting data, verify everything works:

```bash
# Test 1: Current baseline
python tools/analyze_test_board.py --image data/chessboardv2.png
# Should show: 28% accuracy, lots of queens

# Test 2: Main detection (with debug)
python main.py --image data/chessboardv2.png --no-bestmove --debug
# Should detect 31/32 pieces

# Test 3: Verify tools work
python tools/data_collector.py --help
python tools/train.py --help
python tools/evaluate.py --help
```

All should run without errors!

---

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install -r requirements.txt
```

### "Corner detection failed"
Your image may have:
- Obscured corners (pieces sitting on them)
- Poor lighting
- Board not fully visible

Try: `--corner-conf 0.05 --min-corner-dist 30`

### "Training too slow on CPU"
Use Google Colab (free GPU) or reduce epochs:
```bash
python tools/train.py --epochs 100 --batch 8
```

### "Not enough training data"
You need minimum:
- 100 board images
- 200+ pieces per class
- Collect more photos!

---

## Expected Timeline

**Week 1 (This Weekend):**
- ✅ Baseline established (DONE)
- ⬜ Collect 100+ board photos
- ⬜ Run data_collector.py

**Week 2:**
- ⬜ Train model on Colab (2 hours)
- ⬜ Evaluate results
- ⬜ Test on real boards

**Week 3:**
- ⬜ Collect more data for weak classes (if needed)
- ⬜ Retrain with full dataset
- ⬜ Final evaluation

**Goal:** 80%+ accuracy by end of Week 3

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `board_state.py` | Main detection code (modified) |
| `tools/data_collector.py` | Extract pieces from boards |
| `tools/train.py` | Train YOLO model |
| `tools/evaluate.py` | Evaluate model performance |
| `tools/analyze_test_board.py` | Test against known positions |
| `BASELINE_ANALYSIS.md` | Current performance metrics |
| `IMPROVEMENTS_README.md` | Detailed documentation |

---

## What to Collect (Priority Order)

1. **Starting position boards** (100 images) - HIGHEST PRIORITY
   - Fixes row 7 pawn detection
   - Gives 3200 piece examples
   - Easy to verify labels

2. **Mid-game positions** (50 images)
   - Varied piece positions
   - Real-world scenarios

3. **Different chess sets** (focus on variety)
   - Dark wood, light wood
   - Plastic, wooden
   - Different styles

4. **Lighting variations**
   - Natural window light
   - Indoor overhead
   - Shadows present

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Detection Accuracy | 28% | 80% | 95% |
| Detection Recall | 72% | 90% | 98% |
| Queen Precision | 40% | 85% | 95% |
| Row 7 Pawn Recall | 12% | 90% | 98% |

---

## You Are Here 📍

```
[✅ Phase 1] → [📸 Photo Collection] → [🔄 Training] → [📊 Evaluation] → [🚀 Deploy]
   DONE         YOU ARE HERE           2 hours       5 minutes      30 seconds
```

**Next Action:** Take 100 photos of chess boards in starting position! 📸

---

## Questions?

- Check `IMPROVEMENTS_README.md` for detailed docs
- Check `BASELINE_ANALYSIS.md` for error patterns
- Run tools with `--help` for usage

**Good luck! You've got everything you need!** 🎯
