# Baseline Performance Analysis - chessboardv2.png

**Date:** 2025-01-XX
**Image:** data/chessboardv2.png (360x360, starting position, 32 pieces)
**Model:** data/best_transformed_detection.pt (current)

---

## Results Summary

| Metric | Without Preprocessing | With Preprocessing |
|--------|---------------------|-------------------|
| **Accuracy** | 25.0% (8/32) | 28.1% (9/32) |
| **Missing** | 31.2% (10 pieces) | 28.1% (9 pieces) |
| **Misclassified** | 43.8% (14 pieces) | 43.8% (14 pieces) |
| **Detection Count** | 30/32 | 31/32 |

**Conclusion:** Current model is critically underperforming. Preprocessing provides minimal benefit (+3.1%).

---

## Error Analysis

### 1. Queen Overclassification (Most Critical Issue)

**Pieces misclassified as Queens:**
- e1 white **king** → queen
- e8 black **king** → queen
- c1, f1 white **bishops** → queens
- h8 black **rook** → queen
- d2, e2 white **pawns** → queens
- h7 black **pawn** → queen

**Root Cause:** Model is likely:
1. Overtrained on queen examples
2. Using queen as "uncertain" fallback class
3. Confusing tall pieces (king, bishop, queen) due to similar heights

**Solution:**
- Collect balanced dataset (equal samples per class)
- Increase queen confidence threshold to 0.55
- Add class weighting during training to penalize queen false positives

---

### 2. Row 7 Pawn Detection Failure

**Missing black pawns:** b7, c7, d7, e7, f7, g7 (6/8 pawns!)

**Root Cause:** Training dataset likely has:
- Few examples of pawns in starting positions (row 7)
- Most training data from mid-game positions
- Pawns in rows 4-6 overrepresented

**Solution:**
- Specifically collect starting position boards
- Include early-game positions (moves 1-5)
- Augment dataset with synthetic starting positions

---

### 3. Bishop/Pawn Confusion

**Misclassifications:**
- c8 black bishop → pawn
- f8 black bishop → pawn

**Root Cause:**
- Similar heights for bishop and pawn
- Color/texture similarity between dark pieces
- Perspective distortion at board edges

**Solution:**
- Color normalization improvements (LAB space already implemented)
- Collect more bishops at edge positions (a/h files)
- Add shape-based features (bishop cross vs pawn dome)

---

### 4. Knight Misclassifications

**Errors:**
- b8 black knight → king
- g8 black knight → white knight (wrong color!)
- g1 white knight → bishop

**Root Cause:**
- Knight shape is most distinctive but also most variable (angle-dependent)
- Color detection failure (g8 example)

**Solution:**
- Augment with knight rotations (±15°)
- Improve lighting normalization
- More black/white distinction features

---

## Missing Detections Breakdown

**Without Preprocessing:**
- White pawns: b2, g2, h2 (3 pieces)
- Black pawns: b7-g7 (7 pieces, except a7 and h7)
- Black rook: a8 (1 piece)

**With Preprocessing:**
- White knight: b1, pawn h2 (2 pieces)
- Black knight: g8, pawns: b7-g7 (7 pieces)
- Black rook: a8 (1 piece)

**Pattern:** Consistent failure on:
1. Row 7 black pawns (training data issue)
2. Corner pieces (a8 rook, h2 pawn)
3. Edge pieces affected by top margin or grid misalignment

---

## Recommended Improvements Priority

### HIGH PRIORITY (Do First)

1. **Disable preprocessing temporarily** (minimal benefit, adds complexity)
   - Revert to baseline for consistent benchmarking
   - Revisit after model retraining

2. **Lower confidence threshold further** (0.35 → 0.25)
   - Catch missing pieces (10-9 missing is still too many)
   - Accept more false positives initially

3. **Increase queen threshold significantly** (0.45 → 0.60)
   - Combat queen overclassification
   - May reduce false queens by 50%

### MEDIUM PRIORITY (Before Data Collection)

4. **Add per-class confidence matrix**
   ```python
   class_thresholds = {
       'queen': 0.60,  # High threshold (overclassified)
       'pawn': 0.40,   # Medium (some confusion)
       'king': 0.40,   # Medium (confused with queen)
       'bishop': 0.40, # Medium (confused with pawn)
       'knight': 0.35, # Low (distinctive shape)
       'rook': 0.35,   # Low (distinctive shape)
   }
   ```

5. **Fix grid/margin issues for row 7**
   - Investigate why 7/8 pawns missing
   - May need to adjust top margin or grid calculation
   - Test with different board sizes

### LOW PRIORITY (After Data Collection)

6. **Refine preprocessing pipeline**
   - Test individually: CLAHE, color norm, sharpening
   - A/B test each component
   - May provide 5-10% boost after better model

---

## Data Collection Targets

Based on error analysis, prioritize collecting:

1. **Starting Position Boards** (100+ images)
   - Critical for row 7 pawn detection
   - Vary lighting, angles, chess sets

2. **Queens** (50+ examples, but mark clearly)
   - Model needs to STOP defaulting to queen
   - Collect with high-confidence labels only

3. **Bishops** (100+ examples)
   - Especially at board edges (a/h files)
   - Various angles and colors

4. **Kings** (100+ examples)
   - Often confused with queens
   - Need distinctive features emphasized

5. **Knights** (100+ examples)
   - All rotations/angles
   - Equal black and white (color confusion seen)

6. **Edge Positions** (focus area)
   - Pieces on a-file, h-file
   - Rank 7 and 8 specifically

**Total Target:** 2400+ pieces (200 per class minimum)

---

## Metrics to Track After Retraining

| Metric | Current | Target |
|--------|---------|--------|
| Overall Accuracy | 28% | 80%+ |
| Detection Recall | 72% | 95%+ |
| Queen Precision | ~40% | 90%+ |
| Row 7 Pawn Recall | 12.5% | 90%+ |
| Classification F1 | ~35% | 85%+ |

---

## Next Steps

1. ✅ Run baseline analysis (DONE)
2. ⬜ Implement high-priority threshold adjustments
3. ⬜ Test threshold changes on chessboardv2.png
4. ⬜ Collect 100 starting position boards
5. ⬜ Run data_collector.py on collected images
6. ⬜ Train YOLOv8m model (150 epochs)
7. ⬜ Evaluate with tools/evaluate.py
8. ⬜ Compare metrics against baseline
9. ⬜ Iterate if needed

---

**Conclusion:** Current model needs complete retraining. The 28% accuracy is unacceptable for production use. However, we now have:
- Clear baseline metrics
- Identified error patterns
- Specific data collection targets
- Measurable improvement goals

The infrastructure is in place (preprocessing, training pipeline, evaluation tools). Now we collect data and retrain! 🚀
