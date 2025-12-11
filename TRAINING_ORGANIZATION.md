# Training Organization

This document explains how training scripts are organized to avoid confusion.

## Directory Structure

```
chessbot/
├── data/
│   ├── best_corners.pt              # Corner detection model (what we fine-tune)
│   ├── best_transformed_detection.pt # Piece detection model (separate)
│   └── training/
│       ├── board_photos/            # Raw photos for corner training
│       ├── corner_dataset/          # Labeled corner dataset (YOLO format)
│       │   ├── images/              # Training images
│       │   ├── labels/              # Corner labels (4 per image)
│       │   └── data.yaml            # Dataset config
│       └── runs/
│           └── corner_finetune/     # Training outputs
│               └── weights/
│                   └── best.pt      # Fine-tuned corner model
│
├── scripts/
│   ├── collect_corner_training_photos.py  # CORNER detection photo capture
│   ├── label_corners.py                   # CORNER labeling tool
│   └── finetune_corners.py                # CORNER model fine-tuning
│
├── CORNER_FINETUNING.md             # Guide for corner detection fine-tuning
└── TRAINING_ORGANIZATION.md         # This file
```

## Naming Convention

All corner detection training files are prefixed/named clearly:

- **Scripts**: `*_corner_*` or `*_corners_*`
- **Documentation**: `CORNER_*.md`
- **Models**: `best_corners.pt` (not just "best.pt")
- **Datasets**: `corner_dataset/` (not just "dataset/")

## Two Separate Models

The system uses **two different YOLO models**:

### 1. Corner Detection Model (`best_corners.pt`)
- **Purpose**: Find the 4 corners of the chessboard
- **Input**: Raw camera photo
- **Output**: 4 corner points (TL, TR, BR, BL)
- **Training data**: Photos with labeled corners
- **Classes**: 1 class ("corner")

### 2. Piece Detection Model (`best_transformed_detection.pt`)
- **Purpose**: Identify chess pieces on the board
- **Input**: Transformed/warped board image
- **Output**: Piece bounding boxes with class labels
- **Training data**: Photos with labeled pieces
- **Classes**: 12 classes (black/white × 6 piece types)

**IMPORTANT**: These are completely separate models. Fine-tuning one does NOT affect the other.

## Workflow Overview

### Corner Detection Fine-tuning (This Guide)
```
1. Collect photos    → collect_corner_training_photos.py
2. Label corners     → label_corners.py
3. Fine-tune model   → finetune_corners.py
4. Deploy model      → Replace data/best_corners.pt
```

### Piece Detection Fine-tuning (Future)
```
1. Collect photos with pieces
2. Label pieces (12 classes)
3. Fine-tune piece model
4. Deploy model → Replace data/best_transformed_detection.pt
```

## Why This Organization?

**Problem**: Generic names like "collect_training_photos.py" are ambiguous:
- Training for what? Corners? Pieces? Something else?
- Which model does it affect?
- Can I use it for both models?

**Solution**: Specific names make it clear:
- `collect_corner_training_photos.py` → Obviously for corner training
- `finetune_corners.py` → Obviously fine-tunes corner model
- `CORNER_FINETUNING.md` → Obviously about corner fine-tuning

## Quick Reference

| Task | Script | Input | Output |
|------|--------|-------|--------|
| Capture corner photos | `collect_corner_training_photos.py` | Camera | `data/training/board_photos/*.png` |
| Label corners | `label_corners.py` | Board photos | `data/training/corner_dataset/` |
| Fine-tune corners | `finetune_corners.py` | Corner dataset | `data/training/runs/corner_finetune/weights/best.pt` |

## Future: Piece Detection Fine-tuning

When we add piece detection fine-tuning, we'll create:
- `collect_piece_training_photos.py`
- `label_pieces.py`
- `finetune_pieces.py`
- `PIECE_FINETUNING.md`

This keeps corner and piece training completely separate and obvious.
