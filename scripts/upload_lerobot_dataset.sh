#!/bin/bash
# Upload ChessBot LeRobot dataset to HuggingFace Hub
#
# Usage:
#   ./scripts/upload_lerobot_dataset.sh                    # Upload to default repo
#   ./scripts/upload_lerobot_dataset.sh --repo user/name   # Upload to custom repo
#   ./scripts/upload_lerobot_dataset.sh --dry-run          # Preview without uploading
#
# Prerequisites:
#   - huggingface-cli login (or HF_TOKEN environment variable)
#   - pip install huggingface_hub

set -e

# Default configuration
DEFAULT_REPO="exidekat/chessbot-lerobot"
DATASET_PATH="data/lerobot_episodes"
REPO_TYPE="dataset"

# Parse arguments
REPO_ID="$DEFAULT_REPO"
DRY_RUN=false
PRIVATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --help|-h)
            echo "Upload ChessBot LeRobot dataset to HuggingFace Hub"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --repo REPO_ID    HuggingFace repo (default: $DEFAULT_REPO)"
            echo "  --dry-run         Preview files without uploading"
            echo "  --private         Create private repository"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment:"
            echo "  HF_TOKEN          HuggingFace API token (alternative to huggingface-cli login)"
            exit 0
            ;;
        *)
            echo "[X] Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "ChessBot LeRobot Dataset Upload"
echo "=============================================="
echo ""

# Check dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}[X] Dataset not found at: $DATASET_PATH${NC}"
    echo "    Run scripts/convert_to_lerobot.py first to create the dataset"
    exit 1
fi

# Check for required files
REQUIRED_FILES=(
    "meta/info.json"
    "meta/stats.json"
    "data/chunk-000/file-000.parquet"
)

echo "[*] Checking dataset structure..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATASET_PATH/$file" ]; then
        echo -e "${RED}[X] Missing required file: $file${NC}"
        exit 1
    fi
done
echo -e "${GREEN}[OK] Dataset structure valid${NC}"

# Check HuggingFace authentication
echo ""
echo "[*] Checking HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo -e "${GREEN}[OK] Using HF_TOKEN environment variable${NC}"
elif huggingface-cli whoami &>/dev/null; then
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    echo -e "${GREEN}[OK] Logged in as: $HF_USER${NC}"
else
    echo -e "${RED}[X] Not logged in to HuggingFace${NC}"
    echo "    Run: huggingface-cli login"
    echo "    Or set: export HF_TOKEN=your_token"
    exit 1
fi

# Show dataset stats
echo ""
echo "[*] Dataset statistics:"
TOTAL_SIZE=$(du -sh "$DATASET_PATH" | cut -f1)
NUM_FILES=$(find "$DATASET_PATH" -type f | wc -l)
echo "    Path: $DATASET_PATH"
echo "    Size: $TOTAL_SIZE"
echo "    Files: $NUM_FILES"

# Read info.json for episode count
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    TOTAL_EPISODES=$(python3 -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
    TOTAL_FRAMES=$(python3 -c "import json; print(json.load(open('$DATASET_PATH/meta/info.json'))['total_frames'])" 2>/dev/null || echo "?")
    echo "    Episodes: $TOTAL_EPISODES"
    echo "    Frames: $TOTAL_FRAMES"
fi

echo ""
echo "[*] Target repository: $REPO_ID"

# Dry run - just list files
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}[DRY RUN] Would upload the following files:${NC}"
    find "$DATASET_PATH" -type f | sort | while read -r file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "    $SIZE  ${file#$DATASET_PATH/}"
    done
    echo ""
    echo "Run without --dry-run to upload"
    exit 0
fi

# Create README for the dataset
echo ""
echo "[*] Creating dataset README..."
README_PATH="$DATASET_PATH/README.md"
cat > "$README_PATH" << 'EOF'
---
license: mit
task_categories:
  - robotics
tags:
  - lerobot
  - chess
  - manipulation
  - so100
  - vla
---

# ChessBot LeRobot Dataset

Vision-language-action (VLA) training dataset for chess piece manipulation with SO-100 robot arm.

## Dataset Description

This dataset contains teleoperated demonstrations of a robot arm manipulating chess pieces on a physical chessboard. Each episode captures:
- **Global camera**: Overhead view of the chessboard (1280x720)
- **Gripper camera**: Wrist-mounted camera view (224x224)
- **Robot state**: 6-DOF joint positions
- **Actions**: Target joint positions

## Intended Use

Fine-tuning VLA models (PI0, SmolVLA) for chess manipulation tasks with natural language instructions.

## Dataset Structure

```
lerobot_episodes/
  data/           # Parquet files with state/action data
  videos/         # MP4 videos per camera
  meta/           # Episode metadata and statistics
  images/         # Decoded frames (optional)
```

## Citation

If you use this dataset, please cite the ChessBot project.
EOF

echo -e "${GREEN}[OK] README created${NC}"

# Build upload command
UPLOAD_CMD="huggingface-cli upload $REPO_ID $DATASET_PATH . --repo-type $REPO_TYPE"

if [ "$PRIVATE" = true ]; then
    UPLOAD_CMD="$UPLOAD_CMD --private"
fi

# Execute upload
echo ""
echo "[*] Uploading to HuggingFace..."
echo "    Command: $UPLOAD_CMD"
echo ""

if eval "$UPLOAD_CMD"; then
    echo ""
    echo -e "${GREEN}=============================================="
    echo "[OK] Upload complete!"
    echo "=============================================="
    echo ""
    echo "Dataset URL: https://huggingface.co/datasets/$REPO_ID"
    echo ""
    echo "To use in training:"
    echo "  python scripts/vla_finetune.py --dataset $REPO_ID"
    echo "${NC}"
else
    echo ""
    echo -e "${RED}[X] Upload failed${NC}"
    exit 1
fi
