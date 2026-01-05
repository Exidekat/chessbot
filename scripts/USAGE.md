# Script Usage Guide

**NOTE**: This guide uses the new standardized argument names. See STANDARDIZATION.md for migration details from old argument names.

## Setup and Configuration

# Download required YOLO models (run on first setup)
python scripts/download.py

# Create default configuration file
python scripts/create_config.py

# Calibrate SO-100 robot arm (saves to data/so100_config_ttyACM0.csv)
python scripts/create_so100_config.py --port /dev/ttyACM0

# Calibrate second SO-100 robot (saves to data/so100_config_ttyACM1.csv)
python scripts/create_so100_config.py --port /dev/ttyACM1

# Calibrate with custom output path
python scripts/create_so100_config.py --port /dev/ttyACM0 --output data/my_robot_config.csv

# Setup virtual camera device for live overlay streaming (one-time setup, requires sudo)
sudo modprobe v4l2loopback devices=1 video_nr=7 card_label="ChessBot Virtual Cam" exclusive_caps=1

## Robot Control (SO-100)

# Teleoperation with interactive menu - scans ports, loads port-specific configs, shows menu
python scripts/tele_op.py

# Test mode - hold robots at home for 10 seconds then exit (no keyboard input required)
python scripts/tele_op.py --test

# Teleoperation with custom config directory
python scripts/tele_op.py --config-dir data/

# Interactive Menu Options:
#   [1] Exit - Cleanly disconnect all robots
#   [2] Tele-op Leader/Follower - Mirror leader arm movements on follower at 15Hz
#       Mode selection: [1] Home-to-Home (default), [2] EncPos-to-EncPos
#   [3] Adjust Home Positions - Save new home positions to config (press 0-5 to save joint)
#       Mode selection: [1] Manual (default) - disable torque, manually position robot
#                       [2] Tele-op - use leader arm to guide follower

## Camera Capture

# Capture photo with auto-detected camera (1920x1080, saves to results/ with timestamp)
python scripts/capture_photo.py

# Capture photo with specific camera device
python scripts/capture_photo.py --device /dev/video0

# Capture photo with custom resolution (4000x3000 maximum for 4K camera)
python scripts/capture_photo.py --device /dev/video0 --width 4000 --height 3000

# Capture photo with custom output in results/ directory
python scripts/capture_photo.py --output results/my_board.png

# Capture photo immediately without preview window
python scripts/capture_photo.py --no-preview

# Capture photo with specific device, resolution, and output
python scripts/capture_photo.py --device /dev/video0 --width 2592 --height 1944 --output results/board_image.png

## Demo and Testing

# Run full pipeline: capture 4K MJPEG -> 720p photo + detect board + calculate best move (auto-detects camera)
python scripts/best_move_demo.py

# Run full pipeline with debug visualizations enabled
python scripts/best_move_demo.py --debug

# Run full pipeline with specific global camera device (UPDATED ARG NAME)
python scripts/best_move_demo.py --global-camera /dev/video0

# Run full pipeline without calculating best move (no Stockfish required)
python scripts/best_move_demo.py --no-bestmove

# Run full pipeline with custom corner detection parameters (optimized for 720p)
python scripts/best_move_demo.py --debug --corner-conf 0.005 --min-corner-dist 30

# Run full pipeline with custom engine time limit
python scripts/best_move_demo.py --time 2.0

# Run full pipeline with all custom parameters (UPDATED ARG NAMES)
python scripts/best_move_demo.py --global-camera /dev/video0 --debug --corner-conf 0.005 --min-corner-dist 30 --time 2.0 --turn white --rotation right

## VLA Training Data Collection

# Generate stage-by-stage move overlays for VLA training (captures photo, detects board, calculates best move, generates interactive overlays)
python scripts/create_overlay_demo.py

# Generate overlays with specific camera device (UPDATED ARG NAME)
python scripts/create_overlay_demo.py --global-camera /dev/video0

# Generate overlays with custom corner detection parameters
python scripts/create_overlay_demo.py --corner-conf 0.005 --min-corner-dist 30

# Generate overlays for black's turn instead of white's turn (UPDATED ARG NAME)
python scripts/create_overlay_demo.py --turn black

# Generate overlays with board rotation (camera positioned on right side of board)
python scripts/create_overlay_demo.py --rotation right

# Generate overlays with all custom parameters (UPDATED ARG NAMES)
python scripts/create_overlay_demo.py --global-camera /dev/video0 --rotation right --turn black --corner-conf 0.005 --min-corner-dist 30

## Virtual Camera with Live Overlay (Advanced VLA Training)

# Setup virtual camera device (one-time setup, requires sudo)
sudo modprobe v4l2loopback devices=1 video_nr=7 card_label="ChessBot Virtual Cam" exclusive_caps=1

# Stream live 720p feed with move overlays to virtual camera /dev/video7 (continuously updates base image every ~1 second)
python scripts/virtual_overlay_demo.py

# Stream with specific camera device (UPDATED ARG NAME)
python scripts/virtual_overlay_demo.py --global-camera /dev/video0

# Stream with board rotation and custom parameters (UPDATED ARG NAMES)
python scripts/virtual_overlay_demo.py --global-camera /dev/video0 --rotation right --turn black

## VLA Episode Collection (Training Data Recording)

# Prerequisites: v4l2loopback loaded, tele_op.py running in Terminal 1
# Terminal 1: Run tele-op to control robot (required for episode collection)
python scripts/tele_op.py

# Terminal 2: Collect VLA training episodes at 15 FPS (passive recording from tele-op)
python scripts/collect_vla_episodes.py --output data/episodes/

# Collect with specific cameras
python scripts/collect_vla_episodes.py --output data/episodes/ --global-camera /dev/video7 --gripper-camera /dev/video0

# Collect at custom frame rate (30 FPS)
python scripts/collect_vla_episodes.py --output data/episodes/ --fps 30

# Collect with custom chess engine
python scripts/collect_vla_episodes.py --output data/episodes/ --engine /usr/local/bin/stockfish

# Collect with board rotation (camera positioned on right side)
python scripts/collect_vla_episodes.py --output data/episodes/ --rotation right

# Collect without LeRobot (raw file storage fallback)
python scripts/collect_vla_episodes.py --output data/episodes/ --no-lerobot

## VLA Episode Validation (Training Data Review)

# Interactive episode review (list, info, playback, mark good/bad, export)
python scripts/validate_vla_episodes.py --dataset data/episodes/

# List all episodes with basic info
python scripts/validate_vla_episodes.py --dataset data/episodes/ --list

# Show detailed info for specific episode
python scripts/validate_vla_episodes.py --dataset data/episodes/ --info 0

# Playback specific episode (controls: SPACE=pause, A/D=seek, W/S=speed, Q=quit)
python scripts/validate_vla_episodes.py --dataset data/episodes/ --play 0

# Export good episodes only to new directory
python scripts/validate_vla_episodes.py --dataset data/episodes/ --export data/episodes_filtered/

# Force raw file mode (disable LeRobot)
python scripts/validate_vla_episodes.py --dataset data/episodes/ --no-lerobot

## LeRobot Dataset Management

# Convert collected episodes to LeRobot format (required before finetuning)
python scripts/convert_to_lerobot.py --input data/episodes/ --output data/lerobot_episodes/

# Clean dataset by removing consecutive static frames (reduces dataset size ~25-30%)
python scripts/clean_lerobot_dataset.py --input data/lerobot_episodes/ --output data/clean_lerobot_episodes/

# Preview cleaning without creating output (dry run)
python scripts/clean_lerobot_dataset.py --dry-run

# Clean with custom tolerance (default: 0.01 radians)
python scripts/clean_lerobot_dataset.py --input data/lerobot_episodes/ --output data/clean_lerobot_episodes/ --tolerance 0.005

# Upload default dataset to HuggingFace (repo: exidekat/chessbot-lerobot)
./scripts/upload_lerobot_dataset.sh

# Upload cleaned dataset to HuggingFace
./scripts/upload_lerobot_dataset.sh --dataset data/clean_lerobot_episodes/

# Upload dataset to custom HuggingFace repository
./scripts/upload_lerobot_dataset.sh --repo myuser/my-chess-dataset

# Preview upload without actually uploading (dry run)
./scripts/upload_lerobot_dataset.sh --dry-run

# Upload as private repository
./scripts/upload_lerobot_dataset.sh --dataset data/clean_lerobot_episodes/ --private

## VLA Finetuning (Multi-Model: PI0, SmolVLA)

# Fine-tune PI0 model on collected episodes (default)
python scripts/vla_finetune.py --dataset data/lerobot_episodes/

# Fine-tune SmolVLA model on collected episodes
python scripts/vla_finetune.py --model smolvla --dataset data/lerobot_episodes/

# Fine-tune with custom output directory
python scripts/vla_finetune.py --model pi0 --output checkpoints/my_chess_pi0/

# Continue training from latest checkpoint
python scripts/vla_finetune.py --model pi0 --continue

# Resume from specific checkpoint
python scripts/vla_finetune.py --model smolvla --resume checkpoints/chess_smolvla/epoch_0050.pt

# Fine-tune with custom hyperparameters
python scripts/vla_finetune.py --model pi0 --epochs 50 --batch-size 2 --lr 1e-5

# Fine-tune with Weights & Biases logging
python scripts/vla_finetune.py --model pi0

## VLA Deployment (Multi-Model: PI0, SmolVLA)

# Deploy PI0 base model for chess robot control (auto-detects cameras)
python scripts/vla_deploy.py --model pi0 --no-robot

# Deploy SmolVLA base model
python scripts/vla_deploy.py --model smolvla --no-robot

# Deploy with specific cameras (global overhead + gripper)
python scripts/vla_deploy.py --model pi0 --no-robot --global-camera /dev/video4 --gripper-camera /dev/video0

# Deploy with fine-tuned PI0 checkpoint
python scripts/vla_deploy.py --model pi0 --checkpoint checkpoints/chess_pi0/best.pt --no-robot

# Deploy with fine-tuned SmolVLA checkpoint
python scripts/vla_deploy.py --model smolvla --checkpoint checkpoints/chess_smolvla/best.pt --no-robot

# Deploy for black's turn with board rotation
python scripts/vla_deploy.py --model pi0 --no-robot --turn black --rotation right

# Deploy with SO-100 robot arm connected (requires hardware)
python scripts/vla_deploy.py --model pi0 --global-camera /dev/video4 --gripper-camera /dev/video0 --robot-port /dev/ttyUSB0

# Deploy with all custom parameters
python scripts/vla_deploy.py --model pi0 --checkpoint checkpoints/chess_pi0/best.pt --global-camera /dev/video4 --gripper-camera /dev/video0 --turn white --rotation right --corner-conf 0.005 --min-corner-dist 30

## Overlay Generation

# Generate guidance overlay from current state cache
python scripts/generate_overlay.py

# Generate overlay for specific action index
python scripts/generate_overlay.py --action 1

# Check state cache status without generating overlay
python scripts/generate_overlay.py --status

# Generate overlay with custom cache path
python scripts/generate_overlay.py --cache data/state_cache.json

## Corner Detection Fine-tuning (Fix Corner Detection Issues)

# Step 1: Collect 20+ training photos of YOUR chessboard (vary angles, lighting, positions)
python scripts/collect_corner_training_photos.py --device /dev/video0 --count 20

# Step 2: Label the 4 corners in each photo (interactive clicking tool)
python scripts/label_corners.py --input data/training/board_photos

# Step 3: Fine-tune the corner detection model on your labeled data (15-30 min on CPU)
python scripts/finetune_corners.py --data data/training/corner_dataset/data.yaml

# Step 4: Backup original model and deploy fine-tuned model
mv data/best_corners.pt data/best_corners_original.pt
cp data/training/runs/corner_finetune/weights/best.pt data/best_corners.pt

# Step 5: Test fine-tuned corner detection model
python scripts/best_move_demo.py --debug

## Piece Detection Fine-tuning (Fix Piece Recognition Issues)

# Step 1: Collect 30+ training photos with VARIED piece positions on board
python scripts/collect_piece_training_photos.py --device /dev/video0 --count 30

# Step 2: Label pieces in each photo (draw bounding boxes, assign classes)
python scripts/label_pieces.py --input data/training/piece_photos

# Step 3: Fine-tune the piece detection model on your labeled data (30-60 min on CPU)
python scripts/finetune_pieces.py --data data/training/piece_dataset/data.yaml

# Step 4: Backup original model and deploy fine-tuned model
mv data/best_transformed_detection.pt data/best_transformed_detection_original.pt
cp data/training/runs/piece_finetune/weights/best.pt data/best_transformed_detection.pt

# Step 5: Test fine-tuned piece detection model
python scripts/best_move_demo.py --debug

## Visualization Tool

# Start visualization tool in development mode (React HMR + FastAPI reload)
python scripts/start_viz_tool.py --dev

# Start visualization tool in production mode
python scripts/start_viz_tool.py

# Build React app only (no server start)
python scripts/start_viz_tool.py --build-only

# Start visualization tool with custom host and port
python scripts/start_viz_tool.py --host 192.168.1.100 --port 8080

# Start development mode with custom FastAPI port
python scripts/start_viz_tool.py --dev --port 8080
