# Script Usage Guide

## Setup and Configuration

# Download required YOLO models (run on first setup)
python scripts/download.py

# Create default configuration file
python scripts/create_config.py

## Camera Capture

# Capture photo with auto-detected camera (saves to results/ with timestamp)
python scripts/capture_photo.py

# Capture photo with specific camera device
python scripts/capture_photo.py --device /dev/video0

# Capture photo with custom output in results/ directory
python scripts/capture_photo.py --output results/my_board.png

# Capture photo immediately without preview window
python scripts/capture_photo.py --no-preview

# Capture photo with specific device and custom output
python scripts/capture_photo.py --device /dev/video0 --output results/board_image.png

## Demo and Testing

# Run board detection and move calculation demo with default image
python scripts/best_move_demo.py

# Run demo with custom image
python scripts/best_move_demo.py --image data/chessboardv2.png

# Run demo with debug visualizations enabled
python scripts/best_move_demo.py --image data/chessboardv2.png --debug

# Run demo without calculating best move (no Stockfish required)
python scripts/best_move_demo.py --image data/chessboardv2.png --no-bestmove

# Run demo with custom corner detection parameters
python scripts/best_move_demo.py --image data/chessboardv2.png --debug --corner-conf 0.05 --min-corner-dist 40

# Run demo with custom engine time limit
python scripts/best_move_demo.py --image data/chessboardv2.png --time 2.0

## Overlay Generation

# Generate guidance overlay from current state cache
python scripts/generate_overlay.py

# Generate overlay for specific action index
python scripts/generate_overlay.py --action 1

# Check state cache status without generating overlay
python scripts/generate_overlay.py --status

# Generate overlay with custom cache path
python scripts/generate_overlay.py --cache data/state_cache.json

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
