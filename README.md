# ChessBot - Chess Vision Board State System

A computer vision system for detecting and analyzing chess board positions from images using YOLOv8 and python-chess.

## Features

- **Corner Detection**: Automatically detects the four corners of a chessboard using a trained YOLO model
- **Perspective Correction**: Applies perspective transformation to obtain a bird's-eye view of the board
- **Piece Detection**: Identifies all chess pieces and their positions using YOLOv8
- **FEN Notation**: Outputs board state in standard FEN (Forsyth-Edwards Notation) format
- **UCI Integration**: Compatible with python-chess for move generation and analysis
- **Engine Analysis**: Calculate best moves using UCI chess engines (e.g., Stockfish)

## Architecture

The system uses a two-stage YOLO pipeline based on the [real-life-chess-vision](https://github.com/shainisan/real-life-chess-vision) project:

1. **Stage 1 - Corner Detection**: YOLO model detects the four corners of the chessboard
2. **Stage 2 - Perspective Transform**: OpenCV applies perspective correction to normalize the board view
3. **Stage 3 - Piece Detection**: YOLO model detects all pieces on the corrected board (12 classes: b,k,n,p,q,r,B,K,N,P,Q,R)
4. **Stage 4 - Square Matching**: Uses IoU (Intersection over Union) to match detected pieces to board squares
5. **Stage 5 - Output**: Generates FEN notation or python-chess Board object

## Installation

### Prerequisites

- Python 3.8+
- Git
- (Optional) Stockfish or another UCI chess engine for move analysis

### Setup

1. Clone the repository with submodules:
```bash
git clone --recurse-submodules <repository-url>
cd chessbot
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download model parameters:
```bash
python download.py
```

This will:
- Copy the corner detection model from the submodule
- Provide instructions for downloading the piece detection model

4. (Optional) Install Stockfish for engine analysis:
```bash
# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt-get install stockfish

# Windows
# Download from https://stockfishchess.org/download/
```

## Usage

### Quick Start

Place a chess board image at `data/chessboard.png` and run:

```bash
python main.py
```

### Command Line Options

```bash
python main.py --help

Options:
  --image IMAGE         Path to chess board image (default: data/chessboard.png)
  --engine ENGINE       Path to UCI engine (default: stockfish)
  --time TIME          Engine analysis time in seconds (default: 1.0)
  --output {fen,board} Output format: 'fen' or 'board' (default: board)
  --no-bestmove        Skip best move calculation
```

### Examples

```bash
# Analyze a specific image
python main.py --image path/to/chessboard.jpg

# Get FEN notation output
python main.py --output fen

# Use custom engine with longer analysis time
python main.py --engine /usr/local/bin/stockfish --time 5.0

# Skip engine analysis (just detect board state)
python main.py --no-bestmove
```

### Using the BoardState Module

```python
from board_state import BoardState

# Initialize detector
detector = BoardState()

# Take snapshot and get FEN
fen = detector.snapshot("path/to/image.jpg", output_format="fen")
print(f"FEN: {fen}")

# Or get python-chess Board object
board = detector.snapshot("path/to/image.jpg", output_format="board")
print(board)

# Get current state (from last snapshot)
current_board = detector.current(output_format="board")

# Calculate best move
best_move = detector.bestmove(engine_path="stockfish", time_limit=1.0)
print(f"Best move: {best_move.uci()}")
```

## API Reference

### BoardState Class

#### `__init__(corner_model_path, piece_model_path)`
Initialize the board state detector with model paths.

#### `snapshot(image_path, output_format='fen') -> str | chess.Board`
Process an image and detect the board state.

**Parameters:**
- `image_path` (str): Path to the chess board image
- `output_format` (str): 'fen' for FEN notation string, 'board' for chess.Board object

**Returns:** FEN string or chess.Board object

#### `current(output_format='board') -> str | chess.Board`
Get the last snapshot result without reprocessing.

**Parameters:**
- `output_format` (str): 'fen' or 'board'

**Returns:** FEN string or chess.Board object

#### `bestmove(engine_path='stockfish', time_limit=1.0) -> chess.Move`
Calculate the best move using a UCI engine.

**Parameters:**
- `engine_path` (str): Path to UCI engine executable
- `time_limit` (float): Analysis time limit in seconds

**Returns:** chess.Move object or None

## Project Structure

```
chessbot/
├── board_state.py           # Main BoardState module
├── main.py                  # Test driver script
├── download.py              # Model download script
├── requirements.txt         # Python dependencies
├── data/                    # Model files and test images
│   ├── best_cornres.pt      # Corner detection model
│   ├── best_transformed_detection.pt  # Piece detection model
│   └── chessboard.png       # Test image
└── submodules/
    └── real-life-chess-vision/  # Original project submodule
```

## Model Files

The system requires two YOLO model files:

1. **best_cornres.pt** - Corner detection model (~6 MB)
   - Automatically copied from submodule by download.py

2. **best_transformed_detection.pt** - Piece detection model
   - Must be downloaded manually from OneDrive link provided by download.py

## Troubleshooting

### Models not found
```
FileNotFoundError: Corner detection model not found
```
**Solution:** Run `python download.py` to set up model files

### Engine not found
```
FileNotFoundError: Engine not found at 'stockfish'
```
**Solution:** Install Stockfish or specify engine path with `--engine`

### CUDA/GPU errors
```
TypeError: can't convert cuda:0 device type tensor to numpy
```
**Solution:** The code automatically handles this by using `.cpu()` on tensors

### Invalid FEN
If the detected FEN is invalid, this may indicate:
- Poor image quality or lighting
- Obstructed view of the board
- Model confidence too low

Try adjusting the image or confidence thresholds in the model prediction calls.

## Credits

This project is based on the excellent work by [shainisan/real-life-chess-vision](https://github.com/shainisan/real-life-chess-vision).

## License

See the LICENSE file for details.
