# Utils Module

Shared utilities for the chess robot system. Provides thread-safe state management, camera discovery, and terminal input handling.

## Components

### `state_cache.py`

Thread-safe JSON-based state cache for chess robot. Supports multi-source updates from guidance, robot, VLA, and user.

```python
from utils import StateCache

cache = StateCache("data/state_cache.json")

# Read with dotted notation
fen = cache.get("game_state.fen")
is_robot_turn = cache.get("robot_state.is_robot_turn")

# Update with source tracking
cache.update({
    "robot_state": {
        "holding_piece": True,
        "joint_positions": [1.0, 2.0, 3.0, 4.0, 5.0, 0.5]
    }
}, source="robot")

# Robot helpers
current_action = cache.get_current_action()
cache.advance_action()
cache.set_action_status(0, "complete")
```

**State Structure:**
```json
{
  "game_state": {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "turn": "white",
    "transformed_board_path": "data/chessboard_transformed.png"
  },
  "robot_state": {
    "is_robot_turn": false,
    "holding_piece": false,
    "current_move": "e2e4",
    "action_sequence": [...],
    "action_index": 0,
    "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gripper_state": 0.0
  },
  "guidance_state": {
    "best_move": "e2e4",
    "best_move_san": "e4"
  },
  "metadata": {
    "last_updated_by": "guidance",
    "timestamp": 1234567890
  }
}
```

**Features:**
- Thread-safe with locking
- Atomic writes (temp file + rename)
- Dotted notation access
- Source tracking for updates
- Robot helper methods

### `camera_helpers.py`

Camera detection and capture utilities shared across scripts.

```python
from utils.camera_helpers import (
    get_available_cameras,
    select_camera,
    get_camera_index_from_device,
    capture_4k_downscale
)

# Find all cameras
cameras = get_available_cameras()
# Returns: [('/dev/video0', 'WBC-0E01: WBC-0E01'), ...]

# Interactive selection
device = select_camera(cameras)

# Parse device path
index = get_camera_index_from_device("/dev/video0")  # Returns 0

# Capture 4K and downscale to 720p
success = capture_4k_downscale("/dev/video0", Path("board.png"))
```

**Functions:**

| Function | Description |
|----------|-------------|
| `get_available_cameras()` | Detect USB cameras via v4l2-ctl |
| `select_camera(cameras)` | Interactive camera selection prompt |
| `get_camera_index_from_device(path)` | Parse /dev/videoN to OpenCV index |
| `capture_4k_downscale(device, output)` | Capture 4K MJPEG, downscale to 720p |

### `keyboard_input.py`

Non-blocking keyboard input for Linux terminals. Used by tele_op.py and other interactive scripts.

```python
from utils.keyboard_input import KeyboardInput

with KeyboardInput() as kb:
    while True:
        key = kb.get_key(timeout=0.1)
        if key == 'ESC':
            break
        elif key == 'ENTER':
            print("Enter pressed")
        elif key and key.isdigit():
            print(f"Number: {key}")
```

**Features:**
- Non-blocking with timeout
- Context manager for terminal mode restore
- Handles special keys (ESC, ENTER)
- Arrow key escape sequences consumed

## Usage

All utils are exported from the module:

```python
from utils import StateCache
from utils.camera_helpers import get_available_cameras
from utils.keyboard_input import KeyboardInput
```

Or import the module directly:

```python
import utils.camera_helpers as cam
cameras = cam.get_available_cameras()
```
