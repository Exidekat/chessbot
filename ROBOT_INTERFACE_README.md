# Robot Interface Documentation

Complete guide for integrating a robotic chess arm with the chess vision system.

---

## Overview

The robot interface enables a robotic arm to physically execute chess moves with visual guidance and network-based communication. The system handles all move types including captures, castling, en passant, and promotions.

### Features
- ✅ Network communication (TCP/UDP)
- ✅ Visual highlighting on camera feed
- ✅ State machine for move execution
- ✅ Capture sequence handling
- ✅ Castling support (king first, then rook)
- ✅ En passant support
- ✅ Error handling and recovery
- ✅ Extensible and configurable

---

## Quick Start

### 1. Basic Usage

```python
from board_state import BoardState

# Initialize detector
detector = BoardState()
detector.snapshot("board.png")

# Get best move
best_move = detector.bestmove()

# Execute with robot (auto-creates executor)
success = detector.execute_move_with_robot(best_move)

if success:
    print("Move executed successfully!")
```

### 2. Advanced Usage (Reusable Executor)

```python
from board_state import BoardState

detector = BoardState()

# Create reusable executor
executor = detector.get_robot_executor(port=5555)

# Execute multiple moves
for move in moves:
    detector.snapshot("board.png")
    success = detector.execute_move_with_robot(move, robot_executor=executor)

    if not success:
        break

# Clean up when done
executor.cleanup()
```

---

## System Architecture

### Components

```
┌─────────────────┐
│   BoardState    │  ← Main chess vision system
└────────┬────────┘
         │
         ├── execute_move_with_robot()
         │
         ▼
┌─────────────────┐
│  MoveExecutor   │  ← State machine for move execution
└────────┬────────┘
         │
         ├─────────────┬─────────────┬─────────────────┐
         ▼             ▼             ▼                 ▼
┌──────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────────┐
│ RobotComm    │ │ MoveInt... │ │ CoordMapper  │ │  Highlight   │
│  (Network)   │ │  (Logic)   │ │ (Position)   │ │  (Visual)    │
└──────────────┘ └────────────┘ └──────────────┘ └──────────────┘
         │
         ▼
   [Robot Hardware]
```

### File Structure

```
chessbot/
├── board_state.py           # Main vision system (MODIFIED)
├── robot_interface.py       # Robot interface module (NEW)
├── robot_demo.py           # Demo/testing script (NEW)
├── robot_server_example.py # Example robot-side code (NEW)
└── ROBOT_INTERFACE_README.md  # This file
```

---

## Move Execution Flow

### Normal Move (e.g., e2-e4)

```
1. HIGHLIGHT_PIECE_PICKUP
   ├─ Highlight source square (GREEN)
   ├─ Send pickup command to robot
   └─ Set robot_action = True

2. WAIT_PICKUP_PIECE
   └─ Wait for holding_piece = True

3. HIGHLIGHT_PIECE_PLACE
   ├─ Highlight destination square (BLUE)
   ├─ Send place command to robot
   └─ Keep robot_action = True

4. WAIT_PLACE_PIECE
   └─ Wait for holding_piece = False

5. COMPLETE
   ├─ Clear highlights
   ├─ Set robot_action = False
   └─ Update board state
```

### Capture Move (e.g., exd5)

```
1. HIGHLIGHT_CAPTURE_PICKUP
   ├─ Highlight opponent's piece (RED)
   ├─ Send pickup command
   └─ Set robot_action = True

2. WAIT_PICKUP_CAPTURE
   └─ Wait for holding_piece = True

3. HIGHLIGHT_CAPTURE_PLACE
   ├─ Highlight graveyard area (ORANGE)
   ├─ Send place command
   └─ Keep robot_action = True

4. WAIT_PLACE_CAPTURE
   └─ Wait for holding_piece = False

5-8. [Execute normal move for our piece]
    └─ Same as steps 1-4 above

9. COMPLETE
```

### Castling (e.g., O-O)

```
1. Execute king move
   └─ [Normal move sequence for king]

2. Execute rook move
   └─ [Normal move sequence for rook]

3. COMPLETE
```

### En Passant

```
1. Pickup captured pawn (NOTE: not on destination square!)
2. Place captured pawn in graveyard
3. Pickup our pawn
4. Place our pawn on destination
5. COMPLETE
```

---

## Communication Protocol

### Network Setup

**Chessbot (Server):**
- Binds to host:port and listens
- Default: `0.0.0.0:5555` (TCP)
- Sends action commands to robot
- Receives status updates from robot

**Robot (Client):**
- Connects to chessbot server
- Sends status updates periodically (10 Hz recommended)
- Receives and executes action commands

### Message Formats

#### Robot → Chessbot (Status Update)

```json
{
    "type": "status",
    "holding_piece": bool,
    "ready": bool,
    "error": string | null
}
```

**Fields:**
- `holding_piece`: True when robot is holding a piece
- `ready`: True when robot is ready for next action
- `error`: Error message, or null if no error

**Example:**
```json
{
    "type": "status",
    "holding_piece": false,
    "ready": true,
    "error": null
}
```

#### Chessbot → Robot (Action Command)

```json
{
    "type": "action",
    "robot_action": bool,
    "action": "pickup" | "place",
    "target_square": string | null,
    "coordinates": {"x": int, "y": int},
    "is_capture": bool,
    "piece": string
}
```

**Fields:**
- `robot_action`: True when robot should take action
- `action`: "pickup" or "place"
- `target_square`: Chess square notation (e.g., "e4") or null for graveyard
- `coordinates`: Pixel coordinates {"x": 250, "y": 300}
- `is_capture`: Whether this is part of a capture sequence
- `piece`: Piece being moved (e.g., "Q", "p")

**Example:**
```json
{
    "type": "action",
    "robot_action": true,
    "action": "pickup",
    "target_square": "e4",
    "coordinates": {"x": 320, "y": 320},
    "is_capture": false,
    "piece": "P"
}
```

---

## Robot-Side Implementation

### Minimal Robot Client

```python
import socket
import json
import time

# Connect to chessbot
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.1.100", 5555))

# Send status updates
def send_status(holding: bool, ready: bool):
    status = {
        "type": "status",
        "holding_piece": holding,
        "ready": ready,
        "error": null
    }
    sock.sendall(json.dumps(status).encode() + b"\n")

# Main loop
while True:
    # Receive command
    data = sock.recv(4096).decode()
    command = json.loads(data)

    if command["type"] == "action" and command["robot_action"]:
        action = command["action"]
        x, y = command["coordinates"]["x"], command["coordinates"]["y"]

        if action == "pickup":
            # Move to (x, y) and grip
            robot_move_to(x, y)
            robot_grip()
            send_status(holding=True, ready=True)

        elif action == "place":
            # Move to (x, y) and release
            robot_move_to(x, y)
            robot_release()
            send_status(holding=False, ready=True)
```

### Complete Example

See `robot_server_example.py` for a full implementation with threading, error handling, and hardware interfacing patterns.

---

## Visual Highlighting

### Color Scheme

| Action | Color | RGB | Use Case |
|--------|-------|-----|----------|
| Capture Pickup | Red | (255, 0, 0) | Opponent's piece to capture |
| Capture Place | Orange | (255, 165, 0) | Graveyard placement |
| Piece Pickup | Green | (0, 255, 0) | Our piece to move |
| Piece Place | Blue | (0, 0, 255) | Destination square |

### Rendering

Highlights are rendered using OpenCV with alpha blending (default: 0.5 transparency).

**Square Highlighting:**
- Filled rectangle with colored border
- Covers entire chess square

**Graveyard Highlighting:**
- Circular highlight (radius: 30px)
- Placed at right edge of board

---

## Coordinate Mapping

### Pixel Coordinates

The `CoordinateMapper` class converts chess squares to pixel coordinates on the transformed board image.

**Grid Calculation:**
- Uses BoardState's `calculate_grid()` method
- Accounts for perspective transform
- Accounts for top margin (added for piece visibility)

**Square to Pixels:**
```python
mapper = CoordinateMapper(board_image, board_state)
x, y = mapper.square_to_pixels("e4")  # Returns center of square
```

**Square Bounds:**
```python
x1, y1, x2, y2 = mapper.get_square_bounds("e4")  # Returns rectangle
```

**Graveyard:**
```python
x, y = mapper.get_graveyard_coords()  # Right edge, centered vertically
```

### Physical Coordinates

**Robot developers must implement calibration** to convert pixel coordinates to physical robot coordinates.

**Typical calibration involves:**
1. Camera calibration matrix
2. Workspace transformation
3. Scaling based on camera height/angle
4. Board size measurements

**Example:**
```python
def pixels_to_physical(pixel_x, pixel_y):
    # Assume 400mm x 400mm board, 640x640 image
    scale = 400.0 / 640.0  # mm per pixel

    physical_x = pixel_x * scale
    physical_y = pixel_y * scale

    return physical_x, physical_y
```

---

## Testing

### Demo Script

Test the robot interface without physical hardware:

```bash
# Test all components with simulated robot
python robot_demo.py --test all --simulate

# Test specific component
python robot_demo.py --test communication --simulate

# Test with real robot
python robot_demo.py --test execution
```

### Components Tested

1. **MoveInterpreter** - Chess move parsing and type detection
2. **RobotCommunicator** - Network communication
3. **MoveExecutor** - Full move execution sequences

### Simulated Robot

The demo includes a `SimulatedRobot` class that:
- Auto-responds to action commands
- Simulates pickup/place delays
- Sends appropriate status updates
- Perfect for development without hardware

---

## Configuration

### Network Settings

```python
# Create custom executor with specific settings
from robot_interface import RobotCommunicator, MoveExecutor

comm = RobotCommunicator(
    host="192.168.1.100",  # Server IP
    port=5555,              # Server port
    protocol="tcp",         # "tcp" or "udp"
    timeout=30.0           # Response timeout (seconds)
)

executor = MoveExecutor(board_state=detector, robot_comm=comm)
```

### Highlight Colors

```python
from robot_interface import HighlightRenderer

renderer = HighlightRenderer(alpha=0.5)  # Transparency (0.0-1.0)

# Customize colors (BGR format)
renderer.COLORS["capture_pickup"] = (0, 0, 255)  # Red
renderer.COLORS["piece_pickup"] = (0, 255, 0)     # Green
```

### Delays

Adjust simulated delays for testing:

```python
from robot_demo import SimulatedRobot

sim_robot = SimulatedRobot(
    communicator=comm,
    pickup_delay=2.0,   # Seconds
    place_delay=2.0     # Seconds
)
```

---

## Error Handling

### Timeout Handling

If robot doesn't respond within timeout period:
- Move execution aborts
- State machine returns to IDLE
- Error logged

**Configure timeout:**
```python
comm = RobotCommunicator(timeout=30.0)  # 30 seconds
```

### Communication Failures

If network connection fails:
- Retries connection automatically
- Falls back to manual mode (display move, don't execute)
- Logs all failures

### Invalid Moves

Before execution:
- Validates move is legal
- Checks if pieces are in expected positions (via detection)
- Aborts if board state mismatch

---

## API Reference

### BoardState Methods

#### `execute_move_with_robot(move, robot_executor=None, camera_feed=None) → bool`

Execute a chess move with robotic arm.

**Args:**
- `move` (chess.Move): Chess move to execute
- `robot_executor` (MoveExecutor, optional): Reusable executor
- `camera_feed` (VideoCapture, optional): OpenCV camera for overlay

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
detector = BoardState()
detector.snapshot("board.png")
move = detector.bestmove()
success = detector.execute_move_with_robot(move)
```

#### `get_robot_executor(host="0.0.0.0", port=5555) → MoveExecutor`

Create a reusable MoveExecutor instance.

**Args:**
- `host` (str): Server IP address
- `port` (int): Server port

**Returns:**
- `MoveExecutor`: Configured executor

**Example:**
```python
executor = detector.get_robot_executor(port=5555)
# Use executor for multiple moves
executor.cleanup()  # When done
```

### MoveExecutor Methods

#### `execute_move(move, board, camera_feed=None) → bool`

Execute a chess move.

**Args:**
- `move` (chess.Move): Move to execute
- `board` (chess.Board): Current board state
- `camera_feed` (VideoCapture, optional): Camera feed

**Returns:**
- `bool`: Success status

#### `cleanup()`

Clean up resources (stop network communication).

### RobotCommunicator Methods

#### `start()`

Start network server.

#### `stop()`

Stop network server.

#### `send_action(robot_action, action, target_square, coordinates, is_capture, piece)`

Send action command to robot.

#### `get_status() → RobotStatus`

Get current robot status (thread-safe).

#### `wait_for_signal(signal_name, expected_value, timeout=None) → bool`

Wait for specific signal to reach expected value.

**Args:**
- `signal_name` (str): "holding_piece" or "ready"
- `expected_value` (bool): Expected value
- `timeout` (float, optional): Timeout in seconds

**Returns:**
- `bool`: True if signal reached value, False if timeout

---

## Troubleshooting

### Robot Not Connecting

**Problem:** Robot can't connect to chessbot server

**Solutions:**
1. Check firewall settings
2. Verify host/port configuration
3. Test with `telnet <host> <port>`
4. Try UDP instead of TCP
5. Check network connectivity

### Highlights Not Visible

**Problem:** Visual highlights not showing on camera feed

**Solutions:**
1. Verify camera_feed is provided
2. Check if board_image is set correctly
3. Verify grid calculation working
4. Increase alpha value (less transparent)

### Robot Not Responding

**Problem:** Robot receives commands but doesn't move

**Solutions:**
1. Check robot hardware connections
2. Verify coordinate conversion is correct
3. Test with simulated robot first
4. Check robot error messages
5. Increase timeout value

### Move Execution Fails

**Problem:** execute_move_with_robot() returns False

**Solutions:**
1. Check timeout settings (may need longer)
2. Verify robot status updates arriving
3. Test with demo script first
4. Check for network errors in logs
5. Verify move is legal

---

## Best Practices

### 1. Reuse Executor for Multiple Moves

```python
# GOOD - Reuse executor
executor = detector.get_robot_executor()
for move in moves:
    detector.execute_move_with_robot(move, robot_executor=executor)
executor.cleanup()

# BAD - Creates new executor every time
for move in moves:
    detector.execute_move_with_robot(move)  # Slower, more connections
```

### 2. Always Clean Up

```python
try:
    executor = detector.get_robot_executor()
    # ... use executor ...
finally:
    executor.cleanup()  # Always clean up
```

### 3. Handle Errors Gracefully

```python
success = detector.execute_move_with_robot(move)
if not success:
    # Handle failure
    print("Move failed, attempting recovery...")
    # Maybe retry, or switch to manual mode
```

### 4. Test with Simulation First

```python
# Always test with simulated robot before using real hardware
python robot_demo.py --simulate
```

### 5. Calibrate Coordinates

Properly calibrate your camera-to-robot coordinate transformation:
- Measure board size accurately
- Account for camera angle/distortion
- Test with known positions
- Adjust as needed

---

## Examples

### Example 1: Basic Game Loop

```python
from board_state import BoardState
import time

detector = BoardState()
executor = detector.get_robot_executor()

try:
    while True:
        # Detect current position
        detector.snapshot("camera_feed.png", debug=False)

        # Calculate best move
        move = detector.bestmove(time_limit=2.0)

        if not move:
            print("No legal moves - game over!")
            break

        # Execute move
        print(f"Executing move: {move.uci()}")
        success = detector.execute_move_with_robot(move, robot_executor=executor)

        if not success:
            print("Move execution failed!")
            break

        # Wait for opponent's move
        print("Waiting for opponent...")
        time.sleep(10)

finally:
    executor.cleanup()
```

### Example 2: Manual Move Execution

```python
import chess
from board_state import BoardState

detector = BoardState()
detector.snapshot("board.png")

# Create custom move
move = chess.Move.from_uci("e2e4")

# Execute it
detector.execute_move_with_robot(move)
```

### Example 3: With Camera Feed

```python
import cv2
from board_state import BoardState

# Open camera
camera = cv2.VideoCapture(0)

detector = BoardState()
executor = detector.get_robot_executor()

try:
    # Capture frame
    ret, frame = camera.read()
    cv2.imwrite("current_board.png", frame)

    # Detect and execute
    detector.snapshot("current_board.png")
    move = detector.bestmove()

    # Execute with camera feed for live overlay
    detector.execute_move_with_robot(move, robot_executor=executor, camera_feed=camera)

finally:
    executor.cleanup()
    camera.release()
```

---

## Future Enhancements

### Planned Features

- [ ] Multi-camera support
- [ ] Move validation with post-execution detection
- [ ] Undo/redo support
- [ ] Move queue for batch execution
- [ ] Web interface for remote monitoring
- [ ] ROS integration
- [ ] Configurable move speeds
- [ ] Multiple robot support (e.g., white/black robots)

### Contributing

To add features or improvements:
1. Modify `robot_interface.py`
2. Update tests in `robot_demo.py`
3. Update this documentation
4. Test thoroughly with simulated robot
5. Test with real hardware if available

---

## Support

For issues or questions:
1. Check this documentation
2. Run demo script: `python robot_demo.py --help`
3. Check example code: `robot_server_example.py`
4. Review error logs
5. Test with simulated robot first

---

**Version:** 1.0
**Last Updated:** 2025-01
**Compatible with:** BoardState v1.0+
