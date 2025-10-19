# Robot Interface - Quick Start Guide

Get your robotic chess arm running in 5 minutes!

---

## What You Need

### Software (Provided ✅)
- `robot_interface.py` - Core robot interface
- `robot_demo.py` - Testing tool
- `robot_server_example.py` - Robot-side template

### Hardware (You Provide)
- Robotic arm (any type)
- Camera (for board detection)
- Network connection (TCP/UDP)

---

## Step 1: Test Without Hardware (2 minutes)

```bash
# Test with simulated robot
python robot_demo.py --simulate

# You should see:
# ✓ MoveInterpreter tests passing
# ✓ Network communication working
# ✓ Full move execution (e2e4, captures, castling)
```

**Expected Output:**
```
Testing MoveInterpreter
e2e4: normal
e4d5: capture, is_capture=True
e1g1 (castling): castling
  Decomposed: King e1g1, Rook h1f1

Testing Robot Communication
[RobotComm] Server started on 127.0.0.1:5556 (TCP)
[SimRobot] Simulated robot started
Test 1: Pickup action
[SimRobot] Simulating pickup (delay: 1.0s)...
...
```

---

## Step 2: Basic Usage (1 minute)

```python
from board_state import BoardState

# Initialize
detector = BoardState()
detector.snapshot("data/chessboardv2.png")

# Get best move from Stockfish
best_move = detector.bestmove()
print(f"Best move: {best_move.uci()}")

# Execute with robot (auto-connects on port 5555)
success = detector.execute_move_with_robot(best_move)

if success:
    print("✓ Move executed!")
```

---

## Step 3: Implement Your Robot (30-60 minutes)

### 3.1 Start with Example Template

Copy `robot_server_example.py` and modify:

```python
# robot_server_example.py - YOUR ROBOT CODE GOES HERE

class MockRobotArm:
    """REPLACE THIS with your robot hardware!"""

    def move_to(self, x: float, y: float):
        # TODO: Replace with your robot's move command
        # Examples:
        # - serial.write(f"MOVE {x} {y}")
        # - arm.set_position(x, y, z)
        # - ros_publisher.publish(Point(x, y, z))
        print(f"Moving to ({x}, {y})")

    def grip(self):
        # TODO: Close gripper
        # - serial.write("GRIP")
        # - gripper.close()
        print("Gripping")

    def release(self):
        # TODO: Open gripper
        print("Releasing")
```

### 3.2 Implement Coordinate Calibration

```python
def _pixels_to_physical(self, pixel_x: int, pixel_y: int) -> tuple:
    """Convert pixel coords to robot coords."""

    # TODO: Implement YOUR calibration
    # Measure your board size and camera position

    # Example for 400mm x 400mm board:
    BOARD_SIZE_MM = 400.0
    IMAGE_SIZE_PX = 640.0
    scale = BOARD_SIZE_MM / IMAGE_SIZE_PX

    physical_x = pixel_x * scale
    physical_y = pixel_y * scale

    return physical_x, physical_y
```

### 3.3 Run Your Robot

```bash
# Terminal 1: Start your robot client
python robot_server_example.py --host 127.0.0.1 --port 5555

# Terminal 2: Run chessbot demo
python robot_demo.py --test execution
```

---

## Step 4: Full Integration (10 minutes)

### Complete Game Loop

```python
from board_state import BoardState
import time

detector = BoardState()
executor = detector.get_robot_executor(port=5555)

try:
    print("Starting chess game with robot...")

    while True:
        # 1. Detect current board state
        print("\n[1/4] Detecting board...")
        detector.snapshot("camera_feed.png", debug=False)

        # 2. Calculate best move
        print("[2/4] Calculating best move...")
        move = detector.bestmove(time_limit=2.0)

        if not move:
            print("Game over - no legal moves!")
            break

        print(f"[3/4] Executing: {move.uci()}")

        # 3. Execute with robot
        success = detector.execute_move_with_robot(
            move,
            robot_executor=executor  # Reuse executor
        )

        if not success:
            print("✗ Move failed!")
            break

        print("✓ Move complete!")

        # 4. Wait for opponent
        print("[4/4] Waiting for opponent...")
        time.sleep(10)  # Or wait for board change detection

finally:
    executor.cleanup()
    print("\nGame ended.")
```

---

## Communication Protocol

### Your Robot Must:

**1. Connect to chessbot:**
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("192.168.1.100", 5555))
```

**2. Send status updates (10 Hz):**
```python
import json

status = {
    "type": "status",
    "holding_piece": False,  # True when gripping
    "ready": True,
    "error": None
}

sock.sendall(json.dumps(status).encode() + b"\n")
```

**3. Receive and execute commands:**
```python
data = sock.recv(4096)
command = json.loads(data)

if command["type"] == "action":
    action = command["action"]  # "pickup" or "place"
    x = command["coordinates"]["x"]
    y = command["coordinates"]["y"]

    if action == "pickup":
        robot_move_to(x, y)
        robot_grip()
        # Send status with holding_piece=True

    elif action == "place":
        robot_move_to(x, y)
        robot_release()
        # Send status with holding_piece=False
```

---

## Move Sequence Examples

### Normal Move (e2 → e4)

```
Chessbot → Robot: {"action": "pickup", "target_square": "e2", ...}
Robot executes: Move to e2, grip
Robot → Chessbot: {"holding_piece": true}

Chessbot → Robot: {"action": "place", "target_square": "e4", ...}
Robot executes: Move to e4, release
Robot → Chessbot: {"holding_piece": false}
```

### Capture (exd5)

```
1. Pick up opponent's piece on d5
2. Place in graveyard
3. Pick up our piece on e4
4. Place on d5
```

### Castling (O-O)

```
1. Move king (e1 → g1)
2. Move rook (h1 → f1)
```

---

## Troubleshooting

### "Robot not connecting"
```bash
# Check if server is running
netstat -an | grep 5555

# Try localhost first
python robot_server_example.py --host 127.0.0.1 --port 5555
```

### "Move execution fails"
```bash
# Test with simulated robot first
python robot_demo.py --simulate

# Check robot logs for errors
# Increase timeout if robot is slow
```

### "Coordinates wrong"
```python
# Calibrate your coordinate conversion
# Measure board size accurately
# Test with known positions first
```

---

## Next Steps

1. ✅ Test with simulated robot (`robot_demo.py --simulate`)
2. ⬜ Modify `robot_server_example.py` for your hardware
3. ⬜ Calibrate coordinate transformation
4. ⬜ Test pickup/place at known positions
5. ⬜ Run full game loop
6. ⬜ Add camera feed for live visualization

---

## Files Reference

| File | Purpose |
|------|---------|
| `robot_interface.py` | Core interface (don't modify) |
| `robot_demo.py` | Testing tool |
| `robot_server_example.py` | **START HERE** - Modify for your robot |
| `ROBOT_INTERFACE_README.md` | Full documentation |
| `ROBOT_QUICK_START.md` | This guide |

---

## Support

**Test First:**
```bash
python robot_demo.py --help
```

**Read Full Docs:**
```bash
cat ROBOT_INTERFACE_README.md
```

**Example Code:**
```bash
cat robot_server_example.py
```

---

**Ready to build? Start with `robot_demo.py --simulate`!** 🤖♟️
