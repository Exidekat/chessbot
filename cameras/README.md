# Cameras Module

Multi-camera stream management for the chess robot system. Consolidates two real-time camera feeds and one resource-efficient static overlay stream.

## Overview

The cameras module provides a unified interface for accessing all visual streams needed by the system:
1. **Global Camera** - Overhead view for board state detection (WBC-0E01, 1280x720)
2. **Gripper Camera** - Arm-mounted view for VLA (eMeet C950, 224x224)
3. **Guidance Overlay** - Static PNG with visual highlights (loaded on-demand)

**Resource Efficiency**: Only the two physical cameras run continuously with threaded capture. The guidance overlay is loaded from disk only when the guidance system signals via flag file - no polling, no wasted resources.

## Components

### `global_camera.py`
Overhead camera with continuous threaded capture:
```python
from cameras import GlobalCamera

camera = GlobalCamera(camera_id=0, resolution=(1280, 720))
camera.start()

# Get latest frame
frame = camera.get_frame()  # Returns BGR numpy array

# Save snapshot
camera.save_frame("snapshot.png")

camera.stop()
```

**Features:**
- Background threaded capture (non-blocking)
- Thread-safe frame access
- Automatic resolution configuration
- Real-time overhead view for board detection

### `gripper_camera.py`
Arm-mounted camera for close-up views:
```python
from cameras import GripperCamera

camera = GripperCamera(camera_id=0, resolution=(640, 480))
camera.start()

# Get latest frame
frame = camera.get_frame()  # Returns BGR numpy array

camera.stop()
```

**Features:**
- Same interface as GlobalCamera
- Lower resolution (sufficient for gripper feedback)
- Real-time visual feedback during manipulation
- Future: Visual servoing integration

### `overlay_generator.py`
Resource-efficient overlay management using flag-based signaling:
```python
from cameras import OverlayGenerator

overlay_gen = OverlayGenerator(
    overlay_path="data/guidance_overlay.png",
    flag_path="data/overlay_ready.flag"
)

# Check for updates (efficient - only checks file modification time)
if overlay_gen.check_for_update():
    overlay_gen.load_overlay()

# Get current overlay (auto-checks for updates)
overlay = overlay_gen.get_overlay()  # Returns BGR numpy array or None
```

**Signaling Protocol** (used by guidance system):
```python
# Guidance system generates overlay
from guidance import GuidanceSystem

guidance = GuidanceSystem()
guidance.generate_overlay_from_cache()
# → Saves data/guidance_overlay.png
# → Creates data/overlay_ready.flag

# Camera automatically picks up new overlay on next get_overlay_frame()
```

**Resource Efficiency:**
- No continuous polling or camera capture
- Only loads from disk when flag file is updated
- Flag file timestamp check is O(1) filesystem operation
- Minimal memory footprint when no overlay active

### `camera_manager.py`
Unified interface consolidating all three streams:
```python
from cameras import CameraManager

# Initialize with config
config = {
    'global_camera_id': 0,
    'global_resolution': (1280, 720),
    'gripper_camera_id': 0,
    'gripper_resolution': (640, 480),
    'overlay_path': 'data/guidance_overlay.png',
    'overlay_flag_path': 'data/overlay_ready.flag'
}

manager = CameraManager(config)
manager.start()

# Get individual streams
global_frame = manager.get_global_frame()
gripper_frame = manager.get_gripper_frame()
overlay = manager.get_overlay_frame()  # Only loads if flagged

# Get all streams at once
frames = manager.get_all_frames()
# {'global': frame1, 'gripper': frame2, 'overlay': frame3_or_None}

# Check status
status = manager.get_stream_status()
# {'global': True, 'gripper': True, 'overlay': True/False}

# Save snapshots from all streams
manager.save_snapshot("data/snapshots")

manager.stop()
```

**Features:**
- Single unified interface for all streams
- Automatic start/stop of real-time cameras
- Efficient overlay management (flag-based)
- Stream status monitoring
- Snapshot capability

## Usage Patterns

### With Guidance System
```python
from cameras import CameraManager
from guidance import GuidanceSystem

cameras = CameraManager(config)
cameras.start()

guidance = GuidanceSystem()

# Capture board state
global_frame = cameras.get_global_frame()
cv2.imwrite("temp_board.png", global_frame)

# Detect, calculate, and update cache
fen, best_move, actions = guidance.detect_and_calculate(
    "temp_board.png",
    update_cache=True
)

# Generate overlay with action highlights
guidance.generate_overlay_from_cache()

# Camera automatically loads new overlay
overlay = cameras.get_overlay_frame()  # Returns highlighted board
```

### With Robot Control
```python
from cameras import CameraManager
from controls import RobotArm

cameras = CameraManager(config)
cameras.start()

robot = RobotArm(config)

# Visual feedback during movement
while robot.is_moving():
    gripper_view = cameras.get_gripper_frame()
    # Process gripper view for visual servoing
    # Adjust movement if needed
```

### VLA Episode Collection

```bash
# Terminal 1: Run tele-op
python scripts/tele_op.py

# Terminal 2: Collect episodes at 15 FPS
python scripts/collect_vla_episodes.py --output data/episodes/
```

See [scripts/USAGE.md](../scripts/USAGE.md) for full episode collection workflow.

## Additional Utilities

### `live_camera_capture.py`

Threaded camera capture for VLA scripts with minimal latency:

```python
from cameras.live_camera_capture import LiveCameraCapture

capture = LiveCameraCapture("/dev/video0")
capture.start()  # Background thread at 30fps

frame = capture.get_latest_frame()  # Thread-safe, returns 720p BGR

capture.stop()
```

**Features:**
- Configures camera for native 720p MJPEG at 30fps (low latency for real-time VLA)
- Buffer size = 1 for minimal latency
- Thread-safe frame access

**Note:** This is for real-time VLA control loops. For board detection (corner/piece detection), use `capture_4k_downscale()` from `utils.camera_helpers` which captures at 4K and downscales to 720p for better quality.

### `virtual_camera.py`

Output frames to a virtual camera via v4l2loopback + ffmpeg:

```python
from cameras.virtual_camera import VirtualCamera

vcam = VirtualCamera("/dev/video7", width=1280, height=720)
vcam.start()

vcam.write_frame(frame)  # Queue frame for output (non-blocking)

vcam.stop()
```

**Setup (one-time):**
```bash
sudo modprobe v4l2loopback devices=1 video_nr=7 \
    card_label="ChessBot Virtual Cam" exclusive_caps=1
```

**Features:**
- Low-latency streaming via ffmpeg pipe
- Automatic frame resizing if needed
- Used for VLA training with virtual camera input

## Configuration

Example `config.yaml`:
```yaml
cameras:
  global_camera_id: 0
  global_resolution: [1280, 720]

  gripper_camera_id: 1
  gripper_resolution: [640, 480]

  overlay_path: "data/guidance_overlay.png"
  overlay_flag_path: "data/overlay_ready.flag"
```

## Stream Specifications

### Global Camera (WBC-0E01)
- **Purpose**: Board state detection, overall monitoring
- **Output Resolution**: 1280x720 (fixed for YOLO consistency)
- **Capture Modes**:
  - **Board Detection**: 4K MJPEG (3840x2160) downscaled to 720p via LANCZOS4 for superior quality
  - **Real-time VLA**: Native 720p MJPEG at 30fps for low latency
- **Frame Rate**: 30 FPS MJPEG
- **Mounting**: Overhead, bird's-eye view of full board
- **Format**: BGR (OpenCV standard)

### Gripper Camera (eMeet C950)
- **Purpose**: Close-up manipulation feedback, VLA input
- **Capture Resolution**: 640x480 (native camera resolution)
- **Output Resolution**: 224x224 (resized for VLA input - most cameras don't support 224x224 natively)
- **Frame Rate**: 30 FPS
- **Mounting**: On robot arm/gripper
- **Format**: BGR (OpenCV standard)

### Guidance Overlay
- **Purpose**: Visual guidance (move highlights, piece targets)
- **Resolution**: Matches transformed board image (~400-800px)
- **Update Rate**: On-demand (only when guidance updates)
- **Storage**: Static PNG on disk
- **Format**: BGR (OpenCV standard)

## Development Status

**Completed:**
- `global_camera.py` - Threaded capture (WBC-0E01, 1280x720)
- `gripper_camera.py` - Threaded capture (eMeet C950, 224x224)
- `overlay_generator.py` - Flag-based updates
- `camera_manager.py` - Unified interface
- `live_camera_capture.py` - VLA threaded capture utility
- `virtual_camera.py` - v4l2loopback output for VLA training
- Hardware testing with physical cameras

**Planned:**
- Calibration integration
- Visual servoing implementation

## Troubleshooting

### Camera Not Opening
```python
# List available cameras
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

### Low Frame Rate
- Reduce resolution in config
- Check USB bandwidth (use USB 3.0)
- Ensure no other applications using camera

### Overlay Not Updating
- Check flag file exists and is writable
- Verify overlay PNG path is correct
- Ensure guidance system calls `signal_overlay_ready()`

### Thread Safety
All camera classes use thread locks for frame access - safe for multi-threaded use.
