# Cameras Module

Multi-camera stream management for the chess robot system. Consolidates two real-time camera feeds and one resource-efficient static overlay stream.

## Overview

The cameras module provides a unified interface for accessing all visual streams needed by the system:
1. **Global Camera** - Overhead view for board state detection
2. **Gripper Camera** - Arm-mounted view for precision manipulation
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

camera = GripperCamera(camera_id=1, resolution=(640, 480))
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
# Guidance system side
overlay_gen.save_overlay(highlighted_image)  # Saves PNG and creates flag
# or
overlay_gen.signal_overlay_ready()  # Just updates flag timestamp
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
    'gripper_camera_id': 1,
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
from cameras import CameraManager, OverlayGenerator
from guidance import BoardDetector

cameras = CameraManager(config)
cameras.start()

detector = BoardDetector()
overlay_gen = OverlayGenerator()

# Capture board state
global_frame = cameras.get_global_frame()
cv2.imwrite("temp_board.png", global_frame)

# Detect and generate overlay
fen, transformed = detector.detect_board_state("temp_board.png", debug=True)

# Signal overlay ready (guidance could highlight best move squares)
overlay_gen.save_overlay(transformed)  # Cameras will pick it up on next get_overlay_frame()
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

### VLA Data Collection (Future)
```python
from cameras import CameraManager
from vla.data_collection import EpisodeRecorder

cameras = CameraManager(config)
cameras.start()

recorder = EpisodeRecorder()
recorder.start_episode(metadata={'move': 'e2e4'})

# Record synchronized frames throughout episode
while episode_active:
    frames = cameras.get_all_frames()

    recorder.record_timestep(
        global_frame=frames['global'],
        gripper_frame=frames['gripper'],
        overlay_frame=frames['overlay'],  # Guidance highlights
        robot_state=robot.get_state(),
        action=action
    )

recorder.end_episode(success=True)
```

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

### Global Camera
- **Purpose**: Board state detection, overall monitoring
- **Resolution**: 1280x720 (configurable)
- **Frame Rate**: ~30 FPS (hardware dependent)
- **Mounting**: Overhead, bird's-eye view of full board
- **Format**: BGR (OpenCV standard)

### Gripper Camera
- **Purpose**: Close-up manipulation feedback
- **Resolution**: 640x480 (sufficient for gripper view)
- **Frame Rate**: ~30 FPS (hardware dependent)
- **Mounting**: On robot arm/gripper
- **Format**: BGR (OpenCV standard)

### Guidance Overlay
- **Purpose**: Visual guidance (move highlights, piece targets)
- **Resolution**: Matches transformed board image (~400-800px)
- **Update Rate**: On-demand (only when guidance updates)
- **Storage**: Static PNG on disk
- **Format**: BGR (OpenCV standard)

## Development Status

- [x] `global_camera.py` - Complete with threaded capture
- [x] `gripper_camera.py` - Complete with threaded capture
- [x] `overlay_generator.py` - Complete with flag-based updates
- [x] `camera_manager.py` - Complete unified interface
- [x] Module exports and documentation
- [ ] Hardware testing with physical cameras
- [ ] Calibration integration
- [ ] Visual servoing implementation

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
