# Configuration System

YAML-based configuration management for the chess robot system. Provides centralized settings for cameras, models, paths, and system parameters.

## Quick Start

### First-Time Setup

Run the interactive configuration wizard to detect cameras and set up the system:

```bash
python create_config.py
```

The wizard will:
1. **Detect cameras** via unplug/replug method (automatically identifies device IDs)
2. **Configure resolutions** for each camera
3. **Set model paths** and other system settings
4. **Validate** the configuration
5. **Save** to `configs/current.yaml`

### Using Existing Configuration

Modules automatically load configuration from `configs/current.yaml`:

```python
# Camera Manager
from cameras import CameraManager

camera_mgr = CameraManager.from_config()  # Loads from YAML
camera_mgr.start_all()

# Manual config loading
from configs import load_config

config = load_config()  # Loads configs/current.yaml
print(config['cameras']['global_camera_id'])
```

## Configuration File Structure

### Complete Example (`configs/current.yaml`)

```yaml
cameras:
  global_camera_id: 0              # Overhead camera device ID
  global_resolution:
    - 1280
    - 720
  gripper_camera_id: 1             # Arm-mounted camera device ID
  gripper_resolution:
    - 640
    - 480

models:
  corner_model: data/best_cornres.pt
  piece_model: data/best_transformed_detection.pt

paths:
  state_cache: data/state_cache.json
  overlay_image: data/guidance_overlay.png
  overlay_flag: data/overlay_ready.flag
  data_dir: data

visualization:
  host: 0.0.0.0                    # Viz server host
  port: 8000                        # Viz server port

guidance:
  robot_plays_white: false          # Robot color preference
  engine_path: stockfish            # UCI engine path
  engine_time_limit: 1.0            # Engine analysis time (seconds)
  corner_confidence: 0.1            # YOLO corner detection threshold
  min_corner_distance: 50.0         # Min pixel distance between corners
```

## Configuration Fields

### Cameras

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cameras.global_camera_id` | int | 0 | Device ID for overhead camera |
| `cameras.global_resolution` | [int, int] | [1280, 720] | Resolution [width, height] |
| `cameras.gripper_camera_id` | int | 1 | Device ID for gripper camera |
| `cameras.gripper_resolution` | [int, int] | [640, 480] | Resolution [width, height] |

**Note**: Camera IDs must be different. Use `create_config.py` to auto-detect.

### Models

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models.corner_model` | str | data/best_cornres.pt | YOLO corner detection model |
| `models.piece_model` | str | data/best_transformed_detection.pt | YOLO piece detection model |

### Paths

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `paths.state_cache` | str | data/state_cache.json | State cache JSON file |
| `paths.overlay_image` | str | data/guidance_overlay.png | Guidance overlay PNG |
| `paths.overlay_flag` | str | data/overlay_ready.flag | Overlay update flag |
| `paths.data_dir` | str | data | Root data directory |

### Visualization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `visualization.host` | str | 0.0.0.0 | Server bind address |
| `visualization.port` | int | 8000 | Server port |

### Guidance

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `guidance.robot_plays_white` | bool | false | Robot plays as white |
| `guidance.engine_path` | str | stockfish | UCI chess engine |
| `guidance.engine_time_limit` | float | 1.0 | Engine analysis time |
| `guidance.corner_confidence` | float | 0.1 | YOLO corner threshold |
| `guidance.min_corner_distance` | float | 50.0 | Corner separation |

## Camera Detection

### Unplug/Replug Method

The configuration wizard uses OpenCV to auto-detect camera device IDs:

1. **List current cameras**: Enumerate all connected cameras (IDs 0-9)
2. **User unplugs camera**: System detects which ID disappeared
3. **User replugs camera**: System confirms camera reconnection
4. **Test camera**: Display live preview to verify functionality

### Manual Detection

To find camera IDs manually:

```python
import cv2

for camera_id in range(10):
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {camera_id}: Available")
            cv2.imshow(f"Camera {camera_id}", frame)
            cv2.waitKey(2000)  # Display for 2 seconds
        cap.release()
    cv2.destroyAllWindows()
```

## Usage in Code

### Loading Configuration

```python
from configs import load_config, save_config

# Load config (merges with defaults)
config = load_config()

# Access nested values
camera_id = config['cameras']['global_camera_id']
model_path = config['models']['corner_model']

# Modify and save
config['cameras']['global_camera_id'] = 2
save_config(config)
```

### Using with Modules

```python
# Camera Manager (auto-loads from config)
from cameras import CameraManager

camera_mgr = CameraManager.from_config()
camera_mgr.start_all()

# Stream Manager (auto-creates CameraManager from config)
from viz.stream_manager import StreamManager

stream_mgr = StreamManager()  # Uses config internally

# Manual config passing
from cameras import CameraManager
from configs import get_camera_config, get_path_config

cam_config = get_camera_config()
path_config = get_path_config()

config = {
    'global_camera_id': cam_config['global_camera_id'],
    'global_resolution': tuple(cam_config['global_resolution']),
    'gripper_camera_id': cam_config['gripper_camera_id'],
    'gripper_resolution': tuple(cam_config['gripper_resolution']),
    'overlay_path': path_config['overlay_image'],
}

camera_mgr = CameraManager(config)
```

## Validation

Configuration is automatically validated on load/save:

```python
from configs import load_config, validate_config

config = load_config()

errors = validate_config(config)
if errors:
    for error in errors:
        print(f"Error: {error}")
```

### Validation Rules

- **Required fields**: Camera IDs, model paths, essential paths
- **Type checking**: int, float, str, bool, list types enforced
- **Cross-field validation**: Camera IDs must be different
- **File existence**: Model files must exist
- **Format validation**: Resolutions must be [width, height]

## Troubleshooting

### Camera Not Detected

**Symptom**: `create_config.py` shows "No cameras detected"

**Solutions**:
1. Check camera is plugged in and powered
2. Verify camera works in other applications (e.g., webcam app)
3. On Linux, check permissions: `ls -l /dev/video*`
4. Try different USB ports
5. Restart computer to reset USB bus

### Wrong Camera Detected

**Symptom**: Unplug/replug detects wrong camera ID

**Solutions**:
1. Unplug ALL cameras except the one you're configuring
2. Run wizard again
3. Manually test each camera ID:
   ```bash
   python -c "import cv2; cap=cv2.VideoCapture(0); ret,frame=cap.read(); print('Works' if ret else 'Failed'); cap.release()"
   ```

### Validation Errors

**Symptom**: `Configuration validation failed: Required field missing`

**Solution**: Run `create_config.py` to fill in missing required fields

**Symptom**: `cameras.global_camera_id and cameras.gripper_camera_id must be different`

**Solution**: Ensure each camera has unique device ID (rerun camera detection)

### Model Files Not Found

**Symptom**: `models.corner_model: File not found`

**Solution**: Run model download script first:
```bash
python download.py
```

### Config File Corrupted

**Symptom**: `Error parsing YAML config: ...`

**Solutions**:
1. Delete `configs/current.yaml`
2. Run `create_config.py` to regenerate
3. Or manually fix YAML syntax (ensure proper indentation)

## Advanced Usage

### Multiple Configurations

Create environment-specific configs:

```bash
# Development config
cp configs/current.yaml configs/dev.yaml

# Production config
cp configs/current.yaml configs/prod.yaml

# Load specific config
from configs import load_config
config = load_config('configs/prod.yaml')
```

### Environment Variables

Override config values with environment variables:

```python
import os
from configs import load_config

config = load_config()

# Override from environment
config['cameras']['global_camera_id'] = int(os.getenv('GLOBAL_CAMERA_ID', config['cameras']['global_camera_id']))
config['visualization']['port'] = int(os.getenv('VIZ_PORT', config['visualization']['port']))
```

### Programmatic Updates

```python
from configs import load_config, save_config
from configs.config_schema import set_nested

config = load_config()

# Update nested value
set_nested(config, 'cameras.global_resolution', [1920, 1080])
set_nested(config, 'guidance.robot_plays_white', True)

save_config(config)
```

## Dependencies

- `pyyaml` - YAML parsing (install: `pip install pyyaml`)
- `opencv-python` - Camera detection (install: `pip install opencv-python`)

## Files

- `configs/current.yaml` - Active configuration (created by wizard)
- `configs/config_schema.py` - Schema, defaults, validation
- `configs/__init__.py` - Load/save utilities
- `create_config.py` - Interactive setup wizard

## Best Practices

1. **Always run wizard on new systems**: Use `create_config.py` instead of manual editing
2. **Version control**: Commit default config, gitignore machine-specific overrides
3. **Backup configs**: Copy `configs/current.yaml` before major changes
4. **Validate after edits**: Use `validate_config()` if manually editing YAML
5. **Test cameras**: Run camera detection wizard after hardware changes
