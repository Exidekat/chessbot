# Visualization Tool

Real-time web-based visualization system for the chess robot. Displays live camera feeds, guidance overlays, and robot state with filesystem-reactive updates.

## Features

- **Real-time Camera Feeds**: Global overhead camera and gripper-mounted camera
- **Guidance Overlay**: Color-coded action highlights from the guidance system
- **Robot State Display**: Live state cache monitoring with action sequences
- **Filesystem Reactive**: Automatic updates when state/overlay files change
- **WebSocket Updates**: Instant push notifications for state changes
- **Development Mode**: Hot module reloading (HMR) for rapid development
- **Production Mode**: Optimized bundled deployment

## Architecture

```
viz/
├── api.py                 # FastAPI server with WebSocket support
├── stream_manager.py      # Camera frame encoding (JPEG streaming)
├── file_watcher.py        # Filesystem monitoring (watchdog)
└── site/                  # React SPA
    ├── src/
    │   ├── App.jsx        # Main application
    │   ├── components/
    │   │   ├── CameraFeed.jsx    # Camera display component
    │   │   └── StateViewer.jsx   # State cache viewer
    │   └── main.jsx
    ├── package.json
    └── vite.config.js
```

### Backend (FastAPI)

**Endpoints:**
- `GET /api/health` - Health check and system status
- `GET /api/state` - Current state cache (JSON)
- `GET /api/stream/global` - Global camera frame (JPEG)
- `GET /api/stream/gripper` - Gripper camera frame (JPEG)
- `GET /api/stream/overlay` - Guidance overlay (JPEG)
- `GET /api/stats` - System statistics
- `WS /ws` - WebSocket for real-time updates

**File Watching:**
- Monitors `data/state_cache.json` for state updates
- Monitors `data/guidance_overlay.png` for overlay changes
- Monitors `data/overlay_updated.flag` for regeneration signals
- Pushes updates to all connected clients via WebSocket

### Frontend (React + Vite)

**Layout:**
```
┌─────────────────────────────────────────────────┐
│  Chess Robot Visualization    [● Connected]     │
├──────────────┬──────────────┬──────────────────┤
│ Global Cam   │ Overlay      │ Gripper Cam      │
│              │              │                  │
│              │              │                  │
├──────────────┴──────────────┴──────────────────┤
│ Robot State                                    │
│  Game State | Robot State | Guidance | Meta    │
└─────────────────────────────────────────────────┘
```

**Components:**
- `CameraFeed`: Polls JPEG endpoints, auto-refreshes
- `StateViewer`: Displays state cache with organized sections

## Installation

### Backend Dependencies

```bash
# Install Python packages
pip install fastapi uvicorn[standard] watchdog python-socketio
```

### Frontend Dependencies

```bash
# Navigate to site directory
cd viz/site

# Install npm packages
npm install
```

## Usage

### Development Mode (Recommended)

Runs React dev server (port 3000) with HMR and FastAPI (port 8000) with auto-reload:

```bash
python scripts/start_viz_tool.py --dev
```

- React dev server: http://localhost:3000
- FastAPI backend: http://localhost:8000
- API docs: http://localhost:8000/docs

### Production Mode

Builds React app and serves from FastAPI:

```bash
# Build React app
python scripts/start_viz_tool.py --build-only

# Start production server
python scripts/start_viz_tool.py
```

- Web interface: http://localhost:8000

### Custom Configuration

```bash
# Custom host and port
python scripts/start_viz_tool.py --host 192.168.1.100 --port 8080

# Dev mode with custom FastAPI port
python scripts/start_viz_tool.py --dev --port 8080
```

## WebSocket Protocol

The WebSocket endpoint (`ws://localhost:8000/ws`) sends JSON messages:

### Initial State
```json
{
  "type": "initial_state",
  "data": {
    "game_state": {...},
    "robot_state": {...},
    "guidance_state": {...},
    "metadata": {...}
  }
}
```

### Cache Update
```json
{
  "type": "cache_update",
  "data": {
    "game_state": {...},
    "robot_state": {...},
    "guidance_state": {...},
    "metadata": {...}
  }
}
```

### Overlay Update
```json
{
  "type": "overlay_update",
  "path": "data/guidance_overlay.png"
}
```

### Flag Update
```json
{
  "type": "flag_update"
}
```

## Integration with Other Modules

### Guidance System

The guidance system updates the state cache and overlay:

```python
from guidance import GuidanceSystem

system = GuidanceSystem()

# Detect board and calculate move (updates cache)
fen, move, actions = system.detect_and_calculate('data/board.png')

# Generate overlay (triggers flag update → WebSocket notification)
system.generate_overlay_from_cache()
```

### Camera System

The visualization tool integrates with the camera manager:

```python
from cameras import CameraManager
from viz.stream_manager import StreamManager

# Camera manager is initialized by viz server
camera_mgr = CameraManager()
camera_mgr.start_all()

# Stream manager encodes frames for web
stream_mgr = StreamManager(camera_mgr)
jpeg_bytes = stream_mgr.get_global_frame_jpeg()
```

### State Cache

Direct access to state cache for updates:

```python
from utils.state_cache import StateCache

cache = StateCache()

# Update robot state (triggers WebSocket notification)
cache.update({
    "robot_state": {
        "is_robot_turn": True,
        "holding_piece": True,
        "action_index": 1
    }
}, source="robot")
```

## Camera Stream Refresh Rates

- **Global Camera**: 100ms (10 FPS) - Real-time robot monitoring
- **Gripper Camera**: 100ms (10 FPS) - Real-time piece manipulation
- **Guidance Overlay**: 500ms (2 FPS) - Static overlay, updates only when flagged

Refresh rates are configured in `App.jsx` and can be adjusted based on network conditions.

## Troubleshooting

### "No Signal" on Camera Feeds

**Cause**: Camera devices not available or camera manager failed to start

**Solutions**:
1. Check camera connections (USB or network)
2. Verify camera IDs in camera manager configuration
3. Test cameras directly:
   ```python
   from cameras import GlobalCamera
   cam = GlobalCamera(camera_id=0)
   cam.start()
   frame = cam.get_frame()
   ```

### WebSocket Disconnects Frequently

**Cause**: Network issues or server restart

**Solution**: WebSocket auto-reconnects. Check browser console for errors.

### Overlay Not Updating

**Cause**: File watcher not detecting changes or flag file not being touched

**Solutions**:
1. Verify overlay file exists: `data/guidance_overlay.png`
2. Verify flag file exists: `data/overlay_updated.flag`
3. Check file watcher logs in FastAPI output
4. Manually regenerate overlay:
   ```bash
   python scripts/generate_overlay.py
   ```

### React Build Fails

**Cause**: Missing npm dependencies or Node.js version incompatibility

**Solutions**:
1. Ensure Node.js >= 16 installed
2. Delete `node_modules` and reinstall:
   ```bash
   cd viz/site
   rm -rf node_modules package-lock.json
   npm install
   ```

## Development

### Adding New State Fields

1. Update state cache structure in `utils/state_cache.py`
2. Update `StateViewer.jsx` to display new fields
3. Update guidance/robot/VLA modules to populate new fields

### Adding New Camera Streams

1. Add camera class in `cameras/` directory
2. Register in `CameraManager`
3. Add stream endpoint in `viz/api.py`
4. Add `CameraFeed` component in `App.jsx`

### Modifying UI Layout

Edit `viz/site/src/App.css` for styling:
- Grid layout: `.main-content { grid-template-columns: ... }`
- Colors: Modify `.state-item`, `.camera-panel`, etc.
- Responsive design: Add media queries

## Performance Considerations

- **Frame Encoding**: JPEG quality set to 85% (adjustable in `StreamManager`)
- **WebSocket Debouncing**: File changes debounced to 100ms
- **Object URL Cleanup**: Camera feed component properly revokes object URLs
- **Stream Throttling**: Overlay refresh rate lower than real-time cameras

## Security Notes

- **CORS**: Currently allows all origins (development). Restrict in production:
  ```python
  # In api.py
  allow_origins=["http://yourdomain.com"]
  ```
- **Authentication**: No authentication currently implemented. Add middleware for production.
- **File Access**: Server only serves files from `data/` directory

## Future Enhancements

- [ ] Action sequence playback controls (pause, step)
- [ ] Historical state timeline view
- [ ] Recording/playback of game sessions
- [ ] Mobile-responsive layout
- [ ] Multi-robot support (multiple state caches)
- [ ] Authentication and user management
- [ ] HTTPS support
