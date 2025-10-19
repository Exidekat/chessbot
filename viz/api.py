"""
Visualization Tool API Server

FastAPI server with WebSocket support for real-time updates.
Serves camera streams, state cache, and React SPA.
"""

import json
import asyncio
from pathlib import Path
from typing import Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from viz.stream_manager import StreamManager
from viz.file_watcher import FileWatcherManager
from utils.state_cache import StateCache


# Initialize FastAPI app
app = FastAPI(
    title="Chess Robot Visualization Tool",
    description="Real-time visualization of camera feeds and robot state",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in dev (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
stream_manager: Optional[StreamManager] = None
file_watcher: Optional[FileWatcherManager] = None
state_cache: Optional[StateCache] = None

# WebSocket connection management
active_connections: Set[WebSocket] = set()


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup."""
    global stream_manager, file_watcher, state_cache

    print("[VizAPI] Starting visualization server...")

    # Initialize state cache
    state_cache = StateCache()
    print("[VizAPI] State cache loaded")

    # Initialize stream manager
    stream_manager = StreamManager()
    stream_manager.start()

    # Initialize file watcher with callbacks
    file_watcher = FileWatcherManager()

    # Set up callbacks for filesystem events
    file_watcher.set_cache_callback(on_cache_update)
    file_watcher.set_overlay_callback(on_overlay_update)
    file_watcher.set_flag_callback(on_flag_update)

    # Start watching
    file_watcher.start()

    print("[VizAPI] Visualization server ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on server shutdown."""
    print("[VizAPI] Shutting down...")

    if file_watcher:
        file_watcher.stop()

    if stream_manager:
        stream_manager.stop()

    # Close all WebSocket connections
    for connection in active_connections.copy():
        await connection.close()

    print("[VizAPI] Shutdown complete")


# ============================================================================
# Filesystem Event Handlers
# ============================================================================

def on_cache_update(cache_data: dict):
    """Handle state cache updates."""
    print("[VizAPI] Cache updated, notifying clients...")
    # Notify all connected WebSocket clients
    asyncio.create_task(broadcast_message({
        "type": "cache_update",
        "data": cache_data
    }))


def on_overlay_update(overlay_path: str):
    """Handle overlay image updates."""
    print(f"[VizAPI] Overlay updated: {overlay_path}")
    asyncio.create_task(broadcast_message({
        "type": "overlay_update",
        "path": overlay_path
    }))


def on_flag_update():
    """Handle overlay flag updates."""
    print("[VizAPI] Overlay flag updated")
    asyncio.create_task(broadcast_message({
        "type": "flag_update"
    }))


# ============================================================================
# WebSocket Handlers
# ============================================================================

async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients."""
    disconnected = set()

    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            print(f"[VizAPI] Error sending to client: {e}")
            disconnected.add(connection)

    # Remove disconnected clients
    active_connections.difference_update(disconnected)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Sends:
    - cache_update: When state cache changes
    - overlay_update: When overlay image changes
    - flag_update: When overlay flag changes
    """
    await websocket.accept()
    active_connections.add(websocket)
    print(f"[VizAPI] WebSocket client connected ({len(active_connections)} total)")

    try:
        # Send initial state
        if state_cache:
            initial_state = state_cache.get()
            await websocket.send_json({
                "type": "initial_state",
                "data": initial_state
            })

        # Keep connection alive
        while True:
            # Wait for messages from client (mostly for keepalive)
            await websocket.receive_text()

    except WebSocketDisconnect:
        print(f"[VizAPI] WebSocket client disconnected")
    except Exception as e:
        print(f"[VizAPI] WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)
        print(f"[VizAPI] WebSocket clients remaining: {len(active_connections)}")


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - serve React app in production."""
    site_dir = Path(__file__).parent / "site" / "dist"
    index_file = site_dir / "index.html"

    if index_file.exists():
        return FileResponse(index_file)
    else:
        return {
            "message": "Chess Robot Visualization API",
            "docs": "/docs",
            "status": "ready"
        }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "websocket_clients": len(active_connections),
        "streams": stream_manager.get_stream_status() if stream_manager else {}
    }


@app.get("/api/state")
async def get_state():
    """Get current state cache."""
    if state_cache is None:
        return {"error": "State cache not initialized"}

    return state_cache.get()


@app.get("/api/stream/global")
async def get_global_stream():
    """Get current global camera frame as JPEG."""
    if stream_manager is None:
        return Response(status_code=503, content="Stream manager not initialized")

    jpeg_bytes = stream_manager.get_global_frame_jpeg()

    if jpeg_bytes is None:
        # Return placeholder
        jpeg_bytes = stream_manager.get_placeholder_jpeg(text="Global Camera")

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/stream/gripper")
async def get_gripper_stream():
    """Get current gripper camera frame as JPEG."""
    if stream_manager is None:
        return Response(status_code=503, content="Stream manager not initialized")

    jpeg_bytes = stream_manager.get_gripper_frame_jpeg()

    if jpeg_bytes is None:
        # Return placeholder
        jpeg_bytes = stream_manager.get_placeholder_jpeg(text="Gripper Camera")

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/stream/overlay")
async def get_overlay_stream():
    """Get current guidance overlay as JPEG."""
    if stream_manager is None:
        return Response(status_code=503, content="Stream manager not initialized")

    jpeg_bytes = stream_manager.get_overlay_frame_jpeg()

    if jpeg_bytes is None:
        # Return placeholder
        jpeg_bytes = stream_manager.get_placeholder_jpeg(text="Guidance Overlay")

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "websocket_clients": len(active_connections),
        "stream_status": stream_manager.get_stream_status() if stream_manager else {},
        "cache_status": "loaded" if state_cache else "not loaded"
    }


# Mount static files in production (after building React app)
site_dist = Path(__file__).parent / "site" / "dist"
if site_dist.exists():
    app.mount("/", StaticFiles(directory=str(site_dist), html=True), name="static")
