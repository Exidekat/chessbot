"""
Filesystem Watcher for Visualization Tool

Monitors state cache, overlay images, and flag files for changes.
Emits events via callback when files are modified.
"""

import time
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent


class CacheWatcher(FileSystemEventHandler):
    """Watches state cache file for modifications."""

    def __init__(self, cache_path: Path, callback: Callable[[Dict[str, Any]], None]):
        """
        Initialize cache watcher.

        Args:
            cache_path: Path to state_cache.json
            callback: Function to call when cache changes (receives cache dict)
        """
        self.cache_path = cache_path.resolve()
        self.callback = callback
        self.last_modified = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            event_path = Path(event.src_path).resolve()
            if event_path == self.cache_path:
                # Debounce: only trigger if >100ms since last event
                current_time = time.time()
                if current_time - self.last_modified > 0.1:
                    self.last_modified = current_time
                    self._load_and_notify()

    def _load_and_notify(self):
        """Load cache and notify callback."""
        try:
            import json
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)
            self.callback(cache_data)
        except Exception as e:
            print(f"[CacheWatcher] Error loading cache: {e}")


class OverlayWatcher(FileSystemEventHandler):
    """Watches overlay image for modifications."""

    def __init__(self, overlay_path: Path, callback: Callable[[str], None]):
        """
        Initialize overlay watcher.

        Args:
            overlay_path: Path to guidance_overlay.png
            callback: Function to call when overlay changes (receives path string)
        """
        self.overlay_path = overlay_path.resolve()
        self.callback = callback
        self.last_modified = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            event_path = Path(event.src_path).resolve()
            if event_path == self.overlay_path:
                # Debounce
                current_time = time.time()
                if current_time - self.last_modified > 0.1:
                    self.last_modified = current_time
                    self.callback(str(self.overlay_path))


class FlagWatcher(FileSystemEventHandler):
    """Watches flag file for overlay regeneration signals."""

    def __init__(self, flag_path: Path, callback: Callable[[], None]):
        """
        Initialize flag watcher.

        Args:
            flag_path: Path to overlay flag file
            callback: Function to call when flag is updated
        """
        self.flag_path = flag_path.resolve()
        self.callback = callback
        self.last_modified = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            event_path = Path(event.src_path).resolve()
            if event_path == self.flag_path:
                # Debounce
                current_time = time.time()
                if current_time - self.last_modified > 0.1:
                    self.last_modified = current_time
                    self.callback()


class FileWatcherManager:
    """
    Manages all filesystem watchers for the visualization tool.

    Watches:
    - State cache JSON for state updates
    - Overlay PNG for visual updates
    - Flag file for overlay regeneration signals
    """

    def __init__(
        self,
        cache_path: str = "data/state_cache.json",
        overlay_path: str = "data/guidance_overlay.png",
        flag_path: str = "data/overlay_updated.flag"
    ):
        """
        Initialize file watcher manager.

        Args:
            cache_path: Path to state cache JSON
            overlay_path: Path to guidance overlay PNG
            flag_path: Path to overlay update flag file
        """
        self.cache_path = Path(cache_path)
        self.overlay_path = Path(overlay_path)
        self.flag_path = Path(flag_path)

        # Callbacks
        self.on_cache_update: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_overlay_update: Optional[Callable[[str], None]] = None
        self.on_flag_update: Optional[Callable[[], None]] = None

        # Watchdog observers
        self.observer = Observer()
        self.running = False

        # Event handlers
        self.cache_handler = None
        self.overlay_handler = None
        self.flag_handler = None

    def start(self):
        """Start watching filesystem."""
        if self.running:
            print("[FileWatcher] Already running")
            return

        # Create handlers with callbacks
        if self.on_cache_update:
            self.cache_handler = CacheWatcher(self.cache_path, self.on_cache_update)
            self.observer.schedule(
                self.cache_handler,
                str(self.cache_path.parent),
                recursive=False
            )

        if self.on_overlay_update:
            self.overlay_handler = OverlayWatcher(self.overlay_path, self.on_overlay_update)
            self.observer.schedule(
                self.overlay_handler,
                str(self.overlay_path.parent),
                recursive=False
            )

        if self.on_flag_update:
            self.flag_handler = FlagWatcher(self.flag_path, self.on_flag_update)
            self.observer.schedule(
                self.flag_handler,
                str(self.flag_path.parent),
                recursive=False
            )

        # Start observer
        self.observer.start()
        self.running = True
        print("[FileWatcher] Started watching:")
        print(f"  - Cache: {self.cache_path}")
        print(f"  - Overlay: {self.overlay_path}")
        print(f"  - Flag: {self.flag_path}")

    def stop(self):
        """Stop watching filesystem."""
        if not self.running:
            return

        self.observer.stop()
        self.observer.join(timeout=2.0)
        self.running = False
        print("[FileWatcher] Stopped")

    def set_cache_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for cache updates."""
        self.on_cache_update = callback

    def set_overlay_callback(self, callback: Callable[[str], None]):
        """Set callback for overlay updates."""
        self.on_overlay_update = callback

    def set_flag_callback(self, callback: Callable[[], None]):
        """Set callback for flag updates."""
        self.on_flag_update = callback
