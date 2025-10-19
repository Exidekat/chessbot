"""
Visualization Tool Launcher

Starts the web-based visualization tool for chess robot system.
Serves camera feeds, state cache, and React SPA.

Usage:
    # Development mode (React dev server + FastAPI)
    python start_viz_tool.py --dev

    # Production mode (bundled React served from FastAPI)
    python start_viz_tool.py

    # Custom host/port
    python start_viz_tool.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import fastapi
        import uvicorn
        import watchdog
        print("✓ Python dependencies OK")
        return True
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e.name}")
        print("\nInstall with:")
        print("  pip install fastapi uvicorn[standard] watchdog python-socketio")
        return False


def check_npm_install():
    """Check if npm dependencies are installed."""
    site_dir = Path(__file__).parent / "viz" / "site"
    node_modules = site_dir / "node_modules"

    if not node_modules.exists():
        print("✗ React dependencies not installed")
        print(f"\nInstall with:")
        print(f"  cd {site_dir}")
        print("  npm install")
        return False

    print("✓ React dependencies OK")
    return True


def build_react_app():
    """Build React app for production."""
    site_dir = Path(__file__).parent / "viz" / "site"
    dist_dir = site_dir / "dist"

    print("\n" + "=" * 70)
    print("Building React app for production...")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=site_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print("✓ React app built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("✗ npm not found. Please install Node.js")
        return False


def start_dev_mode(host: str, port: int):
    """
    Start development mode with React dev server + FastAPI.

    Runs:
    1. FastAPI server on specified port
    2. React dev server on port 3000 (with proxy to FastAPI)
    """
    print("\n" + "=" * 70)
    print("DEVELOPMENT MODE")
    print("=" * 70)
    print(f"FastAPI server: http://{host}:{port}")
    print("React dev server: http://localhost:3000")
    print("=" * 70)
    print("\nStarting servers...")

    site_dir = Path(__file__).parent / "viz" / "site"

    # Start React dev server in background
    try:
        print("\n[1/2] Starting React dev server...")
        react_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=site_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✓ React dev server started on port 3000")
    except Exception as e:
        print(f"✗ Failed to start React dev server: {e}")
        return

    # Start FastAPI server (blocking)
    try:
        print(f"\n[2/2] Starting FastAPI server on {host}:{port}...")
        import uvicorn
        from viz.api import app

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            reload=True  # Auto-reload on code changes
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Clean up React process
        if react_process:
            react_process.terminate()
            react_process.wait()
            print("✓ React dev server stopped")


def start_production_mode(host: str, port: int):
    """
    Start production mode with bundled React served from FastAPI.

    Serves pre-built React app from dist/ directory.
    """
    print("\n" + "=" * 70)
    print("PRODUCTION MODE")
    print("=" * 70)
    print(f"Server: http://{host}:{port}")
    print("=" * 70)

    # Check if React app is built
    site_dir = Path(__file__).parent / "viz" / "site"
    dist_dir = site_dir / "dist"

    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print("\n⚠ React app not built. Building now...")
        if not build_react_app():
            print("\n✗ Cannot start production mode without build")
            return

    # Start FastAPI server
    try:
        print(f"\nStarting FastAPI server on {host}:{port}...")
        import uvicorn
        from viz.api import app

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nShutting down...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Chess Robot Visualization Tool Launcher"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (React HMR + FastAPI reload)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for FastAPI server (default: 8000)"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build React app, don't start server"
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("Chess Robot Visualization Tool")
    print("=" * 70)

    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        return 1

    if args.dev or not args.build_only:
        if not check_npm_install():
            return 1

    # Build-only mode
    if args.build_only:
        print("\nBuilding React app...")
        return 0 if build_react_app() else 1

    # Start appropriate mode
    if args.dev:
        start_dev_mode(args.host, args.port)
    else:
        start_production_mode(args.host, args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
