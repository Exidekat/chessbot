"""
Get VLM Prompt - Board analysis with JSON output for agent integration.

Runs the full guidance pipeline (capture -> detect -> calculate -> decompose)
and outputs VLM prompt stages as structured JSON. The agent can pass these
stages directly to the /robot/chess_move bridge endpoint.

Usage:
    # One-shot JSON output (agent exec pattern)
    python scripts/get_vlm_prompt.py --json

    # Save to file for agent consumption
    python scripts/get_vlm_prompt.py --json --output /tmp/gen-int/camera/vlm_prompt.json

    # Stream overlay to virtual camera
    python scripts/get_vlm_prompt.py --stream

    # With custom parameters
    python scripts/get_vlm_prompt.py --json --turn black --rotation right --corner-conf 0.005
"""

import argparse
import json
import sys
import time
import threading
import io
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from guidance.board_detector import BoardDetector
from guidance.move_calculator import MoveCalculator
from guidance.move_decomposer import decompose_move

from utils.camera_helpers import (
    capture_720p_yuyv,
    capture_4k_downscale,
    get_default_global_camera,
    DEFAULT_USE_YUYV,
)


def analyze_board(args) -> dict:
    """Run full pipeline and return structured result."""
    import chess

    # 1. Camera capture
    device = args.global_camera or get_default_global_camera()
    if not device:
        return {"success": False, "error": "No camera found"}

    image_path = Path("data/vlm_prompt_capture.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    use_yuyv = DEFAULT_USE_YUYV
    if args.mjpeg:
        use_yuyv = False
    elif args.yuyv:
        use_yuyv = True

    if use_yuyv:
        ok = capture_720p_yuyv(device, image_path)
    else:
        ok = capture_4k_downscale(device, image_path)
    if not ok:
        return {"success": False, "error": "Camera capture failed"}

    # 2. Board detection
    detector = BoardDetector(
        camera_position=args.rotation,
        use_corner_rgb=not args.corner_grayscale,
        use_piece_rgb=not args.piece_grayscale,
    )

    turn = "w" if args.turn == "white" else "b"
    try:
        fen, transformed = detector.detect_board_state(
            str(image_path),
            corner_conf=args.corner_conf,
            debug=True,
            turn=turn,
        )
    except Exception as e:
        return {"success": False, "error": f"Detection failed: {e}"}

    # 3. Parse FEN into board
    try:
        board = chess.Board(fen)
    except ValueError:
        return {"success": False, "error": f"Invalid FEN: {fen}"}

    # 4. Calculate best move
    try:
        calculator = MoveCalculator(engine_path=args.engine)
        move = calculator.calculate_best_move(board, time_limit=args.time)
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Engine not found: {args.engine}",
            "fen": fen,
        }

    if not move:
        return {"success": False, "error": "No legal moves available", "fen": fen}

    # 5. Decompose into stages with VLM prompts
    san = board.san(move)
    stages = decompose_move(board, move)

    return {
        "success": True,
        "fen": fen,
        "best_move": {"uci": move.uci(), "san": san},
        "stages": stages,
        "board_ascii": str(board),
        "camera_rotation": args.rotation,
        "turn": args.turn,
    }


def stream_overlay(detector, stages, device, rotation):
    """Stream overlay to virtual camera until Ctrl+C."""
    from guidance import apply_stage_overlay_to_frame

    try:
        from cameras import LiveCameraCapture, VirtualCamera
    except ImportError:
        print("Camera modules not available for streaming.", file=sys.stderr)
        return

    live = LiveCameraCapture(device)
    live.start()
    time.sleep(1)

    vcam = VirtualCamera("/dev/video7", 1280, 720)
    if not vcam.start():
        print("Virtual camera failed. Run:", file=sys.stderr)
        print("  sudo modprobe v4l2loopback devices=1 video_nr=7 "
              "card_label='ChessBot Virtual Cam' exclusive_caps=1",
              file=sys.stderr)
        live.stop()
        return

    print(f"Streaming {len(stages)} stage(s) to /dev/video7", file=sys.stderr)
    print("View: ffplay -fflags nobuffer -flags low_delay /dev/video7",
          file=sys.stderr)
    print("Press ENTER to advance stages, Ctrl+C to stop", file=sys.stderr)

    transformed_path = "data/chessboard_transformed.png"

    try:
        for i, stage in enumerate(stages):
            print(f"Stage {i+1}/{len(stages)}: {stage['vlm_prompt']}",
                  file=sys.stderr)

            stop = threading.Event()

            def wait_for_enter():
                input()
                stop.set()

            t = threading.Thread(target=wait_for_enter, daemon=True)
            t.start()

            while not stop.is_set():
                frame = live.get_latest_frame()
                if frame is not None:
                    overlayed = apply_stage_overlay_to_frame(
                        frame, stage, transformed_path,
                        detector, detector.perspective_matrix, rotation,
                    )
                    if overlayed is not None:
                        vcam.write_frame(overlayed)
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        live.stop()
        vcam.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Get VLM prompt for chess move (JSON output for agent integration)"
    )
    parser.add_argument("--global-camera", type=str, default=None,
                        help="Camera device (auto-detects WBC-0E01)")
    parser.add_argument("--engine", type=str, default="stockfish",
                        help="Stockfish path (default: stockfish)")
    parser.add_argument("--time", type=float, default=1.0,
                        help="Engine time limit in seconds (default: 1.0)")
    parser.add_argument("--rotation", type=str, default="right",
                        choices=["left", "right", "top", "bottom"],
                        help="Camera rotation (default: right)")
    parser.add_argument("--turn", type=str, default="black",
                        choices=["white", "black"],
                        help="Whose turn (default: black)")
    parser.add_argument("--yuyv", action="store_true",
                        help="Force native 720p YUYV capture")
    parser.add_argument("--mjpeg", action="store_true",
                        help="Use 4K MJPEG -> 720p downscale")
    parser.add_argument("--stream", action="store_true",
                        help="Stream overlay to /dev/video7 until Ctrl+C")
    parser.add_argument("--json", action="store_true",
                        help="Output only JSON (suppress logging to stdout)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON to file instead of stdout")
    parser.add_argument("--corner-conf", type=float, default=0.005,
                        help="Corner detection threshold (default: 0.005)")
    parser.add_argument("--corner-grayscale", action="store_true",
                        help="Use grayscale+CLAHE for corner detection")
    parser.add_argument("--piece-grayscale", action="store_true",
                        help="Use grayscale+CLAHE for piece detection")
    args = parser.parse_args()

    # In --json mode, capture stdout from BoardDetector prints
    if args.json:
        captured_stdout = io.StringIO()
        real_stdout = sys.stdout
        sys.stdout = captured_stdout
        result = analyze_board(args)
        sys.stdout = real_stdout
    else:
        result = analyze_board(args)

    # Output JSON
    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        if not args.json:
            print(f"Written to {args.output}")
    else:
        print(output)

    # Optional: stream overlay to virtual camera
    if args.stream and result.get("success"):
        device = args.global_camera or get_default_global_camera()
        # Re-initialize detector for perspective_matrix
        detector = BoardDetector(
            camera_position=args.rotation,
            use_corner_rgb=not args.corner_grayscale,
            use_piece_rgb=not args.piece_grayscale,
        )
        turn = "w" if args.turn == "white" else "b"
        detector.detect_board_state(
            "data/vlm_prompt_capture.png",
            corner_conf=args.corner_conf,
            debug=True,
            turn=turn,
        )
        stream_overlay(detector, result["stages"], device, args.rotation)

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
