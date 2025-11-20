"""
Generate Overlay Driver Script

Simple CLI tool to generate guidance overlays from state cache.
Can be called by guidance system, robot control, VLA, or user.

Usage:
    python generate_overlay.py                    # Generate from current cache
    python generate_overlay.py --action 0         # Generate for specific action
    python generate_overlay.py --status           # Print cache status
"""

import argparse
import sys
from pathlib import Path

from guidance.guidance_system import GuidanceSystem
from utils.state_cache import StateCache


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate guidance overlay from state cache"
    )
    parser.add_argument(
        "--action",
        type=int,
        default=None,
        help="Generate overlay for specific action index (default: current action from cache)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print cache status and exit (do not generate overlay)"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/state_cache.json",
        help="Path to state cache file (default: data/state_cache.json)"
    )

    args = parser.parse_args()

    # Check if cache exists
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Error: State cache not found at {cache_path}")
        print("Run detection first or initialize cache")
        return 1

    # Initialize guidance system
    guidance = GuidanceSystem(config={'cache_path': str(cache_path)})

    # Print status if requested
    if args.status:
        guidance.print_cache_status()
        return 0

    # Generate overlay
    print("=" * 60)
    print("Generating Guidance Overlay")
    print("=" * 60)

    try:
        if args.action is not None:
            # Generate for specific action
            success = guidance.generate_overlay_for_action(args.action)
        else:
            # Generate from cache (current action)
            success = guidance.generate_overlay_from_cache()

        if success:
            print("=" * 60)
            print("✓ Overlay generated successfully!")
            print("=" * 60)

            # Print what was highlighted
            cache_data = guidance.cache.get()
            robot_state = cache_data.get("robot_state", {})
            action_sequence = robot_state.get("action_sequence", [])
            action_index = robot_state.get("action_index", 0)

            if action_sequence and action_index < len(action_sequence):
                current_action = action_sequence[action_index]
                print(f"\nHighlighted Action:")
                print(f"  Type: {current_action['type']}")
                print(f"  Square: {current_action.get('square', 'graveyard')}")
                print(f"  Piece: {current_action.get('piece', 'N/A')}")
                print(f"  Status: {current_action.get('status', 'unknown')}")
                print(f"\nProgress: Action {action_index + 1}/{len(action_sequence)}")
                print(f"Remaining: {robot_state.get('actions_remaining', 0)} actions\n")

            return 0
        else:
            print("=" * 60)
            print("✗ Failed to generate overlay")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"✗ Error generating overlay: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
