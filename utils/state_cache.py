"""
State Cache Manager

Thread-safe JSON-based cache for chess robot state.
Supports multi-source updates from guidance, robot, VLA, and user.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List
from copy import deepcopy


class StateCache:
    """
    Thread-safe state cache for chess robot system.

    Stores current game state, robot state, and guidance information.
    Supports atomic updates from multiple sources (guidance, robot, VLA, user).
    """

    DEFAULT_STATE = {
        "game_state": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "turn": "white",
            "board_image_path": None,
            "transformed_board_path": None
        },
        "robot_state": {
            "is_robot_turn": False,
            "holding_piece": False,
            "current_move": None,
            "move_type": None,
            "action_sequence": [],
            "action_index": 0,
            "actions_remaining": 0,
            "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # NEW: 6 joints in radians
            "joint_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # NEW: optional velocities
            "gripper_state": 0.0,         # NEW: gripper position (joint 6)
            "last_joint_update": 0.0      # NEW: timestamp of last update
        },
        "guidance_state": {
            "best_move": None,
            "best_move_san": None,
            "evaluation": None
        },
        "metadata": {
            "timestamp": 0.0,
            "last_updated_by": "system",
            "cache_version": "1.0",
            "overlay_path": "data/guidance_overlay.png"
        }
    }

    def __init__(self, cache_path: str = "data/state_cache.json"):
        """
        Initialize state cache.

        Args:
            cache_path: Path to JSON cache file
        """
        self.cache_path = Path(cache_path)
        self.backup_path = self.cache_path.parent / "state_cache_backup.json"
        self.lock = threading.Lock()

        # Ensure data directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize cache
        if self.cache_path.exists():
            self._load()
        else:
            self._initialize_default()

    def _load(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'r') as f:
                self._state = json.load(f)
        except Exception as e:
            print(f"[StateCache] Error loading cache: {e}")
            print("[StateCache] Initializing with default state")
            self._initialize_default()

    def _initialize_default(self):
        """Initialize cache with default state."""
        self._state = deepcopy(self.DEFAULT_STATE)
        self._state["metadata"]["timestamp"] = time.time()
        self._save()

    def _save(self):
        """Save cache to disk (caller must hold lock)."""
        try:
            # Write to temp file first (atomic write)
            temp_path = self.cache_path.parent / f"{self.cache_path.name}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(self._state, f, indent=2)

            # Atomic rename
            temp_path.replace(self.cache_path)

        except Exception as e:
            print(f"[StateCache] Error saving cache: {e}")

    def get(self, key_path: Optional[str] = None) -> Any:
        """
        Get value from cache.

        Args:
            key_path: Dot-separated path (e.g., "robot_state.holding_piece")
                     If None, returns entire cache

        Returns:
            Requested value or entire cache dict
        """
        with self.lock:
            if key_path is None:
                return deepcopy(self._state)

            # Navigate nested dict
            keys = key_path.split('.')
            value = self._state
            for key in keys:
                value = value.get(key)
                if value is None:
                    return None

            return deepcopy(value) if isinstance(value, (dict, list)) else value

    def update(self, data: Dict, source: str = "unknown"):
        """
        Update cache with new data (deep merge).

        Args:
            data: Dictionary with updates (can be nested)
            source: Source of update ("guidance", "robot", "vla", "user")
        """
        with self.lock:
            # Deep merge
            self._deep_merge(self._state, data)

            # Update metadata
            self._state["metadata"]["timestamp"] = time.time()
            self._state["metadata"]["last_updated_by"] = source

            # Save to disk
            self._save()

    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def initialize_game(self, fen: str, robot_plays_white: bool = False):
        """
        Initialize a new game.

        Args:
            fen: Starting FEN position
            robot_plays_white: Whether robot plays white pieces
        """
        turn = "white" if "w" in fen.split()[1] else "black"
        is_robot_turn = (turn == "white" and robot_plays_white) or \
                       (turn == "black" and not robot_plays_white)

        self.update({
            "game_state": {
                "fen": fen,
                "turn": turn
            },
            "robot_state": {
                "is_robot_turn": is_robot_turn,
                "holding_piece": False,
                "current_move": None,
                "move_type": None,
                "action_sequence": [],
                "action_index": 0,
                "actions_remaining": 0
            },
            "guidance_state": {
                "best_move": None,
                "best_move_san": None,
                "evaluation": None
            }
        }, source="system")

    def set_action_status(self, action_index: int, status: str, source: str = "robot"):
        """
        Update status of a specific action.

        Args:
            action_index: Index of action to update
            status: New status ("pending", "in_progress", "complete", "failed")
            source: Source of update
        """
        with self.lock:
            actions = self._state["robot_state"]["action_sequence"]
            if 0 <= action_index < len(actions):
                actions[action_index]["status"] = status
                self._state["metadata"]["timestamp"] = time.time()
                self._state["metadata"]["last_updated_by"] = source
                self._save()

    def advance_action(self, source: str = "robot"):
        """
        Advance to next action in sequence.

        Args:
            source: Source of update
        """
        with self.lock:
            robot_state = self._state["robot_state"]

            # Mark current action as complete
            current_idx = robot_state["action_index"]
            actions = robot_state["action_sequence"]

            if 0 <= current_idx < len(actions):
                actions[current_idx]["status"] = "complete"

            # Advance to next
            robot_state["action_index"] += 1
            robot_state["actions_remaining"] = max(0, robot_state["actions_remaining"] - 1)

            # Update metadata
            self._state["metadata"]["timestamp"] = time.time()
            self._state["metadata"]["last_updated_by"] = source

            self._save()

    def get_current_action(self) -> Optional[Dict]:
        """
        Get the current action to execute.

        Returns:
            Current action dict or None if all complete
        """
        with self.lock:
            robot_state = self._state["robot_state"]
            idx = robot_state["action_index"]
            actions = robot_state["action_sequence"]

            if 0 <= idx < len(actions):
                return deepcopy(actions[idx])
            return None

    def update_joint_positions(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
        gripper_state: float = 0.0,
        source: str = "robot"
    ):
        """
        Update robot joint positions for VLA training data collection.

        This method enables passive observation of robot state by collect_vla_episodes.py
        without requiring a second serial connection to the robot. tele_op.py writes
        joint positions here, and collect_vla_episodes.py reads them.

        Args:
            positions: List of 6 joint angles in radians
            velocities: Optional list of 6 joint velocities (rad/s)
            gripper_state: Gripper position (joint 6 in radians)
            source: Source of update (default: "robot")
        """
        with self.lock:
            self._state["robot_state"]["joint_positions"] = positions
            if velocities:
                self._state["robot_state"]["joint_velocities"] = velocities
            self._state["robot_state"]["gripper_state"] = gripper_state
            self._state["robot_state"]["last_joint_update"] = time.time()
            self._state["metadata"]["timestamp"] = time.time()
            self._state["metadata"]["last_updated_by"] = source
            self._save()

    def validate(self) -> bool:
        """
        Validate cache structure.

        Returns:
            True if valid, False otherwise
        """
        with self.lock:
            required_keys = ["game_state", "robot_state", "guidance_state", "metadata"]

            for key in required_keys:
                if key not in self._state:
                    print(f"[StateCache] Validation failed: missing key '{key}'")
                    return False

            # Check robot state consistency
            robot_state = self._state["robot_state"]
            if len(robot_state["action_sequence"]) > 0:
                expected_remaining = len(robot_state["action_sequence"]) - robot_state["action_index"]
                if robot_state["actions_remaining"] != expected_remaining:
                    print(f"[StateCache] Warning: actions_remaining inconsistent")

            return True

    def create_backup(self):
        """Create backup of current cache."""
        with self.lock:
            try:
                with open(self.backup_path, 'w') as f:
                    json.dump(self._state, f, indent=2)
                print(f"[StateCache] Backup created: {self.backup_path}")
            except Exception as e:
                print(f"[StateCache] Error creating backup: {e}")

    def restore_from_backup(self):
        """Restore cache from backup."""
        if not self.backup_path.exists():
            print(f"[StateCache] No backup found at {self.backup_path}")
            return False

        with self.lock:
            try:
                with open(self.backup_path, 'r') as f:
                    self._state = json.load(f)
                self._save()
                print(f"[StateCache] Restored from backup")
                return True
            except Exception as e:
                print(f"[StateCache] Error restoring from backup: {e}")
                return False

    def reset(self):
        """Reset cache to default state."""
        with self.lock:
            self._state = deepcopy(self.DEFAULT_STATE)
            self._state["metadata"]["timestamp"] = time.time()
            self._state["metadata"]["last_updated_by"] = "system"
            self._save()
            print("[StateCache] Reset to default state")

    def print_status(self):
        """Print current cache status (for debugging)."""
        with self.lock:
            print("\n" + "=" * 60)
            print("State Cache Status")
            print("=" * 60)

            game = self._state["game_state"]
            robot = self._state["robot_state"]
            guidance = self._state["guidance_state"]
            meta = self._state["metadata"]

            print(f"FEN: {game['fen']}")
            print(f"Turn: {game['turn']}")
            print(f"\nRobot Turn: {robot['is_robot_turn']}")
            print(f"Holding Piece: {robot['holding_piece']}")
            print(f"Current Move: {robot['current_move']} ({robot['move_type']})")
            print(f"Action: {robot['action_index'] + 1}/{len(robot['action_sequence'])}")
            print(f"Actions Remaining: {robot['actions_remaining']}")

            if robot['action_sequence']:
                current_action = robot['action_sequence'][robot['action_index']] \
                    if robot['action_index'] < len(robot['action_sequence']) else None
                if current_action:
                    print(f"Current Action: {current_action}")

            print(f"\nBest Move: {guidance['best_move']} ({guidance['best_move_san']})")
            print(f"Evaluation: {guidance['evaluation']}")

            print(f"\nLast Updated: {time.ctime(meta['timestamp'])}")
            print(f"Updated By: {meta['last_updated_by']}")
            print("=" * 60 + "\n")
