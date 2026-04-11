"""
Keyboard Input Module

Non-blocking keyboard input handler for Linux terminals.
"""

import sys
import tty
import termios
import select
from typing import Optional


class KeyboardInput:
    """Non-blocking keyboard input handler for Linux terminals."""

    def __init__(self):
        self.old_settings = None

    def __enter__(self):
        """Set terminal to raw mode for character-by-character input."""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *args):
        """Restore terminal settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get a single keypress with timeout.

        Args:
            timeout: Maximum time to wait for input in seconds

        Returns:
            str: The key pressed, or None if timeout
            Special keys: 'ESC' for escape, 'ENTER' for enter
        """
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            char = sys.stdin.read(1)
            if char == '\x1b':  # Escape sequence
                # Check for arrow keys or plain escape
                rlist2, _, _ = select.select([sys.stdin], [], [], 0.01)
                if rlist2:
                    # Arrow key or other escape sequence - consume and return key name
                    seq = sys.stdin.read(2)
                    arrow_map = {'[A': 'UP', '[B': 'DOWN', '[C': 'RIGHT', '[D': 'LEFT'}
                    return arrow_map.get(seq, 'ESC')
                return 'ESC'
            elif char == '\n' or char == '\r':
                return 'ENTER'
            elif char.isdigit():
                return char
            else:
                return char
        return None
