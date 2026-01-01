#!/usr/bin/env python3
"""
MPS Compatibility Patches for LeRobot PI05

This module provides runtime patches to enable PI05 training on Apple Silicon (MPS).

PyTorch's MPS backend is missing several operations required by PI05:
- aten::_sample_dirichlet (used by Beta distribution sampling)

Rather than using the global PYTORCH_ENABLE_MPS_FALLBACK flag (which causes
significant slowdown), we apply targeted patches that minimize CPU transfers.

Usage:
    Import this module BEFORE loading PI05Policy:

    >>> from vla.mps_patches import apply_mps_patches
    >>> apply_mps_patches()
    >>> from lerobot.policies.pi05 import PI05Policy  # Now uses patched version
"""

from .beta_sampling import sample_beta_mps_compatible, patch_lerobot_beta_sampling


def apply_mps_patches(verbose: bool = True) -> bool:
    """
    Apply all MPS compatibility patches to LeRobot.

    This function should be called once at the start of your script,
    before importing any LeRobot models.

    Args:
        verbose: If True, print patch status messages

    Returns:
        True if all patches applied successfully, False otherwise
    """
    success = True

    if verbose:
        print("=" * 60)
        print("Applying MPS Compatibility Patches")
        print("=" * 60)

    # Patch 1: Beta sampling for flow matching
    if not patch_lerobot_beta_sampling():
        success = False
        if verbose:
            print("[WARN] Beta sampling patch failed")

    if verbose:
        if success:
            print("[OK] All MPS patches applied successfully")
        else:
            print("[WARN] Some patches failed - training may not work on MPS")
        print("=" * 60)
        print()

    return success


__all__ = [
    'apply_mps_patches',
    'sample_beta_mps_compatible',
    'patch_lerobot_beta_sampling',
]
