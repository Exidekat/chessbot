#!/usr/bin/env python3
"""Verify OpenPI installation in cb conda environment."""

import sys


def check_import(module_name, display_name=None):
    """Check if a module can be imported and display its version."""
    display_name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"[X] {display_name}: FAILED - {e}")
        return False


def main():
    """Run verification checks."""
    print("=" * 60)
    print("OpenPI Installation Verification (cb environment)")
    print("=" * 60)
    print()

    success = True

    # Core dependencies
    print("Core Dependencies:")
    success &= check_import('openpi', 'OpenPI')
    success &= check_import('torch', 'PyTorch')
    success &= check_import('jax', 'JAX')
    success &= check_import('flax', 'Flax')
    success &= check_import('transformers', 'Transformers')
    success &= check_import('lerobot', 'LeRobot')
    success &= check_import('cv2', 'OpenCV')
    success &= check_import('PIL', 'Pillow')
    success &= check_import('numpy', 'NumPy')
    success &= check_import('einops', 'Einops')
    success &= check_import('equinox', 'Equinox')

    print()
    print("=" * 60)

    if success:
        print("[OK] All dependencies installed successfully!")

        # Check GPU support
        import torch
        import jax

        print()
        print("GPU Support:")
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  PyTorch CUDA version: {torch.version.cuda}")
            print(f"  PyTorch GPU count: {torch.cuda.device_count()}")
            print(f"  PyTorch GPU 0: {torch.cuda.get_device_name(0)}")

        print(f"  JAX devices: {jax.devices()}")

        # Check Python version
        print()
        print("Python Environment:")
        print(f"  Python version: {sys.version.split()[0]}")
        print(f"  Python executable: {sys.executable}")

    else:
        print("[X] Some dependencies failed to install")
        print()
        print("Troubleshooting:")
        print("  1. Ensure you're in the cb conda environment: conda activate cb")
        print("  2. Install dependencies: pip install -r vla/openpi_requirements.txt")
        print("  3. Install openpi-client: pip install -e submodules/openpi/packages/openpi-client/")
        print("  4. Install openpi: pip install -e submodules/openpi/")
        sys.exit(1)

    print("=" * 60)
    print()
    print("Next Steps:")
    print("  1. Test π₀ model loading: see submodules/openpi/examples/")
    print("  2. Collect episodes: python vla/vla_collect_episodes.py")
    print("  3. Fine-tune model: python vla/vla_finetune.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
