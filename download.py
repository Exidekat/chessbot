"""
Download and setup script for YOLO model parameters.

This script downloads/copies the necessary YOLO model files to the data/ directory.
"""

import shutil
from pathlib import Path
import sys
import urllib.request
import os


def ensure_data_dir():
    """Ensure the data directory exists."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✓ Data directory ready at: {data_dir.absolute()}")
    return data_dir


def copy_corner_model(data_dir: Path) -> bool:
    """
    Copy corner detection model from submodule.

    Args:
        data_dir: Path to data directory

    Returns:
        True if successful, False otherwise
    """
    source = Path("submodules/real-life-chess-vision/best_corners.pt")
    dest = data_dir / "best_cornres.pt"

    if not source.exists():
        print(f"✗ Corner model not found at: {source}")
        print("  Please ensure the submodule is initialized:")
        print("  git submodule update --init --recursive")
        return False

    try:
        shutil.copy2(source, dest)
        file_size = dest.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Corner detection model copied to: {dest}")
        print(f"  Size: {file_size:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Failed to copy corner model: {e}")
        return False


def download_piece_model(data_dir: Path) -> bool:
    """
    Download piece detection model.

    Note: OneDrive links require manual download or special handling.
    This function provides instructions for manual download.

    Args:
        data_dir: Path to data directory

    Returns:
        True if file already exists or download successful, False otherwise
    """
    dest = data_dir / "best_transformed_detection.pt"

    if dest.exists():
        file_size = dest.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Piece detection model already exists: {dest}")
        print(f"  Size: {file_size:.2f} MB")
        return True

    print("\n⚠ Piece detection model needs to be downloaded manually:")
    print("  1. Visit: https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBZ1RLOGU2c0cxclhoSmRrWDY3c1NQRVQ4VWlSeWc%5FZT16dHV1d3k&cid=D75A1BACEEF1CA04&id=D75A1BACEEF1CA04%2168580&parId=D75A1BACEEF1CA04%2168526&o=OneUp")
    print("  2. Download the file 'best_transformed_detection.pt'")
    print(f"  3. Save it to: {dest.absolute()}")
    print("\nAlternatively, if you have the file elsewhere, copy it to the data/ directory.")

    return False


def download_alternative_piece_model(data_dir: Path) -> bool:
    """
    Check if we can use the piece detection model from the submodule repo.

    Args:
        data_dir: Path to data directory

    Returns:
        True if successful, False otherwise
    """
    # Check if there's a model in the submodule
    possible_sources = [
        Path("submodules/real-life-chess-vision/best_transformed_detection.pt"),
        Path("submodules/real-life-chess-vision/best_detection.pt"),
    ]

    dest = data_dir / "best_transformed_detection.pt"

    for source in possible_sources:
        if source.exists():
            try:
                shutil.copy2(source, dest)
                file_size = dest.stat().st_size / (1024 * 1024)  # MB
                print(f"✓ Piece detection model copied from submodule: {dest}")
                print(f"  Size: {file_size:.2f} MB")
                return True
            except Exception as e:
                print(f"✗ Failed to copy piece model: {e}")
                continue

    return False


def verify_models(data_dir: Path) -> bool:
    """
    Verify that all required models are present.

    Args:
        data_dir: Path to data directory

    Returns:
        True if all models present, False otherwise
    """
    required_files = [
        "best_cornres.pt",
        "best_transformed_detection.pt"
    ]

    all_present = True
    print("\n" + "=" * 60)
    print("Model Verification:")
    print("=" * 60)

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {filename}: {file_size:.2f} MB")
        else:
            print(f"✗ {filename}: MISSING")
            all_present = False

    print("=" * 60)
    return all_present


def main():
    """Main download script."""
    print("=" * 60)
    print("Chess Vision Model Download Script")
    print("=" * 60)
    print()

    # Create data directory
    data_dir = ensure_data_dir()
    print()

    # Copy corner detection model
    print("Step 1: Corner Detection Model")
    print("-" * 60)
    corner_success = copy_corner_model(data_dir)
    print()

    # Download/copy piece detection model
    print("Step 2: Piece Detection Model")
    print("-" * 60)
    piece_success = download_alternative_piece_model(data_dir)

    if not piece_success:
        piece_success = download_piece_model(data_dir)
    print()

    # Verify all models
    all_ready = verify_models(data_dir)

    if all_ready:
        print("\n✓ All models ready! You can now run main.py")
        return 0
    else:
        print("\n✗ Some models are missing. Please follow the instructions above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
