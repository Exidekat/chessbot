"""
Image Preprocessing Utilities

Preprocessing functions for consistent image handling across training and inference.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


def preprocess_for_corner_detection(
    image: Union[np.ndarray, Image.Image, str, Path],
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8
) -> np.ndarray:
    """
    Preprocess image for corner detection: grayscale + contrast normalization.

    This preprocessing improves corner detection in dimly lit conditions by:
    1. Converting to grayscale (removes color information, reduces noise)
    2. Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
       to normalize contrast and enhance edge visibility

    IMPORTANT: This preprocessing MUST be applied identically during:
    - Training data preparation
    - Labeling (so labels match what model sees)
    - Inference (deployment)

    Args:
        image: Input image (numpy array BGR, PIL Image, or path to file)
        clahe_clip_limit: CLAHE contrast limit (higher = more contrast, default 3.0)
        clahe_tile_size: CLAHE tile grid size (smaller = more local contrast, default 8)

    Returns:
        Preprocessed image as numpy array (BGR format, 3-channel grayscale)
        The output is 3-channel so it's compatible with YOLO which expects 3 channels.
    """
    # Load image if path provided
    if isinstance(image, (str, Path)):
        img_bgr = cv2.imread(str(image))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image}")
    elif isinstance(image, Image.Image):
        # Convert PIL to BGR numpy
        img_rgb = np.array(image)
        if len(img_rgb.shape) == 2:
            # Already grayscale
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()

    # Step 1: Convert to grayscale
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # Step 2: Apply CLAHE for contrast normalization
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) works well for:
    # - Handling uneven lighting across the board
    # - Enhancing edges in dim conditions
    # - Avoiding over-amplification of noise (unlike regular histogram equalization)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size)
    )
    normalized = clahe.apply(gray)

    # Step 3: Convert back to 3-channel (BGR) for YOLO compatibility
    # YOLO expects 3-channel input, so we replicate the grayscale to all channels
    result = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

    return result


def preprocess_for_piece_detection(
    image: Union[np.ndarray, Image.Image, str, Path],
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8
) -> np.ndarray:
    """
    Preprocess image for piece detection: grayscale + contrast normalization.

    Uses the same grayscale + CLAHE pipeline as corner detection.

    IMPORTANT: This preprocessing MUST be applied identically during:
    - Training data preparation (label_pieces.py)
    - Labeling (so labels match what model sees)
    - Inference (board_detector.py piece detection)

    Args:
        image: Input image (numpy array BGR, PIL Image, or path to file)
               This should be the perspective-transformed board image.
        clahe_clip_limit: CLAHE contrast limit (higher = more contrast, default 3.0)
        clahe_tile_size: CLAHE tile grid size (smaller = more local contrast, default 8)

    Returns:
        Preprocessed image as numpy array (BGR format, 3-channel grayscale)
    """
    return preprocess_for_corner_detection(image, clahe_clip_limit, clahe_tile_size)


def preprocess_dataset_for_pieces(
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8
) -> int:
    """
    Preprocess all images in a directory for piece detection training.

    Args:
        images_dir: Directory containing original transformed board images
        output_dir: Directory to save preprocessed images
        clahe_clip_limit: CLAHE contrast limit
        clahe_tile_size: CLAHE tile grid size

    Returns:
        Number of images preprocessed
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    count = 0
    for img_path in sorted(image_files):
        try:
            # Preprocess
            preprocessed = preprocess_for_piece_detection(
                img_path,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_size=clahe_tile_size
            )

            # Save with same filename
            output_path = output_dir / img_path.name
            save_preprocessed_image(preprocessed, output_path)
            count += 1

        except Exception as e:
            print(f"[WARNING] Failed to preprocess {img_path.name}: {e}")

    return count


def save_preprocessed_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> Path:
    """
    Save preprocessed image to file.

    Args:
        image: Preprocessed image (BGR numpy array)
        output_path: Path to save the image
        quality: JPEG quality if saving as .jpg (default 95)

    Returns:
        Path to saved image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), image)

    return output_path


def preprocess_dataset_for_corners(
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    clahe_clip_limit: float = 3.0,
    clahe_tile_size: int = 8
) -> int:
    """
    Preprocess all images in a directory for corner detection training.

    Args:
        images_dir: Directory containing original images
        output_dir: Directory to save preprocessed images
        clahe_clip_limit: CLAHE contrast limit
        clahe_tile_size: CLAHE tile grid size

    Returns:
        Number of images preprocessed
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

    count = 0
    for img_path in sorted(image_files):
        try:
            # Preprocess
            preprocessed = preprocess_for_corner_detection(
                img_path,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_size=clahe_tile_size
            )

            # Save with same filename
            output_path = output_dir / img_path.name
            save_preprocessed_image(preprocessed, output_path)
            count += 1

        except Exception as e:
            print(f"[WARNING] Failed to preprocess {img_path.name}: {e}")

    return count
