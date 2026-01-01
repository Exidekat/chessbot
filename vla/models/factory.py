"""
VLA Model Factory

Factory function for loading VLA models with a unified interface.
"""

from typing import Optional, Tuple, Any
from pathlib import Path

from .registry import get_model_class, list_models
from .base import VLAModelBase, Tokenizer


def load_vla_model(
    model_name: str = "pi0",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    for_training: bool = False,
) -> Tuple[VLAModelBase, Tokenizer]:
    """
    Factory function to load any registered VLA model.

    This is the primary entry point for loading VLA models. It handles:
    - Model class lookup from registry
    - Checkpoint loading (fine-tuned or base weights)
    - Device placement
    - Tokenizer initialization

    Args:
        model_name: Model type identifier. Options: "pi0", "smolvla".
                   Default: "pi0" for backward compatibility.
        checkpoint_path: Path to fine-tuned checkpoint. If None or doesn't exist,
                        loads base weights from HuggingFace.
        device: Device to run model on ("cuda" or "cpu"). Default: "cuda".
        for_training: If True, set model to training mode. If False, eval mode.

    Returns:
        Tuple of (model_wrapper, tokenizer):
            - model_wrapper: VLAModelBase instance with unified interface
            - tokenizer: Model-specific tokenizer for language prompts

    Raises:
        ValueError: If model_name is not registered

    Example:
        >>> # Load PI0 for inference
        >>> model, tokenizer = load_vla_model("pi0", device="cuda")
        >>> action = model.predict_action(obs, "Pick up the pawn")

        >>> # Load SmolVLA for training
        >>> model, tokenizer = load_vla_model("smolvla", for_training=True)

        >>> # Load fine-tuned checkpoint
        >>> model, tokenizer = load_vla_model(
        ...     "pi0",
        ...     checkpoint_path="checkpoints/chess_pi0/best.pt"
        ... )

        >>> # List available models
        >>> print(list_models())  # ["pi0", "smolvla"]
    """
    # Get model class from registry
    model_cls = get_model_class(model_name)

    print(f"[INFO] Loading {model_name.upper()} model...")

    # Check if checkpoint exists
    if checkpoint_path:
        checkpoint_path_obj = Path(checkpoint_path)
        if checkpoint_path_obj.exists():
            print(f"[INFO] Using checkpoint: {checkpoint_path}")
        else:
            print(f"[INFO] Checkpoint not found: {checkpoint_path}")
            print(f"[INFO] Loading base weights from HuggingFace")
            checkpoint_path = None

    # Load model using class method
    model = model_cls.from_pretrained(
        checkpoint_path=checkpoint_path,
        device=device,
        for_training=for_training,
    )

    # Return model and tokenizer
    return model, model.tokenizer
