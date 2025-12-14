#!/usr/bin/env python3
"""
VLA Model Loading Utilities

Shared utilities for loading π₀.₅ (Pi Zero Point Five) models across different VLA scripts.
This module provides a standardized way to load the model for:
- Deployment/inference (vla_deploy.py)
- Fine-tuning (vla_finetune.py)
- Episode collection (vla_collect_episodes.py)

The π₀.₅ model is from Physical Intelligence's LeRobot implementation.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    from lerobot.policies.pi05 import PI05Policy as PI0Policy
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"[X] Failed to import required packages: {e}")
    print("    Run: conda activate ltx && pip install lerobot transformers")
    sys.exit(1)


def load_pi0_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[PI0Policy, AutoTokenizer]:
    """
    Load π₀.₅ VLA model and PaliGemma tokenizer.

    This function handles:
    - Loading base weights from HuggingFace (lerobot/pi05_base)
    - Loading fine-tuned checkpoints from local paths
    - Device placement (CUDA/CPU)
    - Tokenizer initialization for language conditioning

    Args:
        checkpoint_path: Path to fine-tuned checkpoint. If None or doesn't exist,
                        loads base π₀.₅ weights from HuggingFace.
        device: Device to run model on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        Tuple of (policy, tokenizer):
            - policy: PI05Policy instance ready for inference
            - tokenizer: PaliGemma tokenizer for language prompts

    Note on State Dict Warnings:
        You may see warnings like "Vision embedding key might need handling" or
        "Could not remap state_dict keys". These are INFORMATIONAL and expected when:
        - Loading base weights (not fine-tuned)
        - Cross-embodiment transfer (π₀.₅ trained on diverse robots)
        - Model remapping for SO-100 compatibility

        These warnings do NOT prevent the model from running. Monitor for actual
        errors (KeyError, RuntimeError) during inference if issues arise.

    Example:
        >>> policy, tokenizer = load_pi0_model(device="cuda")
        >>> # Use for inference
        >>> observation = {...}
        >>> action = policy.select_action(observation)

        >>> # Or load fine-tuned checkpoint
        >>> policy, tokenizer = load_pi0_model("checkpoints/chess_pi0.pt")
    """
    # Determine checkpoint to load
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        pretrained_path = checkpoint_path
        is_finetuned = True
    else:
        print(f"Loading base π₀.₅ weights from HuggingFace (lerobot/pi05_base)")
        pretrained_path = "lerobot/pi05_base"
        is_finetuned = False

    # Load π₀.₅ model using LeRobot canonical pattern
    print(f"Loading model from: {pretrained_path}")
    print(f"Target device: {device}")

    policy = PI0Policy.from_pretrained(pretrained_path)

    # Move to device and set to evaluation mode
    policy = policy.to(device)
    policy.eval()

    print(f"[OK] Model loaded and moved to {device}")

    # Load tokenizer for language prompts
    # PaliGemma (π₀.₅'s vision-language backbone) uses Gemma tokenizer
    print("Loading PaliGemma tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    print("[OK] Tokenizer loaded")

    return policy, tokenizer


def get_model_info(policy: PI0Policy) -> dict:
    """
    Get information about the loaded model.

    Args:
        policy: Loaded PI05Policy instance

    Returns:
        Dictionary with model metadata:
            - n_obs_steps: Number of observation history steps
            - n_action_steps: Number of action prediction steps (chunk size)
            - chunk_size: Action chunk size
            - max_action_dim: Maximum action dimensionality
            - max_state_dim: Maximum state dimensionality
            - num_inference_steps: Diffusion sampling steps
            - device: Current device ("cuda" or "cpu")
    """
    config = policy.config

    return {
        "n_obs_steps": config.n_obs_steps,
        "n_action_steps": config.n_action_steps,
        "chunk_size": config.chunk_size,
        "max_action_dim": config.max_action_dim,
        "max_state_dim": config.max_state_dim,
        "num_inference_steps": config.num_inference_steps,
        "device": next(policy.parameters()).device.type,
        "dtype": str(next(policy.parameters()).dtype),
        "total_params": sum(p.numel() for p in policy.parameters()),
        "trainable_params": sum(p.numel() for p in policy.parameters() if p.requires_grad)
    }


if __name__ == "__main__":
    """Quick test of model loading."""
    print("=" * 60)
    print("Testing π₀.₅ Model Loading")
    print("=" * 60)
    print()

    # Test loading base model
    try:
        policy, tokenizer = load_pi0_model(device="cuda" if torch.cuda.is_available() else "cpu")

        print()
        print("=" * 60)
        print("Model Information")
        print("=" * 60)

        info = get_model_info(policy)
        for key, value in info.items():
            print(f"  {key}: {value}")

        print()
        print("=" * 60)
        print("[OK] Model loading test successful!")
        print("=" * 60)

    except Exception as e:
        print(f"[X] Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
