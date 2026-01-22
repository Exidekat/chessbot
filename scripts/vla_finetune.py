#!/usr/bin/env python3
"""
VLA Finetuning Script - Multi-Model Support

Fine-tune PI0 or SmolVLA on collected chess robot episodes using LoRA-style training
(freeze vision encoder, train action head) or full finetuning (all parameters).

Usage:
    # Fine-tune PI0 with LoRA-style training (default - freezes vision/language)
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/

    # Full finetuning (all parameters trainable)
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/ --full

    # Fine-tune SmolVLA
    python scripts/vla_finetune.py --model smolvla --dataset data/lerobot_episodes/

    # With config file
    python scripts/vla_finetune.py --model pi0 --config vla/chess_training.yaml

    # Continue from latest checkpoint
    python scripts/vla_finetune.py --model pi0 --continue

    # Resume from specific checkpoint
    python scripts/vla_finetune.py --model smolvla --resume checkpoints/chess_smolvla/epoch_0050.pt

Requirements:
    - GPU with >22GB VRAM for LoRA finetuning, >40GB for full finetuning
    - Collected episodes in LeRobot format (via collect_vla_episodes.py)
    - LeRobot dependencies installed
"""

# Disable tokenizer parallelism before any imports to avoid fork warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
    from torch.amp import autocast
except ImportError as e:
    print(f"[X] Failed to import PyTorch: {e}")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from vla.models import load_vla_model, list_models
from vla.configs import get_training_config, BaseTrainingConfig
from vla.chess_dataloader import create_dataloaders


def freeze_vision_backbone(model: nn.Module) -> int:
    """
    Freeze vision encoder parameters for LoRA-style training.

    Args:
        model: VLA model

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Freeze parameters with "vision" or "image_encoder" in name
    for name, param in model.named_parameters():
        if any(key in name.lower() for key in ["vision", "image_encoder", "vit", "patch_embed"]):
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count


def freeze_language_backbone(model: nn.Module) -> int:
    """
    Freeze language encoder parameters.

    Args:
        model: VLA model

    Returns:
        Number of frozen parameters
    """
    frozen_count = 0

    # Freeze parameters with "language" or "text" in name
    for name, param in model.named_parameters():
        if any(key in name.lower() for key in ["language", "text_encoder", "embed_tokens"]):
            param.requires_grad = False
            frozen_count += param.numel()

    return frozen_count


def unfreeze_all_params(model: nn.Module) -> int:
    """
    Unfreeze all model parameters for full finetuning.

    Args:
        model: VLA model

    Returns:
        Number of unfrozen parameters
    """
    unfrozen_count = 0

    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += param.numel()

    return unfrozen_count


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
    }


def find_learning_rate(
    model: nn.Module,
    dataloader,
    config,
    tokenizer=None,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_steps: int = 100,
) -> float:
    """
    Find optimal learning rate using the LR range test.

    Gradually increases learning rate and tracks loss. The optimal LR is
    typically where the loss decreases fastest (steepest slope).

    Args:
        model: Model to test
        dataloader: Training dataloader
        config: Training configuration
        tokenizer: Tokenizer for language instructions
        min_lr: Starting learning rate
        max_lr: Maximum learning rate to test
        num_steps: Number of steps to run

    Returns:
        Suggested learning rate
    """
    print("\n" + "=" * 60)
    print("Learning Rate Finder")
    print("=" * 60)

    device = config.device
    camera_keys = config.camera_keys

    # Save model state to restore later
    model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Create optimizer with min_lr
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=min_lr)

    # LR schedule: exponential increase
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)

    model.train()
    losses = []
    lrs = []
    best_loss = float("inf")
    batch_iter = iter(dataloader)

    print(f"Testing LR range: {min_lr:.2e} to {max_lr:.2e}")

    for step in range(num_steps):
        # Get batch (cycle if needed)
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        # Current learning rate
        lr = min_lr * (lr_mult ** step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        optimizer.zero_grad()

        observation_state = batch.get("observation.state")
        target_action = batch.get("action")
        language_instructions = batch.get("language_instruction", ["Move piece"])

        if observation_state is not None:
            observation_state = observation_state.to(device)
        if target_action is not None:
            target_action = target_action.to(device)

        model_batch = {}
        for cam_type in ['global', 'gripper', 'unused']:
            cam_key = camera_keys[cam_type]
            if cam_key in batch:
                model_batch[cam_key] = batch[cam_key].to(device)

        if observation_state is not None:
            model_batch["observation.state"] = observation_state
        if target_action is not None:
            model_batch["action"] = target_action

        if tokenizer is not None and language_instructions:
            lang_list = [str(s) if s else "Move chess piece" for s in language_instructions]
            tokenized = tokenizer(
                lang_list, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            )
            model_batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
            model_batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(device)

        try:
            with autocast("cuda", enabled=config.mixed_precision):
                loss, _ = model.forward(model_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            losses.append(current_loss)
            lrs.append(lr)

            # Stop if loss explodes
            if current_loss > best_loss * 10:
                print(f"  Step {step}: LR={lr:.2e}, Loss={current_loss:.4f} (stopping - loss exploded)")
                break

            if current_loss < best_loss:
                best_loss = current_loss

            if step % 20 == 0:
                print(f"  Step {step}: LR={lr:.2e}, Loss={current_loss:.4f}")

        except Exception as e:
            print(f"  Step {step}: LR={lr:.2e}, Error: {e}")
            break

    # Restore model state
    model.load_state_dict(model_state)

    # Find optimal LR using "1/10th before explosion" heuristic
    if len(losses) < 5:
        print("[WARN] Not enough data points for LR finder")
        return config.learning_rate

    # Find the minimum loss and where loss starts exploding
    min_loss = min(losses)
    min_loss_idx = losses.index(min_loss)

    # Find explosion point: where loss exceeds 2x the minimum
    explosion_idx = len(losses) - 1
    for i in range(min_loss_idx, len(losses)):
        if losses[i] > min_loss * 2:
            explosion_idx = i
            break

    # Optimal LR is typically 1/10th of the explosion LR
    # Or equivalently, go back ~10 steps in log space
    # We'll use the LR at about 1/3 of the way from start to explosion
    # This is more robust than finding steepest descent
    optimal_idx = max(1, int(explosion_idx * 0.6))  # 60% of way to explosion
    suggested_lr = lrs[optimal_idx]

    # Alternative: use LR at minimum loss point, divided by 3 for safety margin
    min_loss_lr = lrs[min_loss_idx]
    safe_lr = min_loss_lr / 3

    # Use the more conservative of the two
    if safe_lr < suggested_lr:
        suggested_lr = safe_lr
        optimal_idx = min_loss_idx

    print(f"\n[LR Finder Results]")
    print(f"  Min loss: {min_loss:.4f} at LR={lrs[min_loss_idx]:.2e} (step {min_loss_idx})")
    print(f"  Explosion: LR={lrs[explosion_idx]:.2e} (step {explosion_idx})")
    print(f"  Suggested: {suggested_lr:.2e}")

    return suggested_lr


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: BaseTrainingConfig,
    path: Path,
    tile_mode: str = "multi_tile",
):
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.to_dict(),
        "model_name": config.model_name,
        "full_finetuning": not (config.freeze_vision_encoder or config.freeze_language_encoder),
        "tile_mode": tile_mode,  # Save for deployment consistency
    }

    torch.save(checkpoint, path)
    print(f"[SAVE] Checkpoint saved: {path}")


def load_checkpoint(
    model: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    device: str = "cuda",
) -> int:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into. If None, skip model loading
               (useful when model was already loaded from checkpoint via factory).
        optimizer: Optimizer to load state into. If None, skip optimizer loading.
        path: Path to checkpoint file.
        device: Device for loading.

    Returns:
        Starting epoch number
    """
    if not path.exists():
        print(f"[WARN] Checkpoint not found: {path}")
        return 0

    # Load to CPU first to avoid OOM
    checkpoint = torch.load(path, map_location="cpu")

    # Load model weights (skip if model is None - already loaded via factory)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[OK] Loaded model weights from: {path}")

    # Check if training mode changed (LoRA <-> full)
    ckpt_full = checkpoint.get("full_finetuning", False)

    # Try to load optimizer state (may fail if param count changed)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[OK] Loaded optimizer state from checkpoint")
        except ValueError as e:
            if "doesn't match the size" in str(e):
                print(f"[WARN] Optimizer state incompatible (training mode changed)")
                print(f"       Checkpoint was {'full' if ckpt_full else 'LoRA-style'} finetuning")
                print(f"       Starting with fresh optimizer state")
            else:
                raise

    # Get starting epoch
    epoch = checkpoint.get("epoch", 0) + 1
    print(f"[OK] Resuming from epoch {epoch}")

    # Clean up
    del checkpoint

    return epoch


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.

    Looks for epoch_XXXX.pt files and returns the one with highest epoch number.
    Falls back to best.pt or final.pt if no epoch checkpoints found.

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None

    # Look for epoch_XXXX.pt files
    epoch_checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    if epoch_checkpoints:
        # Sort by epoch number (extract from filename)
        def get_epoch_num(p: Path) -> int:
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return 0
        epoch_checkpoints.sort(key=get_epoch_num, reverse=True)
        return epoch_checkpoints[0]

    # Fallback to best.pt or final.pt
    for name in ["best.pt", "final.pt"]:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate

    return None


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    config: BaseTrainingConfig,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    device = config.device

    total_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    accumulation_steps = config.gradient_accumulation_steps

    optimizer.zero_grad()

    # Get camera keys from config
    camera_keys = config.camera_keys

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device - using model-specific camera keys
        observation_state = batch.get("observation.state")
        target_action = batch.get("action")
        language_instructions = batch.get("language_instruction", ["Move piece"] * (len(target_action) if target_action is not None else 1))

        if observation_state is not None:
            observation_state = observation_state.to(device)
        if target_action is not None:
            target_action = target_action.to(device)

        # Forward pass with mixed precision
        with autocast("cuda", enabled=config.mixed_precision):
            # Build batch dict for model with dynamic camera keys
            model_batch = {}

            # Move all camera images to device (handles PI0 and SmolVLA keys)
            for cam_type in ['global', 'gripper', 'unused']:
                cam_key = camera_keys[cam_type]
                if cam_key in batch:
                    model_batch[cam_key] = batch[cam_key].to(device)

            if observation_state is not None:
                model_batch["observation.state"] = observation_state
            if target_action is not None:
                model_batch["action"] = target_action

            # Tokenize language instructions
            if tokenizer is not None and language_instructions:
                lang_list = [str(s) if s else "Move chess piece" for s in language_instructions]
                tokenized = tokenizer(
                    lang_list,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                model_batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
                model_batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(device)

            # Model forward pass - both PI0 and SmolVLA return (loss, loss_dict)
            try:
                loss, loss_dict = model.forward(model_batch)
                loss = loss / accumulation_steps

            except Exception as e:
                print(f"[WARN] Model forward failed: {e}")
                print(f"       Batch keys: {list(model_batch.keys())}")
                raise

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()

        # Accumulate metrics
        batch_loss = loss.item() * accumulation_steps
        batch_action_loss = loss_dict.get("loss", loss.item())
        total_loss += batch_loss
        total_action_loss += batch_action_loss
        num_batches += 1

        # Aggressive memory cleanup
        del loss, loss_dict, model_batch
        if device == "cuda":
            torch.cuda.empty_cache()

        # Log progress
        if (batch_idx + 1) % config.log_every_n_steps == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | Loss: {avg_loss:.6f}")

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_action_loss": total_action_loss / max(num_batches, 1),
    }


def validate(
    model: nn.Module,
    dataloader,
    config: BaseTrainingConfig,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Validate model on held-out data.

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    device = config.device

    total_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0

    # Get camera keys from config
    camera_keys = config.camera_keys

    with torch.no_grad():
        for batch in dataloader:
            language_instructions = batch.get("language_instruction", ["Move piece"])

            # Move batch to device - using model-specific camera keys
            observation_state = batch.get("observation.state")
            target_action = batch.get("action")

            if observation_state is not None:
                observation_state = observation_state.to(device)
            if target_action is not None:
                target_action = target_action.to(device)

            # Build batch dict with dynamic camera keys
            model_batch = {}

            # Move all camera images to device (handles PI0 and SmolVLA keys)
            for cam_type in ['global', 'gripper', 'unused']:
                cam_key = camera_keys[cam_type]
                if cam_key in batch:
                    model_batch[cam_key] = batch[cam_key].to(device)

            if observation_state is not None:
                model_batch["observation.state"] = observation_state
            if target_action is not None:
                model_batch["action"] = target_action

            # Tokenize language instructions
            if tokenizer is not None and language_instructions:
                lang_list = [str(s) if s else "Move chess piece" for s in language_instructions]
                tokenized = tokenizer(
                    lang_list,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                )
                model_batch["observation.language.tokens"] = tokenized["input_ids"].to(device)
                model_batch["observation.language.attention_mask"] = tokenized["attention_mask"].bool().to(device)

            try:
                # Use autocast for mixed precision (same as training)
                with autocast("cuda", enabled=config.mixed_precision):
                    loss, loss_dict = model.forward(model_batch)
                total_loss += loss.item()
                total_action_loss += loss_dict.get("loss", loss.item())
                num_batches += 1

            except Exception as e:
                print(f"[WARN] Validation forward failed: {e}")
                continue

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "val_action_loss": total_action_loss / max(num_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VLA on chess robot episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fine-tune PI0 with LoRA-style training (default)
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/

    # Full finetuning (all parameters)
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/ --full

    # With early stopping (stop if no improvement for 10 epochs)
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/ --early-stopping --patience 10

    # Find optimal learning rate before training
    python scripts/vla_finetune.py --dataset data/lerobot_episodes/ --find-lr

    # Fine-tune SmolVLA
    python scripts/vla_finetune.py --model smolvla --dataset data/lerobot_episodes/

    # Continue from latest checkpoint
    python scripts/vla_finetune.py --model pi0 --continue
        """
    )

    parser.add_argument(
        "--model", type=str, default="pi0",
        choices=list_models(),
        help="Model type to fine-tune (default: pi0)"
    )
    parser.add_argument("--dataset", type=str, default="data/lerobot_episodes",
                        help="Path to LeRobot dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for checkpoints (default: checkpoints/chess_{model})")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to specific checkpoint to resume from")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue from latest checkpoint in output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--full", action="store_true",
                        help="Full finetuning (all parameters trainable, no freezing)")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Enable early stopping when validation loss stops improving")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement, default: 10)")
    parser.add_argument("--find-lr", action="store_true",
                        help="Run learning rate finder before training")
    parser.add_argument("--tile-mode", type=str, default="multi_tile",
                        choices=["multi_tile", "letterbox"],
                        help="Global camera tiling mode (default: multi_tile)")

    args = parser.parse_args()

    # Load model-specific configuration
    config = get_training_config(args.model)

    # Load from YAML if specified
    if args.config:
        config = type(config).from_yaml(args.config)

    # Override with command line args
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output:
        # User specified custom output directory
        config._checkpoint_dir = args.output
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config._learning_rate = args.lr
    if args.no_wandb:
        config.use_wandb = False
    if args.full:
        # Full finetuning: disable all freezing
        config.freeze_vision_encoder = False
        config.freeze_language_encoder = False

    # Validate configuration
    errors = config.validate()
    if errors:
        print("[X] Configuration errors:")
        for error in errors:
            print(f"    - {error}")
        sys.exit(1)

    # Print configuration
    config.print_summary()

    # Initialize wandb (optional)
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
            )
        except ImportError:
            print("[WARN] wandb not installed, disabling logging")
            config.use_wandb = False

    # Determine checkpoint path BEFORE loading model
    # This allows us to load directly from checkpoint instead of base weights
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args.continue_training:
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"\n[CONTINUE] Found checkpoint: {resume_path}")
        else:
            print(f"\n[CONTINUE] No checkpoints found in {checkpoint_dir}, starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\n[RESUME] Using checkpoint: {resume_path}")
        else:
            print(f"\n[WARN] Checkpoint not found: {resume_path}, starting fresh")
            resume_path = None

    # Load model
    print("\n" + "=" * 60)
    training_mode = "FULL" if args.full else "LoRA-style"
    print(f"Loading Model ({args.model.upper()}) - {training_mode} finetuning")
    print("=" * 60)

    # Build norm_stats_path from dataset path
    norm_stats_path = Path(config.dataset_path) / "norm_stats.json"
    if not norm_stats_path.exists():
        norm_stats_path = None
        print(f"[WARN] No norm_stats.json found in dataset - model will not have normalizer")
    else:
        norm_stats_path = str(norm_stats_path)

    # Load model using factory - pass checkpoint_path to load directly from checkpoint
    model_wrapper, tokenizer = load_vla_model(
        model_name=args.model,
        checkpoint_path=str(resume_path) if resume_path else None,
        device=config.device,
        for_training=True,
        norm_stats_path=norm_stats_path,
    )

    # Get the underlying policy for training
    model = model_wrapper.policy

    # Handle parameter freezing based on training mode
    if args.full:
        # Full finetuning: unfreeze ALL parameters (some models freeze by default)
        unfrozen = unfreeze_all_params(model)
        if unfrozen > 0:
            print(f"[OK] Unfroze {unfrozen:,} parameters for full finetuning")
        else:
            print("[OK] Full finetuning enabled - all parameters already trainable")
    else:
        # LoRA-style: freeze vision and/or language encoders
        if config.freeze_vision_encoder:
            frozen_vision = freeze_vision_backbone(model)
            print(f"[OK] Froze vision encoder ({frozen_vision:,} parameters)")

        if config.freeze_language_encoder:
            frozen_lang = freeze_language_backbone(model)
            print(f"[OK] Froze language encoder ({frozen_lang:,} parameters)")

    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,} ({param_counts['trainable_pct']:.1f}%)")
    print(f"  Frozen: {param_counts['frozen']:,}")

    # Create dataloaders with model-specific image size
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, val_loader, _ = create_dataloaders(
        dataset_path=config.dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        val_split=config.val_split,
        image_size=config.image_size,  # Model-specific: 224 for PI0, 256 for SmolVLA
        model_camera_keys=config.camera_keys,  # Model-specific camera key mapping
        state_dim=config.state_dim,  # Model-specific: 32 for PI0, 6 for SmolVLA
        global_tile_mode=args.tile_mode,  # multi_tile (default) or letterbox
    )

    # Log dataset format (video vs PNG)
    dataset_format = "video" if train_loader.dataset.uses_video else "PNG images"
    print(f"\nDataset format: {dataset_format}")
    print(f"Image size: {config.image_size}")
    print(f"Tile mode: {args.tile_mode}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Run learning rate finder if requested
    if args.find_lr:
        suggested_lr = find_learning_rate(
            model=model,
            dataloader=train_loader,
            config=config,
            tokenizer=tokenizer,
        )
        print(f"\n[LR Finder] Updating learning rate: {config.learning_rate:.2e} -> {suggested_lr:.2e}")
        config._learning_rate = suggested_lr

    # Create optimizer - 8-bit AdamW for PI0 (saves VRAM), standard for SmolVLA
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.model == "pi0":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            print(f"[OK] Using 8-bit AdamW optimizer for PI0 (lr={config.learning_rate})")
        except ImportError:
            print("[WARN] bitsandbytes not installed, using standard AdamW")
            optimizer = AdamW(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
    else:
        # SmolVLA uses standard AdamW
        optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        print(f"[OK] Using AdamW optimizer for {args.model.upper()} (lr={config.learning_rate})")

    # Warmup + cosine annealing scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 10,
        T_mult=2,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )

    # Mixed precision - both PI0 and SmolVLA now use BFloat16
    # GradScaler is NOT needed for bfloat16, only for float16
    scaler = None
    if config.mixed_precision:
        print("[OK] Using native BFloat16 mixed precision")

    # Load optimizer state from checkpoint (model weights already loaded above)
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(
            model=None,  # Skip model loading - already loaded via load_vla_model
            optimizer=optimizer,
            path=resume_path,
            device=config.device,
        )

    # Compile model (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        print("\n[OK] Compiling model with torch.compile...")
        model = torch.compile(model)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    if args.early_stopping:
        print(f"  Early stopping enabled (patience={args.patience})")
    print("=" * 60)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = start_epoch  # Track the last completed epoch
    val_metrics = None  # Initialize to detect if training loop ran

    # Check if already past target epochs
    if start_epoch >= config.num_epochs:
        print(f"\n[WARN] Already at epoch {start_epoch}, but num_epochs={config.num_epochs}")
        print(f"       Use --epochs {start_epoch + 10} to continue training")
        sys.exit(0)

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        final_epoch = epoch + 1

        print(f"\n--- Epoch {epoch + 1}/{config.num_epochs} ---")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, epoch + 1, scaler, tokenizer
        )

        # Validate
        val_metrics = validate(model, val_loader, config, tokenizer)

        # Update scheduler
        scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n  Train Loss: {train_metrics['train_loss']:.6f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
        print(f"  LR: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")

        # Log to wandb
        if config.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "lr": current_lr,
                **train_metrics,
                **val_metrics,
            })

        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics["val_loss"], config,
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                tile_mode=args.tile_mode,
            )

        # Save best model and track improvement
        if config.save_best and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics["val_loss"], config,
                checkpoint_dir / "best.pt",
                tile_mode=args.tile_mode,
            )
            print(f"  [BEST] New best val loss: {best_val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping:
                print(f"  [EARLY STOP] No improvement for {epochs_without_improvement}/{args.patience} epochs")

        # Early stopping check
        if args.early_stopping and epochs_without_improvement >= args.patience:
            print(f"\n[EARLY STOP] Stopping training - no improvement for {args.patience} epochs")
            break

    # Save final model (only if training actually ran)
    if val_metrics is not None:
        save_checkpoint(
            model, optimizer, final_epoch, val_metrics["val_loss"], config,
            checkpoint_dir / "final.pt",
            tile_mode=args.tile_mode,
        )
    else:
        print("\n[WARN] No training epochs completed - skipping final checkpoint")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel: {args.model.upper()}")
    if best_val_loss < float("inf"):
        print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
