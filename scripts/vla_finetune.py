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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: BaseTrainingConfig,
    path: Path,
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
    }

    torch.save(checkpoint, path)
    print(f"[SAVE] Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
    device: str = "cuda",
) -> int:
    """
    Load training checkpoint.

    Returns:
        Starting epoch number
    """
    if not path.exists():
        print(f"[WARN] Checkpoint not found: {path}")
        return 0

    # Load to CPU first to avoid OOM (model already on GPU)
    checkpoint = torch.load(path, map_location="cpu")

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded model weights from: {path}")

    # Check if training mode changed (LoRA <-> full)
    ckpt_full = checkpoint.get("full_finetuning", False)
    current_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    ckpt_config = checkpoint.get("config", {})

    # Try to load optimizer state (may fail if param count changed)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[OK] Loaded optimizer state")
        except ValueError as e:
            if "doesn't match the size" in str(e):
                print(f"[WARN] Optimizer state incompatible (training mode changed)")
                print(f"       Checkpoint was {'full' if ckpt_full else 'LoRA-style'} finetuning")
                print(f"       Starting with fresh optimizer state")
            else:
                raise

    # Clean up CPU checkpoint
    epoch = checkpoint.get("epoch", 0) + 1
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
    parser.add_argument("--dataset", type=str, default="data/episodes",
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

    # Load model using factory
    model_wrapper, tokenizer = load_vla_model(
        model_name=args.model,
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
    )

    print(f"\nImage size: {config.image_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

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

    # Mixed precision scaler - only for PI0 (SmolVLA uses BFloat16 which is incompatible with GradScaler)
    # GradScaler is only needed for float16, not bfloat16
    if config.mixed_precision and args.model == "pi0":
        scaler = torch.amp.GradScaler("cuda")
        print("[OK] Using GradScaler for mixed precision (PI0)")
    else:
        scaler = None
        if config.mixed_precision:
            print("[OK] Using native BFloat16 mixed precision (SmolVLA)")

    # Setup checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    resume_path = None

    if args.continue_training:
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"\n[CONTINUE] Found latest checkpoint: {resume_path}")
        else:
            print(f"\n[CONTINUE] No checkpoints found in {checkpoint_dir}, starting fresh")
    elif args.resume:
        resume_path = Path(args.resume)

    if resume_path:
        start_epoch = load_checkpoint(model, optimizer, resume_path, config.device)

    # Compile model (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        print("\n[OK] Compiling model with torch.compile...")
        model = torch.compile(model)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

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
                checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            )

        # Save best model
        if config.save_best and val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics["val_loss"], config,
                checkpoint_dir / "best.pt"
            )
            print(f"  [BEST] New best val loss: {best_val_loss:.6f}")

    # Save final model
    save_checkpoint(
        model, optimizer, config.num_epochs, val_metrics["val_loss"], config,
        checkpoint_dir / "final.pt"
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel: {args.model.upper()}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
