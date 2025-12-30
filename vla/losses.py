"""
VLA Loss Functions

Loss functions for π₀.₅ finetuning on chess robot tasks.
Primary loss: Action MSE (behavioral cloning)
Auxiliary loss: Language attention regularization
"""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLALoss(nn.Module):
    """
    Combined loss for VLA training.

    Primary component: Action MSE loss (behavioral cloning)
    Auxiliary component: Language conditioning loss (ensures model uses prompts)
    """

    def __init__(
        self,
        action_weight: float = 1.0,
        auxiliary_weight: float = 0.1,
        use_smooth_l1: bool = False,
        action_dim: int = 6,
    ):
        """
        Initialize VLA loss.

        Args:
            action_weight: Weight for action prediction loss
            auxiliary_weight: Weight for auxiliary losses
            use_smooth_l1: Use smooth L1 instead of MSE (more robust to outliers)
            action_dim: Number of action dimensions (6 for SO-100)
        """
        super().__init__()
        self.action_weight = action_weight
        self.auxiliary_weight = auxiliary_weight
        self.use_smooth_l1 = use_smooth_l1
        self.action_dim = action_dim

    def forward(
        self,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined VLA loss.

        Args:
            predicted_actions: Model predicted actions (B, action_dim) or (B, T, action_dim)
            target_actions: Ground truth actions (B, action_dim) or (B, T, action_dim)
            attention_weights: Optional attention weights for language conditioning
            language_mask: Optional mask for language tokens

        Returns:
            Dictionary with:
                - loss: Total combined loss
                - action_loss: Action prediction loss
                - auxiliary_loss: Auxiliary regularization loss
        """
        # Flatten if temporal dimension present
        if predicted_actions.dim() == 3:
            B, T, D = predicted_actions.shape
            predicted_actions = predicted_actions.reshape(-1, D)
            target_actions = target_actions.reshape(-1, D)

        # Action loss (primary)
        if self.use_smooth_l1:
            action_loss = F.smooth_l1_loss(predicted_actions, target_actions)
        else:
            action_loss = F.mse_loss(predicted_actions, target_actions)

        # Per-joint losses for logging
        per_joint_losses = {}
        for i in range(min(self.action_dim, predicted_actions.shape[-1])):
            joint_loss = F.mse_loss(predicted_actions[:, i], target_actions[:, i])
            per_joint_losses[f"joint_{i}_loss"] = joint_loss.detach()

        # Auxiliary loss (language attention regularization)
        auxiliary_loss = torch.tensor(0.0, device=predicted_actions.device)
        if attention_weights is not None and language_mask is not None:
            # Encourage model to attend to language tokens
            # Compute average attention to language vs non-language
            lang_attention = (attention_weights * language_mask.unsqueeze(-1)).sum() / (language_mask.sum() + 1e-8)
            # Maximize language attention (minimize negative)
            auxiliary_loss = -lang_attention * 0.1

        # Combined loss
        total_loss = self.action_weight * action_loss + self.auxiliary_weight * auxiliary_loss

        return {
            "loss": total_loss,
            "action_loss": action_loss.detach(),
            "auxiliary_loss": auxiliary_loss.detach(),
            **per_joint_losses,
        }


def compute_action_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute masked MSE loss for actions.

    Args:
        predicted: Predicted actions (B, D) or (B, T, D)
        target: Target actions (B, D) or (B, T, D)
        mask: Optional mask (B,) or (B, T) for valid samples

    Returns:
        Scalar MSE loss
    """
    mse = (predicted - target) ** 2

    if mask is not None:
        if mask.dim() < mse.dim():
            mask = mask.unsqueeze(-1)  # (B,) -> (B, 1) or (B, T) -> (B, T, 1)
        mse = mse * mask
        return mse.sum() / (mask.sum() * mse.shape[-1] + 1e-8)

    return mse.mean()


def compute_action_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,  # ~5.7 degrees in radians
) -> Dict[str, float]:
    """
    Compute action prediction accuracy metrics.

    Args:
        predicted: Predicted actions (B, D)
        target: Target actions (B, D)
        threshold: Error threshold for "correct" prediction (radians)

    Returns:
        Dictionary with accuracy metrics per joint and overall
    """
    with torch.no_grad():
        errors = torch.abs(predicted - target)

        # Per-joint accuracy
        metrics = {}
        for i in range(errors.shape[-1]):
            joint_errors = errors[:, i]
            accuracy = (joint_errors < threshold).float().mean().item()
            metrics[f"joint_{i}_acc"] = accuracy
            metrics[f"joint_{i}_mae"] = joint_errors.mean().item()

        # Overall accuracy (all joints within threshold)
        all_within = (errors < threshold).all(dim=-1).float().mean().item()
        metrics["overall_accuracy"] = all_within
        metrics["overall_mae"] = errors.mean().item()

        return metrics


class ActionChunkLoss(nn.Module):
    """
    Loss for action chunk prediction (π₀ style).

    Predicts N future actions at once, with temporal weighting
    (closer actions weighted more heavily).
    """

    def __init__(
        self,
        chunk_size: int = 16,
        temporal_decay: float = 0.9,
    ):
        """
        Initialize action chunk loss.

        Args:
            chunk_size: Number of future actions to predict
            temporal_decay: Decay factor for temporal weighting
        """
        super().__init__()
        self.chunk_size = chunk_size

        # Create temporal weights (exponential decay)
        weights = torch.tensor([temporal_decay ** i for i in range(chunk_size)])
        weights = weights / weights.sum()  # Normalize
        self.register_buffer("temporal_weights", weights)

    def forward(
        self,
        predicted_chunk: torch.Tensor,  # (B, chunk_size, action_dim)
        target_chunk: torch.Tensor,      # (B, chunk_size, action_dim)
    ) -> torch.Tensor:
        """Compute temporally-weighted action chunk loss."""
        # Per-timestep MSE
        mse = (predicted_chunk - target_chunk) ** 2  # (B, T, D)
        mse = mse.mean(dim=-1)  # (B, T)

        # Apply temporal weights
        weighted_mse = mse * self.temporal_weights.unsqueeze(0)  # (B, T)

        return weighted_mse.sum(dim=-1).mean()


def create_loss_fn(config) -> VLALoss:
    """
    Create loss function from training config.

    Args:
        config: ChessTrainingConfig instance

    Returns:
        VLALoss instance
    """
    return VLALoss(
        action_weight=config.action_loss_weight,
        auxiliary_weight=config.auxiliary_loss_weight,
        use_smooth_l1=False,
        action_dim=6,  # SO-100 has 6 joints
    )


if __name__ == "__main__":
    """Test loss functions."""
    print("Testing VLA Loss Functions")
    print("=" * 60)

    # Create dummy data
    batch_size = 4
    action_dim = 6

    predicted = torch.randn(batch_size, action_dim)
    target = torch.randn(batch_size, action_dim)

    # Test VLALoss
    loss_fn = VLALoss()
    losses = loss_fn(predicted, target)

    print("\nVLALoss outputs:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")

    # Test action accuracy
    print("\nAction accuracy metrics:")
    metrics = compute_action_accuracy(predicted, target)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test chunk loss
    print("\nActionChunkLoss:")
    chunk_size = 8
    predicted_chunk = torch.randn(batch_size, chunk_size, action_dim)
    target_chunk = torch.randn(batch_size, chunk_size, action_dim)

    chunk_loss_fn = ActionChunkLoss(chunk_size=chunk_size)
    chunk_loss = chunk_loss_fn(predicted_chunk, target_chunk)
    print(f"  chunk_loss: {chunk_loss.item():.6f}")

    print("\n[OK] Loss function tests passed!")
