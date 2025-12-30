#!/usr/bin/env python3
"""
VLA Model Evaluation

Evaluate fine-tuned π₀.₅ checkpoint on held-out chess episodes.

Usage:
    python vla/evaluate.py --checkpoint checkpoints/chess_pi0/best.pt --dataset data/episodes/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

try:
    import torch
except ImportError as e:
    print(f"[X] Failed to import PyTorch: {e}")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from vla.chess_dataloader import ChessEpisodeDataset, collate_fn
from vla.losses import compute_action_accuracy
from vla.vla_load_model import load_pi0_model


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    device: str = "cuda",
    batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on the validation set.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to LeRobot dataset
        device: Device to run evaluation on
        batch_size: Batch size for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("VLA Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    model, tokenizer = load_pi0_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()

    # Load validation data
    print(f"\nLoading validation data from: {dataset_path}")
    val_dataset = ChessEpisodeDataset(
        dataset_path=dataset_path,
        split="val",
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")

    # Evaluate
    print("\nEvaluating...")
    all_predictions = []
    all_targets = []
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move to device
            observation_image = batch.get("observation.image")
            observation_state = batch.get("observation.state")
            target_action = batch.get("action")

            if observation_image is not None:
                observation_image = observation_image.to(device)
            if observation_state is not None:
                observation_state = observation_state.to(device)
            if target_action is not None:
                target_action = target_action.to(device)

            # Build observation
            observation = {}
            if observation_image is not None:
                observation["image"] = observation_image
            if observation_state is not None:
                observation["state"] = observation_state

            # Forward pass
            try:
                model_output = model(observation)

                if hasattr(model_output, "actions"):
                    predicted_actions = model_output.actions
                elif isinstance(model_output, dict) and "actions" in model_output:
                    predicted_actions = model_output["actions"]
                elif isinstance(model_output, torch.Tensor):
                    predicted_actions = model_output
                else:
                    predicted_actions = model.select_action(observation)

            except Exception as e:
                print(f"[WARN] Forward pass failed: {e}")
                continue

            # Collect predictions
            all_predictions.append(predicted_actions.cpu())
            all_targets.append(target_action.cpu())

            # Compute per-batch metrics
            metrics = compute_action_accuracy(predicted_actions, target_action)
            all_metrics.append(metrics)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    # Aggregate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute final metrics
    results = {}

    # Overall MSE
    mse = ((all_predictions - all_targets) ** 2).mean().item()
    results["mse"] = mse
    results["rmse"] = np.sqrt(mse)

    # Per-joint metrics
    for i in range(all_predictions.shape[-1]):
        joint_mse = ((all_predictions[:, i] - all_targets[:, i]) ** 2).mean().item()
        joint_mae = (all_predictions[:, i] - all_targets[:, i]).abs().mean().item()
        results[f"joint_{i}_mse"] = joint_mse
        results[f"joint_{i}_mae"] = joint_mae
        results[f"joint_{i}_rmse"] = np.sqrt(joint_mse)

    # Accuracy at different thresholds
    for threshold in [0.05, 0.1, 0.2]:  # radians
        errors = (all_predictions - all_targets).abs()
        accuracy = (errors < threshold).all(dim=-1).float().mean().item()
        results[f"accuracy@{threshold}rad"] = accuracy

    # Average batch metrics
    if all_metrics:
        for key in all_metrics[0].keys():
            results[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])

    return results


def print_results(results: Dict[str, Any]):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  MSE: {results.get('mse', 0):.6f}")
    print(f"  RMSE: {results.get('rmse', 0):.6f}")

    # Accuracy at thresholds
    print("\nAccuracy at Thresholds:")
    for key, value in results.items():
        if "accuracy@" in key:
            threshold = key.split("@")[1]
            print(f"  {threshold}: {value:.4f} ({value * 100:.1f}%)")

    # Per-joint metrics
    print("\nPer-Joint RMSE (radians):")
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist Pitch", "Wrist Roll", "Gripper"]
    for i in range(6):
        rmse = results.get(f"joint_{i}_rmse", 0)
        mae = results.get(f"joint_{i}_mae", 0)
        print(f"  Joint {i} ({joint_names[i]}): RMSE={rmse:.4f}, MAE={mae:.4f}")


def compare_to_baseline(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Compare results to baseline metrics.

    Baseline: Random model (predicts mean action)
    """
    comparisons = {}

    # Expected random baseline RMSE ~1.0 for normalized actions
    random_rmse = 1.0
    model_rmse = results.get("rmse", float("inf"))

    if model_rmse < random_rmse:
        improvement = (1 - model_rmse / random_rmse) * 100
        comparisons["vs_random"] = f"+{improvement:.1f}% better than random"
    else:
        degradation = (model_rmse / random_rmse - 1) * 100
        comparisons["vs_random"] = f"{degradation:.1f}% worse than random"

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="data/episodes",
                        help="Path to LeRobot dataset")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"[X] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"[X] Dataset not found: {args.dataset}")
        sys.exit(1)

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Print results
    print_results(results)

    # Compare to baseline
    comparisons = compare_to_baseline(results)
    print("\nBaseline Comparison:")
    for key, value in comparisons.items():
        print(f"  {key}: {value}")

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_path}")

    print("\n" + "=" * 60)
    print("[OK] Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
