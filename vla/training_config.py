"""
VLA Training Configuration

Chess-specific training configuration for π₀.₅ LoRA finetuning.
Supports YAML loading and dataclass-based defaults.
"""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List


@dataclass
class ChessTrainingConfig:
    """
    Training configuration for chess VLA finetuning.

    Default values are tuned for LoRA finetuning on ~1000 episodes
    with 22GB VRAM (RTX 3090 / A5000 class GPU).
    """

    # Dataset
    dataset_path: str = "data/episodes"
    val_split: float = 0.1  # 10% validation holdout
    num_workers: int = 4

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Model
    base_model: str = "lerobot/pi05_base"
    freeze_vision_encoder: bool = True  # LoRA-style: freeze vision, train action head
    freeze_language_encoder: bool = True  # Freeze PaliGemma language tower

    # Checkpoints
    checkpoint_dir: str = "checkpoints/chess_pi0"
    save_every_n_epochs: int = 10
    save_best: bool = True  # Save best model by val loss

    # Logging
    wandb_project: str = "chessbot-vla"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 10
    use_wandb: bool = False  # Disabled by default

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True  # Use fp16/bf16 for memory efficiency
    compile_model: bool = False  # torch.compile for speed (requires PyTorch 2.0+)

    # Loss weights
    action_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.1  # Language attention regularization

    # Data augmentation
    augment_color: bool = True  # Random color jitter
    augment_spatial: bool = False  # Disabled for chess (board position matters)
    augment_temporal: bool = False  # Disabled (action timing matters)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ChessTrainingConfig":
        """Load config from YAML file, merging with defaults."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            print(f"[WARN] Config file not found: {yaml_path}, using defaults")
            return cls()

        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}

        return cls(**yaml_config)

    def to_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

        print(f"[OK] Config saved to: {yaml_path}")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check paths
        dataset_path = Path(self.dataset_path)
        if not dataset_path.exists():
            errors.append(f"Dataset path does not exist: {self.dataset_path}")

        # Check numeric ranges
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")

        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")

        if not 0.0 <= self.val_split < 1.0:
            errors.append(f"val_split must be in [0, 1), got {self.val_split}")

        if self.num_epochs < 1:
            errors.append(f"num_epochs must be >= 1, got {self.num_epochs}")

        # Check device
        if self.device not in ["cuda", "cpu"]:
            errors.append(f"device must be 'cuda' or 'cpu', got {self.device}")

        return errors

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Chess VLA Training Configuration")
        print("=" * 60)
        print()

        sections = [
            ("Dataset", ["dataset_path", "val_split", "num_workers"]),
            ("Training", ["batch_size", "gradient_accumulation_steps", "learning_rate",
                         "weight_decay", "num_epochs", "warmup_steps", "max_grad_norm"]),
            ("Model", ["base_model", "freeze_vision_encoder", "freeze_language_encoder"]),
            ("Checkpoints", ["checkpoint_dir", "save_every_n_epochs", "save_best"]),
            ("Hardware", ["device", "mixed_precision", "compile_model"]),
            ("Logging", ["wandb_project", "use_wandb", "log_every_n_steps"]),
        ]

        config_dict = self.to_dict()
        for section_name, keys in sections:
            print(f"{section_name}:")
            for key in keys:
                if key in config_dict:
                    print(f"  {key}: {config_dict[key]}")
            print()


# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "chess_training.yaml"


def get_default_config() -> ChessTrainingConfig:
    """Get default training configuration."""
    return ChessTrainingConfig()


def load_config(config_path: Optional[str] = None) -> ChessTrainingConfig:
    """
    Load training configuration.

    Args:
        config_path: Path to YAML config file. If None, tries default path,
                    then falls back to hardcoded defaults.

    Returns:
        ChessTrainingConfig instance
    """
    if config_path:
        return ChessTrainingConfig.from_yaml(config_path)

    if DEFAULT_CONFIG_PATH.exists():
        return ChessTrainingConfig.from_yaml(str(DEFAULT_CONFIG_PATH))

    return ChessTrainingConfig()


if __name__ == "__main__":
    """Test configuration loading and saving."""
    print("Testing ChessTrainingConfig")
    print()

    # Create default config
    config = ChessTrainingConfig()
    config.print_summary()

    # Validate
    errors = config.validate()
    if errors:
        print("Validation Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("[OK] Configuration is valid")

    # Save example config
    example_path = Path("vla/chess_training_example.yaml")
    config.to_yaml(str(example_path))
    print(f"\nExample config saved to: {example_path}")
