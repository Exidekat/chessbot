"""
Base Training Configuration

Abstract base class for model-specific training configurations.
"""

import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class BaseTrainingConfig(ABC):
    """
    Base training configuration for all VLA models.

    This class defines the common configuration options shared across all models.
    Model-specific configs (PI0TrainingConfig, SmolVLATrainingConfig) inherit from
    this and override model-specific defaults.

    Subclasses must implement:
        - model_name: str property
        - learning_rate: float property (model-specific default)
        - image_size: Tuple[int, int] property
        - checkpoint_dir: str property
        - base_model: str property (HuggingFace path)
    """

    # Dataset (common)
    dataset_path: str = "data/episodes"
    val_split: float = 0.1  # 10% validation holdout
    num_workers: int = 4

    # Training (common structure, values may be overridden)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 16
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Freezing (common)
    freeze_vision_encoder: bool = True  # LoRA-style: freeze vision
    freeze_language_encoder: bool = True  # Freeze language tower

    # Checkpoints (common structure)
    save_every_n_epochs: int = 10
    save_best: bool = True  # Save best model by val loss

    # Logging (common)
    wandb_project: str = "chessbot-vla"
    wandb_run_name: Optional[str] = None
    log_every_n_steps: int = 100
    use_wandb: bool = False  # Disabled by default

    # Hardware (common)
    device: str = "cuda"
    mixed_precision: bool = True  # Use fp16/bf16 for memory efficiency
    compile_model: bool = False  # torch.compile for speed

    # Loss weights (common)
    action_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.1

    # Data augmentation (common)
    augment_color: bool = True  # Random color jitter
    augment_spatial: bool = False  # Disabled for chess (board position matters)
    augment_temporal: bool = False  # Disabled (action timing matters)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model type identifier (e.g., 'pi0', 'smolvla')."""
        ...

    @property
    @abstractmethod
    def learning_rate(self) -> float:
        """Model-specific default learning rate."""
        ...

    @property
    @abstractmethod
    def image_size(self) -> Tuple[int, int]:
        """Model-specific input image resolution."""
        ...

    @property
    @abstractmethod
    def checkpoint_dir(self) -> str:
        """Model-specific checkpoint directory."""
        ...

    @property
    @abstractmethod
    def base_model(self) -> str:
        """HuggingFace model path for base weights."""
        ...

    @property
    @abstractmethod
    def camera_keys(self) -> dict:
        """Model-specific camera key mapping (global, gripper, unused)."""
        ...

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Model-specific state dimension (6 for SmolVLA, 32 for PI0)."""
        ...

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseTrainingConfig":
        """
        Load config from YAML file, merging with defaults.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Configuration instance with YAML overrides applied
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            print(f"[WARN] Config file not found: {yaml_path}, using defaults")
            return cls()

        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}

        # Filter out abstract properties that can't be set
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in yaml_config.items() if k in valid_fields}

        return cls(**filtered_config)

    def to_yaml(self, yaml_path: str):
        """
        Save config to YAML file.

        Args:
            yaml_path: Path to save YAML config
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Include computed properties in output
        config_dict = self.to_dict()
        config_dict["model_name"] = self.model_name
        config_dict["learning_rate"] = self.learning_rate
        config_dict["image_size"] = list(self.image_size)
        config_dict["checkpoint_dir"] = self.checkpoint_dir
        config_dict["base_model"] = self.base_model

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"[OK] Config saved to: {yaml_path}")

    def to_dict(self) -> dict:
        """Convert dataclass fields to dictionary."""
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
        print(f"VLA Training Configuration ({self.model_name.upper()})")
        print("=" * 60)
        print()

        sections = [
            ("Model", [
                ("model_name", self.model_name),
                ("base_model", self.base_model),
                ("image_size", self.image_size),
                ("checkpoint_dir", self.checkpoint_dir),
            ]),
            ("Dataset", [
                ("dataset_path", self.dataset_path),
                ("val_split", self.val_split),
                ("num_workers", self.num_workers),
            ]),
            ("Training", [
                ("batch_size", self.batch_size),
                ("gradient_accumulation_steps", self.gradient_accumulation_steps),
                ("learning_rate", self.learning_rate),
                ("weight_decay", self.weight_decay),
                ("num_epochs", self.num_epochs),
                ("warmup_steps", self.warmup_steps),
                ("max_grad_norm", self.max_grad_norm),
            ]),
            ("Freezing", [
                ("freeze_vision_encoder", self.freeze_vision_encoder),
                ("freeze_language_encoder", self.freeze_language_encoder),
            ]),
            ("Checkpoints", [
                ("save_every_n_epochs", self.save_every_n_epochs),
                ("save_best", self.save_best),
            ]),
            ("Hardware", [
                ("device", self.device),
                ("mixed_precision", self.mixed_precision),
                ("compile_model", self.compile_model),
            ]),
            ("Logging", [
                ("wandb_project", self.wandb_project),
                ("use_wandb", self.use_wandb),
                ("log_every_n_steps", self.log_every_n_steps),
            ]),
        ]

        for section_name, items in sections:
            print(f"{section_name}:")
            for key, value in items:
                print(f"  {key}: {value}")
            print()
