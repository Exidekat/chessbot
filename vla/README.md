# VLA Module

Vision-Language-Action model integration using Physical Intelligence's π₀ (pi-zero) for end-to-end robot control with color-conditioned prompts.

## Overview

The VLA (Vision-Language-Action) approach represents the evolution beyond symbolic guidance:
- **Guidance** (current): Symbolic YOLO detection → rule-based move execution
- **VLA** (integration): π₀ end-to-end learned policy from camera pixels to robot actions

**Status**: OpenPI submodule integrated, VLA scripts in development.

## Quick Start

```bash
# 1. Initialize OpenPI submodule
git submodule update --init --recursive

# 2. Install OpenPI dependencies to ltx conda environment (CONDA-SAFE METHOD)
conda activate ltx
pip install -r vla/openpi_requirements.txt
pip install -e submodules/openpi/packages/openpi-client/
pip install -e submodules/openpi/

# 3. Verify installation
python vla/verify_openpi.py

# 4. Collect training episodes
python vla/vla_collect_episodes.py --output data/episodes/

# 5. Fine-tune π₀ on chess data
python vla/vla_finetune.py --episodes data/episodes/ --output checkpoints/

# 6. Deploy fine-tuned model
python vla/vla_deploy.py --checkpoint checkpoints/chess_pi0.pt
```

**Important:** Do NOT use `uv sync` as recommended by OpenPI documentation - it creates conflicting virtual environments. Always use the conda-safe pip installation method above. See `vla/INSTALL_OPENPI.md` for detailed installation instructions, troubleshooting, and dependency resolution.

## Planned Architecture

### Control Mode Switching
```yaml
# config.yaml
control_mode: 'guidance'  # Current: rule-based with YOLO
# control_mode: 'vla'     # Future: end-to-end learned policy
```

The system will support both modes, allowing:
- Development and testing with guidance (reliable, explainable)
- Gradual migration to VLA (more flexible, learns from data)
- Hybrid approaches (VLA for manipulation, guidance for verification)

## Planned Components

### `vla/model/`

#### `pi0_wrapper.py`
HuggingFace Pi0 model integration:
```python
from vla.model import Pi0Wrapper

model = Pi0Wrapper(
    model_id="physical-intelligence/pi0-preview",
    device="cuda"
)

# Load checkpoint
model.load_pretrained("checkpoints/chess_pi0.pt")

# Predict action from observation
observation = {
    'global_camera': global_frame,      # (H, W, 3)
    'gripper_camera': gripper_frame,    # (H, W, 3)
    'overlay': overlay_frame,           # (H, W, 3) - optional guidance
    'robot_state': robot.get_state(),   # Joint angles, gripper state
    'language': "Pick up white pawn"    # Natural language instruction
}

action = model.predict(observation)
# Returns: {
#     'position': (x, y, z),
#     'orientation': (roll, pitch, yaw),
#     'gripper': 0.0-1.0,
#     'done': bool
# }
```

#### `chess_adapter.py`
Chess-specific adaptations and preprocessing:
```python
from vla.model import ChessAdapter

adapter = ChessAdapter()

# Preprocess cameras for VLA
processed = adapter.preprocess_observation(
    global_frame, gripper_frame, robot_state
)

# Postprocess VLA actions
safe_action = adapter.postprocess_action(
    raw_action, safety_bounds, current_state
)

# Action space constraints
adapter.set_workspace_bounds(x_min, x_max, y_min, y_max, z_min, z_max)
adapter.set_velocity_limits(max_vel, max_accel)
```

#### `inference.py`
VLA inference pipeline:
```python
from vla.model import VLAInference

inference = VLAInference(model, adapter, safety)

# Single-step inference
action = inference.step(observation)

# Multi-step rollout with re-planning
success = inference.execute_move(
    chess_move,
    board_state,
    max_steps=50,
    replan_interval=10
)
```

### `vla/data_collection/`

#### `episode_recorder.py`
Record robot episodes for VLA training:
```python
from vla.data_collection import EpisodeRecorder

recorder = EpisodeRecorder(output_dir="episodes/")

# Start episode
recorder.start_episode(metadata={
    'move': 'e2e4',
    'move_type': 'normal',
    'difficulty': 'easy',
    'board_fen': fen,
    'operator': 'human'  # or 'guidance'
})

# Record timesteps during execution
while not done:
    recorder.record_timestep(
        global_frame=cameras.get_global_frame(),
        gripper_frame=cameras.get_gripper_frame(),
        overlay_frame=cameras.get_overlay_frame(),
        robot_state=robot.get_state(),
        action=action,
        reward=reward  # Success/failure signal
    )

# End episode
recorder.end_episode(success=True)
# Saves to: episodes/episode_001/
#   - metadata.json
#   - global_frames/ (images)
#   - gripper_frames/ (images)
#   - overlay_frames/ (images)
#   - robot_states.npy
#   - actions.npy
#   - rewards.npy
```

#### `trajectory_logger.py`
Log full trajectories with annotations:
```python
from vla.data_collection import TrajectoryLogger

logger = TrajectoryLogger()

# Log with language annotations
logger.log_move(
    episode_data,
    language_annotation="Pick up the white pawn from e2",
    segmentation="pickup",  # pickup, transport, place, return
    quality_score=0.9  # Human or automated quality rating
)
```

#### `dataset_builder.py`
Build VLA datasets from episodes:
```python
from vla.data_collection import DatasetBuilder

builder = DatasetBuilder()

# Convert episodes to training format
dataset = builder.build_from_episodes(
    episode_dirs="episodes/",
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

# Apply augmentations
dataset = builder.augment(
    dataset,
    spatial_aug=True,  # Rotation, flip
    color_aug=True,    # Lighting variations
    temporal_aug=True  # Speed variation
)

# Save in VLA format
builder.save(dataset, "vla_dataset/chess_manipulation_v1")
```

#### `storage.py`
Efficient episode storage and retrieval:
```python
from vla.data_collection import EpisodeStorage

storage = EpisodeStorage("episodes/")

# Query episodes
easy_episodes = storage.query(
    move_type='normal',
    success=True,
    difficulty='easy'
)

# Load episode data
episode = storage.load_episode("episode_001")

# Statistics
stats = storage.get_statistics()
# {
#     'total_episodes': 1000,
#     'success_rate': 0.85,
#     'avg_steps_per_episode': 45,
#     'move_type_distribution': {...}
# }
```

### `vla/training/`

#### `train_vla.py`
VLA model training script:
```bash
python -m vla.training.train_vla \
    --dataset vla_dataset/chess_manipulation_v1 \
    --base_model physical-intelligence/pi0-preview \
    --output_dir checkpoints/chess_pi0 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

#### `data_loader.py`
Episode data loading for training:
```python
from vla.training import VLADataLoader

loader = VLADataLoader(
    dataset_dir="vla_dataset/chess_manipulation_v1",
    batch_size=32,
    shuffle=True,
    augment=True
)

for batch in loader:
    # batch contains:
    # - global_frames: (B, T, H, W, 3)
    # - gripper_frames: (B, T, H, W, 3)
    # - robot_states: (B, T, state_dim)
    # - actions: (B, T, action_dim)
    # - language: (B,) list of strings
    pass
```

#### `config.py`
Training configuration:
```python
from vla.training import VLAConfig

config = VLAConfig(
    base_model="physical-intelligence/pi0-preview",
    image_size=(224, 224),
    sequence_length=16,
    action_dim=7,  # (x, y, z, roll, pitch, yaw, gripper)
    state_dim=14,  # Joint angles + gripper state

    # Training
    learning_rate=1e-4,
    batch_size=32,
    epochs=100,
    gradient_accumulation_steps=4,

    # Augmentation
    spatial_aug_prob=0.5,
    color_aug_prob=0.3,

    # Chess-specific
    chess_mode=True,
    board_size=8,
    piece_types=12
)
```

## Data Collection Strategy

### Phase 1: Teleoperation
Collect episodes via human teleoperation:
```python
from vla.data_collection import TeleoperationRecorder

teleop = TeleoperationRecorder(robot, cameras)

# Human controls robot via joystick/VR
# System records all observations and actions
teleop.start_recording()
# ... human executes chess move ...
teleop.stop_recording()
```

### Phase 2: Guidance Bootstrap
Use symbolic guidance system to generate training data:
```python
from guidance import GuidanceSystem
from vla.data_collection import GuidanceBootstrap

guidance = GuidanceSystem(config)
bootstrap = GuidanceBootstrap(guidance, robot, cameras)

# Generate 1000s of episodes automatically
bootstrap.collect_episodes(
    num_episodes=5000,
    randomize_positions=True,
    add_noise=0.01  # Small execution noise for diversity
)
```

### Phase 3: Self-Play Improvement
VLA improves through self-play:
```python
from vla.training import SelfPlayTrainer

trainer = SelfPlayTrainer(vla_model, robot, cameras)

# VLA executes, learns from successes/failures
trainer.self_improve(
    num_iterations=10,
    episodes_per_iteration=500,
    success_threshold=0.9
)
```

## Comparison: Guidance vs VLA

| Aspect | Guidance (Current) | VLA (Future) |
|--------|-------------------|--------------|
| **Input** | YOLO detections + chess rules | Raw camera pixels + language |
| **Output** | Discrete move commands | Continuous robot actions |
| **Training Data** | 100s of board images | 1000s of robot episodes |
| **Interpretability** | High (every decision traceable) | Low (learned black box) |
| **Flexibility** | Fixed chess rules | Learns from data |
| **Generalization** | Poor (new boards/pieces) | Good (visual adaptation) |
| **Reliability** | High (deterministic) | Variable (learned) |
| **Development Time** | Moderate | High |

## Development Roadmap

### Stage 1: Data Collection (Current)
- [x] Episode recorder skeleton
- [ ] Implement episode storage format
- [ ] Integrate with guidance system
- [ ] Collect 100 teleoperation episodes

### Stage 2: VLA Integration
- [ ] Pi0 model wrapper
- [ ] Chess-specific adapter
- [ ] Inference pipeline
- [ ] Test with pre-trained Pi0

### Stage 3: Training
- [ ] Dataset builder
- [ ] Training pipeline
- [ ] Evaluate on test set
- [ ] Compare to guidance baseline

### Stage 4: Deployment
- [ ] Safety validation
- [ ] Hybrid guidance+VLA mode
- [ ] Production deployment
- [ ] Continuous improvement

## Configuration

Example `config.yaml` for VLA mode:
```yaml
control_mode: 'vla'

vla:
  model_id: "physical-intelligence/pi0-preview"
  checkpoint: "checkpoints/chess_pi0_best.pt"
  device: "cuda"

  inference:
    max_steps: 50
    replan_interval: 10
    safety_checks: true

  observation:
    global_resolution: [224, 224]
    gripper_resolution: [224, 224]
    include_overlay: true  # Include guidance overlay as hint
    state_history: 4  # Number of past states to condition on

  action:
    position_scale: 0.001  # Scale for small precise movements
    velocity_limit: 0.1  # m/s
    gripper_threshold: 0.5  # Open/close threshold
```

## Future Enhancements

- Multi-task learning (chess + other manipulation tasks)
- Language-conditioned execution ("Castle kingside")
- Failure recovery policies
- Sim-to-real transfer with domain randomization
- Active learning (query human on uncertain moves)

## References

- [Pi0: Physical Intelligence Foundation Model](https://www.physicalintelligence.company/)
- [OpenVLA: Open-source Vision-Language-Action Models](https://openvla.github.io/)
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/)
