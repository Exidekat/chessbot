# VLA Module

Vision-Language-Action model integration using Physical Intelligence's pi0.5 for end-to-end robot control with color-conditioned prompts.

## Overview

The VLA (Vision-Language-Action) approach represents the evolution beyond symbolic guidance:
- **Guidance** (current): Symbolic YOLO detection -> rule-based move execution
- **VLA** (future): pi0.5 end-to-end learned policy from camera pixels to robot actions

**Status**: Episode collection, finetuning, and deployment implemented.

## Quick Start

```bash
# 1. Initialize OpenPI submodule
git submodule update --init --recursive

# 2. Install OpenPI dependencies (CONDA-SAFE METHOD)
conda activate ltx
pip install -r vla/openpi_requirements.txt
pip install -e submodules/openpi/packages/openpi-client/
pip install -e submodules/openpi/

# 3. Verify installation
python vla/verify_openpi.py

# 4. Collect training episodes (requires tele_op.py running)
python scripts/collect_vla_episodes.py --output data/episodes/

# 5. Validate collected episodes
python scripts/validate_vla_episodes.py --dataset data/episodes/

# 6. Fine-tune pi0.5 on collected episodes
python vla/vla_finetune.py --dataset data/episodes/

# 7. Evaluate fine-tuned model
python vla/evaluate.py --checkpoint checkpoints/chess_pi0/best.pt

# 8. Deploy fine-tuned model for inference
python vla/vla_deploy.py --checkpoint checkpoints/chess_pi0/best.pt
```

**Important:** Do NOT use `uv sync` as recommended by OpenPI docs - it creates conflicting virtual environments. Use conda-safe pip installation above.

## Implemented Components

### Episode Collection (`scripts/collect_vla_episodes.py`)

Passive recording of tele-op sessions for VLA training:

```bash
# Terminal 1: Run tele-op
python scripts/tele_op.py

# Terminal 2: Collect episodes at 15 FPS
python scripts/collect_vla_episodes.py --output data/episodes/

# With specific cameras
python scripts/collect_vla_episodes.py --output data/episodes/ \
    --global-camera /dev/video4 \
    --gripper-camera /dev/video0
```

**Features:**
- Reads robot joint positions from state_cache.json (passive - no robot control)
- Records synchronized global + gripper camera frames
- Generates VLM prompts from board detection + move decomposition
- Saves to LeRobot dataset format (or raw files if LeRobot unavailable)
- Retry logic for sensor reading failures

**LeRobot Dataset Schema:**
```python
{
    "observation.images.global": (720, 1280, 3),  # BGR, with color-conditioned overlay
    "observation.images.gripper": (224, 224, 3),  # BGR, resized from 640x480
    "observation.state": (6,),                    # joint positions in radians
    "action": (6,),                               # target joint positions
    "task": str                                   # VLM prompt
}
```

**Camera Capture Pipeline:**
- **Board Detection**: Uses 4K MJPEG -> 720p downscale via `capture_4k_downscale()` for consistent quality with training data
- **Real-time Recording**: Uses native 720p MJPEG at 30fps via `LiveCameraCapture` for low latency
- **Gripper Camera**: Captures at 640x480, resized to 224x224 for VLA input

### Episode Validation (`scripts/validate_vla_episodes.py`)

Interactive tool for reviewing and exporting collected episodes:

```bash
# Interactive review
python scripts/validate_vla_episodes.py --dataset data/episodes/

# List episodes
python scripts/validate_vla_episodes.py --dataset data/episodes/ --list

# Playback episode
python scripts/validate_vla_episodes.py --dataset data/episodes/ --play 0

# Export good episodes
python scripts/validate_vla_episodes.py --dataset data/episodes/ --export data/filtered/
```

### Model Loading (`vla/vla_load_model.py`)

Load pi0.5 model and PaliGemma tokenizer:

```python
from vla.vla_load_model import load_pi05_model

model, tokenizer = load_pi05_model()
```

### Deployment (`vla/vla_deploy.py`)

Deploy pi0.5 for inference with board detection:

```bash
# Without robot (visualization only)
python vla/vla_deploy.py --no-robot

# With SO-100 robot
python vla/vla_deploy.py --robot-port /dev/ttyACM0
```

**Pipeline:**
1. Capture from global + gripper cameras
2. Detect board state with YOLO
3. Calculate best move with Stockfish
4. Decompose into stages with VLM prompts
5. Generate color-conditioned overlay
6. Run pi0.5 inference
7. Execute actions on robot (if connected)

### Finetuning (`vla/vla_finetune.py`)

LoRA-style finetuning of pi0.5 on collected chess episodes:

```bash
# Basic finetuning with defaults
python vla/vla_finetune.py --dataset data/episodes/

# With config file
python vla/vla_finetune.py --config vla/chess_training.yaml

# Resume from checkpoint
python vla/vla_finetune.py --resume checkpoints/chess_pi0/epoch_50.pt

# Custom hyperparameters
python vla/vla_finetune.py --dataset data/episodes/ \
    --epochs 200 \
    --batch-size 8 \
    --lr 5e-6
```

**Training approach:**
- LoRA-style: Freeze vision encoder, train action head
- Requires ~22GB VRAM (RTX 3090 / A5000 class)
- Mixed precision (fp16) for memory efficiency
- Cosine annealing with warmup

**Configuration (`vla/chess_training.yaml`):**
```yaml
batch_size: 4
gradient_accumulation_steps: 4  # Effective batch = 16
learning_rate: 1.0e-5
num_epochs: 100
freeze_vision_encoder: true
freeze_language_encoder: true
```

### Evaluation (`vla/evaluate.py`)

Evaluate fine-tuned checkpoints on held-out episodes:

```bash
python vla/evaluate.py --checkpoint checkpoints/chess_pi0/best.pt --dataset data/episodes/
```

**Metrics:**
- Per-joint MSE and MAE
- Accuracy at thresholds (0.05, 0.1, 0.2 radians)
- Comparison to random baseline

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

## Development Status

**Completed:**
- Episode collection with LeRobot format
- Episode validation and export tool
- pi0.5 model loading
- Deployment script with board detection integration
- Color-conditioned VLM prompt generation
- Fine-tuning pipeline (LoRA-style)
- Evaluation utilities

**In Progress:**
- Collect sufficient teleoperation episodes (6 collected, target 100+)

**Planned:**
- Hybrid guidance+VLA mode
- Self-improvement via play

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
