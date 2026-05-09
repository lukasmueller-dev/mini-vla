# mini-vla-maniskill

Minimal visuomotor imitation learning pipeline for ManiSkill.

**Goal:** train a behavior cloning (BC) policy on `PickCube-v1` using RGB
camera observations and robot proprioceptive state, then evaluate it by rolling
the policy out in simulation.

---

## Architecture

```
RGB image (H×W×3)          Robot state (qpos + qvel)
      │                              │
  CNN Encoder                  MLP Encoder
  (3 conv layers)            (2 hidden layers)
      │                              │
  image_embed (256-d)         state_embed (64-d)
      └──────────┬───────────────────┘
                 │  concatenate (320-d)
            Action Head
          (MLP, 2 hidden layers)
                 │
          action vector (7-d)
```

All dimensions are configurable in `configs/pickcube_rgb_state_bc.yaml`.

---

## Setup

```bash
# Create a virtual environment (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate

# Install the package and core dependencies
pip install -e .

# Optional: add Weights & Biases support
pip install -e ".[wandb]"

# Download ManiSkill assets (first run)
python -m mani_skill.utils.download_asset PickCube-v1
```

---

## Commands

### 1. Inspect the environment

```bash
python scripts/inspect_env.py
# custom env / control mode:
python scripts/inspect_env.py --env PickCube-v1 --control-mode pd_ee_delta_pose
```

Prints observation space, action space, camera keys, image shape,
proprioception shape, control mode, and reward mode.

### 2. Collect / validate demos

```bash
# Print collection instructions:
python scripts/collect_or_load_demos.py

# Validate an existing HDF5 demo file:
python scripts/collect_or_load_demos.py --validate --demo-path data/demos/demo.h5
```

To replay ManiSkill's built-in motion-planner demos into the required format:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <MANISKILL_ASSET_PATH>/demos/PickCube-v1/motionplanning/trajectory.h5 \
    --obs-mode rgb+state \
    --target-control-mode pd_ee_delta_pose \
    -o data/demos/PickCube-v1_pd_ee_delta_pose.h5
```

### 3. Train

```bash
python scripts/train_rgb_state_bc.py --config configs/pickcube_rgb_state_bc.yaml

# Override individual hyperparameters:
python scripts/train_rgb_state_bc.py \
    --config configs/pickcube_rgb_state_bc.yaml \
    --override num_epochs=50 learning_rate=1e-3 use_wandb=false
```

Checkpoints are saved to `checkpoints/` every `checkpoint_every` epochs.

### 4. Evaluate

```bash
# Use the latest checkpoint automatically:
python scripts/evaluate_rgb_state_bc.py --config configs/pickcube_rgb_state_bc.yaml

# Specify a checkpoint:
python scripts/evaluate_rgb_state_bc.py \
    --config configs/pickcube_rgb_state_bc.yaml \
    --checkpoint checkpoints/epoch_0100.pt \
    --episodes 20
```

Reports success rate, average reward, and average episode length.
Rollout videos are saved to `videos/` when `save_video: true`.

### 5. WandB

```bash
wandb login   # once
# then set use_wandb: true in the config, or:
python scripts/train_rgb_state_bc.py \
    --config configs/pickcube_rgb_state_bc.yaml \
    --override use_wandb=true wandb_project=my-project
```

---

## Config

All hyperparameters live in `configs/pickcube_rgb_state_bc.yaml`.
Key fields:

| Field | Default | Description |
|---|---|---|
| `env_id` | `PickCube-v1` | ManiSkill environment |
| `control_mode` | `pd_ee_delta_pose` | Robot control mode |
| `image_size` | `[128, 128]` | H×W to resize frames |
| `num_epochs` | `100` | Training epochs |
| `batch_size` | `256` | Mini-batch size |
| `learning_rate` | `3e-4` | AdamW LR |
| `eval_episodes` | `50` | Rollout episodes |
| `use_wandb` | `true` | Enable W&B logging |

---

## Project Structure

```
mini-vla-maniskill/
├── configs/
│   └── pickcube_rgb_state_bc.yaml   # all hyperparameters
├── scripts/
│   ├── inspect_env.py               # diagnostic tool
│   ├── collect_or_load_demos.py     # demo collection guide + HDF5 validator
│   ├── train_rgb_state_bc.py        # training entry point
│   └── evaluate_rgb_state_bc.py    # evaluation entry point
└── src/mini_vla/
    ├── models.py                    # BCPolicy (CNN + MLP + action head)
    ├── datasets.py                  # BCDataset + load_maniskill_demos TODO
    ├── train.py                     # training loop
    ├── eval.py                      # rollout evaluation
    └── utils.py                     # config, logging, checkpointing, wandb
```

---

## TODO – Future Extensions

- [ ] **Language conditioning** – add a language encoder (e.g. CLIP text encoder)
      and condition the action head on a task description embedding.
- [ ] **Diffusion policy** – replace the MSE action head with a diffusion-based
      action decoder for multi-modal action distributions.
- [ ] **Data augmentation** – random crops, color jitter, random erasing on RGB.
- [ ] **Multi-camera fusion** – concatenate embeddings from wrist + base cameras.
- [ ] **Recurrent policy** – add LSTM/GRU over the embedding to handle partial
      observability.
- [ ] **Real-robot transfer** – domain randomisation, sim2real gap analysis.
