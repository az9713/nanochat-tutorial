---
name: train
description: Interactive training launcher - guides through configuration and starts training
---

# Interactive Training Launcher

Guide the user through nanochat training configuration.

## Step 1: Determine Training Type

Ask the user:
- **Pretraining**: Train base language model from scratch
- **SFT**: Fine-tune for chat/instruction following
- **Quick test**: Fast iteration for experimentation

## Step 2: Check Environment

Run these checks:

```bash
# Check Python environment
python --version
which python

# Check PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check for nanochat
python -c "import nanochat; print('nanochat import OK')"
```

## Step 3: Gather Configuration

### For Pretraining

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--depth` | 12 (dev), 20 (default), 24 (production) | Model size |
| `--device-batch-size` | Based on GPU memory | See /config-memory |
| `--total-batch-size` | 524288 | Effective batch size |
| `--run` | Unique name | For wandb logging |

### For SFT

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--model-tag` | From base training | Which base model |
| `--step` | Latest | Checkpoint step |

### For Quick Testing

Use these settings for fast iteration:
```bash
--depth=12
--run=dummy  # No wandb
--core-metric-every=999999
--sample-every=-1
--save-every=-1
```

## Step 4: Generate Command

### Single GPU Pretraining

```bash
python -m scripts.base_train \
    --depth=DEPTH \
    --device-batch-size=BATCH \
    --total-batch-size=524288 \
    --run=RUN_NAME \
    --model-tag=MODEL_TAG
```

### Multi-GPU Pretraining

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m scripts.base_train -- \
    --depth=DEPTH \
    --device-batch-size=BATCH \
    --total-batch-size=524288 \
    --run=RUN_NAME \
    --model-tag=MODEL_TAG
```

### SFT

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
    -m scripts.chat_sft -- \
    --model-tag=BASE_MODEL_TAG \
    --step=STEP
```

### Quick Test (5 min run)

```bash
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --depth=12 \
    --run="d12_test" \
    --model-tag="d12_test" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

## Step 5: Pre-flight Checklist

Before starting training, verify:

```bash
# 1. Data exists
ls -la ~/.cache/nanochat/fineweb_edu/ | head

# 2. Tokenizer exists
ls -la ~/.cache/nanochat/tokenizer/

# 3. GPU memory is free
nvidia-smi

# 4. Disk space for checkpoints
df -h ~/.cache/nanochat/

# 5. wandb configured (if using)
wandb login --verify
```

## Step 6: Start Training

After confirmation, run the generated command.

Monitor with:
```bash
# Terminal 1: Training
torchrun ...

# Terminal 2: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 3: Wandb dashboard (if enabled)
# Open https://wandb.ai
```

## Common Training Scenarios

### Scenario 1: First Time User
```bash
# 1. Setup environment
uv sync --extra gpu

# 2. Train tokenizer (downloads data too)
python -m scripts.tok_train

# 3. Quick test run
python -m scripts.base_train --depth=12 --run=dummy --device-batch-size=4
```

### Scenario 2: Resume Training
```bash
# Resume from checkpoint
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train -- \
    --model-tag=MY_MODEL \
    --resume
```

### Scenario 3: Full Speedrun (8×H100)
```bash
# Complete pipeline in ~3 hours
bash runs/speedrun.sh
```

### Scenario 4: Consumer GPU (RTX 4090)
```bash
# Single GPU training
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --run=my_run \
    --model-tag=d12_4090
```

## Training Progress Reference

Expected timelines (8×H100):

| Stage | Duration | Output |
|-------|----------|--------|
| Tokenizer | 5 min | ~/.cache/nanochat/tokenizer/ |
| Base train d24 | 2.5 hrs | ~/.cache/nanochat/base_checkpoints/ |
| SFT | 15 min | ~/.cache/nanochat/chatsft_checkpoints/ |
| Eval | 10 min | Metrics to wandb |

## Troubleshooting

If training fails to start:
1. `/gpu-check` - Verify GPU setup
2. `/verify-data` - Check data integrity
3. `/debug-oom` - If OOM errors
4. `/debug-distributed` - If multi-GPU issues
