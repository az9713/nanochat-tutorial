---
name: debug-stuck
description: Diagnose training that hangs or gets stuck - GC pauses, dataloader issues, deadlocks
---

# Training Stuck Debugger

Diagnose and fix training that hangs or becomes unresponsive.

## Step 1: Identify Hang Type

Ask the user:
1. Does training hang at **start** or **during training**?
2. Does it hang at a **specific step** or **random steps**?
3. Is GPU **utilization 0%** or **non-zero**?

## Step 2: Diagnose by Pattern

### Pattern A: Hang at Start (Before First Step)

**Check 1: NCCL Initialization**
```bash
# Run with debug output
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

**Check 2: Data Loading**
```bash
# Test data loading independently
python -c "
from nanochat.dataloader import create_dataloader
dl = create_dataloader(split='train', batch_size=32)
batch = next(iter(dl))
print(f'Batch shape: {batch.shape}')
"
```

**Check 3: Model Initialization**
```bash
# Test model creation
python -c "
from nanochat.gpt import GPT
model = GPT.from_name('d12')
print(f'Model created: {sum(p.numel() for p in model.parameters())} params')
"
```

---

### Pattern B: Hang During Training (GPU 0%)

**Most Likely: Garbage Collection Pause**

This is a known issue in Python - the garbage collector can pause training for seconds to minutes.

**Diagnostic:**
```bash
# Enable GC debugging
python -c "
import gc
gc.set_debug(gc.DEBUG_STATS)
# Run training...
"
```

**Solutions:**
```bash
# Option 1: Increase GC threshold
export PYTHONGC_THRESHOLD=100000
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Option 2: Disable GC during training (advanced)
# Add to training script:
import gc
gc.disable()  # Before training loop
# gc.collect()  # Manually at checkpoints only
```

---

### Pattern C: Hang During Training (GPU > 0%)

**Possible: Deadlock in Collective Operation**

**Diagnostic:**
```bash
# Check which collective is hanging
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL torchrun ...
```

**Solutions:**
- Ensure all ranks execute same collective ops
- Check for conditional code paths that differ by rank
- Add explicit barriers around suspicious code

---

### Pattern D: Hang at Specific Step (Reproducible)

**Possible: Data Issue at That Sample**

**Diagnostic:**
```python
# Calculate which data sample at step N
samples_per_step = batch_size * seq_len
sample_idx = step * samples_per_step

# Find the parquet shard
shard_size = 100000  # Typical
shard_idx = sample_idx // shard_size
offset_in_shard = sample_idx % shard_size

print(f"Check shard {shard_idx}, row {offset_in_shard}")
```

**Solutions:**
- Verify that specific shard's integrity
- Skip problematic samples
- Re-download corrupted data

---

### Pattern E: Random Hangs (Not Reproducible)

**Possible: Hardware Issue**

**Diagnostic:**
```bash
# Check GPU health
nvidia-smi -q | grep -A 5 "ECC Errors"

# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv -l 1

# Check for thermal throttling
nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv -l 1
```

**Possible: Network Fluctuation (Multi-node)**

**Diagnostic:**
```bash
# Check network between nodes
ping -c 100 other_node_ip
# Look for packet loss or high latency spikes
```

---

## Quick Diagnostic Script

Run this to gather diagnostic info:

```bash
#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi

echo "=== GPU Topology ==="
nvidia-smi topo -m

echo "=== GPU Memory ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

echo "=== Running Processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

echo "=== Python Processes ==="
ps aux | grep python

echo "=== NCCL Environment ==="
env | grep NCCL

echo "=== PyTorch Info ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

---

## Common Fixes Summary

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Hang at start, GPU 0% | NCCL init | Check network, firewall |
| Hang mid-train, GPU 0% | GC pause | `PYTHONGC_THRESHOLD=100000` |
| Hang mid-train, GPU 100% | Deadlock | Check collective ops |
| Hang at specific step | Bad data | Verify shard integrity |
| Random hangs | Hardware | Check temps, ECC errors |

---

## Emergency Recovery

If training is stuck and you need to recover:

```bash
# 1. Check if checkpoint was saved
ls -la ~/.cache/nanochat/base_checkpoints/*/

# 2. Kill the stuck process
pkill -9 -f "python.*base_train"

# 3. Resume from checkpoint
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --model-tag=YOUR_TAG --resume
```

Note: nanochat doesn't currently have Ctrl+C checkpoint saving (issue #xxx), so you may lose progress since last checkpoint.
