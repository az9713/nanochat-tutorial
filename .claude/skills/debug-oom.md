---
name: debug-oom
description: Diagnose and fix CUDA Out of Memory errors in nanochat training
---

# OOM Error Debugger

When the user reports CUDA out of memory errors, systematically diagnose and fix.

## Step 1: Gather Information

Run these commands to understand the current state:

```bash
# Check GPU memory
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

# Check if any processes are using GPU memory
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Step 2: Calculate Memory Requirements

For nanochat with a given depth, calculate memory usage:

### Model Memory (bf16)
- Parameters ≈ `12 × depth² × aspect_ratio²` (default aspect_ratio=64)
- For depth=12: ~106M params → ~212MB
- For depth=20: ~295M params → ~590MB
- For depth=24: ~425M params → ~850MB

### Optimizer Memory
- AdamW states: 2× model size (momentum + variance in fp32)
- Total optimizer: ~4× model parameters in bytes

### Activation Memory (the big one!)
- Per layer: `batch_size × seq_len × hidden_dim × ~10`
- For depth=24, batch=32, seq=1024: ~12GB activations

### Total Formula
```
Total ≈ model_params × 2 (bf16)
      + model_params × 8 (optimizer)
      + batch × seq × hidden × layers × 10 (activations)
      + ~1GB (CUDA overhead)
```

## Step 3: Recommend Fix

Based on GPU memory and depth, suggest optimal `--device-batch-size`:

| GPU Memory | depth=12 | depth=20 | depth=24 |
|------------|----------|----------|----------|
| 8GB        | 2-4      | 1-2      | 1        |
| 16GB       | 8-16     | 4-8      | 2-4      |
| 24GB       | 16-32    | 8-16     | 4-8      |
| 40GB       | 32-64    | 16-32    | 8-16     |
| 80GB       | 64+      | 32-64    | 16-32    |

## Step 4: Apply Fix

Explain to user:

```bash
# Reduce device batch size, keep total batch size for training dynamics
python -m scripts.base_train --depth=24 \
    --device-batch-size=8 \      # Reduced from default
    --total-batch-size=524288    # Unchanged - uses gradient accumulation

# The training will use gradient accumulation automatically:
# grad_accum_steps = total_batch_size / (device_batch_size × seq_len × num_gpus)
```

## Step 5: Additional Options if Still OOM

1. **Reduce sequence length** (if acceptable for your use case):
   ```bash
   --seq-len=512  # Default is 1024
   ```

2. **Use gradient checkpointing** (trades compute for memory):
   - Not currently in nanochat, but could be added

3. **Reduce depth** for experimentation:
   ```bash
   --depth=12  # Quick iteration mode
   ```

4. **Check for memory leaks**:
   ```bash
   # Monitor memory over time
   watch -n 1 nvidia-smi
   ```
   If memory grows continuously, there may be a tensor accumulation bug.

## Common OOM Patterns

### Pattern 1: OOM at Start
- Model + optimizer don't fit
- Solution: Reduce depth or use smaller GPU

### Pattern 2: OOM After First Batch
- Activations don't fit
- Solution: Reduce device-batch-size

### Pattern 3: OOM Gradually During Training
- Memory leak (tensors not freed)
- Check: Are you storing tensors in a list?
- Check: Are gradients being detached properly?

### Pattern 4: OOM During Evaluation
- Eval uses larger batch than training
- Solution: Reduce eval batch size or use `torch.no_grad()`

## Example Diagnosis

```
User: CUDA out of memory with RTX 3090 (24GB) and depth=24

Analysis:
- depth=24 → ~425M params
- Model bf16: 850MB
- Optimizer: 3.4GB
- Overhead: ~1GB
- Available for activations: ~19GB

With device_batch_size=32, seq=1024:
- Hidden dim = 24 × 64 = 1536
- Activations ≈ 32 × 1024 × 1536 × 24 × 10 ≈ 12GB ✓

Should fit! But if using default batch_size of 64:
- Activations ≈ 24GB ✗

Recommendation: --device-batch-size=16 for headroom
```
