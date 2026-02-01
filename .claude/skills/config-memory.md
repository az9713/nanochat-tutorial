---
name: config-memory
description: Calculate optimal batch size and memory configuration for your GPU
---

# Memory Configuration Optimizer

Calculate the optimal training configuration for your GPU memory.

## Step 1: Identify Your GPU

```bash
# Get GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Step 2: Memory Calculation

### Model Memory Formula

For nanochat with depth `d` and aspect_ratio `a` (default 64):

```
hidden_dim = d × a
n_heads = d
n_kv_heads = max(1, d // 4)  # GQA
n_layers = d

# Approximate parameter count
params ≈ vocab × hidden + layers × (4 × hidden² + 3 × hidden × hidden/4)
       ≈ 50257 × (d×64) + d × (4 × (d×64)² + 3 × (d×64) × (d×16))
```

**Quick reference:**

| Depth | Hidden | Params | Model Size (bf16) |
|-------|--------|--------|-------------------|
| 12 | 768 | ~106M | ~212MB |
| 16 | 1024 | ~188M | ~376MB |
| 20 | 1280 | ~295M | ~590MB |
| 24 | 1536 | ~425M | ~850MB |

### Optimizer Memory

AdamW requires:
- Momentum (fp32): model_params × 4 bytes
- Variance (fp32): model_params × 4 bytes
- **Total: ~8× model params in bytes**

### Activation Memory

This is the big one and depends on batch size:

```
Per layer ≈ batch × seq × hidden × 10  (rough estimate with attention)
Total ≈ layers × batch × seq × hidden × 10
```

### Total Memory Requirement

```
Total = Model (bf16) + Optimizer (fp32) + Activations + CUDA overhead (~1GB)
```

## Step 3: Configuration Calculator

Given GPU memory, calculate optimal config:

```python
def calculate_config(gpu_memory_gb, depth=24, seq_len=1024):
    """Calculate optimal batch size for given GPU and model."""

    hidden = depth * 64
    # Rough param estimate
    params = 50257 * hidden + depth * (4 * hidden**2 + 0.75 * hidden**2)

    # Memory components (in GB)
    model_mem = params * 2 / 1e9  # bf16
    optimizer_mem = params * 8 / 1e9  # AdamW fp32 states
    overhead = 1.0  # CUDA overhead

    available_for_activations = gpu_memory_gb - model_mem - optimizer_mem - overhead

    # Activation memory per sample per layer
    # Rough: hidden * seq * 10 bytes per layer
    activation_per_sample = depth * hidden * seq_len * 10 / 1e9

    max_batch = int(available_for_activations / activation_per_sample)

    # Round down to power of 2 for efficiency
    batch_size = 2 ** (max_batch.bit_length() - 1)
    batch_size = max(1, min(batch_size, 64))  # Clamp to reasonable range

    return {
        'gpu_memory': gpu_memory_gb,
        'depth': depth,
        'model_params': f"{params/1e6:.0f}M",
        'model_memory': f"{model_mem:.1f}GB",
        'optimizer_memory': f"{optimizer_mem:.1f}GB",
        'available_for_activations': f"{available_for_activations:.1f}GB",
        'recommended_batch_size': batch_size,
        'safe_batch_size': max(1, batch_size // 2),  # Extra headroom
    }

# Examples
for gpu_mem in [8, 16, 24, 40, 80]:
    config = calculate_config(gpu_mem, depth=24)
    print(f"{gpu_mem}GB GPU: batch={config['recommended_batch_size']} "
          f"(safe: {config['safe_batch_size']})")
```

## Step 4: Recommended Configurations

### Consumer GPUs

| GPU | Memory | depth=12 | depth=20 | depth=24 |
|-----|--------|----------|----------|----------|
| RTX 3060 | 12GB | batch=16 | batch=4 | batch=2 |
| RTX 3080 | 10GB | batch=8 | batch=2 | batch=1 |
| RTX 3090 | 24GB | batch=32 | batch=8 | batch=4 |
| RTX 4070 | 12GB | batch=16 | batch=4 | batch=2 |
| RTX 4080 | 16GB | batch=16 | batch=8 | batch=4 |
| RTX 4090 | 24GB | batch=32 | batch=16 | batch=8 |

### Professional/Cloud GPUs

| GPU | Memory | depth=20 | depth=24 | Notes |
|-----|--------|----------|----------|-------|
| A10 | 24GB | batch=8 | batch=4 | Good for dev |
| A100 40GB | 40GB | batch=16 | batch=8 | |
| A100 80GB | 80GB | batch=32 | batch=16 | |
| H100 | 80GB | batch=32 | batch=16 | Best MFU |

## Step 5: Complete Training Command

```bash
# Example for RTX 4090 (24GB) with depth=24
python -m scripts.base_train \
    --depth=24 \
    --device-batch-size=8 \
    --total-batch-size=524288 \
    --run=my_experiment

# This will use gradient accumulation:
# 524288 / (8 * 1024) = 64 accumulation steps per GPU
```

## Gradient Accumulation Explained

nanochat maintains effective batch size via gradient accumulation:

```
total_batch_size = device_batch_size × seq_len × num_gpus × grad_accum_steps

grad_accum_steps = total_batch_size / (device_batch_size × seq_len × num_gpus)
```

**Example:**
- total_batch_size = 524288 (default)
- device_batch_size = 8
- seq_len = 1024
- num_gpus = 1

grad_accum_steps = 524288 / (8 × 1024 × 1) = 64

This means 64 forward/backward passes before optimizer step.

**Tradeoffs:**
- Higher grad_accum = slower (more passes per step)
- But mathematically identical to larger batch
- Memory stays constant

## Troubleshooting

### Still OOM?

1. **Reduce batch size further**: Try 1, 2, 4...
2. **Reduce sequence length**: `--seq-len=512`
3. **Reduce depth**: `--depth=12` for experimentation
4. **Check for memory leaks**: Monitor with `nvidia-smi -l 1`

### Underutilizing GPU?

If GPU memory utilization is low:
1. Increase batch size
2. Use larger model depth
3. Check if bottlenecked on data loading

### Quick Memory Check During Training

```bash
# Monitor memory continuously
watch -n 1 nvidia-smi

# Or get just memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```
