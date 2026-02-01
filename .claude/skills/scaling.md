---
name: scaling
description: Model scaling calculator - compute tokens, depth, and training budget tradeoffs
---

# Scaling Calculator

Calculate optimal model size, training tokens, and compute budget for nanochat.

## Quick Reference: Model Sizes

| Depth | Params | Hidden | Heads | KV Heads | Equivalent |
|-------|--------|--------|-------|----------|------------|
| 12 | ~106M | 768 | 12 | 3 | GPT-2 Small |
| 16 | ~188M | 1024 | 16 | 4 | - |
| 20 | ~295M | 1280 | 20 | 5 | GPT-2 Medium |
| 24 | ~425M | 1536 | 24 | 6 | GPT-2 Large |
| 32 | ~756M | 2048 | 32 | 8 | - |
| 48 | ~1.7B | 3072 | 48 | 12 | GPT-2 XL+ |

## Scaling Laws

### Chinchilla Optimal (Tokens = 20 × Params)

For compute-optimal training:
```
tokens_optimal = 20 × model_params
```

| Depth | Params | Optimal Tokens |
|-------|--------|----------------|
| 12 | 106M | 2.1B |
| 20 | 295M | 5.9B |
| 24 | 425M | 8.5B |

### nanochat Default (Tokens = 10.5 × Params)

nanochat uses a ratio of 10.5 for faster training:
```python
# Default in nanochat
target_param_data_ratio = 10.5
tokens = model_params × target_param_data_ratio
```

| Depth | Params | Default Tokens | Training Time (8×H100) |
|-------|--------|----------------|------------------------|
| 12 | 106M | 1.1B | ~20 min |
| 20 | 295M | 3.1B | ~1.5 hr |
| 24 | 425M | 4.5B | ~2.5 hr |

## Compute Budget Calculator

### FLOPS per Token

For transformer forward pass:
```
flops_forward ≈ 2 × params × seq_len
flops_backward ≈ 4 × params × seq_len
flops_total ≈ 6 × params × seq_len
```

### Total FLOPS for Training

```
total_flops = 6 × params × tokens
```

| Depth | Params | Tokens | FLOPS |
|-------|--------|--------|-------|
| 12 | 106M | 1.1B | 7.0e17 |
| 20 | 295M | 3.1B | 5.5e18 |
| 24 | 425M | 4.5B | 1.1e19 |

### GPU Hours

| GPU | TFLOPS (bf16) | FLOPS/hour |
|-----|---------------|------------|
| RTX 4090 | 165 | 5.9e17 |
| A100 | 312 | 1.1e18 |
| H100 | 989 | 3.6e18 |

At ~40% MFU (Model FLOPS Utilization):
```
gpu_hours = total_flops / (tflops × 1e12 × 3600 × 0.4)
```

| Depth | 1×RTX 4090 | 1×A100 | 8×H100 |
|-------|------------|--------|--------|
| 12 | 3 hrs | 1.6 hrs | 0.3 hrs |
| 20 | 24 hrs | 12 hrs | 1.5 hrs |
| 24 | 48 hrs | 24 hrs | 3 hrs |

## Interactive Calculator

```python
def calculate_scaling(
    depth: int = 24,
    aspect_ratio: int = 64,
    param_data_ratio: float = 10.5,
    gpu: str = "H100",
    num_gpus: int = 8,
    mfu: float = 0.4,
):
    """Calculate training requirements."""

    # Model size
    hidden = depth * aspect_ratio
    # Approximate parameter count
    vocab_size = 50257
    params = vocab_size * hidden + depth * (4 * hidden**2 + 0.75 * hidden**2)

    # Tokens
    tokens = int(params * param_data_ratio)

    # FLOPS
    total_flops = 6 * params * tokens

    # GPU specs (TFLOPS bf16)
    gpu_tflops = {
        "RTX_4090": 165,
        "A10": 125,
        "A100": 312,
        "H100": 989,
    }

    tflops = gpu_tflops.get(gpu, 312)
    effective_tflops = tflops * mfu * num_gpus

    # Time
    seconds = total_flops / (effective_tflops * 1e12)
    hours = seconds / 3600

    # Cost estimate
    gpu_cost_per_hour = {
        "RTX_4090": 0.50,
        "A10": 0.75,
        "A100": 1.50,
        "H100": 2.00,
    }
    cost = hours * gpu_cost_per_hour.get(gpu, 1.50) * num_gpus

    return {
        "depth": depth,
        "hidden_dim": hidden,
        "params": f"{params/1e6:.0f}M",
        "tokens": f"{tokens/1e9:.1f}B",
        "total_flops": f"{total_flops:.2e}",
        "gpu": f"{num_gpus}×{gpu}",
        "hours": f"{hours:.1f}",
        "estimated_cost": f"${cost:.0f}",
    }

# Examples
for depth in [12, 20, 24]:
    result = calculate_scaling(depth=depth)
    print(f"depth={depth}: {result['params']} params, "
          f"{result['tokens']} tokens, "
          f"{result['hours']} hours, "
          f"{result['estimated_cost']}")
```

## Tradeoff Analysis

### More Tokens vs Larger Model

Given fixed compute budget:

| Strategy | Pros | Cons |
|----------|------|------|
| Larger model, fewer tokens | Better final loss | Undertrained |
| Smaller model, more tokens | Fully trained | Lower capacity |
| Chinchilla optimal | Balanced | Standard |

### nanochat's Choice

nanochat uses 10.5× ratio (below Chinchilla's 20×):
- Faster training (saves money)
- Slightly undertrained but still good
- Good for learning/experimentation

For production, consider:
```bash
# Use Chinchilla optimal
python -m scripts.base_train --target-param-data-ratio=20
```

## Scaling Experiments

### Quick Comparison Run

Test different depths with same compute:

```bash
# Each run uses ~same FLOPS
# depth=12, more tokens
python -m scripts.base_train --depth=12 --target-flops=1e18 --run=d12_scale

# depth=16, balanced
python -m scripts.base_train --depth=16 --target-flops=1e18 --run=d16_scale

# depth=20, fewer tokens
python -m scripts.base_train --depth=20 --target-flops=1e18 --run=d20_scale
```

Compare CORE scores to find optimal depth for your compute.

## MFU (Model FLOPS Utilization)

MFU measures how efficiently you're using GPU compute:

```
MFU = actual_FLOPS / peak_theoretical_FLOPS
```

Expected MFU:
- Single GPU: 40-50%
- Multi-GPU (8×): 35-45%
- Multi-node: 25-35%

Factors affecting MFU:
- Batch size (larger = better MFU)
- Sequence length
- Memory bandwidth
- Communication overhead

### Checking MFU

Look for MFU in training logs or calculate:
```python
# From training step time
tokens_per_step = batch_size * seq_len
flops_per_step = 6 * params * tokens_per_step
flops_per_second = flops_per_step / step_time
mfu = flops_per_second / (peak_tflops * 1e12)
```

## Decision Tree

```
Q: How much compute do you have?
│
├── < $10 (experimentation)
│   └── depth=12, default settings
│       ~1-2 hours on consumer GPU
│
├── $10-50 (learning)
│   └── depth=20, default settings
│       ~6-12 hours on A100
│
├── $50-100 (serious training)
│   └── depth=24, default settings
│       ~3 hours on 8×H100
│
└── $100+ (production)
    └── depth=24, Chinchilla ratio=20
        ~6 hours on 8×H100
```

## Command Line Examples

```bash
# Specify by FLOPS budget
python -m scripts.base_train --depth=24 --target-flops=1e19

# Specify by token count
python -m scripts.base_train --depth=24 --num-iterations=30000

# Specify by data ratio
python -m scripts.base_train --depth=24 --target-param-data-ratio=20

# Quick experiment
python -m scripts.base_train --depth=12 --num-iterations=1000 --run=quick
```
