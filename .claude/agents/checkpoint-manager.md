---
name: checkpoint-manager
description: Manage and compare nanochat checkpoints - list, inspect, compare, clean up
tools: [Bash, Read]
---

# Checkpoint Manager Agent

Manage nanochat model checkpoints.

## Checkpoint Locations

```bash
# Base model checkpoints
BASE_DIR="${HOME}/.cache/nanochat/base_checkpoints"

# SFT checkpoints
SFT_DIR="${HOME}/.cache/nanochat/chatsft_checkpoints"
```

## Actions

### List All Checkpoints

```bash
echo "=== Base Model Checkpoints ==="
for dir in ~/.cache/nanochat/base_checkpoints/*/; do
    if [ -d "$dir" ]; then
        model_tag=$(basename "$dir")
        ckpt_count=$(ls -1 "$dir"/*.pt 2>/dev/null | wc -l)
        latest=$(ls -t "$dir"/*.pt 2>/dev/null | head -1)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)

        echo "$model_tag: $ckpt_count checkpoints, $size"
        if [ -n "$latest" ]; then
            echo "  Latest: $(basename $latest)"
        fi
    fi
done

echo ""
echo "=== SFT Checkpoints ==="
for dir in ~/.cache/nanochat/chatsft_checkpoints/*/; do
    if [ -d "$dir" ]; then
        model_tag=$(basename "$dir")
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "$model_tag: $size"
    fi
done
```

### Inspect Checkpoint

```python
#!/usr/bin/env python
"""Inspect a nanochat checkpoint."""
import sys
import torch
from pathlib import Path

def inspect_checkpoint(path):
    path = Path(path)
    print(f"Inspecting: {path}")
    print(f"Size: {path.stat().st_size / 1e6:.1f} MB")
    print()

    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    print("=== Contents ===")
    for key in sorted(ckpt.keys()):
        value = ckpt[key]
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")
        elif hasattr(value, 'shape'):
            print(f"  {key}: tensor {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Model details
    if 'model' in ckpt:
        print("\n=== Model ===")
        model_state = ckpt['model']
        total_params = sum(v.numel() for v in model_state.values())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Parameter count: {len(model_state)} tensors")

        # Infer depth
        layer_keys = [k for k in model_state.keys() if 'layers.' in k]
        if layer_keys:
            layer_nums = [int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()]
            depth = max(layer_nums) + 1 if layer_nums else 'unknown'
            print(f"  Depth: {depth}")

    # Training state
    if 'step' in ckpt:
        print(f"\n=== Training State ===")
        print(f"  Step: {ckpt['step']}")

    if 'optimizer' in ckpt:
        print(f"  Optimizer: present")

    if 'config' in ckpt:
        print(f"\n=== Config ===")
        for k, v in ckpt['config'].items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ckpt.py <checkpoint.pt>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
```

### Compare Checkpoints

```python
def compare_checkpoints(path1, path2):
    """Compare two checkpoints."""
    import torch

    ckpt1 = torch.load(path1, map_location='cpu', weights_only=False)
    ckpt2 = torch.load(path2, map_location='cpu', weights_only=False)

    print(f"Comparing:")
    print(f"  1: {path1}")
    print(f"  2: {path2}")
    print()

    # Compare steps
    step1 = ckpt1.get('step', 'N/A')
    step2 = ckpt2.get('step', 'N/A')
    print(f"Steps: {step1} vs {step2}")

    # Compare model weights
    if 'model' in ckpt1 and 'model' in ckpt2:
        model1 = ckpt1['model']
        model2 = ckpt2['model']

        # Check for same keys
        keys1 = set(model1.keys())
        keys2 = set(model2.keys())

        if keys1 != keys2:
            print(f"Different keys!")
            print(f"  Only in 1: {keys1 - keys2}")
            print(f"  Only in 2: {keys2 - keys1}")
        else:
            print(f"Same model structure ({len(keys1)} parameters)")

            # Compute weight differences
            total_diff = 0
            for key in keys1:
                diff = (model1[key] - model2[key]).abs().mean().item()
                total_diff += diff

            avg_diff = total_diff / len(keys1)
            print(f"Average weight difference: {avg_diff:.6f}")
```

### Clean Up Old Checkpoints

```bash
#!/bin/bash
# clean_checkpoints.sh - Remove old checkpoints, keep latest N

MODEL_TAG=$1
KEEP=${2:-3}  # Default: keep 3

if [ -z "$MODEL_TAG" ]; then
    echo "Usage: clean_checkpoints.sh <model_tag> [keep_count]"
    exit 1
fi

CKPT_DIR="${HOME}/.cache/nanochat/base_checkpoints/${MODEL_TAG}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi

# List all checkpoints sorted by modification time
CKPTS=$(ls -t "$CKPT_DIR"/*.pt 2>/dev/null)
TOTAL=$(echo "$CKPTS" | wc -l)

if [ "$TOTAL" -le "$KEEP" ]; then
    echo "Only $TOTAL checkpoints, keeping all (threshold: $KEEP)"
    exit 0
fi

# Delete old ones
echo "Found $TOTAL checkpoints, keeping $KEEP newest"
echo "$CKPTS" | tail -n +$((KEEP + 1)) | while read ckpt; do
    echo "Deleting: $(basename $ckpt)"
    rm "$ckpt"
done

# Report
NEW_TOTAL=$(ls -1 "$CKPT_DIR"/*.pt 2>/dev/null | wc -l)
echo "Remaining: $NEW_TOTAL checkpoints"
```

### Calculate Storage Usage

```bash
echo "=== Checkpoint Storage Usage ==="
echo ""

# Base checkpoints
if [ -d ~/.cache/nanochat/base_checkpoints ]; then
    echo "Base checkpoints:"
    du -sh ~/.cache/nanochat/base_checkpoints/*/
    echo "Total: $(du -sh ~/.cache/nanochat/base_checkpoints | cut -f1)"
fi

echo ""

# SFT checkpoints
if [ -d ~/.cache/nanochat/chatsft_checkpoints ]; then
    echo "SFT checkpoints:"
    du -sh ~/.cache/nanochat/chatsft_checkpoints/*/
    echo "Total: $(du -sh ~/.cache/nanochat/chatsft_checkpoints | cut -f1)"
fi

echo ""

# Total nanochat cache
echo "Total nanochat cache: $(du -sh ~/.cache/nanochat | cut -f1)"
```

## Output Formats

### Checkpoint List

```
=== Checkpoints ===

Model: baseline
  Location: ~/.cache/nanochat/base_checkpoints/baseline/
  Checkpoints: 5
  Size: 4.2 GB
  Latest: step_30000.pt (850 MB)
  Steps: 5000, 10000, 15000, 20000, 30000

Model: exp-001
  Location: ~/.cache/nanochat/base_checkpoints/exp-001/
  Checkpoints: 2
  Size: 1.7 GB
  Latest: step_10000.pt (850 MB)
  Status: Training stopped early
```

### Checkpoint Details

```
=== Checkpoint Details ===

File: step_30000.pt
Size: 850 MB
Step: 30000
Depth: 24
Parameters: 425M

Config:
  depth: 24
  lr: 0.02
  batch_size: 524288

Model Layers:
  embedding: 50257 × 1536
  layers: 24 transformer blocks
  lm_head: 1536 × 50257 (tied)
```

## Best Practices

1. **Checkpoint Frequency**
   - Default: every 5000 steps
   - For long runs: every 10000 steps
   - For debugging: every 1000 steps

2. **Storage Management**
   - Keep 3-5 most recent checkpoints
   - Archive important checkpoints to cloud storage
   - Delete failed experiment checkpoints

3. **Backup Strategy**
   - Copy best checkpoints to separate location
   - Use cloud storage for important models
   - Document which checkpoints are important
