---
name: fix-checkpoint
description: Recover from checkpoint issues - corrupted, missing, version mismatch
---

# Checkpoint Recovery

Diagnose and fix checkpoint-related issues in nanochat.

## Checkpoint Locations

```bash
# Base model checkpoints
~/.cache/nanochat/base_checkpoints/{model_tag}/

# SFT checkpoints
~/.cache/nanochat/chatsft_checkpoints/{model_tag}/

# Tokenizer
~/.cache/nanochat/tokenizer/
```

## Common Issues

### Issue 1: "Checkpoint not found"

**Symptom:** Training or inference fails with checkpoint not found error.

**Diagnostic:**
```bash
# List available checkpoints
ls -la ~/.cache/nanochat/base_checkpoints/

# Check specific model tag
ls -la ~/.cache/nanochat/base_checkpoints/YOUR_TAG/
```

**Solutions:**
```bash
# Option 1: Use a different step
python -m scripts.base_train --model-tag=YOUR_TAG --step=EARLIER_STEP

# Option 2: Start fresh
python -m scripts.base_train --model-tag=NEW_TAG

# Option 3: Download pretrained (if available)
# Check if Karpathy released any pretrained models
```

---

### Issue 2: "Tensor size mismatch"

**Symptom:** Loading checkpoint fails with size mismatch error.

**Cause:** Model architecture changed between save and load.

**Diagnostic:**
```python
import torch

# Load checkpoint to inspect
ckpt = torch.load('checkpoint.pt', map_location='cpu')

# Check saved config
if 'config' in ckpt:
    print(ckpt['config'])

# Check state dict keys and shapes
for k, v in ckpt['model'].items():
    print(f"{k}: {v.shape}")
```

**Solutions:**
```bash
# Option 1: Use matching depth
python -m scripts.base_train --depth=XX  # Match saved depth

# Option 2: Start fresh if architecture changed
python -m scripts.base_train --model-tag=NEW_TAG
```

---

### Issue 3: Corrupted Checkpoint

**Symptom:** `RuntimeError: storage has wrong size` or pickle errors.

**Cause:** Incomplete save (Ctrl+C during save, disk full, etc.)

**Diagnostic:**
```python
import torch

try:
    ckpt = torch.load('checkpoint.pt', map_location='cpu')
    print("Checkpoint loads successfully")
except Exception as e:
    print(f"Corruption detected: {e}")
```

**Solutions:**
```bash
# Option 1: Use previous checkpoint
ls -la ~/.cache/nanochat/base_checkpoints/YOUR_TAG/
# Look for earlier step numbers

# Option 2: Delete and restart
rm -rf ~/.cache/nanochat/base_checkpoints/YOUR_TAG/
python -m scripts.base_train --model-tag=YOUR_TAG
```

---

### Issue 4: Optimizer State Issues

**Symptom:** Training resumes but loss spikes or diverges.

**Cause:** Optimizer state corrupted or mismatched.

**Solutions:**
```bash
# Option 1: Resume without optimizer state
# (modify code to skip optimizer loading)

# Option 2: Reset learning rate schedule
python -m scripts.base_train --model-tag=YOUR_TAG --resume \
    --warmup-iters=1000  # Re-warmup
```

---

### Issue 5: Missing Step in Checkpoint

**Symptom:** Want to resume from step X but only have step Y.

**Diagnostic:**
```bash
# List available steps
ls ~/.cache/nanochat/base_checkpoints/YOUR_TAG/*.pt
```

**Solutions:**
```bash
# Use closest available step
python -m scripts.base_train --model-tag=YOUR_TAG --step=CLOSEST_STEP

# Or start fresh and run longer
python -m scripts.base_train --model-tag=NEW_TAG \
    --num-iterations=MORE_STEPS
```

---

## Checkpoint Inspection Script

```python
#!/usr/bin/env python
"""Inspect a nanochat checkpoint."""
import sys
import torch

def inspect_checkpoint(path):
    print(f"Loading: {path}")
    ckpt = torch.load(path, map_location='cpu')

    print("\n=== Keys ===")
    for k in ckpt.keys():
        print(f"  {k}")

    if 'config' in ckpt:
        print("\n=== Config ===")
        for k, v in ckpt['config'].items():
            print(f"  {k}: {v}")

    if 'step' in ckpt:
        print(f"\n=== Training State ===")
        print(f"  Step: {ckpt['step']}")

    if 'model' in ckpt:
        print(f"\n=== Model ===")
        total_params = 0
        for k, v in ckpt['model'].items():
            params = v.numel()
            total_params += params
            print(f"  {k}: {v.shape} ({params:,} params)")
        print(f"\nTotal parameters: {total_params:,}")

    if 'optimizer' in ckpt:
        print(f"\n=== Optimizer ===")
        print(f"  Type: {type(ckpt['optimizer'])}")
        if isinstance(ckpt['optimizer'], dict):
            print(f"  Keys: {list(ckpt['optimizer'].keys())}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint.pt>")
        sys.exit(1)
    inspect_checkpoint(sys.argv[1])
```

---

## Safe Checkpoint Practices

### Manual Checkpoint Save

If you need to save mid-training (nanochat doesn't have Ctrl+C save):

```python
# Add to training script or run interactively
import torch
import signal

def save_checkpoint_signal(signum, frame):
    print("Saving emergency checkpoint...")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': current_step,
        'config': config,
    }, 'emergency_checkpoint.pt')
    print("Saved!")
    exit(0)

signal.signal(signal.SIGINT, save_checkpoint_signal)
```

### Checkpoint Backup

```bash
# Before resuming, backup current checkpoint
cp -r ~/.cache/nanochat/base_checkpoints/YOUR_TAG \
      ~/.cache/nanochat/base_checkpoints/YOUR_TAG_backup

# Verify backup
ls -la ~/.cache/nanochat/base_checkpoints/YOUR_TAG_backup/
```

### Disk Space Check

```bash
# Checkpoints can be large, verify space
df -h ~/.cache/nanochat/

# Typical checkpoint sizes:
# depth=12: ~500MB
# depth=20: ~1.5GB
# depth=24: ~2GB
```

---

## Quick Reference

| Problem | Quick Fix |
|---------|-----------|
| Not found | Check path, use different step |
| Size mismatch | Match --depth to saved model |
| Corrupted | Use earlier checkpoint |
| Loss spike on resume | Re-warmup, skip optimizer state |
| No recent checkpoint | Modify save frequency |
