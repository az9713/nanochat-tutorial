---
name: debug-loss
description: Analyze training loss curves for anomalies - spikes, plateaus, divergence, NaN
---

# Loss Curve Analyzer

Diagnose and fix training loss issues in nanochat.

## Step 1: Identify the Problem Type

Ask the user which issue they're experiencing:

1. **Loss Spike** - Sudden jump in loss value
2. **Loss Plateau** - Loss stops decreasing
3. **Loss Divergence** - Loss increases continuously
4. **NaN/Inf Loss** - Loss becomes undefined
5. **Loss Oscillation** - Loss swings wildly

## Step 2: Diagnose by Symptom

### Loss Spike (sudden increase)

**Possible causes:**
1. **Corrupted batch** (most likely)
   - A malformed data sample caused bad gradients
   - Check: Which step? Which data shard?

2. **Learning rate scheduler jump**
   - Check the LR schedule at that step

3. **Numerical overflow** (rare with bf16)
   - Large activations causing overflow

**Diagnostic commands:**
```bash
# Check which data shard was being processed
# Step to shard mapping: shard_idx = (step * batch_size * seq_len) // shard_size

# Verify data integrity
python -c "
import pyarrow.parquet as pq
# Check the suspicious shard
table = pq.read_table('~/.cache/nanochat/fineweb_edu/shard_XXXX.parquet')
print(f'Rows: {len(table)}')
print(table.schema)
"
```

**Solutions:**
- Re-download corrupted shard
- Add gradient clipping: `--grad-clip=1.0`
- Skip bad batch (requires code modification)

---

### Loss Plateau (stops decreasing)

**Possible causes:**
1. **Learning rate too low**
   - Already in "flat" region of LR schedule

2. **Saddle point**
   - Model stuck in local minimum

3. **Data exhaustion**
   - Seeing same data repeatedly

4. **Capacity limit**
   - Model too small for task

**Diagnostic commands:**
```bash
# Check current learning rate
# Look in wandb or parse logs for current LR value

# Check data iteration
# How many epochs through data?
```

**Solutions:**
- Increase LR warmup or peak LR
- Add noise to gradients (some optimizers do this)
- Use larger model (increase depth)
- Check if you've seen all data

---

### Loss Divergence (keeps increasing)

**Possible causes:**
1. **Learning rate too high** (most common)
   - Gradients overshooting

2. **Weight initialization issue**
   - Unstable starting point

3. **Optimizer bug**
   - Momentum/variance accumulation issue

**Diagnostic:**
```python
# Check gradient magnitudes
# Add this to training loop temporarily:
total_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"Gradient norm: {total_norm}")
```

**Solutions:**
- Reduce learning rate by 2-10x
- Increase warmup steps
- Add gradient clipping
- Check weight initialization

---

### NaN/Inf Loss

**Possible causes:**
1. **Division by zero**
   - Softmax overflow/underflow

2. **Log of zero/negative**
   - Check cross-entropy inputs

3. **Exploding gradients**
   - Gradients became inf, then weights

**Diagnostic:**
```python
# Find where NaN first appears
import torch
torch.autograd.set_detect_anomaly(True)  # Slow but finds NaN source
```

**Solutions:**
- Enable gradient clipping: `--grad-clip=1.0`
- Check for numerical stability in loss function
- Use bf16 instead of fp16 (more range)
- Reduce learning rate

---

### Loss Oscillation

**Possible causes:**
1. **Batch size too small**
   - High gradient variance

2. **Learning rate too high**
   - Overshooting repeatedly

3. **Data ordering issues**
   - Non-shuffled data with patterns

**Solutions:**
- Increase batch size (or gradient accumulation)
- Reduce learning rate
- Ensure proper data shuffling

## Step 3: Visualization

If user has wandb:
```bash
# Check loss curve in wandb dashboard
# Look for: spikes, trends, plateaus
```

If no wandb, parse logs:
```bash
# Extract loss values from stdout
grep "loss:" training.log | awk '{print $NF}' > losses.txt

# Simple plot with Python
python -c "
import matplotlib.pyplot as plt
losses = [float(l) for l in open('losses.txt')]
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
print('Saved to loss_curve.png')
"
```

## Step 4: Compare to Expected

For nanochat with FineWeb-Edu:

| Depth | Expected Final Loss | Training Steps |
|-------|--------------------:|---------------:|
| 12    | ~3.0-3.2           | ~10K           |
| 20    | ~2.8-3.0           | ~20K           |
| 24    | ~2.6-2.8           | ~30K           |

If your loss is significantly higher, something is wrong.

## Quick Reference

| Symptom | First Thing to Check | Quick Fix |
|---------|---------------------|-----------|
| Spike | Data integrity | `--grad-clip=1.0` |
| Plateau | Learning rate | Increase LR |
| Diverge | LR too high | Reduce LR by 5x |
| NaN | Gradient explosion | `--grad-clip=1.0` + reduce LR |
| Oscillate | Batch too small | Increase batch/grad accum |
