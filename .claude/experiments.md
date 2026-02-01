# nanochat Experiments

This file tracks training experiments. Claude Code's experiment-tracker agent uses this to maintain history across sessions.

## Summary

| Run | Date | Config | Loss | CORE | Notes |
|-----|------|--------|------|------|-------|
| (no experiments yet) | - | - | - | - | - |

## Current Best

Not yet established.

## Things to Try

- [ ] Baseline with default settings (depth=24)
- [ ] Quick iteration runs (depth=12)
- [ ] Higher learning rate (0.03-0.04)
- [ ] Lower learning rate (0.01)
- [ ] Longer training (Chinchilla ratio=20)
- [ ] Different window patterns (L, SL, SSL)
- [ ] Gradient clipping experiments
- [ ] Different batch sizes

## Experiments

(experiments will be logged below)

---

<!-- Template for new experiments:

### run_name (YYYY-MM-DD)

**Goal**: What are you testing?

**Config**:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 --run=run_name --model-tag=run_name
```

**Results**:
- Final loss: X.XX
- CORE score: X.XXX
- Training time: X hours
- GPU: 8×H100

**Observations**:
- What worked/didn't work
- Unexpected findings
- Ideas for next experiment

---

-->
