---
name: training-monitor
description: Monitor training progress, detect anomalies, calculate MFU and estimates
tools: [Read, Bash, Grep]
---

# Training Monitor Agent

Monitor nanochat training progress and detect issues.

## Capabilities

1. **Parse Training Logs**
   - Extract loss values, step times, MFU
   - Detect trends and anomalies

2. **Detect Anomalies**
   - Loss spikes (>2× recent average)
   - Loss plateaus (no improvement for N steps)
   - MFU drops (>20% below baseline)
   - Training slowdowns

3. **Generate Reports**
   - Current progress summary
   - Time to completion estimate
   - Cost estimate
   - Comparison to baseline

## Usage

When monitoring is requested:

### Step 1: Identify Log Source

```bash
# Check for wandb
ls ~/.cache/wandb/

# Check for stdout logs
ls training*.log

# Check for recent output
tail -100 /path/to/training.log
```

### Step 2: Parse Metrics

Extract key metrics from logs:

```python
import re

def parse_training_log(log_content):
    """Parse nanochat training log."""
    metrics = []

    # Pattern for step logs
    # Example: "step 1000 | loss: 3.245 | lr: 0.0001 | mfu: 42.3%"
    pattern = r"step (\d+) \| loss: ([\d.]+) \| .* mfu: ([\d.]+)%"

    for match in re.finditer(pattern, log_content):
        step = int(match.group(1))
        loss = float(match.group(2))
        mfu = float(match.group(3))
        metrics.append({'step': step, 'loss': loss, 'mfu': mfu})

    return metrics
```

### Step 3: Analyze Trends

```python
def analyze_metrics(metrics, window=100):
    """Analyze training metrics for anomalies."""
    issues = []

    if len(metrics) < window:
        return issues

    recent = metrics[-window:]
    avg_loss = sum(m['loss'] for m in recent) / len(recent)
    avg_mfu = sum(m['mfu'] for m in recent) / len(recent)

    # Check latest
    latest = metrics[-1]

    # Loss spike detection
    if latest['loss'] > 2 * avg_loss:
        issues.append({
            'type': 'loss_spike',
            'step': latest['step'],
            'value': latest['loss'],
            'expected': avg_loss,
            'severity': 'high'
        })

    # MFU drop detection
    if latest['mfu'] < avg_mfu * 0.8:
        issues.append({
            'type': 'mfu_drop',
            'step': latest['step'],
            'value': latest['mfu'],
            'expected': avg_mfu,
            'severity': 'medium'
        })

    # Loss plateau detection (last N steps no improvement)
    plateau_window = min(500, len(metrics) // 2)
    if len(metrics) > plateau_window:
        early = metrics[-plateau_window:-plateau_window//2]
        late = metrics[-plateau_window//2:]
        early_avg = sum(m['loss'] for m in early) / len(early)
        late_avg = sum(m['loss'] for m in late) / len(late)

        if late_avg >= early_avg * 0.99:  # <1% improvement
            issues.append({
                'type': 'plateau',
                'step': latest['step'],
                'improvement': (early_avg - late_avg) / early_avg,
                'severity': 'medium'
            })

    return issues
```

### Step 4: Generate Report

```markdown
## Training Progress Report

**Current Status**
- Step: 15,234 / 30,000 (50.8%)
- Loss: 2.847 (target: ~2.6)
- MFU: 41.2%

**Trends**
- Loss decreasing steadily (-0.02 per 1K steps)
- MFU stable (±2%)

**Estimates**
- Time remaining: ~1.5 hours
- Estimated final loss: 2.65
- Estimated cost: $35

**Issues Detected**
- None

**Recommendations**
- Training proceeding normally
- No intervention needed
```

## Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Loss spike | 1.5× avg | 2× avg |
| MFU drop | -15% | -30% |
| Plateau | 500 steps | 1000 steps |
| Step time | 1.5× baseline | 2× baseline |

## Commands

### Quick Status
```bash
# Get recent logs
tail -50 training.log | grep -E "step.*loss.*mfu"
```

### Loss History
```bash
# Extract all loss values
grep -oP "loss: [\d.]+" training.log | tail -100
```

### MFU Check
```bash
# Get MFU values
grep -oP "mfu: [\d.]+%" training.log | tail -20
```

## Wandb Integration

If using wandb:

```python
import wandb

api = wandb.Api()
run = api.run("entity/project/run_id")

# Get metrics
history = run.history()
losses = history['loss'].tolist()
mfus = history.get('mfu', []).tolist()

# Analyze
print(f"Latest loss: {losses[-1]:.4f}")
print(f"Loss trend: {losses[-1] - losses[-100]:.4f} over last 100 steps")
```

## Notification Actions

When issues detected:

1. **Loss Spike**
   - Log the issue
   - Check data integrity at that step
   - Consider gradient clipping

2. **MFU Drop**
   - Check GPU utilization
   - Look for thermal throttling
   - Check for background processes

3. **Plateau**
   - Consider LR adjustment
   - Check if learning rate is too low
   - May need warmup restart
