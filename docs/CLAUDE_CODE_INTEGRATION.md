# Claude Code Integration for nanochat

This document describes the Claude Code integration for nanochat, providing AI-assisted debugging, training guidance, and educational tools.

## Research-Driven Design

This integration was designed based on real pain points discovered through systematic research:

### Sources Analyzed

1. **nanochat GitHub Issues & Discussions**
   - [Discussion #216](https://github.com/karpathy/nanochat/discussions/216) - Cloud GPU rental questions
   - [Issue #344](https://github.com/karpathy/nanochat/issues) - Data integrity problems with FineWeb downloads
   - Training hang issues reported by multiple users
   - Checkpoint recovery questions

2. **LLM Training Workflow Pain Points**
   - [LLM Workflow Pain Points (Laurent Charignon)](https://blog.laurentcharignon.com/post/2025-09-30-llm-workflow-part1-pain-points/) - Common frustrations in LLM development

3. **Distributed Training Debugging**
   - [NCCL Debugging Guide (Medium)](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34) - Comprehensive NCCL troubleshooting
   - [Gradient Accumulation Bugs (Unsloth)](https://unsloth.ai/blog/gradient) - Subtle training bugs

4. **Loss Curve Diagnostics**
   - [Google ML Testing & Debugging](https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic) - Interpreting training metrics

### Pain Points Addressed

| Pain Point | Source | Solution |
|------------|--------|----------|
| OOM errors | Common in issues | `/debug-oom` with GPU-specific recommendations |
| Training hangs | GitHub issues, NCCL guide | `/debug-stuck`, `/debug-distributed` |
| Hidden torchrun errors | NCCL debugging guide | `/debug-distributed` with @record decorator |
| Loss interpretation | ML debugging resources | `/debug-loss` with pattern recognition |
| Data corruption | Issue #344 | `/verify-data` validator |
| Cloud GPU confusion | Discussion #216 | `/rent-gpu` comparison guide |
| Lost experiment context | Workflow pain points | Experiment tracker agent |
| Slow onboarding | General feedback | `/explain` with ASCII diagrams |

## Quick Start

After cloning nanochat, Claude Code will automatically have access to these capabilities:

```bash
# Start Claude Code in the nanochat directory
cd nanochat
claude

# Use slash commands
> /train           # Interactive training launcher
> /debug-oom       # Fix CUDA memory errors
> /explain gpt     # Architecture explanation with diagrams
> /gpu-check       # Validate GPU environment
```

## Available Skills (Slash Commands)

### Troubleshooting

| Command | Description |
|---------|-------------|
| `/debug-oom` | Diagnose and fix CUDA Out of Memory errors |
| `/debug-loss` | Analyze training loss curves for anomalies |
| `/debug-distributed` | Debug multi-GPU training issues (NCCL, torchrun) |
| `/debug-stuck` | Diagnose training that hangs or gets stuck |
| `/fix-checkpoint` | Recover from checkpoint issues |
| `/verify-data` | Validate FineWeb-Edu dataset integrity |

### Setup & Configuration

| Command | Description |
|---------|-------------|
| `/train` | Interactive training launcher with configuration |
| `/gpu-check` | Comprehensive GPU environment validation |
| `/config-memory` | Calculate optimal batch size for your GPU |
| `/scaling` | Model scaling calculator (compute, tokens, cost) |
| `/rent-gpu` | Cloud GPU rental guide |

### Education

| Command | Description |
|---------|-------------|
| `/explain <module>` | Architecture explainer with ASCII diagrams |

Modules: `gpt`, `attention`, `mlp`, `optimizer`, `dataloader`, `rotary`, `tokenizer`, `engine`

## Example Workflows

### Workflow 1: First Time Setup

```
You: I just cloned nanochat, how do I get started?

Claude: [Uses /train skill]
Let me guide you through the setup...

1. Install dependencies:
   uv sync --extra gpu

2. Train tokenizer (downloads data):
   python -m scripts.tok_train

3. Quick test run:
   python -m scripts.base_train --depth=12 --run=test
```

### Workflow 2: OOM Error

```
You: I'm getting CUDA out of memory with depth=24

Claude: [Uses /debug-oom skill]
Let me analyze your GPU...

Your RTX 4090 has 24GB. With depth=24:
- Model: 850MB
- Optimizer: 3.4GB
- Activations at batch=32: ~12GB

Recommendation: --device-batch-size=8

This will use gradient accumulation (64 steps) to maintain
the same effective batch size.
```

### Workflow 3: Understanding the Code

```
You: /explain attention

Claude: [Generates ASCII diagram]

Here's how CausalSelfAttention works in nanochat:

    Input x: [B, T, D]
        │
        ├───────┬───────┐
        ▼       ▼       ▼
       Q       K       V
        │       │       │
        ▼       ▼       │
     QK-Norm QK-Norm    │
        │       │       │
        ▼       ▼       │
      RoPE    RoPE      │
        │       │       │
        └───┬───┘       │
            │           ▼
      Scaled Dot-Product Attention
            │
            ▼
        Output

Key features:
- GQA: 24 query heads, 6 KV heads
- QK normalization for stability
- RoPE for positional encoding
```

### Workflow 4: Cloud Training

```
You: /rent-gpu

Claude: [Uses /rent-gpu skill]

Recommended options for nanochat:

| Provider | GPU | Cost | Best For |
|----------|-----|------|----------|
| Lambda | H100 | $2/hr | Production |
| RunPod | A100 | $1.50/hr | Serious training |
| Vast.ai | 4090 | $0.40/hr | Experimentation |

For full speedrun (8×H100, ~3hrs): ~$48 on Lambda

Quick setup:
ssh ubuntu@<ip>
curl -sSL .../setup.sh | bash
bash runs/speedrun.sh
```

## Agents

Claude Code includes specialized agents that can be invoked for complex tasks:

### Training Monitor

Monitors training progress and detects anomalies:
- Loss spike detection
- MFU tracking
- Time-to-completion estimates

### Experiment Tracker

Maintains experiment history in `.claude/experiments.md`:
- Records hyperparameters and results
- Tracks what's been tried
- Compares runs

### Data Validator

Verifies FineWeb-Edu dataset integrity:
- Detects corrupted shards
- Validates parquet files
- Reports issues

### Checkpoint Manager

Manages model checkpoints:
- Lists available checkpoints
- Inspects checkpoint contents
- Cleans up old checkpoints

## Hooks

The integration includes automation hooks that run at specific times:

### Pre-Training Check

Before training commands, verifies:
- GPU availability and memory
- Dataset existence
- Disk space

### Post-Training Notification

After training completes:
- Desktop notification (Linux/macOS)
- Logging to `~/.cache/nanochat/training.log`

### Session Initialization

At session start:
- Shows environment status
- Lists quick commands
- Checks virtual environment

## File Structure

```
.claude/
├── skills/                    # Slash command implementations
│   ├── debug-oom.md
│   ├── debug-loss.md
│   ├── debug-distributed.md
│   ├── debug-stuck.md
│   ├── fix-checkpoint.md
│   ├── verify-data.md
│   ├── config-memory.md
│   ├── train.md
│   ├── gpu-check.md
│   ├── explain.md
│   ├── rent-gpu.md
│   └── scaling.md
├── agents/                    # Specialized agents
│   ├── training-monitor.md
│   ├── experiment-tracker.md
│   ├── data-validator.md
│   └── checkpoint-manager.md
├── hooks/                     # Automation hooks
│   ├── pre-train-check.sh
│   ├── post-train-notify.sh
│   ├── pre-tool-check.sh
│   └── session-init.sh
├── settings.json              # Hook configuration
└── experiments.md             # Experiment log (created on use)
```

## Customization

### Adding Custom Skills

Create a new `.md` file in `.claude/skills/`:

```markdown
---
name: my-skill
description: What this skill does
---

# My Custom Skill

Instructions for Claude when this skill is invoked...
```

### Modifying Hooks

Edit files in `.claude/hooks/` to customize behavior.

### Experiment Tracking

Initialize experiment tracking:

```bash
# Create experiments.md
cat > .claude/experiments.md << 'EOF'
# nanochat Experiments

## Summary

| Run | Date | Config | Loss | Notes |
|-----|------|--------|------|-------|

## Things to Try

- [ ] Baseline with defaults
- [ ] Higher learning rate
- [ ] Longer training

## Experiments

(experiments logged here)
EOF
```

## Troubleshooting

### Skills Not Loading

Ensure you're in the nanochat root directory:
```bash
cd /path/to/nanochat
claude
```

### Hooks Not Running

Make hook scripts executable:
```bash
chmod +x .claude/hooks/*.sh
```

### GPU Not Detected

Run GPU check:
```
> /gpu-check
```

## Contributing

To add new skills or improve existing ones:

1. Create/edit files in `.claude/skills/`
2. Test by invoking the skill
3. Submit PR with your improvements

## Research Sources

The skills and agents in this integration were designed based on analysis of real user pain points:

### nanochat-Specific Issues
- **GitHub Issues**: Training hangs, checkpoint recovery, data corruption
- **GitHub Discussions**: Cloud GPU rental (#216), hardware requirements
- **Common Questions**: OOM errors, loss interpretation, scaling decisions

### LLM Training General Resources
- [LLM Workflow Pain Points](https://blog.laurentcharignon.com/post/2025-09-30-llm-workflow-part1-pain-points/) - Laurent Charignon's analysis of common LLM development frustrations
- [Gradient Accumulation Bugs](https://unsloth.ai/blog/gradient) - Subtle bugs in gradient accumulation implementations
- [NCCL Debugging Guide](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34) - Comprehensive distributed training troubleshooting
- [ML Testing & Debugging](https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic) - Google's guide to interpreting training metrics

### Key Insights Applied
1. **OOM is the #1 issue** - Most users hit memory limits before anything else
2. **torchrun hides errors** - The @record decorator is essential for debugging
3. **GC pauses cause hangs** - Python garbage collection can pause training for minutes
4. **Data corruption is silent** - Truncated downloads go unnoticed until training fails
5. **Context is lost between sessions** - Experiment tracking prevents repeating mistakes

## Related Resources

- [nanochat README](../README.md)
- [nanochat CLAUDE.md](../CLAUDE.md)
- [Claude Code Documentation](https://docs.anthropic.com/claude-code)
