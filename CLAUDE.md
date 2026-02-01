# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a minimal, educational full-stack LLM training harness created by Andrej Karpathy. It demonstrates the complete pipeline for training ChatGPT-like models: tokenization, pretraining, supervised fine-tuning (SFT), reinforcement learning (RL), evaluation, and inference with a web UI. Designed for a single GPU node (optimized for 8×H100), it can train a GPT-2 capability model in ~3 hours for approximately $73.

## Development Commands

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra gpu    # For CUDA GPUs
uv sync --extra cpu    # For CPU-only

# Activate the virtual environment
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### Full Training Pipeline (8×H100, ~3 hours)
```bash
bash runs/speedrun.sh
```

### Individual Training Stages
```bash
# Tokenizer training
python -m scripts.tok_train
python -m scripts.tok_eval

# Base model pretraining (distributed on 8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24

# Single GPU (uses gradient accumulation automatically)
python -m scripts.base_train --depth=12

# Supervised fine-tuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

### Inference
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"   # CLI single response
python -m scripts.chat_cli                               # Interactive CLI
python -m scripts.chat_web                               # Web UI at localhost:8000
```

### Testing
```bash
pytest                           # All tests
pytest tests/test_engine.py -v   # Single test file
pytest -m "not slow"             # Skip slow tests
```

### Quick Research Iteration (~5 min runs)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 --run="d12" --model-tag="d12" \
    --core-metric-every=999999 --sample-every=-1 --save-every=-1
```

## Architecture Overview

### Model (`nanochat/gpt.py`)
- GPT Transformer with rotary embeddings (no learned positional embeddings)
- QK normalization for stable training
- ReLU² activation in MLP layers
- RMSNorm without learnable parameters
- Group-Query Attention (GQA) for efficient inference
- Sliding window attention (configurable via `--window-pattern`)
- Value embeddings (ResFormer-style) on alternating layers
- Flash Attention 3 on Hopper+ GPUs, PyTorch SDPA fallback elsewhere
- Model size controlled by depth: `model_dim = depth × aspect_ratio (default 64)`

### Optimizer (`nanochat/optim.py`)
- Combined MuonAdamW optimizer
- Muon optimizer for matrix parameters (transformer weights)
- AdamW for embedding parameters
- Learning rates scale by `1/√dmodel` for AdamW parameters
- Weight decay scales by `1/depth²`

### Data Pipeline
- `nanochat/dataset.py`: Downloads FineWeb-Edu shards from HuggingFace
- `nanochat/dataloader.py`: Tokenizing distributed dataloader with BOS-aligned best-fit packing
- `nanochat/tokenizer.py`: BPE tokenizer (GPT-4 style) with rustbpe training and tiktoken inference

### Evaluation System
- CORE metric (`nanochat/core_eval.py`): Primary benchmark (from DCLM paper)
- Bits-per-byte (`nanochat/loss_eval.py`): Loss normalization metric
- Task evaluations in `tasks/`: ARC, GSM8K, MMLU, HumanEval, SpellingBee

### Inference Engine (`nanochat/engine.py`)
- KV cache management for efficient autoregressive generation
- Batched generation with temperature/top-k sampling
- Built-in Python calculator tool support for math expressions

### Web Server (`scripts/chat_web.py`)
- FastAPI-based server with multi-GPU worker pool
- Streaming SSE responses
- ChatGPT-style UI in `nanochat/ui.html`

## Key Conventions

### Script Invocation
All scripts run as Python modules from the project root:
```bash
python -m scripts.base_train    # NOT python scripts/base_train.py
```

### Distributed Training
- Use `torchrun` for multi-GPU: `torchrun --standalone --nproc_per_node=8 -m scripts.base_train`
- Single GPU works without `torchrun` (auto gradient accumulation)
- Arguments after `--` are passed to the script

### Logging
- `--run=dummy` disables wandb logging
- `--run=myrun` enables wandb with run name "myrun"

### Model Scaling
- `--depth` is the primary complexity dial
- Common depths: 12 (fast iteration), 20 (default), 24 (GPT-2 grade)
- Model dimensions derive from depth automatically

### Checkpoints
- Base models: `~/.cache/nanochat/base_checkpoints/{model_tag}/`
- SFT models: `~/.cache/nanochat/sft_checkpoints/{model_tag}/` (saved as `chatsft_checkpoints`)
- Override with `NANOCHAT_BASE_DIR` environment variable

### Training Horizon
Set via one of (in order of precedence):
1. `--num-iterations`: Explicit step count
2. `--target-flops`: Compute budget
3. `--target-param-data-ratio`: Data-to-parameters ratio (default 10.5, Chinchilla=20)

## File Organization

```
nanochat/           # Core library modules
  gpt.py            # GPT model architecture
  engine.py         # Inference engine with KV cache
  tokenizer.py      # BPE tokenizer wrapper
  dataloader.py     # Distributed data loading
  optim.py          # MuonAdamW optimizer

scripts/            # Training and inference scripts
  base_train.py     # Pretraining
  chat_sft.py       # Supervised fine-tuning
  chat_web.py       # Web UI server
  chat_cli.py       # CLI interface

tasks/              # Evaluation task implementations
runs/               # Shell scripts for common workflows
tests/              # Pytest test files
```

## Common Issues

### Out of Memory
Reduce `--device-batch-size` (try 16, 8, 4, 2, or 1).

### Flash Attention Not Available
The code falls back to PyTorch SDPA automatically. For sliding window attention efficiency, use `--window-pattern L`.

### Tokenizer Not Found
Run tokenizer training first: `python -m scripts.tok_train`

### Model Checkpoint Not Found
Run base training first, or specify `--model-tag` and `--step` explicitly.

## Claude Code Skills

This repository includes Claude Code skills for AI-assisted development. These skills were designed based on real pain points discovered through research of:
- nanochat GitHub issues and discussions (#216, #344, etc.)
- [LLM Workflow Pain Points](https://blog.laurentcharignon.com/post/2025-09-30-llm-workflow-part1-pain-points/)
- [NCCL Debugging Guide](https://medium.com/@devaru.ai/debugging-nccl-errors-in-distributed-training-a-comprehensive-guide-28df87512a34)
- [Gradient Accumulation Bugs](https://unsloth.ai/blog/gradient)

### Troubleshooting Skills
| Command | Description |
|---------|-------------|
| `/debug-oom` | Fix CUDA out of memory errors |
| `/debug-loss` | Analyze training loss anomalies |
| `/debug-distributed` | Debug multi-GPU/NCCL issues |
| `/debug-stuck` | Diagnose training hangs |
| `/fix-checkpoint` | Recover from checkpoint issues |
| `/verify-data` | Validate dataset integrity |

### Setup & Education Skills
| Command | Description |
|---------|-------------|
| `/train` | Interactive training launcher |
| `/gpu-check` | Validate GPU environment |
| `/config-memory` | Calculate optimal batch size |
| `/scaling` | Model scaling calculator |
| `/explain <module>` | Architecture explainer with diagrams |
| `/rent-gpu` | Cloud GPU rental guide |

See [docs/CLAUDE_CODE_INTEGRATION.md](docs/CLAUDE_CODE_INTEGRATION.md) for full documentation.

## Experiment Log

Track your experiments below. Claude will reference this for context.

### Template
| Date | Run | Config | Loss | CORE | Notes |
|------|-----|--------|------|------|-------|
| YYYY-MM-DD | run_name | d=XX, lr=XX | X.XX | X.XXX | Notes |

### Current Best
Not yet established. Run experiments and update this section.

### Things to Try
- [ ] Baseline with default settings (d=24)
- [ ] Quick iteration runs (d=12)
- [ ] Different learning rates
- [ ] Longer training (Chinchilla ratio)
- [ ] Different window patterns

### Experiments
(Add your experiments here)
