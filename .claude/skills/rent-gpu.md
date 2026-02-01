---
name: rent-gpu
description: Guide for renting cloud GPUs for nanochat training
---

# Cloud GPU Rental Guide

Find and rent GPUs for nanochat training, addressing GitHub discussion #216.

## Quick Recommendations

| Budget | Provider | GPU | Cost | Best For |
|--------|----------|-----|------|----------|
| Low | Vast.ai | RTX 4090 | ~$0.40/hr | Experimentation |
| Medium | Lambda | A10 | ~$0.75/hr | Development |
| Medium | RunPod | A100 40GB | ~$1.50/hr | Serious training |
| High | Lambda | H100 | ~$2.00/hr | Production runs |
| Max | AWS/GCP | 8×H100 | ~$25/hr | Full speedrun |

## Provider Details

### 1. Lambda Labs (Recommended for Beginners)

**Pros:**
- Simple pricing, no hidden fees
- Good documentation
- Pre-installed CUDA/PyTorch
- SSH and Jupyter access

**Cons:**
- Limited availability (H100s sell out)
- US regions only

**Pricing (2024):**
- A10: $0.75/hr
- A100 40GB: $1.29/hr
- A100 80GB: $1.89/hr
- H100: $1.99/hr

**Getting Started:**
```bash
# 1. Sign up at https://lambdalabs.com/cloud
# 2. Add SSH key
# 3. Launch instance
# 4. SSH in:
ssh ubuntu@<instance-ip>

# 5. Clone nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 6. Setup environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --extra gpu

# 7. Run training
source .venv/bin/activate
python -m scripts.base_train --depth=12 --run=lambda_test
```

---

### 2. RunPod (Flexible, Good for Spot)

**Pros:**
- Spot instances (up to 80% off)
- Wide GPU selection
- Templates available
- Pay-per-second billing

**Cons:**
- Spot can be interrupted
- UI can be confusing

**Pricing (On-Demand):**
- RTX 4090: $0.69/hr
- A100 80GB: $1.99/hr
- H100: $3.99/hr

**Spot Pricing:**
- RTX 4090: $0.34/hr
- A100: ~$0.70/hr

**Template for nanochat:**
Create a template with:
```dockerfile
# Use PyTorch base
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install uv
RUN pip install uv

# Clone nanochat
RUN git clone https://github.com/karpathy/nanochat.git /workspace/nanochat

WORKDIR /workspace/nanochat
RUN uv sync --extra gpu
```

---

### 3. Vast.ai (Cheapest, Marketplace)

**Pros:**
- Peer-to-peer marketplace
- Cheapest rates available
- Wide GPU variety

**Cons:**
- Variable reliability
- No SLA
- Must verify machines

**Tips:**
- Filter for "Verified" hosts
- Check uptime history
- Use on-demand for important runs

**Pricing:**
- RTX 4090: $0.25-0.50/hr
- A100: $0.80-1.50/hr
- H100: $1.80-2.50/hr

**Selection criteria:**
```
- Reliability: >99%
- DLPerf: High
- Inet speed: >100 Mbps
- Disk: >100GB
```

---

### 4. AWS (Enterprise, 8×H100 Clusters)

**Pros:**
- Enterprise SLA
- p5.48xlarge: 8×H100
- Spot for cost savings

**Cons:**
- Complex setup
- Expensive on-demand
- Quota approval needed

**Pricing:**
- p5.48xlarge (8×H100): ~$98/hr on-demand
- Spot: ~$30-40/hr (variable)

**Setup for nanochat speedrun:**
```bash
# Launch p5.48xlarge with Deep Learning AMI
# SSH in, then:

cd ~
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Use conda from DLAMI
conda activate pytorch
pip install uv
uv sync --extra gpu

# Run speedrun
source .venv/bin/activate
bash runs/speedrun.sh
```

---

### 5. Google Cloud (Flexible, Good Spot)

**Pros:**
- Good spot availability
- Preemptible VMs (like spot)
- Global regions

**Cons:**
- Complex networking
- Quotas required

**Pricing (a2-ultragpu-8g, 8×A100):**
- On-demand: ~$40/hr
- Preemptible: ~$12/hr

---

## Cost Estimation for nanochat

### Full Speedrun (GPT-2 equivalent)

| Hardware | Time | Cost |
|----------|------|------|
| 8×H100 (Lambda) | ~3 hrs | ~$48 |
| 8×H100 (AWS Spot) | ~3 hrs | ~$100 |
| 8×A100 | ~6 hrs | ~$60 |
| 1×A100 | ~24 hrs | ~$45 |
| 1×RTX 4090 | ~48 hrs | ~$35 |

### Quick Experimentation (depth=12)

| Hardware | Time | Cost |
|----------|------|------|
| 1×RTX 4090 | ~2 hrs | ~$1 |
| 1×A100 | ~1 hr | ~$2 |

---

## Region Considerations

### US (Most Options)
- Lambda: Oregon, Texas
- AWS: us-east-1, us-west-2
- RunPod: Multiple US DCs

### EU (GDPR Compliant)
- OVHCloud: France (good for EU)
- CoreWeave: EU regions available
- Scaleway: France
- AWS: eu-west-1, eu-central-1

### Asia
- AWS: ap-northeast-1 (Tokyo)
- GCP: asia-east1 (Taiwan)

---

## Setup Checklist

Before renting:

1. **SSH Key Ready**
   ```bash
   # Generate if needed
   ssh-keygen -t ed25519 -C "your@email.com"
   cat ~/.ssh/id_ed25519.pub
   # Add to provider
   ```

2. **Budget Alert**
   - Set spending limits in provider dashboard
   - Most providers support alerts

3. **Data Transfer Plan**
   - Dataset: Download on instance (faster)
   - Results: Use rsync/scp
   - Large files: Consider S3/GCS

4. **Snapshot Strategy**
   - Save checkpoints to persistent storage
   - Or sync to cloud storage

---

## Quick Start Script

Run this on any cloud instance:

```bash
#!/bin/bash
# nanochat_setup.sh - One-liner cloud setup

set -e

echo "=== nanochat Cloud Setup ==="

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Clone nanochat
git clone https://github.com/karpathy/nanochat.git ~/nanochat
cd ~/nanochat

# Setup environment
uv sync --extra gpu
source .venv/bin/activate

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Quick test
python -m scripts.base_train --depth=12 --run=dummy --num-iterations=10

echo "=== Setup Complete! ==="
echo "Run: cd ~/nanochat && source .venv/bin/activate"
```

Usage:
```bash
curl -sSL https://your-url/nanochat_setup.sh | bash
```

---

## Troubleshooting Cloud Issues

### "No GPUs available"
- Try different region
- Use spot/preemptible
- Try different provider

### "Connection timeout"
- Check security group/firewall
- Verify SSH key
- Check instance status

### "Slow data download"
- Use wget with --continue
- Download in parallel
- Use provider's object storage

### "Instance terminated unexpectedly"
- Spot instance preempted
- Save checkpoints frequently
- Use persistent storage
