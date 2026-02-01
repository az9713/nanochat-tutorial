---
name: gpu-check
description: Validate GPU environment for nanochat training
---

# GPU Environment Validator

Comprehensive check of GPU setup for nanochat training.

## Quick Check

Run all checks at once:

```bash
#!/bin/bash
echo "=== nanochat GPU Environment Check ==="
echo

# 1. NVIDIA Driver
echo "1. NVIDIA Driver"
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
echo

# 2. CUDA Version
echo "2. CUDA Version"
nvcc --version 2>/dev/null | grep release || echo "nvcc not in PATH (ok if using PyTorch)"
echo

# 3. PyTorch CUDA
echo "3. PyTorch CUDA Support"
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN version: {torch.backends.cudnn.version()}')
"
echo

# 4. GPU Details
echo "4. GPU Details"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
echo

# 5. GPU Topology (for multi-GPU)
echo "5. GPU Topology"
nvidia-smi topo -m 2>/dev/null || echo "  Single GPU or topology not available"
echo

# 6. Flash Attention
echo "6. Flash Attention"
python -c "
try:
    import flash_attn
    print(f'  flash_attn version: {flash_attn.__version__}')
except ImportError:
    print('  flash_attn not installed (will use PyTorch SDPA fallback)')
"
echo

# 7. Memory Status
echo "7. Current GPU Memory"
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv
echo

# 8. Running Processes
echo "8. GPU Processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "  No GPU processes"
echo

echo "=== Check Complete ==="
```

## Detailed Checks

### Check 1: NVIDIA Driver Version

```bash
nvidia-smi
```

**Minimum requirements:**
- Driver 525+ for H100
- Driver 515+ for A100
- Driver 470+ for older GPUs

**If outdated:**
```bash
# Ubuntu
sudo apt update
sudo apt install nvidia-driver-535

# Or use CUDA installer
```

### Check 2: PyTorch CUDA Compatibility

```python
import torch

# Basic check
assert torch.cuda.is_available(), "CUDA not available!"

# Detailed info
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")

# Test computation
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print(f"Matrix multiply test: OK ({y.shape})")
```

### Check 3: Multi-GPU Communication

```python
import torch
import torch.distributed as dist
import os

# Test NCCL
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

dist.init_process_group('nccl')
print(f"NCCL backend: OK")
dist.destroy_process_group()
```

For multi-GPU:
```bash
# Test with torchrun
torchrun --standalone --nproc_per_node=2 -c "
import torch.distributed as dist
dist.init_process_group('nccl')
print(f'Rank {dist.get_rank()}/{dist.get_world_size()}: OK')
dist.destroy_process_group()
"
```

### Check 4: Flash Attention

```python
# Check Flash Attention availability
try:
    from flash_attn import flash_attn_func
    print("Flash Attention 2: Available")
except ImportError:
    print("Flash Attention 2: Not installed")
    print("  Install with: pip install flash-attn --no-build-isolation")

# Check Flash Attention 3 (Hopper+)
try:
    from flash_attn_interface import flash_attn_func as flash3
    print("Flash Attention 3: Available")
except ImportError:
    print("Flash Attention 3: Not available (requires H100/H200)")
```

### Check 5: Memory Bandwidth Test

```python
import torch
import time

def memory_bandwidth_test(size_gb=1):
    """Measure GPU memory bandwidth."""
    n = int(size_gb * 1e9 / 4)  # float32
    x = torch.randn(n, device='cuda')

    # Warmup
    for _ in range(3):
        y = x * 2

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(10):
        y = x * 2

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # 2 reads + 1 write per element
    bytes_transferred = n * 4 * 3 * 10
    bandwidth = bytes_transferred / elapsed / 1e9

    return bandwidth

bw = memory_bandwidth_test()
print(f"Memory bandwidth: {bw:.1f} GB/s")

# Expected:
# H100: ~3000 GB/s
# A100: ~2000 GB/s
# RTX 4090: ~1000 GB/s
# RTX 3090: ~936 GB/s
```

### Check 6: Compute Performance Test

```python
import torch
import time

def matmul_test(size=4096):
    """Measure TFLOPS."""
    a = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(size, size, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    start = time.perf_counter()

    iters = 100
    for _ in range(iters):
        c = torch.matmul(a, b)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = 2 * size**3 * iters
    tflops = flops / elapsed / 1e12

    return tflops

tflops = matmul_test()
print(f"Compute performance: {tflops:.1f} TFLOPS (bf16)")

# Expected:
# H100: ~1000 TFLOPS
# A100: ~300 TFLOPS
# RTX 4090: ~165 TFLOPS
# RTX 3090: ~71 TFLOPS
```

## Common Issues

### Issue: "CUDA not available"

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "NCCL error"

```bash
# Check for conflicting NCCL versions
python -c "import torch; print(torch.cuda.nccl.version())"

# Set debug output
export NCCL_DEBUG=INFO
```

### Issue: Low Performance

```bash
# Check power state
nvidia-smi -q | grep "Performance State"

# Check clocks
nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv

# Enable persistence mode (may need sudo)
nvidia-smi -pm 1
```

### Issue: GPU Not Detected

```bash
# List PCI devices
lspci | grep -i nvidia

# Check driver loaded
lsmod | grep nvidia

# Reinstall driver
sudo apt purge nvidia-*
sudo apt install nvidia-driver-535
sudo reboot
```

## Summary Output

After running checks, provide summary:

```
=== GPU Environment Summary ===

GPUs: 8× NVIDIA H100 80GB
Driver: 535.129.03
CUDA: 12.1
PyTorch: 2.2.0
Flash Attention: 2.5.0

Status: READY for training

Recommended config:
  --depth=24
  --device-batch-size=16
  --nproc_per_node=8
```
