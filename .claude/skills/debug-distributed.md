---
name: debug-distributed
description: Debug multi-GPU training issues - NCCL timeouts, torchrun errors, GPU assignment
---

# Distributed Training Debugger

Diagnose and fix multi-GPU training issues in nanochat.

## Common Issues and Solutions

### Issue 1: "ChildFailedError" with No Details

**Symptom:** torchrun fails but doesn't show the actual error.

**Cause:** torchrun captures stderr and only shows generic failure message.

**Solution:**
```bash
# Add error recording decorator to main function
# In the training script, add:
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
    # ... training code
```

Or run with more verbose output:
```bash
# Set environment variables for debugging
TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

---

### Issue 2: NCCL Timeout

**Symptom:** Training hangs, then fails with "NCCL timeout" or "watchdog timeout".

**Possible causes:**

1. **Tensor size mismatch across ranks**
   - One GPU has different sized tensor than others

2. **Network issues between GPUs**
   - InfiniBand/NVLink problems

3. **One GPU is much slower**
   - Thermal throttling, faulty GPU

4. **Deadlock in code**
   - Missing barrier, conditional collective

**Diagnostic steps:**
```bash
# 1. Check all GPUs are healthy
nvidia-smi

# 2. Check GPU topology
nvidia-smi topo -m

# 3. Set longer timeout for debugging
export NCCL_IB_TIMEOUT=1000  # Increase from default 23

# 4. Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL  # Focus on collectives

# 5. Run training again
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
```

**Solutions:**
- Increase timeout: `NCCL_IB_TIMEOUT=1000`
- Check for asymmetric conditions in code
- Ensure all GPUs receive same batch size
- Reduce batch size (smaller allreduce)

---

### Issue 3: GPU Assignment Conflicts

**Symptom:** "CUDA error: invalid device ordinal" or wrong GPU being used.

**Cause:** Manual CUDA_VISIBLE_DEVICES conflicts with torchrun's assignment.

**Solution:**
```bash
# DON'T do this:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ...

# DO this instead - let torchrun handle GPU assignment:
torchrun --standalone --nproc_per_node=4 -m scripts.base_train

# If you need specific GPUs, use this:
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 ...
# BUT don't also set local_rank in code
```

---

### Issue 4: Hangs at Specific Step

**Symptom:** Training runs fine, then hangs at same step consistently.

**Possible causes:**

1. **Barrier mismatch**
   - Some ranks skip a collective operation

2. **Conditional logging/saving**
   - Only rank 0 does something that includes collective

3. **Data exhaustion on one rank**
   - One rank runs out of data before others

**Diagnostic:**
```bash
# Add logging to identify where hang occurs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL

# Add barriers with logging in code:
print(f"Rank {rank}: Before barrier at step {step}")
dist.barrier()
print(f"Rank {rank}: After barrier at step {step}")
```

**Solutions:**
- Ensure all ranks execute same collective ops
- Check checkpoint saving code (often only rank 0)
- Verify data loader provides same number of batches

---

### Issue 5: "Address already in use"

**Symptom:** `RuntimeError: Address already in use`

**Cause:** Previous training left zombie processes.

**Solution:**
```bash
# Kill any orphaned processes
pkill -f "python.*base_train"
pkill -f torchrun

# Or find and kill specific process
lsof -i :29500  # Default master port
kill -9 <PID>

# Use different port for new run
torchrun --master-port=29501 --standalone --nproc_per_node=8 -m scripts.base_train
```

---

### Issue 6: OOM on Only Some GPUs

**Symptom:** One GPU runs OOM while others have free memory.

**Cause:** Unbalanced memory usage across ranks.

**Diagnostic:**
```bash
# Monitor all GPUs
watch -n 0.5 nvidia-smi
```

**Solutions:**
```bash
# 1. Ensure equal batch distribution
# In dataloader, verify batch_size is same across ranks

# 2. Check for rank 0 only allocations
# E.g., metrics, logging tensors

# 3. Force memory balancing
export CUDA_MEMORY_FRACTION=0.9  # Leave headroom
```

---

## Quick Reference

### Environment Variables

```bash
# Debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,INIT,NET

# Timeouts
export NCCL_IB_TIMEOUT=1000
export NCCL_ASYNC_ERROR_HANDLING=1

# Network
export NCCL_SOCKET_IFNAME=eth0  # Specific interface
export NCCL_IB_DISABLE=1        # Disable InfiniBand

# Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Minimal Test Script

If issues persist, test with minimal distributed script:

```python
# test_distributed.py
import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Simple allreduce test
    tensor = torch.ones(1000, 1000, device=device)
    dist.all_reduce(tensor)

    print(f"Rank {rank}: Success! Tensor sum = {tensor.sum()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with:
```bash
torchrun --standalone --nproc_per_node=8 test_distributed.py
```

### Decision Tree

```
Training hangs?
├── At start → Check NCCL init, network
├── At specific step → Check data/collective mismatch
└── Random step → Check GPU health, thermals

Error message?
├── "Address in use" → Kill zombies, change port
├── "NCCL timeout" → Increase timeout, check network
├── "invalid device" → Don't set CUDA_VISIBLE_DEVICES manually
└── "ChildFailedError" → Add @record decorator, NCCL_DEBUG=INFO
```
