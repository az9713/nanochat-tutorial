#!/bin/bash
# Pre-training validation hook
# Runs before any training command to verify environment

set -e

# Only run for training-related commands
case "$CLAUDE_BASH_COMMAND" in
    *base_train*|*chat_sft*|*torchrun*)
        ;;
    *)
        exit 0
        ;;
esac

echo "[Pre-train check]"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)

    echo "  GPUs: $GPU_COUNT"
    echo "  Free memory: ${FREE_MEM}MB"

    if [ "$FREE_MEM" -lt 10000 ]; then
        echo "  WARNING: Low GPU memory (${FREE_MEM}MB free)"
    fi
else
    echo "  WARNING: nvidia-smi not found"
fi

# Check data directory
DATA_DIR="${HOME}/.cache/nanochat/fineweb_edu"
if [ ! -d "$DATA_DIR" ]; then
    echo "  WARNING: Dataset not found at $DATA_DIR"
    echo "  Run: python -m scripts.tok_train"
fi

# Check tokenizer
TOK_DIR="${HOME}/.cache/nanochat/tokenizer"
if [ ! -d "$TOK_DIR" ]; then
    echo "  WARNING: Tokenizer not found at $TOK_DIR"
fi

# Check disk space
CACHE_DIR="${HOME}/.cache/nanochat"
if [ -d "$CACHE_DIR" ]; then
    AVAIL=$(df -BG "$CACHE_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "$AVAIL" -lt 10 ]; then
        echo "  WARNING: Low disk space (${AVAIL}GB available)"
    fi
fi

echo "[Pre-train check complete]"
