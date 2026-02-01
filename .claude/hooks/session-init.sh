#!/bin/bash
# Session initialization hook
# Runs at the start of each Claude Code session

echo "=== nanochat Development Environment ==="

# Check if in nanochat directory
if [ -f "pyproject.toml" ] && grep -q "nanochat" pyproject.toml 2>/dev/null; then
    echo "Project: nanochat"
else
    echo "Note: Not in nanochat root directory"
fi

# Check virtual environment
if [ -d ".venv" ]; then
    echo "Virtual env: .venv (use 'source .venv/bin/activate')"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual env: $VIRTUAL_ENV (active)"
else
    echo "Virtual env: Not found (run 'uv sync --extra gpu')"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "GPU: $GPU_NAME ($GPU_MEM)"
else
    echo "GPU: Not detected (CPU mode)"
fi

# Quick tips
echo ""
echo "Quick commands:"
echo "  /train       - Start training"
echo "  /debug-oom   - Fix memory issues"
echo "  /explain     - Explain architecture"
echo "  /gpu-check   - Verify GPU setup"

echo "==="
