#!/bin/bash
# Generic pre-tool check hook
# Performs basic validation before Bash commands

# This hook is intentionally minimal to avoid slowing down operations
# Add specific checks as needed

# Example: Warn before destructive git commands
case "$CLAUDE_BASH_COMMAND" in
    *"git reset --hard"*|*"git clean -f"*)
        echo "[WARNING] Destructive git command detected"
        ;;
    *"rm -rf"*"cache"*)
        echo "[WARNING] Removing cache directory"
        ;;
esac

exit 0
