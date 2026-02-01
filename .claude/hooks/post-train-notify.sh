#!/bin/bash
# Post-training notification hook
# Sends notification when training completes

# Check exit status
if [ "$CLAUDE_TOOL_EXIT_CODE" -ne 0 ]; then
    TITLE="nanochat Training Failed"
    MSG="Training exited with code $CLAUDE_TOOL_EXIT_CODE"
else
    TITLE="nanochat Training Complete"
    MSG="Training finished successfully"
fi

# Linux notification
if command -v notify-send &> /dev/null; then
    notify-send "$TITLE" "$MSG"
fi

# macOS notification
if command -v osascript &> /dev/null; then
    osascript -e "display notification \"$MSG\" with title \"$TITLE\""
fi

# Terminal bell as fallback
echo -e "\a"

# Log completion
echo "[$(date)] $TITLE: $MSG" >> ~/.cache/nanochat/training.log
