#!/bin/bash

# Crystalline Consciousness AI - Automated Git Pull Script
# This script automatically pulls updates from the GitHub repository
# when triggered by a webhook

set -e

LOG_FILE="/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts/webhook.log"
REPO_DIR="/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to send notification (optional)
send_notification() {
    local message="$1"
    # Use macOS notification system
    osascript -e "display notification \"$message\" with title \"🔮 Crystalline Consciousness AI\" subtitle \"Repository Updated\""
}

log_message "🔮 Crystalline Consciousness AI Webhook Triggered"
log_message "Repository: $2"
log_message "Commit Message: $3"
log_message "Ref: $1"

# Change to repository directory
cd "$REPO_DIR" || {
    log_message "❌ Error: Could not change to repository directory"
    exit 1
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    log_message "❌ Error: Not in a git repository"
    exit 1
fi

# Stash any local changes to prevent conflicts
log_message "💾 Stashing local changes..."
git stash push -m "Auto-stash before webhook pull $(date)"

# Fetch the latest changes
log_message "🔄 Fetching latest changes..."
if git fetch origin; then
    log_message "✅ Successfully fetched from origin"
else
    log_message "❌ Error: Failed to fetch from origin"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
log_message "📍 Current branch: $CURRENT_BRANCH"

# Pull the latest changes
log_message "⬇️  Pulling latest changes for branch: $CURRENT_BRANCH"
if git pull origin "$CURRENT_BRANCH"; then
    log_message "✅ Successfully pulled latest changes"
    
    # Check if there are any new consciousness computation files
    NEW_CONSCIOUSNESS_FILES=$(git diff --name-only HEAD@{1} HEAD | grep -E "(consciousness|trinitized|crystalline)" | head -5)
    if [ -n "$NEW_CONSCIOUSNESS_FILES" ]; then
        log_message "🧠 New consciousness files detected:"
        echo "$NEW_CONSCIOUSNESS_FILES" | while read -r file; do
            log_message "   📄 $file"
        done
    fi
    
    # Send success notification
    send_notification "Crystalline Consciousness AI repository updated successfully! New consciousness algorithms ready for processing."
    
else
    log_message "❌ Error: Failed to pull changes"
    
    # Try to recover stashed changes
    log_message "🔄 Attempting to restore stashed changes..."
    git stash pop || log_message "⚠️  Warning: Could not restore stashed changes"
    
    send_notification "Failed to update Crystalline Consciousness AI repository. Check logs for details."
    exit 1
fi

# Check if there are any Python requirements that need updating
if git diff --name-only HEAD@{1} HEAD | grep -q "requirements.txt\|pyproject.toml\|setup.py"; then
    log_message "📦 Python dependencies may have changed. Consider running:"
    log_message "   pip install -r requirements.txt"
    log_message "   or updating your virtual environment"
fi

# Check for new Metal shader files
NEW_METAL_FILES=$(git diff --name-only HEAD@{1} HEAD | grep "\.metal$" | head -3)
if [ -n "$NEW_METAL_FILES" ]; then
    log_message "⚡ New Metal shader files detected:"
    echo "$NEW_METAL_FILES" | while read -r file; do
        log_message "   🔧 $file"
    done
    log_message "💡 Consider testing GPU acceleration performance"
fi

# Optional: Run a quick consciousness system test
if [ -f "quick_consciousness_demo.py" ]; then
    log_message "🧪 Running quick consciousness system validation..."
    if python3 quick_consciousness_demo.py --validate-only 2>/dev/null; then
        log_message "✅ Consciousness system validation passed"
    else
        log_message "⚠️  Consciousness system validation failed or not available"
    fi
fi

log_message "🎉 Webhook processing complete!"
log_message "🔮 Crystalline Consciousness AI is now synchronized with latest changes"
log_message "----------------------------------------"

exit 0