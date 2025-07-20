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

# Pull the latest changes - prioritize main branch for primary sync
if [ "$CURRENT_BRANCH" = "main" ]; then
    log_message "⬇️  Pulling latest changes for main branch"
    PULL_BRANCH="main"
else
    log_message "⬇️  Pulling latest changes for branch: $CURRENT_BRANCH"
    PULL_BRANCH="$CURRENT_BRANCH"
fi

if git pull origin "$PULL_BRANCH"; then
    log_message "✅ Successfully pulled latest changes"
    
    # Check if there are any new consciousness research files
    NEW_CONSCIOUSNESS_FILES=$(git diff --name-only HEAD@{1} HEAD | grep -E "(consciousness|crystalline|resonant|holographic)" | head -5)
    if [ -n "$NEW_CONSCIOUSNESS_FILES" ]; then
        log_message "🧠 New consciousness research files detected:"
        echo "$NEW_CONSCIOUSNESS_FILES" | while read -r file; do
            log_message "   📄 $file"
        done
    fi
    
    # Check for new PDF research documents
    NEW_PDF_FILES=$(git diff --name-only HEAD@{1} HEAD | grep "\.pdf$" | head -3)
    if [ -n "$NEW_PDF_FILES" ]; then
        log_message "📚 New research PDFs detected:"
        echo "$NEW_PDF_FILES" | while read -r file; do
            log_message "   📖 $file"
        done
    fi
    
    # Send success notification
    send_notification "Crystalline Consciousness AI research updated successfully! New documentation and visualizations ready."
    
else
    log_message "❌ Error: Failed to pull changes"
    
    # Try to recover stashed changes
    log_message "🔄 Attempting to restore stashed changes..."
    git stash pop || log_message "⚠️  Warning: Could not restore stashed changes"
    
    send_notification "Failed to update Crystalline Consciousness AI repository. Check logs for details."
    exit 1
fi

# Check if there are any Python visualization updates
if git diff --name-only HEAD@{1} HEAD | grep -q "\.py$"; then
    log_message "🐍 Python visualization scripts may have changed. Consider running:"
    log_message "   python3 -m pip install -r requirements.txt (if available)"
    log_message "   or updating visualization dependencies"
fi

# Check for new documentation updates
NEW_DOCS=$(git diff --name-only HEAD@{1} HEAD | grep -E "(docs/|\.md$)" | head -3)
if [ -n "$NEW_DOCS" ]; then
    log_message "📝 New documentation detected:"
    echo "$NEW_DOCS" | while read -r file; do
        log_message "   📃 $file"
    done
fi

log_message "🎉 Webhook processing complete!"
log_message "🔮 Crystalline Consciousness AI research is now synchronized with latest changes"
log_message "----------------------------------------"

exit 0