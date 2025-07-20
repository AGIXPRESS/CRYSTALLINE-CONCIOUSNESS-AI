#!/bin/bash

# Crystalline Consciousness AI - Stop Webhook System
# This script stops all webhook system processes

echo "🛑 Stopping Crystalline Consciousness AI Webhook System..."

SCRIPT_DIR="/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts"
cd "$SCRIPT_DIR" || {
    echo "❌ Error: Could not change to script directory"
    exit 1
}

# Stop webhook process
if [ -f "webhook.pid" ]; then
    WEBHOOK_PID=$(cat webhook.pid)
    echo "🔇 Stopping webhook (PID: $WEBHOOK_PID)..."
    kill $WEBHOOK_PID 2>/dev/null || true
    rm -f webhook.pid
else
    echo "🔍 Stopping any webhook processes..."
    pkill -f "webhook.*hooks.json" || true
fi

# Stop ngrok process
if [ -f "ngrok.pid" ]; then
    NGROK_PID=$(cat ngrok.pid)
    echo "🌐 Stopping ngrok (PID: $NGROK_PID)..."
    kill $NGROK_PID 2>/dev/null || true
    rm -f ngrok.pid
else
    echo "🔍 Stopping any ngrok processes..."
    pkill -f "ngrok.*http" || true
fi

# Clean up temporary files
echo "🧹 Cleaning up temporary files..."
rm -f webhook_url.txt ngrok.log

echo ""
echo "✅ Crystalline Consciousness AI Webhook System Stopped"
echo ""
echo "📊 Logs preserved:"
echo "   webhook.log - Contains webhook activity history"
echo ""
echo "🔮 Consciousness research synchronization is now inactive"