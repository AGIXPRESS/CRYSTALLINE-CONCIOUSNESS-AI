#!/bin/bash

# Crystalline Consciousness AI - Start Webhook System
# This script starts the complete webhook system for real-time GitHub synchronization

set -e

SCRIPT_DIR="/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts"
WEBHOOK_PORT=9000
LOG_FILE="$SCRIPT_DIR/webhook.log"

echo "🔮 Starting Crystalline Consciousness AI Webhook System..."

# Change to script directory
cd "$SCRIPT_DIR" || {
    echo "❌ Error: Could not change to script directory"
    exit 1
}

# Check if webhook tool is installed
if ! command -v webhook &> /dev/null; then
    echo "❌ Error: 'webhook' tool not found. Please install it with:"
    echo "   brew install webhook"
    exit 1
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Error: 'ngrok' tool not found. Please install it with:"
    echo "   brew install ngrok"
    exit 1
fi

# Check if hooks.json exists
if [ ! -f "hooks.json" ]; then
    echo "❌ Error: hooks.json not found"
    exit 1
fi

# Kill any existing webhook or ngrok processes
echo "🧹 Cleaning up existing processes..."
pkill -f "webhook.*hooks.json" || true
pkill -f "ngrok.*http.*$WEBHOOK_PORT" || true
sleep 2

# Start webhook listener in background
echo "📡 Starting webhook listener on port $WEBHOOK_PORT..."
webhook -hooks hooks.json -port $WEBHOOK_PORT -verbose > webhook.log 2>&1 &
WEBHOOK_PID=$!
echo $WEBHOOK_PID > webhook.pid
echo "   Webhook PID: $WEBHOOK_PID"

# Wait for webhook to start
sleep 3

# Start ngrok tunnel in background
echo "🌐 Starting ngrok tunnel..."
ngrok http $WEBHOOK_PORT > ngrok.log 2>&1 &
NGROK_PID=$!
echo $NGROK_PID > ngrok.pid
echo "   Ngrok PID: $NGROK_PID"

# Wait for ngrok to establish tunnel
echo "⏳ Waiting for ngrok tunnel to establish..."
sleep 5

# Get ngrok public URL
NGROK_URL=""
for i in {1..10}; do
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'tunnels' in data and len(data['tunnels']) > 0:
        print(data['tunnels'][0]['public_url'])
    else:
        print('')
except:
    print('')
" 2>/dev/null)
    
    if [ -n "$NGROK_URL" ]; then
        break
    fi
    echo "   Attempt $i/10: Waiting for ngrok..."
    sleep 2
done

if [ -z "$NGROK_URL" ]; then
    echo "❌ Error: Could not get ngrok URL"
    echo "   Check if ngrok is authenticated: ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

# Save webhook URL
WEBHOOK_URL="$NGROK_URL/hooks/crystalline-consciousness-auto-update"
echo "$WEBHOOK_URL" > webhook_url.txt

echo ""
echo "✅ Crystalline Consciousness AI Webhook System Started Successfully!"
echo ""
echo "📋 System Information:"
echo "   Webhook Port: $WEBHOOK_PORT"
echo "   Webhook PID: $WEBHOOK_PID"
echo "   Ngrok PID: $NGROK_PID"
echo ""
echo "🔗 Your webhook URL:"
echo "   $WEBHOOK_URL"
echo ""
echo "⚙️  Configure this URL in GitHub:"
echo "   1. Go to: https://github.com/AGIXPRESS/CRYSTALLINE-CONCIOUSNESS-AI/settings/hooks"
echo "   2. Click 'Add webhook'"
echo "   3. Payload URL: $WEBHOOK_URL"
echo "   4. Content type: application/json"
echo "   5. Secret: crystalline-consciousness-secret-2024"
echo "   6. Events: Just the push event"
echo "   7. Active: ✅ Checked"
echo ""
echo "📊 Monitor the system:"
echo "   tail -f webhook.log"
echo ""
echo "🛑 Stop the system:"
echo "   ./stop_webhook_system.sh"
echo ""
echo "🔮 Consciousness research synchronization is now active!"