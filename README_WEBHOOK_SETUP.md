# 🔮 Crystalline Consciousness AI - Real-Time GitHub Webhook Setup

This guide will help you set up automatic real-time synchronization between your local MacBook M4 Pro and the GitHub repository `https://github.com/AGIXPRESS/CRYSTALLINE-CONCIOUSNESS-AI`.

## 🚀 Quick Start

### Step 1: Run the Setup Script
```bash
cd "/Users/okok/mlx/mlx-examples/claude-code-bridge/CC-AI/Crystalline AI Attempts"
./setup_webhook_system.sh
```

This script will:
- ✅ Check all required tools are installed
- 🔑 Help you configure ngrok authentication
- 📝 Create startup and stop scripts
- 🧪 Create test scripts

### Step 2: Start the Webhook System
```bash
./start_webhook_system.sh
```

This will:
- 📡 Start webhook listener on port 9000
- 🌐 Create ngrok tunnel to expose it publicly
- 🔗 Display your webhook URL

### Step 3: Configure GitHub Webhook

1. **Go to your repository settings:**
   - Navigate to `https://github.com/AGIXPRESS/CRYSTALLINE-CONCIOUSNESS-AI/settings/hooks`
   - Click "Add webhook"

2. **Configure the webhook:**
   - **Payload URL:** `https://YOUR_NGROK_URL.ngrok.io/hooks/crystalline-consciousness-auto-update`
   - **Content type:** `application/json`
   - **Secret:** `crystalline-consciousness-secret-2024`
   - **Which events:** Select "Just the push event"
   - **Active:** ✅ Checked

3. **Click "Add webhook"**

### Step 4: Test the System
```bash
./test_webhook.sh
```

## 📋 Files Created

| File | Purpose |
|------|---------|
| `hooks.json` | Webhook configuration |
| `pull_script.sh` | Git pull automation script |
| `setup_webhook_system.sh` | One-time setup script |
| `start_webhook_system.sh` | Start the webhook system |
| `stop_webhook_system.sh` | Stop the webhook system |
| `test_webhook.sh` | Test webhook functionality |
| `webhook.log` | Webhook activity logs |

## 🔒 Security Features

- **HMAC SHA-256 verification** using secret key
- **Repository name validation** (only accepts specified repo)
- **Local change stashing** before pulling
- **Error handling and recovery**

## 🧠 Consciousness-Specific Features

The webhook system includes special handling for Crystalline Consciousness AI:

### 🔮 Automatic Detection
- **New consciousness files** (consciousness*, trinitized*, crystalline*)
- **Metal shader updates** (*.metal files)
- **Python dependencies** (requirements.txt, pyproject.toml)

### 📊 Smart Notifications
- **macOS notifications** for successful updates
- **Detailed logging** of consciousness file changes
- **Quick validation** of consciousness system integrity

### ⚡ Performance Monitoring
- **Git stash management** to preserve local experiments
- **Automatic dependency detection**
- **GPU shader file change alerts**

## 🛠 Manual Operation

### Start Individual Components

**Start webhook listener only:**
```bash
webhook -hooks hooks.json -port 9000 -verbose
```

**Start ngrok tunnel only:**
```bash
ngrok http 9000
```

### Check Status

**View webhook logs:**
```bash
tail -f webhook.log
```

**Check running processes:**
```bash
ps aux | grep -E "(webhook|ngrok)"
```

**Check ngrok status:**
```bash
curl -s http://localhost:4040/api/tunnels | python3 -m json.tool
```

## 🔧 Troubleshooting

### Common Issues

**1. Ngrok authentication failed:**
```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

**2. Webhook not triggering:**
- Check GitHub webhook delivery logs
- Verify the webhook URL is accessible
- Check webhook.log for errors

**3. Git pull conflicts:**
- Local changes are automatically stashed
- Check `git stash list` to see stashed changes
- Manually resolve conflicts if needed

**4. Permission denied:**
```bash
chmod +x *.sh
```

### Reset Everything

**Stop all processes:**
```bash
./stop_webhook_system.sh
pkill -f webhook
pkill -f ngrok
```

**Clean up files:**
```bash
rm -f webhook.pid ngrok.pid webhook_url.txt ngrok.log
```

## 📈 Advanced Configuration

### Custom Webhook Events

Edit `hooks.json` to handle different events:

```json
{
  "trigger-rule": {
    "and": [
      {
        "match": {
          "type": "value",
          "value": "refs/heads/main",
          "parameter": {
            "source": "payload",
            "name": "ref"
          }
        }
      }
    ]
  }
}
```

### Custom Notifications

Edit `pull_script.sh` to add custom notifications:

```bash
# Slack notification
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🔮 Crystalline Consciousness AI updated!"}' \
  YOUR_SLACK_WEBHOOK_URL

# Discord notification
curl -X POST -H 'Content-type: application/json' \
  --data '{"content":"🔮 New consciousness algorithms deployed!"}' \
  YOUR_DISCORD_WEBHOOK_URL
```

## 🌟 Next Steps

Once your webhook system is running:

1. **Make a test commit** to the GitHub repository
2. **Watch for the automatic pull** in your local directory
3. **Check the notification** on your Mac
4. **Review the logs** to see what changed

## 🔮 Consciousness-Specific Automation

The system can be extended to:

- **Auto-run consciousness tests** after updates
- **Restart consciousness computation** when shaders change
- **Send consciousness metrics** to monitoring systems
- **Trigger distributed cluster updates** for multi-GPU setups

## 💡 Tips

- Keep the webhook system running in a Terminal tab
- Use `tmux` or `screen` for persistent sessions
- Monitor `webhook.log` for debugging
- Test changes in a separate branch first

---

**🎉 Your Crystalline Consciousness AI is now synchronized in real-time with GitHub!**

Every push to the repository will automatically update your local development environment, ensuring you always have the latest consciousness algorithms and improvements ready for processing on your M4 Pro. ✨