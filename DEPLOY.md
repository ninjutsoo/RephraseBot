# Deploy Your Rephrase Bot (Fast & Scalable)

## 🚀 Quick Deploy (5 minutes)

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `MyRephraseBot` (or any name)
3. Set to **Private** (recommended)
4. Click **"Create repository"**

### Step 2: Upload Your Code

1. On the new repo page, click **"uploading an existing file"**
2. Upload these files from your project folder:
   - `main.py`
   - `requirements.txt`
   - `render.yaml`
   - `README.md`
   - `.gitignore`
3. Click **"Commit changes"**

### Step 3: Deploy to Cloud Provider

1. Connect your GitHub repository to your preferred hosting service (e.g., Render).
2. The deployment will be auto-detected via `render.yaml`.

### Step 4: Add Environment Variables

In your service dashboard, add these variables:

| Key | Value |
|-----|-------|
| `TELEGRAM_BOT_TOKEN` | Get from @BotFather on Telegram |
| `GEMINI_API_KEY` | Get from your AI provider console |
| `WEBHOOK_SECRET` | Any random string (e.g., `whk_` + random characters) |
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | Any random string (e.g., `tgh_` + random characters) |
| `ALLOWED_FORWARD_CHANNEL` | **(Optional)** Channel username (without @) or ID to restrict sources. |
| `RATE_LIMIT_SECONDS` | **(Optional)** Cooldown between messages per user. Default: `0` (disabled) |

Click **"Save Changes"**. The service will auto-deploy.

### Step 5: Update Telegram Webhook

Run this in your terminal (replace `YOUR_SERVICE_URL` and `YOUR_BOT_TOKEN`):

```bash
$BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
$WEBHOOK_URL = "https://YOUR_SERVICE_URL.com/webhook/YOUR_WEBHOOK_SECRET"
$SECRET_TOKEN = "YOUR_TELEGRAM_WEBHOOK_SECRET_TOKEN"

# For PowerShell:
Invoke-RestMethod -Uri "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" `
  -Method Post `
  -Body @{url=$WEBHOOK_URL; secret_token=$SECRET_TOKEN}
```

Verify:
```bash
Invoke-RestMethod -Uri "https://api.telegram.org/bot$BOT_TOKEN/getWebhookInfo"
```

### Step 6: Test!

Forward a message from an authorized channel to your bot. It should reply with a rephrased version!

---

## 🔒 Security Note

**If your keys are ever exposed:**
1. Regenerate your Telegram bot token via @BotFather.
2. Rotate your AI API keys.
3. Update environment variables in your hosting dashboard immediately.

---

## 🎉 Done!

Your bot is now running and ready to process messages.
