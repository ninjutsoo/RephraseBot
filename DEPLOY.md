# Deploy Your Telegram Bot to Render (Free, 24/7)

## 🚀 Quick Deploy (5 minutes, no Git needed)

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `FreeIranXBot` (or any name)
3. Set to **Private** (keeps your code safe)
4. Click **"Create repository"**

### Step 2: Upload Your Code

1. On the new repo page, click **"uploading an existing file"**
2. Drag and drop these files from `C:\Users\mrosh\OneDrive\Desktop\FreeIranXBot`:
   - `main.py`
   - `requirements.txt`
   - `render.yaml`
   - `README.md`
   - `.gitignore`
3. Click **"Commit changes"**

### Step 3: Deploy to Render

1. Go to https://render.com/
2. Click **"Get Started for Free"** → Sign up with GitHub
3. After signup, click **"New +"** → **"Web Service"**
4. Click **"Connect GitHub"** → Select your `FreeIranXBot` repo
5. Render will auto-detect `render.yaml`. Click **"Apply"**

### Step 4: Add Environment Variables

In Render dashboard, go to your service → **Environment** tab → Add these:

| Key | Value |
|-----|-------|
| `TELEGRAM_BOT_TOKEN` | Get from @BotFather on Telegram |
| `GEMINI_API_KEY` | Get from https://aistudio.google.com/apikey |
| `WEBHOOK_SECRET` | Any random string (e.g., `whk_` + random characters) |
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | Any random string (e.g., `tgh_` + random characters) |

Click **"Save Changes"**. Render will auto-deploy.

### Step 5: Get Your Render URL

1. Wait ~2 minutes for deployment to finish
2. Copy your service URL (looks like: `https://freeiranxbot.onrender.com`)

### Step 6: Update Telegram Webhook

Run this in PowerShell (replace `YOUR_RENDER_URL`):

```powershell
$BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
$WEBHOOK_URL = "https://YOUR_RENDER_URL.onrender.com/webhook/YOUR_WEBHOOK_SECRET"
$SECRET_TOKEN = "YOUR_TELEGRAM_WEBHOOK_SECRET_TOKEN"

Invoke-RestMethod -Uri "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" `
  -Method Post `
  -Body @{url=$WEBHOOK_URL; secret_token=$SECRET_TOKEN}
```

Verify:
```powershell
Invoke-RestMethod -Uri "https://api.telegram.org/bot$BOT_TOKEN/getWebhookInfo"
```

### Step 7: Test!

Forward a message to your bot in Telegram. It should reply instantly with a rephrased version!

---

## 🔒 Security Note

**After deploying, you should:**
1. Regenerate your Telegram bot token via @BotFather
2. Create a new Gemini API key
3. Update both in Render's environment variables
4. Update the webhook URL with the new token

This prevents the keys you posted in chat from being misused.

---

## ⚙️ Render Free Tier Limits

- ✅ **750 hours/month** (enough for 24/7 operation)
- ⚠️ Service spins down after 15 min inactivity (wakes up in ~30s on first request)
- ✅ Auto-redeploys when you push to GitHub

To keep it "warm" (prevent spin-down), use a free uptime monitor like:
- https://uptimerobot.com/ (ping your bot every 5 minutes)

---

## 🎉 Done!

Your bot is now running 24/7 on Render's free tier. No credit card, no payment needed!
