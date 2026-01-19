# RephraseBot 🔄

**A Telegram bot that intelligently rephrases forwarded messages to avoid spam detection while preserving meaning, links, and mentions.**

Built with FastAPI, Google Gemini AI, and designed for Render's free tier.

---

## 🎯 Purpose

This bot helps you share the same message multiple times without triggering spam filters. Perfect for:
- 📢 Activism and awareness campaigns
- 📣 Community announcements
- 🔁 Cross-posting content to multiple channels
- ✍️ Creating natural variations of the same message

---

## ✨ Key Features

### 🛡️ **Anti-Spam Technology**
- **Advanced variation engine**: Each rephrase is structurally and stylistically different
- **171+ style combinations**: Randomly selected tone, length, and sentence structure
- **Dynamic randomness**: Variable AI temperature (0.65-0.95) ensures uniqueness
- **Smart preservation**: Keeps @mentions, #hashtags, URLs, and numbers intact

### 🎨 **Intelligent Rephrasing**
- ✅ Preserves core meaning and facts
- ✅ Maintains all mentions, hashtags, and links
- ✅ Varies sentence structure, word choice, and length
- ✅ Changes tone and rhythm each time
- ✅ Natural-sounding output

### 🔐 **Security & Control**
- **Channel filtering**: Only processes messages from specific Telegram channels
- **Rate limiting**: 30-second cooldown per user (configurable)
- **Webhook authentication**: Secret token validation
- **No data storage**: Processes messages in-memory only

---

## 🚀 How It Works

### Input (Original Message):
```
🔥 Join the protest tomorrow at 5 PM! 
We need everyone at Freedom Square. 
#IranRevolution2026 @FreedomNow
```

### Output Examples (Different Each Time):

**Variation 1** (Concise):
```
Tomorrow 5 PM - Freedom Square. Everyone needed. 🔥
#IranRevolution2026 @FreedomNow
```

**Variation 2** (Formal):
```
We're organizing a demonstration scheduled for tomorrow 
evening at 5 PM. Your presence at Freedom Square is 
essential. 🔥 #IranRevolution2026 @FreedomNow
```

**Variation 3** (Restructured):
```
Freedom Square tomorrow - that's where we gather at 5 PM. 
Your participation matters. 🔥 #IranRevolution2026 @FreedomNow
```

### What Gets Preserved:
- ✅ `@mentions` → Exact usernames/channels
- ✅ `#hashtags` → All hashtags unchanged
- ✅ URLs → Complete links intact
- ✅ Numbers, dates, times → Preserved exactly
- ✅ Core message meaning → Same intent and facts

### What Gets Varied:
- 🔄 Sentence structure and order
- 🔄 Word choice (synonyms)
- 🔄 Tone (formal ↔ casual)
- 🔄 Length (±30% variation)
- 🔄 Punctuation and rhythm

---

## 📂 Project Structure

```
RephraseBot/
├── main.py              # Core bot logic and FastAPI app
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
├── env.example          # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── DEPLOY.md           # Step-by-step deployment guide
```

## 💻 Local Development

1. Create a local env file (don't commit it). Use `env.example` as a template.
2. Install dependencies and run:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

3. Use [ngrok](https://ngrok.com/) to expose localhost for Telegram webhook testing

## ⚙️ Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Get from [@BotFather](https://t.me/BotFather) on Telegram |
| `GEMINI_API_KEY` | Get from [Google AI Studio](https://aistudio.google.com/apikey) |
| `WEBHOOK_SECRET` | Random string used in webhook URL path (e.g., `whk_yourRandomString`) |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | None | Secret token for webhook verification |
| `ALLOWED_FORWARD_CHANNEL` | `tweeterstormIranrevolution2026` | Channel username or ID to restrict forwarding from |
| `RATE_LIMIT_SECONDS` | `30` | Cooldown time between user requests (0 to disable) |
| `SYSTEM_INSTRUCTION` | See `main.py` | Custom AI rephrasing instructions |
| `GEMINI_MODEL` | `models/gemini-2.5-flash` | Gemini model to use |

---

## 🎨 Anti-Spam Variation Strategy

The bot uses **multi-layered randomization** to ensure each rephrase is unique:

### 1. **Style Selection** (171 variations)
Random selection from categories like:
- Tone: formal, casual, urgent, calm, professional
- Structure: concise, detailed, restructured
- Voice: active, passive, mixed
- Perspective: neutral, empathetic, assertive

### 2. **AI Temperature** (Dynamic)
- Range: `0.65 - 0.95` (randomized per request)
- Higher = more creative variations
- Lower = more conservative changes

### 3. **Top-P Sampling** (Nucleus Sampling)
- Range: `0.85 - 0.95` (randomized)
- Controls diversity of word choices

### 4. **Structural Transformations**
- Sentence reordering
- Clause restructuring  
- Length variation (±30%)
- Punctuation pattern changes

### 5. **Protection Mechanism**
Before rephrasing, the bot:
1. Identifies and masks all @mentions, #hashtags, URLs, numbers
2. Replaces them with `__PROTECTED_0__`, `__PROTECTED_1__`, etc.
3. Sends masked text to AI
4. Restores original values after rephrasing

This ensures critical elements are **never altered** by the AI.

## 🚀 Quick Deploy

See **[DEPLOY.md](DEPLOY.md)** for detailed step-by-step instructions.

### Summary:
1. **Push to GitHub** (keep repo private to protect your code)
2. **Deploy to Render** (free tier, auto-detected via `render.yaml`)
3. **Set environment variables** in Render dashboard
4. **Configure Telegram webhook** with your Render URL

### Webhook Setup

After deployment, configure Telegram to send updates to your bot:

```bash
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook" \
  -d "url=https://your-service.onrender.com/webhook/<YOUR_WEBHOOK_SECRET>" \
  -d "secret_token=<YOUR_SECRET_TOKEN>"
```

**Verify:**
```bash
curl -s "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getWebhookInfo"
```

---

## 🛡️ Rate Limiting

**Default:** 30 seconds between messages per user

**User Experience:**
```
User forwards message #1 → ✅ Processed
User forwards message #2 (5 seconds later) → ❌ "⏱️ Please wait 25 seconds..."
User forwards message #3 (35 seconds later) → ✅ Processed
```

**Configure:**
- Set `RATE_LIMIT_SECONDS=60` for 1 minute cooldown
- Set `RATE_LIMIT_SECONDS=0` to disable (not recommended)

---

## 🔐 Security Features

### Channel Filtering
Only processes messages forwarded from `@tweeterstormIranrevolution2026` by default.

**To change:**
- Set `ALLOWED_FORWARD_CHANNEL=yourchannel` in environment variables
- Or leave empty to allow any channel

### Secret Validation
- Webhook URL includes random secret path
- Optional Telegram secret header verification
- No persistent data storage

### Best Practices
- ✅ Keep repository private on GitHub
- ✅ Rotate API keys regularly
- ✅ Never commit `.env` files
- ✅ Use Render's environment variables for secrets

**⚠️ If you ever exposed your API keys** (in chat, public repo, etc.):
1. **Telegram:** Message [@BotFather](https://t.me/BotFather) → regenerate token
2. **Gemini:** Delete and create new key at [Google AI Studio](https://aistudio.google.com/apikey)
3. Update both in Render environment variables immediately

---

## 🧪 Testing

### Health Check
```bash
curl https://your-service.onrender.com/
# Response: {"ok": true}
```


---

## 📊 Technical Details

### Stack
- **Backend:** FastAPI (Python 3.13+)
- **AI:** Google Gemini 2.5 Flash
- **Deployment:** Render (free tier)
- **Webhook:** Telegram Bot API

### Dependencies
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `httpx` - Async HTTP client
- `google-genai` - Gemini AI SDK

### Performance
- ⚡ Sub-second response time (Gemini API)
- 🌐 Global CDN via Render
- 💾 Zero database overhead (stateless)
- 🔄 Auto-scales on Render

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is open source. Use responsibly and ethically.

---

## ⚠️ Disclaimer

This tool is designed for legitimate use cases like activism, community organizing, and content distribution. Users are responsible for compliance with local laws and Telegram's Terms of Service.

---

## 🆘 Troubleshooting

### Bot doesn't respond
- ✅ Check Render logs for errors
- ✅ Verify webhook is set: `curl .../getWebhookInfo`
- ✅ Ensure environment variables are set correctly
- ✅ Check if message is from allowed channel

### "403 PERMISSION_DENIED" error
- Your Gemini API key is invalid or leaked
- Generate a new key at [Google AI Studio](https://aistudio.google.com/apikey)

### Rate limit issues
- Adjust `RATE_LIMIT_SECONDS` in Render environment variables
- Default is 30 seconds per user

---

## 📞 Support

For deployment help, see [DEPLOY.md](DEPLOY.md).

For issues, check Render logs: Dashboard → Your Service → Logs tab.

