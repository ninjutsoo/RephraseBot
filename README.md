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
- **463,050+ unique variations** per message via 6-dimensional randomization
- **Dynamic AI creativity** with 3-tier temperature control (conservative to aggressive)
- **Smart preservation** of @mentions, #hashtags, URLs, and numbers
- **Nearly undetectable** as repeated content by spam filters

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
🔥 Join the event tomorrow at 5 PM! 
We need everyone at City Square. 
#YourHashtag @YourChannel
```

### Output Examples (Different Each Time):

**Variation 1** (Concise):
```
Tomorrow 5 PM - City Square. Everyone needed. 🔥
#YourHashtag @YourChannel
```

**Variation 2** (Formal):
```
We're organizing a gathering scheduled for tomorrow 
evening at 5 PM. Your presence at City Square is 
essential. 🔥 #YourHashtag @YourChannel
```

**Variation 3** (Restructured):
```
City Square tomorrow - that's where we meet at 5 PM. 
Your participation matters. 🔥 #YourHashtag @YourChannel
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
| `ALLOWED_FORWARD_CHANNEL` | (Set in code) | Channel username or ID to restrict forwarding from |
| `RATE_LIMIT_SECONDS` | `30` | Cooldown time between user requests (0 to disable) |
| `SYSTEM_INSTRUCTION` | See `main.py` | Custom AI rephrasing instructions |
| `GEMINI_MODEL` | `models/gemini-2.5-flash` | Gemini model to use |

---

## 🎨 Anti-Spam Variation Strategy

The bot uses **6-dimensional randomization** to ensure each rephrase is unique:

### Multi-Dimensional Parameters
Each message is rephrased with randomly selected:

1. **Structure** - How sentences are organized (3 options)
2. **Length** - Target output length variation (5 options)  
3. **Tone** - Formality and emotion level (6 options)
4. **Sentence Style** - Short vs long, punctuation patterns (5 options)
5. **Word Choice** - Vocabulary complexity and variety (6 options)
6. **Additional Style** - From 171 curated style variations

**Total combinations:** 3 × 5 × 6 × 5 × 6 × 171 = **463,050 variations**

### AI Randomness Levels
- **Conservative** (temp: 0.6-0.8) - Safe, readable changes
- **Moderate** (temp: 0.8-1.1) - Balanced creativity
- **Aggressive** (temp: 1.1-1.4) - Maximum variation

### Protection Mechanism
Critical elements (@mentions, #hashtags, URLs, numbers) are:
1. Masked as `__PROTECTED_0__`, `__PROTECTED_1__`, etc.
2. Sent safely through AI rephrasing
3. Restored perfectly in the output

**Result:** Nearly impossible for spam filters to detect repeated messages.

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
Restrict the bot to only process messages from specific channels.

**Configure:**
- Set `ALLOWED_FORWARD_CHANNEL` in code or environment variables
- Use channel username (without @) or channel ID
- Leave empty to allow any channel

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

