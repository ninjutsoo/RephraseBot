# RephraseBot 🔄

**A professional utility for generating natural variations of text content to ensure message diversity while preserving critical elements like links and tags.**

Built with FastAPI, Google Gemini AI, and designed for efficient cloud deployment.

---

## 🎯 Purpose

This bot helps you share the same message multiple times with unique phrasing to maintain engagement and visibility. Perfect for:
- 📢 Multi-channel announcements
- 📣 Community updates
- 🔁 Diverse content distribution
- ✍️ Creating natural variations of standard messages

---

## ✨ Key Features

### 🛡️ **Advanced Variation Technology**
- **463,050+ unique variations** per message via 6-dimensional randomization
- **Dynamic AI creativity** with 3-tier temperature control (conservative to aggressive)
- **Smart preservation** of @mentions, #hashtags, URLs, and numbers
- **Nearly undetectable** as repeated content by standard pattern matching

### 🎨 **Intelligent Rephrasing**
- ✅ Preserves core meaning and facts
- ✅ Maintains all mentions, hashtags, and links
- ✅ **Strict 280-character limit** for platform compatibility
- ✅ Varies sentence structure, word choice, and length
- ✅ Changes tone and rhythm each time
- ✅ Natural-sounding output

### 🔐 **Security & Control**
- **Channel filtering**: Restrict processing to specific authorized sources
- **Rate limiting**: Optional cooldown per user (disabled by default)
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

---

## 📂 Project Structure

```
RephraseBot/
├── main.py              # Core bot logic and FastAPI app
├── requirements.txt     # Python dependencies
├── render.yaml          # Cloud deployment configuration
├── env.example          # Environment variables template
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── DEPLOY.md           # Step-by-step deployment guide
```

---

## ⚙️ Configuration

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Get from [@BotFather](https://t.me/BotFather) on Telegram |
| `GEMINI_API_KEY` | Get from [AI Provider Console](https://aistudio.google.com/apikey) |
| `WEBHOOK_SECRET` | Random string used in webhook URL path (e.g., `whk_yourRandomString`) |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_WEBHOOK_SECRET_TOKEN` | None | Secret token for webhook verification |
| `ALLOWED_FORWARD_CHANNEL` | None | Channel username or ID to restrict forwarding from |
| `RATE_LIMIT_SECONDS` | `0` | Cooldown time between user requests (0 = disabled) |
| `SYSTEM_INSTRUCTION` | See `main.py` | Custom AI rephrasing instructions |
| `GEMINI_MODEL` | `models/gemini-2.0-flash` | AI model to use |

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

---

## 🛡️ Rate Limiting (Optional)

**Default:** Disabled (0 seconds)

To enable rate limiting, set `RATE_LIMIT_SECONDS` to your desired cooldown (e.g., `30` for 30 seconds).

---

## 🔐 Security Features

### Channel Filtering
Restrict the bot to only process messages from specific channels.

**Configure:**
- Set `ALLOWED_FORWARD_CHANNEL` in environment variables
- Use channel username (without @) or channel ID
- Leave empty to allow any channel

### Best Practices
- ✅ Keep repository private on GitHub
- ✅ Rotate API keys regularly
- ✅ Never commit `.env` files
- ✅ Use secure environment variables for secrets

**⚠️ If you ever exposed your API keys** (in chat, public repo, etc.):
1. **Telegram:** Message [@BotFather](https://t.me/BotFather) → regenerate token
2. **Gemini:** Delete and create new key at [AI Provider Console](https://aistudio.google.com/apikey)
3. Update environment variables immediately

---

## 🧪 Testing

### Health Check
```bash
curl https://your-service-url.com/
# Response: {"ok": true}
```

---

## ⚠️ Disclaimer

This tool is designed for legitimate use cases. Users are responsible for compliance with local laws and platform Terms of Service.
