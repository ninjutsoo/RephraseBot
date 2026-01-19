# FreeIranXBot (Telegram → FastAPI webhook → Gemini)

This bot **rephrases forwarded messages**. It aims to keep **names/tags/links/numbers the same** while changing the wording each time.

## What it does

- If you **forward a message** to the bot, it replies with a **rephrased version**.
- If you send a **non-forwarded** message, it asks you to forward one.
- It masks and preserves these tokens exactly:
  - `@mentions`
  - `#hashtags`
  - URLs (including `t.me/...`)
  - numbers/dates/times-like tokens
- It asks Gemini for **5 rewrites** in a randomly chosen style, then **randomly picks 1**, so you get lots of variety over repeated forwards.

## Security note (important)

If you ever pasted your **Telegram bot token** or **Gemini API key** into chat or a repo, **rotate them immediately**:
- **Telegram**: talk to `@BotFather` → regenerate token.
- **Gemini**: delete/regenerate the API key in Google AI Studio / Google Cloud console.

Never commit secrets to git. Use Render environment variables.

## Local run

1. Create a local env file (don’t commit it). Use `env.example` as a template.
2. Install and run:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Environment variables

Required:
- **`TELEGRAM_BOT_TOKEN`**
- **`GEMINI_API_KEY`**
- **`WEBHOOK_SECRET`** (random long string; used in the URL path)

Optional but recommended:
- **`TELEGRAM_WEBHOOK_SECRET_TOKEN`**: if set, the server will verify the request header
  `X-Telegram-Bot-Api-Secret-Token` (Telegram can send this if you set it in `setWebhook`).
- **`SYSTEM_INSTRUCTION`**: extra constraints for rewriting behavior.
- **`GEMINI_MODEL`**: defaults to `gemini-2.0-flash`.

## Deploy on Render

This repo includes `render.yaml`.

1. Push to GitHub.
2. On Render: New → Web Service → connect repo.
3. Add environment variables (Render → Service → Environment):
   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `WEBHOOK_SECRET`
   - (optional) `TELEGRAM_WEBHOOK_SECRET_TOKEN`
4. Deploy. You’ll get a URL like `https://YOUR-SERVICE.onrender.com`.

## Set Telegram webhook

Webhook endpoint:
- `https://YOUR-SERVICE.onrender.com/webhook/<WEBHOOK_SECRET>`

### Basic

```bash
curl -s "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
  -d "url=https://YOUR-SERVICE.onrender.com/webhook/<WEBHOOK_SECRET>"
```

### Recommended (adds secret header verification)

```bash
curl -s "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook" \
  -d "url=https://YOUR-SERVICE.onrender.com/webhook/<WEBHOOK_SECRET>" \
  -d "secret_token=<TELEGRAM_WEBHOOK_SECRET_TOKEN>"
```

Verify:

```bash
curl -s "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo"
```

