## Cursor Cloud specific instructions

**RephraseBot** is a single-file FastAPI Telegram bot (`main.py`) that rephrases text using Google Gemini AI. See `README.md` for full feature documentation.

### Required environment variables

The app crashes on import if these are missing (uses `os.environ["..."]`):
- `TELEGRAM_BOT_TOKEN` — Telegram bot token
- `GEMINI_API_KEY` — Google Gemini API key
- `WEBHOOK_SECRET` — random string for webhook URL path

For local dev without real credentials, set dummy values:
```sh
export TELEGRAM_BOT_TOKEN="test_token" GEMINI_API_KEY="test_key" WEBHOOK_SECRET="dev_secret"
```
The health check (`GET /`) works with dummy tokens. Webhook calls (`POST /webhook/{WEBHOOK_SECRET}`) will fail on the Telegram API call but the request parsing and routing logic still runs.

### Running the dev server

```sh
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```
- Health check: `curl http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`

### Linting

No lint config is committed. Use `python3 -m ruff check main.py` for quick lint checks. Pre-existing issues exist (bare `except`, f-strings without placeholders).

### Tests

No automated test suite exists in the repo.

### Gotchas

- `uvicorn` is installed to the user site-packages; use `python3 -m uvicorn` instead of bare `uvicorn`.
- Supabase is optional — the app runs fine without `SUPABASE_URL`/`SUPABASE_KEY` (logs a warning on startup).
- On startup, the app calls the Telegram API to register bot commands. With dummy tokens this "succeeds" silently (Telegram returns an error but the app continues).
