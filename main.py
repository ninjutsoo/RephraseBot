import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request

# Gemini (Google AI / google-genai)
from google import genai

try:
    # Newer google-genai versions
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    types = None  # type: ignore


app = FastAPI()

# Required env vars
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]  # random string you choose

# Optional hardening: Telegram secret header check
TELEGRAM_WEBHOOK_SECRET_TOKEN = os.environ.get("TELEGRAM_WEBHOOK_SECRET_TOKEN")

# Optional: Only rephrase messages forwarded from a specific channel
# Can be channel username (e.g., "mychannel") or channel ID (e.g., "-1001234567890")
ALLOWED_FORWARD_CHANNEL = os.environ.get("ALLOWED_FORWARD_CHANNEL", "tweeterstormIranrevolution2026")

# Rate limiting: seconds a user must wait between requests
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "30"))

# In-memory storage for rate limiting (user_id -> last_request_timestamp)
user_last_request: Dict[int, float] = {}

SYSTEM_INSTRUCTION = os.environ.get(
    "SYSTEM_INSTRUCTION",
    "You are an assistant for my Telegram bot.\n"
    "Task: rewrite the forwarded message in a different wording each time.\n"
    "Hard rules:\n"
    "- Preserve meaning.\n"
    "- Preserve ALL @mentions, #hashtags, URLs, and numbers EXACTLY.\n"
    "- Preserve names / proper nouns; if unsure, keep them unchanged.\n"
    "- Preserve line breaks and overall structure.\n"
    "- Output ONLY the rewritten message (no preface).",
)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


STYLES: List[str] = [
    "slightly more formal",
    "slightly more casual",
    "more concise",
    "slightly more detailed but not longer than the original",
    "active voice where possible",
    "neutral tone, newsroom style",
    "friendly tone",
    "direct and punchy",
    "calm and diplomatic",
    "empathetic but concise",
    "confident tone without exaggeration",
    "avoid filler words",
    "avoid clichés",
    "avoid slang",
    "lightly conversational",
    "professional tone",
    "simple everyday language",
    "clear and structured",
    "keep rhythm similar to original",
    "keep punctuation pattern similar to original",
    "keep sentence count similar to original",
    "use synonyms where safe",
    "reorder clauses when safe",
    "slightly more persuasive",
    "slightly more neutral",
    "slightly more assertive",
    "slightly softer/less harsh",
    "sound like a helpful moderator",
    "sound like a clear announcer",
    "sound like a careful editor",
    "sound like a pragmatic planner",
    "sound like a concise spokesperson",
    "sound like a friendly teammate",
    "sound like a calm advisor",
    "sound like a factual reporter",
    "sound like a polite reminder",
    "sound like a brief update",
    "sound like a short note",
    "sound like a message to a group chat",
    "avoid changing emphasis",
    "keep emotional intensity the same",
    "keep urgency level the same",
    "keep hedging/certainty level the same",
    "avoid changing intent",
    "avoid adding new claims",
    "avoid removing any key detail",
    "prefer shorter words",
    "prefer clearer verbs",
    "avoid repeating the same word",
    "vary sentence starters",
    "smooth transitions",
    "tighten phrasing",
    "remove redundancy",
    "slightly more polished",
    "slightly more raw/unfiltered but respectful",
    "avoid passive constructions",
    "use mild parallelism",
    "keep formatting as-is (bullets/lines)",
    "keep emojis exactly as-is",
    "keep capitalization pattern similar",
    "keep abbreviations unchanged",
    "keep quoted text unchanged",
    "keep parentheticals unchanged",
    "keep dates unchanged",
    "keep times unchanged",
    "keep locations unchanged (do not rename)",
    "keep organization names unchanged",
    "keep person names unchanged",
    "keep product names unchanged",
    "keep usernames unchanged",
    "keep hashtags unchanged",
    "keep links unchanged",
    "keep numbers unchanged",
    "prefer simple punctuation",
    "avoid extra exclamation points",
    "avoid extra question marks",
    "do not add emojis",
    "do not add hashtags",
    "do not add @mentions",
    "avoid rhetorical questions",
    "avoid moralizing",
    "avoid inflammatory language",
    "avoid charged adjectives",
    "slightly more respectful",
    "slightly more optimistic",
    "slightly more cautious",
    "slightly more urgent",
    "slightly more measured",
    "clarify references (but don’t add info)",
    "avoid ambiguity (but keep meaning)",
    "keep the same point of view",
    "keep the same tense",
    "keep the same ordering of key points",
    "keep the same call-to-action",
    "make it easier to read quickly",
    "make it sound more natural",
    "avoid uncommon words",
    "avoid technical jargon",
    "use a slightly more modern phrasing",
    "use a slightly more traditional phrasing",
    "use a slightly more journalistic phrasing",
    "use a slightly more conversational phrasing",
    "use a slightly more formal phrasing",
    "use a slightly more concise phrasing",
    "use a slightly more vivid phrasing (without adding details)",
    "use stronger verbs (without changing meaning)",
    "swap clause order to vary rhythm",
    "tighten opener",
    "tighten closer",
    "reduce hedges if present (without changing certainty)",
    "keep hedges if present",
    "avoid intensifiers (very/really)",
    "keep line breaks exactly",
    "keep bullet markers exactly",
    "keep the first and last sentence structure similar",
    "keep short sentences short",
    "break run-on sentences",
    "combine choppy sentences",
    "ensure it reads like a native speaker",
    "avoid repetitive phrasing",
    "use more concrete wording (without adding facts)",
    "use more neutral wording (without losing intent)",
]


PROTECTED_PATTERN = re.compile(
    r"("  # capture each token
    r"https?://\S+|"  # URLs
    r"t\.me/\S+|"  # Telegram short links
    r"@\w{1,64}|"  # @mentions
    r"#\w+|"  # hashtags
    r"\$[A-Za-z]{2,}|"  # cashtags-ish
    r"\b\d+(?:[.,:/-]\d+)*\b"  # numbers/dates/times-ish
    r")",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MaskedText:
    masked: str
    placeholders: List[Tuple[str, str]]  # (placeholder, original)


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted(spans, key=lambda s: (s[0], s[1]))
    if not ordered:
        return []
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for s, e in ordered[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def mask_protected(text: str) -> MaskedText:
    spans: List[Tuple[int, int]] = [(m.start(), m.end()) for m in PROTECTED_PATTERN.finditer(text)]
    spans = _merge_spans(spans)
    if not spans:
        return MaskedText(masked=text, placeholders=[])

    out: List[str] = []
    placeholders: List[Tuple[str, str]] = []
    last = 0
    for i, (s, e) in enumerate(spans):
        out.append(text[last:s])
        ph = f"__PROTECTED_{i}__"
        out.append(ph)
        placeholders.append((ph, text[s:e]))
        last = e
    out.append(text[last:])
    return MaskedText(masked="".join(out), placeholders=placeholders)


def unmask(text: str, placeholders: List[Tuple[str, str]]) -> str:
    out = text
    for ph, original in placeholders:
        out = out.replace(ph, original)
    return out


def contains_all_placeholders(text: str, placeholders: List[Tuple[str, str]]) -> bool:
    return all(ph in text for ph, _ in placeholders)


def is_forwarded(message: dict) -> bool:
    # Telegram has multiple forward-related fields depending on API version.
    return any(
        key in message
        for key in (
            "forward_origin",
            "forward_from",
            "forward_sender_name",
            "forward_from_chat",
            "forward_from_message_id",
            "forward_signature",
            "forward_date",
        )
    )


def is_forwarded_from_allowed_channel(message: dict) -> bool:
    """Check if message is forwarded from the allowed channel (if configured)."""
    if not ALLOWED_FORWARD_CHANNEL:
        # No restriction - allow all forwarded messages
        return True
    
    # Check forward_origin (newer API format)
    forward_origin = message.get("forward_origin")
    if forward_origin:
        if forward_origin.get("type") == "channel":
            chat = forward_origin.get("chat", {})
            channel_id = str(chat.get("id", ""))
            channel_username = chat.get("username", "").lower()
            
            # Match by ID or username
            allowed = ALLOWED_FORWARD_CHANNEL.lower()
            if allowed.startswith("-") or allowed.isdigit():
                # Matching by ID
                return channel_id == ALLOWED_FORWARD_CHANNEL
            else:
                # Matching by username (without @ prefix)
                allowed = allowed.lstrip("@")
                return channel_username == allowed
    
    # Check forward_from_chat (older API format)
    forward_from_chat = message.get("forward_from_chat")
    if forward_from_chat and forward_from_chat.get("type") == "channel":
        channel_id = str(forward_from_chat.get("id", ""))
        channel_username = forward_from_chat.get("username", "").lower()
        
        allowed = ALLOWED_FORWARD_CHANNEL.lower()
        if allowed.startswith("-") or allowed.isdigit():
            return channel_id == ALLOWED_FORWARD_CHANNEL
        else:
            allowed = allowed.lstrip("@")
            return channel_username == allowed
    
    return False


def check_rate_limit(user_id: int) -> Optional[int]:
    """
    Check if user is rate limited.
    Returns None if allowed, or seconds remaining if rate limited.
    """
    if RATE_LIMIT_SECONDS <= 0:
        # Rate limiting disabled
        return None
    
    current_time = time.time()
    last_request_time = user_last_request.get(user_id)
    
    if last_request_time is None:
        # First request from this user
        return None
    
    time_elapsed = current_time - last_request_time
    
    if time_elapsed < RATE_LIMIT_SECONDS:
        # Still rate limited
        seconds_remaining = int(RATE_LIMIT_SECONDS - time_elapsed) + 1
        return seconds_remaining
    
    # Rate limit expired
    return None


def update_rate_limit(user_id: int) -> None:
    """Record the current timestamp for this user."""
    user_last_request[user_id] = time.time()


async def telegram_send_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
        "allow_sending_without_reply": True,
    }
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()


def build_prompt(masked_text: str, style: str) -> str:
    # Simpler prompt: ask for 1 complete rewrite to avoid truncation
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "Important: The text contains placeholders like __PROTECTED_0__. "
        "You MUST keep every placeholder EXACTLY unchanged and in the correct position.\n\n"
        f"Rewrite the following text with this style: {style}.\n\n"
        "Output ONLY the rewritten text, nothing else. Do NOT truncate or cut off mid-sentence.\n\n"
        "Text:\n"
        f"{masked_text}"
    )


def _parse_response(text: str) -> List[str]:
    # Since we now ask for 1 rewrite (not 5), just return the full text as a single candidate
    cleaned = text.strip()
    if not cleaned:
        return []
    return [cleaned]


def gemini_generate_candidates(prompt: str) -> List[str]:
    # Randomness knobs (if supported by the installed google-genai version)
    temperature = random.uniform(0.65, 0.95)
    top_p = random.uniform(0.85, 0.95)

    kwargs = {}
    if types is not None:
        try:
            kwargs["config"] = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=2048,  # Increased to allow longer responses
            )
        except Exception:
            # Fall back if signature differs in this installed version.
            kwargs = {}

    # Retry logic for rate limits (429)
    import time
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                **kwargs,
            )
            text = (getattr(resp, "text", None) or "").strip()
            return _parse_response(text)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < 2:
                    wait_time = (2 ** attempt) * 2  # 2s, 4s
                    time.sleep(wait_time)
                    continue
            raise  # Re-raise if not a rate limit or final attempt


@app.get("/")
def health():
    return {"ok": True}


@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def webhook(req: Request):
    # Optional hardening: verify Telegram secret header (if you set it during setWebhook)
    if TELEGRAM_WEBHOOK_SECRET_TOKEN:
        header = req.headers.get("x-telegram-bot-api-secret-token")
        if header != TELEGRAM_WEBHOOK_SECRET_TOKEN:
            return {"ok": True}

    update = await req.json()

    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True}

    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return {"ok": True}

    # Prefer text; fall back to caption (forwarded media captions).
    user_text = message.get("text") or message.get("caption") or ""
    if not user_text.strip():
        await telegram_send_message(chat_id, "Forward a text message (or a media caption) to rephrase it.")
        return {"ok": True}

    # Simple commands
    if user_text.strip().lower() in ("/start", "start"):
        await telegram_send_message(chat_id, "Bot is running. Forward a message to rephrase it.")
        return {"ok": True}

    # Only act on forwarded messages (per your requirement)
    if not is_forwarded(message):
        await telegram_send_message(chat_id, "Please forward a message to me, and I'll rephrase it.")
        return {"ok": True}
    
    # Check if message is from allowed channel (if configured)
    if not is_forwarded_from_allowed_channel(message):
        if ALLOWED_FORWARD_CHANNEL:
            await telegram_send_message(
                chat_id, 
                f"⚠️ This bot only works with messages forwarded from @{ALLOWED_FORWARD_CHANNEL}.\n\n"
                f"Please forward a message from that channel to use this bot."
            )
        return {"ok": True}
    
    # Check rate limit (use from_user.id for user-specific limiting)
    user_id = (message.get("from") or {}).get("id")
    if user_id:
        seconds_remaining = check_rate_limit(user_id)
        if seconds_remaining is not None:
            await telegram_send_message(
                chat_id,
                f"⏱️ Please wait {seconds_remaining} second{'s' if seconds_remaining != 1 else ''} before sending another message.\n\n"
                f"Rate limit: 1 message per {RATE_LIMIT_SECONDS} seconds."
            )
            return {"ok": True}

    masked = mask_protected(user_text)
    style = random.choice(STYLES)
    prompt = build_prompt(masked.masked, style=style)

    try:
        candidates = gemini_generate_candidates(prompt)
        print(f"DEBUG: Got {len(candidates)} candidates: {candidates}")
        if not candidates:
            raise RuntimeError("no_candidates")

        # Prefer candidates that preserved all placeholders so we don't corrupt tags/links.
        good = [c for c in candidates if contains_all_placeholders(c, masked.placeholders)]
        print(f"DEBUG: {len(good)} good candidates (with all placeholders)")
        chosen = random.choice(good or candidates).strip()
        print(f"DEBUG: Chosen candidate: {chosen}")

        rewritten = unmask(chosen, masked.placeholders).strip()
        print(f"DEBUG: After unmask: {rewritten}")
        
        # Safety check: if rewritten is suspiciously shorter than original (likely truncated), use original
        if not rewritten or len(rewritten) < len(user_text) * 0.6:
            print(f"DEBUG: Response too short ({len(rewritten)} vs {len(user_text)}), using original")
            rewritten = user_text.strip()

        # Telegram message limit safety
        if len(rewritten) > 3500:
            rewritten = rewritten[:3500] + "\n\n[truncated]"

        await telegram_send_message(chat_id, rewritten)
        
        # Update rate limit timestamp after successful processing
        if user_id:
            update_rate_limit(user_id)

    except Exception as exc:
        import traceback
        traceback.print_exc()  # Log to console
        await telegram_send_message(chat_id, f"Error: {type(exc).__name__} - {str(exc)[:100]}")

    return {"ok": True}

