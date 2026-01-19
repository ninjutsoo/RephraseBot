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
# Set to 0 to disable rate limiting
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "0"))

# In-memory storage for rate limiting (user_id -> last_request_timestamp)
user_last_request: Dict[int, float] = {}

SYSTEM_INSTRUCTION = os.environ.get(
    "SYSTEM_INSTRUCTION",
    "You are an assistant that rephrases messages to avoid spam detection.\n\n"
    "CRITICAL HARD LIMIT: The final output MUST be under 280 characters.\n"
    "If the message is long, you MUST condense it, remove filler, and summarize to fit.\n"
    "NEVER exceed 280 characters.\n\n"
    "CORE GOAL: The rephrased message must convey the EXACT same meaning but be "
    "structurally and stylistically DIFFERENT from the original.\n\n"
    "MENTION/TAG HANDLING (CRITICAL):\n"
    "- Do NOT change the role of @mentions. If a mention is used as a tag (e.g. at start/end), keep it as a tag.\n"
    "- IF a message starts with a block of @mentions, KEEP them at the start. Do not weave them into the sentence.\n"
    "- IF a message ends with a block of tags/mentions, KEEP them at the end.\n"
    "- Do NOT make a mention the subject/actor if it wasn't one (e.g. don't change '@POTUS...' to '@POTUS said...').\n"
    "- Keep mentions and hashtags in their approximate original positions (start vs end) if possible, or integrate naturally without changing meaning.\n\n"
    "MUST PRESERVE EXACTLY (these will be marked as __PROTECTED_N__):\n"
    "- All @mentions, #hashtags, URLs, and numbers\n"
    "- Names of people, organizations, locations\n\n"
    "MUST PRESERVE:\n"
    "- Core message and meaning\n"
    "- Key facts and claims\n"
    "- Call-to-action intent\n\n"
    "HARD LIMIT:\n"
    "- The final output MUST be under 280 characters (Twitter/X limit).\n"
    "- If the original is longer, condense it intelligently while keeping key info.\n\n"
    "MUST VARY SIGNIFICANTLY:\n"
    "- Sentence structure (reorder, combine, split sentences)\n"
    "- Word choice (use synonyms, different phrasing)\n"
    "- Paragraph structure (can reorder if logical)\n"
    "- Sentence length and rhythm\n"
    "- Transitional phrases\n"
    "- Opening and closing\n\n"
    "ALLOWED MODIFICATIONS:\n"
    "- Remove filler words and redundancy\n"
    "- Add natural connecting phrases\n"
    "- Expand brief points with natural elaboration (without adding new facts)\n"
    "- Compress verbose sections\n"
    "- Change passive to active voice or vice versa\n"
    "- Vary punctuation style\n"
    "- Change from questions to statements or vice versa (if meaning preserved)\n\n"
    "OUTPUT: Only the rewritten text, no preamble or explanation.",
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


def build_prompt(masked_text: str, style: str, force_short: bool = False) -> str:
    # Multi-dimensional randomization for better spam evasion
    
    # Random structural changes
    structure = random.choice([
        "significantly restructure (reorder sentences/paragraphs if logical)",
        "moderately restructure (reorder some clauses)",
        "minimal restructure (keep mostly same order)"
    ])
    
    # Random length variation
    # Logic: If strict shortening needed (retry) OR original is close to limit, force shorter options.
    if force_short:
        length = "CRITICAL: MAKE IT SHORTER. Remove filler words. Condense sentences. STRICTLY UNDER 280 CHARS."
    elif len(masked_text) > 240:
        # Original is already close to limit, so NEVER ask to make it longer
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "compress into fewer sentences",
            "condense strictly to fit under 280 chars",
            "keep similar length but vary sentence lengths"
        ])
    else:
        # Original is short enough to allow expansion
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "make 10-20% longer by expanding key points naturally",
            "keep similar length but vary sentence lengths",
            "compress into fewer sentences",
            "break into more sentences"
        ])
    
    # Random tone
    tone = random.choice([
        "very formal and academic",
        "casual and friendly",
        "urgent and direct",
        "calm and measured",
        "enthusiastic and energetic",
        "matter-of-fact neutral"
    ])
    
    # Random sentence structure
    sentence_style = random.choice([
        "use short, punchy sentences",
        "use longer, flowing sentences",
        "mix short and long sentences",
        "start sentences differently each time",
        "vary punctuation patterns"
    ])
    
    # Random word strategy
    word_strategy = random.choice([
        "maximize synonym usage",
        "prefer simpler/common words",
        "prefer more sophisticated vocabulary",
        "mix formal and informal words",
        "use more action verbs",
        "use more descriptive adjectives (sparingly)"
    ])
    
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        "Important: The text contains placeholders like __PROTECTED_0__. "
        "You MUST keep every placeholder EXACTLY unchanged and in the correct position.\n\n"
        "Rewrite with these requirements:\n"
        f"- Structure: {structure}\n"
        f"- Length: {length}\n"
        f"- Tone: {tone}\n"
        f"- Sentence style: {sentence_style}\n"
        f"- Word choice: {word_strategy}\n"
        f"- Additional style: {style}\n"
        "- MAX LENGTH: 280 characters (strict)\n\n"
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
    # Multi-level randomness strategy for better variation
    randomness_level = random.choice(['conservative', 'moderate', 'aggressive'])
    
    if randomness_level == 'conservative':
        temperature = random.uniform(0.6, 0.8)
        top_p = random.uniform(0.85, 0.95)
    elif randomness_level == 'moderate':
        temperature = random.uniform(0.8, 1.1)
        top_p = random.uniform(0.8, 0.95)
    else:  # aggressive
        temperature = random.uniform(1.1, 1.4)
        top_p = random.uniform(0.7, 0.9)

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
    # Prompt is built inside the loop to allow modification on retry

    try:
        rewritten = ""
        
        # Retry logic: Attempt 1 (Standard) -> Check -> Attempt 2 (Strict Shortening)
        max_attempts = 2
        
        for attempt_idx in range(max_attempts):
            # If this is a retry (attempt_idx > 0), force strict shortening
            force_short = (attempt_idx > 0)
            
            current_prompt = build_prompt(masked.masked, style=style, force_short=force_short)
            
            if force_short:
                print(f"DEBUG: Output too long ({len(rewritten)} chars). Retrying with strict length constraint.")
                current_prompt += (
                    f"\n\nSYSTEM ALERT: Your previous output was {len(rewritten)} characters long."
                    "\nMAXIMUM ALLOWED IS 280 CHARACTERS."
                    "\nREWRITE IT NOW TO BE SIGNIFICANTLY SHORTER."
                )
            
            candidates = gemini_generate_candidates(current_prompt)
            print(f"DEBUG: Got {len(candidates)} candidates")
            
            if not candidates:
                if attempt_idx == max_attempts - 1:
                    raise RuntimeError("no_candidates")
                continue

            # Prefer candidates that preserved all placeholders so we don't corrupt tags/links.
            good = [c for c in candidates if contains_all_placeholders(c, masked.placeholders)]
            chosen = random.choice(good or candidates).strip()
            
            temp_rewritten = unmask(chosen, masked.placeholders).strip()
            
            if len(temp_rewritten) <= 280:
                rewritten = temp_rewritten
                print(f"DEBUG: Success on attempt {attempt_idx+1} ({len(rewritten)} chars)")
                break
            
            # Keep the best effort so far
            rewritten = temp_rewritten
        
        # If still over 280 after retries, log warning
        if len(rewritten) > 280:
            print(f"WARNING: Failed to get under 280 chars after {max_attempts} attempts. Final length: {len(rewritten)}")

        print(f"DEBUG: Final output: {rewritten}")
        
        # Relaxed safety check: Allow 30-200% of original length for variation
        # EXCEPTION: If original > 280 and rewritten <= 280, allow it even if < 30% (successful compression)
        min_length = len(user_text) * 0.3
        max_length = len(user_text) * 2.0
        
        is_compression_attempt = len(user_text) > 280
        is_successful_compression = (is_compression_attempt and len(rewritten) <= 280)
        
        # If we are trying to compress for X, accept the result if it's shorter than original
        # and at least 10 chars long, ignoring the 30% min_length rule.
        if is_compression_attempt:
             if len(rewritten) < 10:
                 print("WARNING: Rewrite too short/empty, using original")
                 rewritten = user_text.strip()
             # Else: Keep rewritten (even if 290 chars, better than 1000)
        
        elif (not rewritten or len(rewritten) < min_length):
            # Only fallback if truly broken (less than 30% of original)
            print(f"WARNING: Rewrite too short ({len(rewritten)} vs {len(user_text)}), using original")
            rewritten = user_text.strip()
        elif len(rewritten) > max_length:
            # If way too long, might be hallucination - use original
            print(f"WARNING: Rewrite too long ({len(rewritten)} vs {len(user_text)}), using original")
            rewritten = user_text.strip()

        # Telegram message limit safety
        if len(rewritten) > 280:
             print(f"WARNING: Output exceeded 280 chars ({len(rewritten)}), truncating/checking.")
             # We can't easily truncate without cutting words, so we'll leave it 
             # but log it. Or strictly cut it if you prefer.
             # For now, let's trust the AI but add a visual warning if testing manually.
             pass

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

