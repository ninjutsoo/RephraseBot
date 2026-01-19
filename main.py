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
ALLOWED_FORWARD_CHANNEL = os.environ.get("ALLOWED_FORWARD_CHANNEL")

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
    "CORE GOAL: The rephrased message must convey the EXACT same meaning/intent but be "
    "structurally and stylistically DIFFERENT from the original.\n\n"
    "NUANCE PRESERVATION (CRITICAL):\n"
    "- Preserve the underlying intent and meaning exactly.\n"
    "- Ensure the subject-object relationship remains the same (who did what to whom).\n"
    "- If a phrase is metaphorical, keep the metaphor's meaning, not just the words.\n"
    "- You MAY change the tone/style/vocabulary as requested, provided the core meaning remains accurate.\n\n"
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

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")

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


def extract_tag_blocks(text: str) -> Tuple[str, str, str]:
    """
    Separates a message into (start_tags, content, end_tags).
    start_tags: continuous block of mentions/hashtags at the very beginning
    end_tags: continuous block of mentions/hashtags at the very end
    content: the middle part to be rephrased
    
    Example:
    "@User1 Hi there #tag" -> ("@User1 ", "Hi there", " #tag")
    """
    # Tokenize by splitting on whitespace, but keep delimiters to reconstruct
    # Actually, simpler: Use regex to find prefix/suffix blocks
    
    # Pattern for a tag line/block:
    # A block is a sequence of (@mention|#hashtag) separated by whitespace
    
    # We'll tokenize and peel off from start and end
    tokens = text.split()
    if not tokens:
        return "", "", ""
        
    def is_tag(t):
        return t.startswith("@") or t.startswith("#")
        
    # Find start block
    start_idxs = []
    for i, t in enumerate(tokens):
        if is_tag(t):
            start_idxs.append(i)
        else:
            break
            
    # Find end block
    end_idxs = []
    for i in range(len(tokens) - 1, -1, -1):
        if i < len(start_idxs): # Don't overlap
            break
        if is_tag(tokens[i]):
            end_idxs.append(i)
        else:
            break
    end_idxs.reverse()
    
    # Reconstruct strings based on original text is tricky with simple split.
    # Better strategy: Regex match for "leading tags" and "trailing tags"
    
    # Regex to match a tag plus following whitespace
    # ^(\s*([@#][\w\d_]+)\s+)*
    
    # Let's try a safer index-based slice on the original string
    # We need to identify the character boundaries of the content.
    
    # 1. Leading tags
    # Match start of string, optional whitespace, then repeated (tag + whitespace)
    leading_pattern = re.compile(r"^\s*((?:[@#][\w\d_]+\s*)+)", re.DOTALL)
    trailing_pattern = re.compile(r"((?:\s*[@#][\w\d_]+)+)\s*$", re.DOTALL)
    
    start_block = ""
    end_block = ""
    content = text
    
    m_start = leading_pattern.match(text)
    if m_start:
        start_block = m_start.group(1)
        content = text[m_start.end():]
        
    m_end = trailing_pattern.search(content)
    if m_end:
        # Check if the whole remaining content is tags (overlap case)
        # If content is empty or just whitespace, we shouldn't strip end too
        if not content.strip():
             # Already captured in start, or effectively empty
             pass
        else:
            end_block = m_end.group(1)
            content = content[:m_end.start()]
            
    return start_block, content, end_block


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
    """
    Swaps placeholders back to original tags, ensuring they don't
    touch adjacent words by inserting a space only when necessary.
    
    This works on PLACEHOLDERS before unmasking to avoid corrupting the tags themselves.
    """
    for ph, original in placeholders:
        # 1. If there's a letter/number directly BEFORE the placeholder, add a space
        # Example: "Thanks__PROTECTED_0__" -> "Thanks __PROTECTED_0__"
        text = re.sub(rf"([a-zA-Z0-9])({re.escape(ph)})", r"\1 \2", text)

        # 2. If there's a letter/number directly AFTER the placeholder, add a space
        # Example: "__PROTECTED_0__thanks" -> "__PROTECTED_0__ thanks"
        text = re.sub(rf"({re.escape(ph)})([a-zA-Z0-9])", r"\1 \2", text)

        # Perform the actual replacement
        text = text.replace(ph, original)

    return text


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


def extract_tweet_id(text: str) -> Optional[str]:
    """
    Extract tweet ID from X/Twitter URLs.
    Supports formats:
    - https://twitter.com/username/status/1234567890
    - https://x.com/username/status/1234567890
    - https://mobile.twitter.com/username/status/1234567890
    """
    pattern = r'(?:twitter\.com|x\.com|mobile\.twitter\.com)/(?:\w+)/status/(\d+)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


async def telegram_send_message(chat_id: int, text: str, reply_markup: Optional[dict] = None) -> None:
    """
    Send a message to a Telegram chat.
    
    Args:
        chat_id: The chat ID to send to
        text: The message text
        reply_markup: Optional inline keyboard markup
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
        "allow_sending_without_reply": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup
    
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(url, json=payload)
        r.raise_for_status()


def build_prompt(masked_text: str, style: str, force_short: bool = False, max_chars: int = 280) -> str:
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
        length = f"CRITICAL: MAKE IT SHORTER. Remove filler words. Condense sentences. STRICTLY UNDER {max_chars} CHARS."
    elif len(masked_text) > (max_chars - 30):
        # Very close to limit: ONLY allow shortening
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "compress into fewer sentences",
            f"condense strictly to fit under {max_chars} chars"
        ])
    elif len(masked_text) > (max_chars - 60):
        # Moderately close: Allow shortening or keeping same length
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "compress into fewer sentences",
            f"condense strictly to fit under {max_chars} chars",
            "keep similar length but vary sentence lengths"
        ])
    elif len(masked_text) < 120:
        # Very short: Prioritize expansion or keeping same length
        length = random.choice([
            f"make 20-40% longer by expanding key points naturally (BUT UNDER {max_chars} CHARS)",
            "add relevant context or adjectives to make it more descriptive",
            "keep similar length but vary sentence lengths",
            "break into more sentences"
        ])
    else:
        # Safe zone (120 - 220 chars): Full variety
        length = random.choice([
            "make 20-30% shorter by removing filler",
            f"make 10-20% longer by expanding key points naturally (BUT UNDER {max_chars} CHARS)",
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
        f"- MAX LENGTH: {max_chars} characters (strict)\n\n"
        "Output ONLY the rewritten text, nothing else. Do NOT truncate or cut off mid-sentence.\n\n"
        "Text:\n"
        f"{masked_text}\n\n"
        f"REMINDER: OUTPUT MUST BE UNDER {max_chars} CHARACTERS."
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
        # Reduced max from 1.4 to 1.25 to improve instruction following (length limits)
        temperature = random.uniform(1.1, 1.25)
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
        await telegram_send_message(chat_id, "Send a text message (or media caption) to rephrase it.")
        return {"ok": True}

    # Simple commands
    if user_text.strip().lower() in ("/start", "start"):
        await telegram_send_message(chat_id, "Bot is running. Send any message to rephrase it.")
        return {"ok": True}
    
    # Check if message contains X/Twitter link (early detection)
    tweet_id = extract_tweet_id(user_text)
    original_text_with_link = user_text  # Keep original for context
    
    # If X link detected, remove it from the text before rephrasing
    if tweet_id:
        # Remove X/Twitter URLs from the text
        pattern = r'https?://(?:twitter\.com|x\.com|mobile\.twitter\.com)/\S+\s*'
        user_text = re.sub(pattern, '', user_text, flags=re.IGNORECASE).strip()
        print(f"DEBUG: Detected X link (tweet_id={tweet_id}), removed URL from text for rephrasing")

    # TEMPORARILY DISABLED: Forward requirement
    # Only act on forwarded messages (per your requirement)
    # if not is_forwarded(message):
    #     await telegram_send_message(chat_id, "Please forward a message to me, and I'll rephrase it.")
    #     return {"ok": True}
    
    # TEMPORARILY DISABLED: Channel restriction
    # Check if message is from allowed channel (if configured)
    # if not is_forwarded_from_allowed_channel(message):
    #     if ALLOWED_FORWARD_CHANNEL:
    #         await telegram_send_message(
    #             chat_id, 
    #             "⚠️ This bot is restricted to specific authorized sources.\n\n"
    #             "Please ensure the message is forwarded from the correct channel."
    #         )
    #     return {"ok": True}
    
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

    # Split tags from content to prevent AI from messing them up
    start_tags, content_body, end_tags = extract_tag_blocks(user_text)
    
    if not content_body.strip():
        # Edge case: Message is ONLY tags. Just return original.
        await telegram_send_message(chat_id, user_text)
        return {"ok": True}

    # Calculate available space for the body
    overhead = len(start_tags) + len(end_tags)
    available_chars = 280 - overhead
    
    if available_chars < 10:
         # Too many tags, can't rephrase safely. Return original.
         await telegram_send_message(chat_id, user_text)
         return {"ok": True}

    masked = mask_protected(content_body)
    style = random.choice(STYLES)
    # Prompt is built inside the loop to allow modification on retry

    try:
        rewritten_body = ""
        
        # Retry logic: Attempt 1 (Standard) -> Check -> Attempt 2 (Strict Shortening)
        max_attempts = 2
        
        for attempt_idx in range(max_attempts):
            # If this is a retry (attempt_idx > 0), force strict shortening
            force_short = (attempt_idx > 0)
            
            # Pass available_chars to build_prompt
            current_prompt = build_prompt(masked.masked, style=style, force_short=force_short, max_chars=available_chars)
            
            if force_short:
                print(f"DEBUG: Output too long ({len(rewritten_body)} chars). Retrying with strict length constraint.")
                current_prompt += (
                    f"\n\nSYSTEM ALERT: Your previous output was {len(rewritten_body)} characters long."
                    f"\nMAXIMUM ALLOWED IS {available_chars} CHARACTERS."
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
            
            temp_rewritten_body = unmask(chosen, masked.placeholders).strip()
            
            if len(temp_rewritten_body) <= available_chars:
                rewritten_body = temp_rewritten_body
                print(f"DEBUG: Success on attempt {attempt_idx+1} ({len(rewritten_body)} chars)")
                break
            
            # Keep the best effort so far
            rewritten_body = temp_rewritten_body
        
        # Reassemble final message with guaranteed spacing
        start_tags_clean = start_tags.rstrip()
        end_tags_clean = end_tags.lstrip()
        rewritten_body_clean = rewritten_body.strip()
        
        final_message = start_tags_clean
        if start_tags_clean and rewritten_body_clean:
            # Ensure space/newline between start tags and body
            final_message += " "
        final_message += rewritten_body_clean
        
        if end_tags_clean:
            if final_message:
                # Ensure space/newline before end tags
                final_message += " "
            final_message += end_tags_clean
        
        # If still over 280 after retries, log warning
        if len(final_message) > 280:
            print(f"WARNING: Failed to get under 280 chars after {max_attempts} attempts. Final length: {len(final_message)}")

        print(f"DEBUG: Final output: {final_message}")
        
        # Relaxed safety check: Allow 30-200% of original length for variation
        # EXCEPTION: If original > 280 and rewritten <= 280, allow it even if < 30% (successful compression)
        min_length = len(user_text) * 0.3
        max_length = len(user_text) * 2.0
        
        is_compression_attempt = len(user_text) > 280
        is_successful_compression = (is_compression_attempt and len(final_message) <= 280)
        
        # If we are trying to compress for X, accept the result if it's shorter than original
        # and at least 10 chars long, ignoring the 30% min_length rule.
        if is_compression_attempt:
             if len(final_message) < 10:
                 print("WARNING: Rewrite too short/empty, using original")
                 final_message = user_text.strip()
             # Else: Keep rewritten (even if 290 chars, better than 1000)
        
        elif (not final_message or len(final_message) < min_length):
            # Only fallback if truly broken (less than 30% of original)
            print(f"WARNING: Rewrite too short ({len(final_message)} vs {len(user_text)}), using original")
            final_message = user_text.strip()
        elif len(final_message) > max_length:
            # If way too long, might be hallucination - use original
            print(f"WARNING: Rewrite too long ({len(final_message)} vs {len(user_text)}), using original")
            final_message = user_text.strip()

        # Telegram message limit safety
        if len(final_message) > 280:
             print(f"WARNING: Output exceeded 280 chars ({len(final_message)}), truncating/checking.")

        if len(final_message) > 3500:
            final_message = final_message[:3500] + "\n\n[truncated]"

        # If X link was detected earlier, create reply button with ONLY the rephrased text
        reply_markup = None
        
        if tweet_id:
            # Create X Web Intent URL with rephrased text pre-filled (without X link)
            from urllib.parse import quote
            
            # Ensure final_message doesn't contain any X links
            clean_reply = re.sub(
                r'https?://(?:twitter\.com|x\.com|mobile\.twitter\.com)/\S+\s*',
                '',
                final_message,
                flags=re.IGNORECASE
            ).strip()
            
            # URL-encode the clean rephrased message for X
            encoded_reply = quote(clean_reply)
            
            # Reply intent: Opens X reply composer with text pre-filled
            # Note: Web Intent API still uses twitter.com domain (official X API endpoint)
            reply_intent_url = f"https://twitter.com/intent/tweet?in_reply_to={tweet_id}&text={encoded_reply}"
            
            # Quote tweet intent: Include the tweet URL directly in the text
            # X automatically converts tweet URLs to embedded quote cards
            # Using x.com domain (current X platform standard)
            original_tweet_url = f"https://x.com/i/status/{tweet_id}"
            quote_text_with_url = f"{clean_reply} {original_tweet_url}"
            encoded_quote_text = quote(quote_text_with_url)
            quote_intent_url = f"https://twitter.com/intent/tweet?text={encoded_quote_text}"
            
            # Create inline keyboard with two buttons (Reply and Quote)
            reply_markup = {
                "inline_keyboard": [
                    [
                        {
                            "text": "💬 Reply on X",
                            "url": reply_intent_url
                        },
                        {
                            "text": "🔁 Quote on X",
                            "url": quote_intent_url
                        }
                    ]
                ]
            }
            print(f"DEBUG: Adding X buttons (tweet_id={tweet_id}) - Reply & Quote with clean text ({len(clean_reply)} chars)")
            
            # For X posts, show the clean reply text in Telegram too (without the X link)
            final_message = clean_reply

        await telegram_send_message(chat_id, final_message, reply_markup=reply_markup)
        
        # Update rate limit timestamp after successful processing
        if user_id:
            update_rate_limit(user_id)

    except Exception as exc:
        import traceback
        traceback.print_exc()  # Log to console
        await telegram_send_message(chat_id, f"Error: {type(exc).__name__} - {str(exc)[:100]}")

    return {"ok": True}