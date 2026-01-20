import json
import os
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Gemini (Google AI / google-genai)
from google import genai

try:
    # Newer google-genai versions
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover
    types = None  # type: ignore


def calculate_text_similarity(original: str, rephrased: str) -> float:
    """
    Calculate similarity score between original and rephrased text using multiple metrics
    that mimic spam detection algorithms. Returns a score from 0.0 (completely different)
    to 1.0 (identical).

    Uses a combination of:
    - Jaccard similarity (word set overlap)
    - Cosine similarity (word frequency vectors)
    - Normalized word overlap ratio
    """

    def preprocess_text(text: str) -> List[str]:
        """Preprocess text for similarity calculation."""
        # Convert to lowercase, remove punctuation, split into words
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = [word.strip() for word in text.split() if word.strip() and len(word) > 1]
        return words

    original_words = preprocess_text(original)
    rephrased_words = preprocess_text(rephrased)

    if not original_words or not rephrased_words:
        return 0.0

    # 1. Jaccard similarity (intersection over union of word sets)
    original_set = set(original_words)
    rephrased_set = set(rephrased_words)
    intersection = original_set & rephrased_set
    union = original_set | rephrased_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # 2. Cosine similarity using word frequency vectors
    original_freq = Counter(original_words)
    rephrased_freq = Counter(rephrased_words)

    # Create combined vocabulary
    all_words = set(original_freq.keys()) | set(rephrased_freq.keys())

    # Create frequency vectors
    original_vector = [original_freq.get(word, 0) for word in all_words]
    rephrased_vector = [rephrased_freq.get(word, 0) for word in all_words]

    # Calculate dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(original_vector, rephrased_vector))
    original_magnitude = sqrt(sum(a * a for a in original_vector))
    rephrased_magnitude = sqrt(sum(b * b for b in rephrased_vector))

    cosine = dot_product / (original_magnitude * rephrased_magnitude) if original_magnitude * rephrased_magnitude > 0 else 0.0

    # 3. Normalized word overlap ratio (considering frequency)
    total_original_words = len(original_words)
    total_rephrased_words = len(rephrased_words)
    overlap_count = sum(min(original_freq[word], rephrased_freq[word]) for word in intersection)

    # Normalize by average length to make it symmetric
    avg_length = (total_original_words + total_rephrased_words) / 2
    overlap_ratio = overlap_count / avg_length if avg_length > 0 else 0.0

    # 4. Length ratio penalty (texts that are too different in length get lower scores)
    length_ratio = min(len(original_words), len(rephrased_words)) / max(len(original_words), len(rephrased_words))
    length_penalty = 0.8 + 0.2 * length_ratio  # Scale from 0.8 to 1.0

    # Combine metrics with weights (tuned for spam detection similarity)
    # Jaccard: 0.3, Cosine: 0.4, Overlap: 0.3, with length penalty
    combined_score = (jaccard * 0.3 + cosine * 0.4 + overlap_ratio * 0.3) * length_penalty

    return min(1.0, max(0.0, combined_score))


# Global rate limiting to protect server from overload
request_times = deque(maxlen=1000)
MAX_REQUESTS_PER_SECOND = 15  # Adjust based on server capacity

class GlobalRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Server-level rate limiting to prevent overload.
    Limits total requests across ALL users to protect server resources.
    Exempt users (RATE_LIMIT_EXEMPT_USERS) bypass this limit.
    """
    async def dispatch(self, request: Request, call_next):
        # Only apply to webhook endpoint
        if "/webhook/" not in str(request.url):
            return await call_next(request)
        
        # Check if user is exempt from rate limiting
        is_exempt = False
        try:
            # Read request body to check user_id (without consuming it permanently)
            body_bytes = await request.body()
            update = json.loads(body_bytes)
            message = update.get("message") or update.get("edited_message")
            user_id = (message.get("from") or {}).get("id") if message else None
            
            # If user is exempt, bypass global rate limit
            if user_id and user_id in EXEMPT_USER_IDS:
                is_exempt = True
                print(f"DEBUG: User {user_id} is exempt from global rate limiting")
        except Exception as e:
            # If parsing fails, continue with normal rate limiting
            print(f"DEBUG: Error parsing request for exemption check: {e}")
        
        # Exempt users bypass the global rate limit
        if is_exempt:
            # Restore request body for downstream handlers
            async def receive():
                return {"type": "http.request", "body": body_bytes}
            request._receive = receive
            return await call_next(request)
        
        now = time.time()
        
        # Remove requests older than 1 second
        while request_times and now - request_times[0] > 1.0:
            request_times.popleft()
        
        # Check if we're over the limit (only for non-exempt users)
        if len(request_times) >= MAX_REQUESTS_PER_SECOND:
            # Silently accept (Telegram will handle retries)
            # Don't process the request to save resources
            return {"ok": True}
        
        request_times.append(now)
        
        # Restore request body for downstream handlers
        try:
            if 'body_bytes' in locals():
                async def receive():
                    return {"type": "http.request", "body": body_bytes}
                request._receive = receive
        except:
            pass
        
        return await call_next(request)


app = FastAPI()

# Add global rate limiting middleware
app.add_middleware(GlobalRateLimitMiddleware)

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
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "30"))

# Rate limit exemption: User IDs that bypass rate limiting (comma-separated)
# Example: "123456789,987654321"
RATE_LIMIT_EXEMPT_USERS = os.environ.get("RATE_LIMIT_EXEMPT_USERS", "")
EXEMPT_USER_IDS = set(int(uid.strip()) for uid in RATE_LIMIT_EXEMPT_USERS.split(",") if uid.strip())

# In-memory storage for rate limiting (user_id -> last_request_timestamp)
user_last_request: Dict[int, float] = {}

SYSTEM_INSTRUCTION = os.environ.get(
    "SYSTEM_INSTRUCTION",
    "You are an assistant that aggressively rephrases messages to avoid spam detection while maintaining core intent.\n\n"
    "CRITICAL HARD LIMIT: The final output MUST be under 280 characters. NEVER exceed this limit.\n\n"
    "CORE GOAL: The rephrased message must convey the SAME core meaning/intent but be "
    "structurally and stylistically VERY DIFFERENT from the original.\n\n"
    "INTENT PRESERVATION (ESSENTIAL):\n"
    "- Keep the fundamental message and purpose intact\n"
    "- Maintain subject-object relationships (who did what to whom)\n"
    "- Preserve call-to-action elements\n"
    "- Don't add new information or change the core argument\n\n"
    "MENTION/TAG HANDLING (CRITICAL):\n"
    "- Do NOT change the role of @mentions. If a mention is used as a tag (e.g. at start/end), keep it as a tag.\n"
    "- IF a message starts with a block of @mentions, KEEP them at the start. Do not weave them into the sentence.\n"
    "- IF a message ends with a block of tags/mentions, KEEP them at the end.\n"
    "- Do NOT make a mention the subject/actor if it wasn't one (e.g. don't change '@POTUS...' to '@POTUS said...').\n"
    "- Keep mentions and hashtags in their approximate original positions (start vs end) if possible, or integrate naturally without changing meaning.\n\n"
    "MUST PRESERVE EXACTLY (these will be marked as __PROTECTED_N__):\n"
    "- All @mentions and #hashtags\n\n"
    "SHOULD PRESERVE (can be paraphrased naturally):\n"
    "- Names of people, organizations, locations (can use synonyms/descriptions)\n"
    "- Numbers and dates (can be rounded, approximated, or expressed differently)\n"
    "- Key facts and concepts (paraphrase freely, keep meaning)\n\n"
    "MUST PRESERVE:\n"
    "- Core message and meaning\n"
    "- General claims and their direction\n"
    "- Call-to-action intent\n\n"
    "OUTPUT: Only the rewritten text, no preamble or explanation.",
)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")

# Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


STYLES: List[str] = [
    # Tone variations
    "very formal and academic",
    "casual and conversational",
    "urgent and action-oriented",
    "calm and measured",
    "enthusiastic and energetic",
    "matter-of-fact and neutral",
    "authoritative and confident",
    "empathetic and understanding",
    "skeptical and questioning",
    "optimistic and positive",
    "cautious and careful",
    "bold and assertive",
    "diplomatic and balanced",
    "passionate and intense",
    "detached and objective",
    "warm and inviting",
    "sharp and incisive",
    
    # Voice and perspective
    "active voice throughout",
    "passive voice where natural",
    "mix of active and passive",
    "first person perspective",
    "second person perspective",
    "third person perspective",
    "imperative mood",
    "subjunctive mood where appropriate",
    
    # Structural variations
    "start with the conclusion (if it doesn't change meaning)",
    "build up to the main point",
    "use parallel structure",
    "vary sentence complexity",
    "use fragments for emphasis (sparingly)",
    "combine related ideas",
    "separate distinct points",
    "flow as continuous prose",
    
    # Word choice strategies
    "prefer shorter, simpler words",
    "use more sophisticated vocabulary",
    "mix formal and informal language",
    "use concrete, specific terms",
    "use abstract, conceptual terms",
    "prefer action verbs",
    "use descriptive adjectives",
    "avoid repetition of words",
    "use synonyms extensively",
    "prefer Anglo-Saxon words",
    "prefer Latin-derived words",
    "use idiomatic expressions",
    "avoid idioms, use literal language",
    
    # Style personas
    "sound like a news anchor",
    "sound like a social media influencer",
    "sound like a business executive",
    "sound like a teacher explaining",
    "sound like a friend giving advice",
    "sound like a journalist reporting",
    "sound like a blogger writing",
    "sound like a scientist explaining",
    "sound like a poet expressing",
    "sound like a lawyer arguing",
    "sound like a marketer selling",
    "sound like a critic reviewing",
    
    # Punctuation and formatting
    "use varied punctuation",
    "prefer periods and commas",
    "use semicolons and colons",
    "use dashes and parentheses",
    "minimal punctuation",
    "expressive punctuation",
    "keep original formatting (if it doesn't interfere with tags)",
    "vary formatting while preserving tag positions",
    
    # Emphasis and rhythm
    "emphasize the beginning",
    "emphasize the end",
    "create rhythm through repetition",
    "avoid repetition entirely",
    "use alliteration naturally",
    "vary word length patterns",
    "create contrast between ideas",
    "build momentum toward conclusion",
    
    # Clarity and flow
    "prioritize clarity above all",
    "prioritize natural flow",
    "use transitions liberally",
    "minimal transitions, direct connections",
    "explain concepts simply",
    "assume reader knowledge",
    "add context naturally (without adding new facts)",
    "remove unnecessary context while keeping key facts",
    
    # Specific techniques
    "use rhetorical questions (if original intent allows)",
    "avoid questions, use statements",
    "use metaphors and similes naturally",
    "avoid figurative language",
    "use examples and illustrations (without adding new facts)",
    "stick to abstract concepts",
    "quantify with numbers (can round/approximate)",
    "qualify with descriptive language",
    "use time references naturally",
    "paraphrase time references differently",
    "use location references naturally",
    "paraphrase location references differently",
]


PROTECTED_PATTERN = re.compile(
    r"("  # capture each token
    r"@\w{1,64}|"  # @mentions (e.g., @POTUS, @username)
    r"#\w+"  # hashtags (e.g., #IranRevolution2026)
    r")",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MaskedText:
    masked: str
    placeholders: List[Tuple[str, str]]  # (placeholder, original)


def apply_random_tag_removal(text: str, removal_chance: float = 0.4) -> str:
    """
    Apply random removal to ALL tag sequences in the text (start, middle, end).
    Tags are identified as consecutive @ or # mentions separated only by whitespace.
    
    When multiple tags are found together, each has a removal_chance probability
    of being removed. At least one tag per sequence is always kept.
    
    Args:
        text: Full text containing tags like "@user1 @user2 text @user3"
        removal_chance: Probability (0.0-1.0) of removing each tag in a sequence
    
    Returns:
        Text with some tags randomly removed from sequences
    """
    if not text.strip():
        return text
    
    # Pattern to find sequences of tags (2 or more consecutive tags)
    # Matches: @tag1 @tag2 or #tag1 #tag2 #tag3, etc.
    tag_sequence_pattern = re.compile(r'((?:[@#][\w\d_]+\s*){2,})')
    
    def process_sequence(match):
        sequence = match.group(0)
        # Find all individual tags in this sequence
        individual_tags = re.findall(r'[@#][\w\d_]+', sequence)
        
        if len(individual_tags) <= 1:
            return sequence
        
        # Apply 40% removal chance to each tag
        kept_tags = [tag for tag in individual_tags if random.random() > removal_chance]
        
        # Always keep at least one tag
        if not kept_tags:
            kept_tags = [random.choice(individual_tags)]
        
        # Reconstruct with spaces, preserve trailing space if original had it
        result = " ".join(kept_tags)
        if sequence.endswith(" "):
            result += " "
        
        return result
    
    # Apply removal to all tag sequences
    return tag_sequence_pattern.sub(process_sequence, text)


def extract_tag_blocks(text: str) -> Tuple[str, str, str]:
    """
    Separates a message into (start_tags, content, end_tags).
    start_tags: continuous block of mentions/hashtags at the very beginning
    end_tags: continuous block of mentions/hashtags at the very end
    content: the middle part to be rephrased
    
    Example:
    "@User1 Hi there #tag" -> ("@User1 ", "Hi there", " #tag")
    """
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

    # SAFETY: Remove any leftover unreplaced placeholders (AI hallucinations)
    # This catches cases where AI generates __PROTECTED_N__ that don't exist
    text = re.sub(r'__PROTECTED_\d+__\s*', '', text)
    
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
        # Rate limiting disabled globally
        return None
    
    # Check if user is exempt from rate limiting
    if user_id in EXEMPT_USER_IDS:
        print(f"DEBUG: User {user_id} is exempt from rate limiting")
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


async def telegram_send_message(chat_id: int, text: str, reply_markup: Optional[dict] = None, parse_mode: Optional[str] = None) -> None:
    """
    Send a message to a Telegram chat.
    
    Args:
        chat_id: The chat ID to send to
        text: The message text
        reply_markup: Optional inline keyboard markup
        parse_mode: Optional parse mode for formatting (e.g., "MarkdownV2", "HTML")
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
        "allow_sending_without_reply": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
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
        "minimal restructure (keep mostly same order)",
        "move the main point to the beginning (if it doesn't change meaning)",
        "move the main point to the end (if it doesn't change meaning)",
        "group related ideas together in new ways",
        "reorganize ideas while maintaining logical flow",
        "split long ideas into separate sentences",
        "combine short ideas into longer sentences",
        "restructure to emphasize different aspects"
    ])
    
    # Random length variation
    # Logic: If strict shortening needed (retry) OR original is close to limit, force shorter options.
    if force_short:
        length = random.choice([
            f"CRITICAL: MAKE IT SHORTER. Remove filler words. Condense sentences. STRICTLY UNDER {max_chars} CHARS.",
            f"URGENT: Cut it down significantly. Remove all unnecessary words. Must be under {max_chars} chars.",
            f"EMERGENCY: Compress aggressively. Eliminate redundancy. Maximum {max_chars} characters."
        ])
    elif len(masked_text) > (max_chars - 30):
        # Very close to limit: ONLY allow shortening
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "compress into fewer sentences",
            f"condense strictly to fit under {max_chars} chars",
            "cut unnecessary words and phrases",
            "trim down to essential points only",
            "reduce verbosity while keeping meaning"
        ])
    elif len(masked_text) > (max_chars - 60):
        # Moderately close: Allow shortening or keeping same length
        length = random.choice([
            "make 20-30% shorter by removing filler",
            "compress into fewer sentences",
            f"condense strictly to fit under {max_chars} chars",
            "keep similar length but vary sentence lengths",
            "slightly reduce length by tightening phrasing",
            "trim excess while maintaining all key points"
        ])
    elif len(masked_text) < 120:
        # Very short: Prioritize expansion or keeping same length
        length = random.choice([
            f"make 20-40% longer by expanding key points naturally (BUT UNDER {max_chars} CHARS)",
            f"add relevant context or adjectives to make it more descriptive (STAY UNDER {max_chars} CHARS)",
            "keep similar length but vary sentence lengths",
            "break into more sentences",
            f"elaborate on key concepts without adding new facts (UNDER {max_chars} CHARS)",
            f"expand with natural connecting phrases (MAX {max_chars} CHARACTERS)",
            f"add descriptive details to enhance clarity (MUST BE UNDER {max_chars} CHARS)"
        ])
    else:
        # Safe zone (120 - 220 chars): Full variety
        length = random.choice([
            "make 20-30% shorter by removing filler",
            f"make 10-20% longer by expanding key points naturally (BUT UNDER {max_chars} CHARS)",
            "keep similar length but vary sentence lengths",
            "compress into fewer sentences",
            "break into more sentences",
            "make it significantly more concise",
            f"expand naturally with additional context (UNDER {max_chars} CHARS)",
            "vary length dramatically from original (but stay under limit)",
            "tighten phrasing while preserving all meaning",
            f"add elaboration where it enhances understanding (MAX {max_chars} CHARACTERS)"
        ])
    
    # Random tone
    tone = random.choice([
        "very formal and academic",
        "casual and friendly",
        "urgent and direct",
        "calm and measured",
        "enthusiastic and energetic",
        "matter-of-fact neutral",
        "conversational and approachable",
        "professional and polished",
        "informal and relaxed",
        "authoritative and confident",
        "empathetic and understanding",
        "skeptical and questioning",
        "optimistic and positive",
        "cautious and careful",
        "bold and assertive",
        "diplomatic and balanced",
        "passionate and intense",
        "detached and objective",
        "warm and inviting",
        "sharp and incisive"
    ])
    
    # Random sentence structure
    sentence_style = random.choice([
        "use short, punchy sentences",
        "use longer, flowing sentences",
        "mix short and long sentences",
        "start sentences differently each time",
        "vary punctuation patterns",
        "use parallel sentence structures",
        "vary sentence beginnings (start with different parts of speech)",
        "use rhetorical questions strategically (only if original intent allows)",
        "combine simple sentences into complex ones",
        "break complex sentences into simpler ones",
        "use fragments for emphasis (sparingly)",
        "vary between active and passive voice",
        "use inverted sentence order occasionally (if natural)",
        "create rhythm through sentence length variation",
        "use semicolons and colons for variety",
        "vary the position of key information in sentences (maintain meaning)"
    ])
    
    # Random word strategy
    word_strategy = random.choice([
        "maximize synonym usage",
        "prefer simpler/common words",
        "prefer more sophisticated vocabulary",
        "mix formal and informal words",
        "use more action verbs",
        "use more descriptive adjectives (sparingly)",
        "replace nouns with verbs where possible",
        "use concrete nouns instead of abstract ones",
        "prefer specific words over generic ones",
        "use idiomatic expressions naturally",
        "vary word length (mix short and long words)",
        "avoid repeating the same root words",
        "use stronger, more precise verbs",
        "replace adjectives with more specific nouns",
        "use metaphors and similes naturally",
        "prefer Anglo-Saxon words over Latin-derived",
        "prefer Latin-derived words over Anglo-Saxon",
        "use colloquialisms and everyday language",
        "use technical or specialized terms when appropriate",
        "vary between literal and figurative language"
    ])
    
    # Check if there are placeholders in the text
    has_placeholders = "__PROTECTED_" in masked_text
    placeholder_instruction = ""
    if has_placeholders:
        placeholder_instruction = (
            "CRITICAL: The text contains placeholders like __PROTECTED_0__, __PROTECTED_1__, etc. "
            "You MUST keep every placeholder EXACTLY as-is in your output. "
            "Do NOT create new placeholders. Do NOT remove existing placeholders. "
            "Do NOT modify placeholders in any way.\n\n"
        )
    else:
        placeholder_instruction = (
            "CRITICAL: Do NOT add any placeholders like __PROTECTED_0__ to your output. "
            "Write natural text only.\n\n"
        )
    
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"{placeholder_instruction}"
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

    # Get user_id early to check for exemptions
    user_id = (message.get("from") or {}).get("id")
    is_exempt_user = user_id and user_id in EXEMPT_USER_IDS

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

    # Forward requirement: Only act on forwarded messages (exempt users bypass this)
    if not is_exempt_user:
        if not is_forwarded(message):
            await telegram_send_message(chat_id, "Please forward a message to me, and I'll rephrase it.")
            return {"ok": True}
        
        # Channel restriction: Check if message is from allowed channel (if configured)
        if ALLOWED_FORWARD_CHANNEL:
            if not is_forwarded_from_allowed_channel(message):
                channel_display = f"@{ALLOWED_FORWARD_CHANNEL}" if not ALLOWED_FORWARD_CHANNEL.startswith("-") else ALLOWED_FORWARD_CHANNEL
                await telegram_send_message(
                    chat_id, 
                    f"⚠️ This bot is being used only on messages <b>forwarded</b> from <b>{channel_display}</b>.",
                    parse_mode="HTML"
                )
                return {"ok": True}
    
    # Check rate limit (use from_user.id for user-specific limiting)
    if user_id:
        seconds_remaining = check_rate_limit(user_id)
        if seconds_remaining is not None:
            await telegram_send_message(
                chat_id,
                f"⏱️ Please wait {seconds_remaining} second{'s' if seconds_remaining != 1 else ''} before sending another message.\n\n"
                f"Rate limit: 1 message per {RATE_LIMIT_SECONDS} seconds."
            )
            return {"ok": True}

    # Apply random tag removal to ALL tag sequences (start, middle, end)
    # This applies 40% removal chance to each tag in sequences of 2+ tags
    user_text = apply_random_tag_removal(user_text, removal_chance=0.4)

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
        
        # Generate 2 candidates and select the one with lowest similarity score
        all_candidates = []
        
        # Generate 2 candidates with varying styles for maximum diversity
        for candidate_num in range(2):
            # Use different style for each candidate to increase diversity
            style = random.choice(STYLES)
            current_prompt = build_prompt(masked.masked, style=style, force_short=False, max_chars=available_chars)
            
            candidates = gemini_generate_candidates(current_prompt)
            print(f"DEBUG: Candidate {candidate_num + 1}: Got {len(candidates)} outputs from Gemini")
            
            if candidates:
                all_candidates.extend(candidates)
            
        if not all_candidates:
                    raise RuntimeError("no_candidates")
        
        print(f"DEBUG: Total candidates generated: {len(all_candidates)}")

        # Filter candidates that preserved all placeholders
        good_candidates = [c for c in all_candidates if contains_all_placeholders(c, masked.placeholders)]
        candidates_to_evaluate = good_candidates if good_candidates else all_candidates
        
        # Unmask all candidates
        unmasked_candidates = [(unmask(c.strip(), masked.placeholders).strip(), c) for c in candidates_to_evaluate]
        
        # Filter candidates that fit within available_chars
        valid_candidates = [(unmasked, original) for unmasked, original in unmasked_candidates if len(unmasked) <= available_chars]
        
        if not valid_candidates:
            # If no candidates fit, take the shortest one and try to work with it
            print("DEBUG: No candidates fit within limit, taking shortest")
            valid_candidates = [min(unmasked_candidates, key=lambda x: len(x[0]))]
        
        print(f"DEBUG: Valid candidates (under {available_chars} chars): {len(valid_candidates)}")
        
        # Calculate similarity scores for all valid candidates
        candidate_scores = []
        for unmasked, original in valid_candidates:
            similarity = calculate_text_similarity(content_body, unmasked)
            candidate_scores.append((unmasked, similarity))
            print(f"DEBUG: Candidate similarity: {similarity:.3f}, length: {len(unmasked)}")
        
        # Select the candidate with the LOWEST similarity score (most different from original)
        # This helps evade spam detection while maintaining intent
        rewritten_body = min(candidate_scores, key=lambda x: x[1])[0]
        best_similarity = min(candidate_scores, key=lambda x: x[1])[1]
        
        print(f"DEBUG: Selected best candidate with similarity {best_similarity:.3f} ({len(rewritten_body)} chars) - LOWEST score")
        
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
        
        # If still over 280 after candidate selection, log warning
        if len(final_message) > 280:
            print(f"WARNING: Output exceeded 280 chars. Final length: {len(final_message)}")

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