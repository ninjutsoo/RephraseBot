import json
import os
import random
import re
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

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
    """st 
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
    "- Keep the fundamental message, purpose, and any opening greetings/salutations (e.g., 'Dear', 'Esteemed Members') intact\n"
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

# Supabase client (optional, for logging/analytics)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized")
else:
    print("⚠ Supabase credentials not found - database logging disabled")


def log_to_supabase(table_name: str, data: dict) -> bool:
    """
    Helper function to safely log data to Supabase.
    Returns True if successful, False otherwise.
    """
    if not supabase_client:
        return False
    
    try:
        supabase_client.table(table_name).insert(data).execute()
        return True
    except Exception as e:
        print(f"⚠ Failed to log to Supabase table '{table_name}': {e}")
        return False


# In-memory buffer for batch writes to activity_logs
activity_buffer: List[dict] = []
BATCH_SIZE = 100  # Flush when buffer reaches this size


def track_user(user_id: int, username: Optional[str], first_name: str, 
               last_name: Optional[str], language_code: Optional[str]) -> None:
    """
    Track user - insert if new, update if username/name changed.
    Only writes to DB if user is new or info changed.
    """
    if not supabase_client:
        return
    
    try:
        # Check if user exists
        result = supabase_client.table("users").select("*").eq("user_id", user_id).execute()
        
        if not result.data:
            # NEW USER - insert
            supabase_client.table("users").insert({
                "user_id": user_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "language_code": language_code,
            }).execute()
            print(f"✓ New user tracked: {user_id} (@{username or 'no_username'})")
        else:
            # EXISTING USER - check if info changed
            existing = result.data[0]
            if (existing.get("username") != username or 
                existing.get("first_name") != first_name or
                existing.get("last_name") != last_name):
                supabase_client.table("users").update({
                    "username": username,
                    "first_name": first_name,
                    "last_name": last_name,
                }).eq("user_id", user_id).execute()
                print(f"✓ User info updated: {user_id}")
    except Exception as e:
        print(f"⚠ Failed to track user {user_id}: {e}")


def log_activity(user_id: int, action_type: str, **kwargs) -> None:
    """
    Log activity to buffer (batched writes for efficiency).
    
    Args:
        user_id: Telegram user ID
        action_type: Type of action (e.g., 'rephrase_success', 'rate_limited')
        **kwargs: Additional fields (original_length, error_message, etc.)
    """
    if not supabase_client:
        return
    
    from datetime import datetime, timezone
    
    activity_buffer.append({
        "user_id": user_id,
        "action_type": action_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    })
    
    # Flush if buffer is full
    if len(activity_buffer) >= BATCH_SIZE:
        flush_activity_logs()


def flush_activity_logs() -> None:
    """Flush activity buffer to Supabase (batch insert)."""
    if not supabase_client or not activity_buffer:
        return
    
    try:
        supabase_client.table("activity_logs").insert(activity_buffer).execute()
        count = len(activity_buffer)
        activity_buffer.clear()
        print(f"✓ Flushed {count} activity logs to Supabase")
    except Exception as e:
        print(f"⚠ Failed to flush activity logs: {e}")
        activity_buffer.clear()  # Clear to prevent memory buildup


def is_pro_user(user_id: int) -> bool:
    """Check if user has active Pro subscription or trial."""
    if not supabase_client:
        return False
    
    try:
        result = supabase_client.table("users").select("is_pro, pro_expires_at, trial_ends_at").eq("user_id", user_id).execute()
        
        if not result.data:
            return False
        
        user = result.data[0]
        
        # Check if pro (lifetime or not expired)
        if user.get("is_pro"):
            expires = user.get("pro_expires_at")
            if not expires:  # Lifetime
                return True
            # Check if not expired
            from datetime import datetime, timezone
            if datetime.fromisoformat(expires.replace('Z', '+00:00')) > datetime.now(timezone.utc):
                return True
        
        # Check if trial is active
        trial_ends = user.get("trial_ends_at")
        if trial_ends:
            from datetime import datetime, timezone
            if datetime.fromisoformat(trial_ends.replace('Z', '+00:00')) > datetime.now(timezone.utc):
                return True
        
        return False
    except Exception as e:
        print(f"⚠ Failed to check pro status: {e}")
        return False


def get_user_preferences(user_id: int) -> Optional[dict]:
    """Load user's saved preferences from database."""
    if not supabase_client:
        return None
    
    try:
        result = supabase_client.table("users").select("preferred_tone, preferred_length, preferred_variation").eq("user_id", user_id).execute()
        
        if not result.data:
            return None
        
        prefs = result.data[0]
        return {
            "tone": prefs.get("preferred_tone"),
            "length": prefs.get("preferred_length"),
            "variation": prefs.get("preferred_variation")
        }
    except Exception as e:
        print(f"⚠ Failed to load preferences: {e}")
        return None


def save_user_preferences(user_id: int, tone: Optional[str] = None, 
                         length: Optional[str] = None, variation: Optional[str] = None) -> bool:
    """Save user's preferences to database."""
    if not supabase_client:
        return False
    
    try:
        update_data = {}
        if tone is not None:
            update_data["preferred_tone"] = tone
        if length is not None:
            update_data["preferred_length"] = length
        if variation is not None:
            update_data["preferred_variation"] = variation
        
        if update_data:
            supabase_client.table("users").update(update_data).eq("user_id", user_id).execute()
            print(f"✓ Saved preferences for user {user_id}")
            return True
        return False
    except Exception as e:
        print(f"⚠ Failed to save preferences: {e}")
        return False


EST = ZoneInfo("America/New_York")


def get_weekly_activity_report() -> List[Tuple[str, int, int]]:
    """
    Fetch activity_logs for the past 7 days (EST), compute per-day new and active user counts.
    New = first activity within that week is on that day. Active = distinct users with activity that day.
    Returns list of (date_str, new_count, active_count) ordered by date (oldest first).
    """
    if not supabase_client:
        return []
    try:
        now_est = datetime.now(EST)
        end_est = now_est.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_est = (now_est - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        start_utc = start_est.astimezone(timezone.utc).isoformat()
        end_utc = end_est.astimezone(timezone.utc).isoformat()

        # Fetch all rows (Supabase default limit is 1000; paginate to get full week)
        all_data: List[dict] = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                supabase_client.table("activity_logs")
                .select("user_id, timestamp")
                .gte("timestamp", start_utc)
                .lte("timestamp", end_utc)
                .range(offset, offset + page_size - 1)
                .execute()
            )
            chunk = result.data or []
            all_data.extend(chunk)
            if len(chunk) < page_size:
                break
            offset += page_size

        if not all_data:
            return _empty_week_rows(start_est)

        # Users who had activity before this week (pre-fill "seen" so "new" = first appearance in window)
        seen_user_ids: set = set()
        offset_pre = 0
        while True:
            result_pre = (
                supabase_client.table("activity_logs")
                .select("user_id")
                .lt("timestamp", start_utc)
                .range(offset_pre, offset_pre + page_size - 1)
                .execute()
            )
            chunk_pre = result_pre.data or []
            for row in chunk_pre:
                if row.get("user_id") is not None:
                    seen_user_ids.add(int(row["user_id"]))
            if len(chunk_pre) < page_size:
                break
            offset_pre += page_size

        # (user_id, date_est) for each activity in the week
        active_by_day: Dict[str, set] = defaultdict(set)
        for row in all_data:
            uid = int(row["user_id"])
            ts = row["timestamp"]
            if not ts:
                continue
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(EST)
            day_str = dt.strftime("%Y-%m-%d")
            active_by_day[day_str].add(uid)

        # New on day D = first appearance in window (same as notebook N-day: day_user_ids - seen_user_ids)
        out: List[Tuple[str, int, int]] = []
        for i in range(7):
            d = start_est + timedelta(days=i)
            day_str = d.strftime("%Y-%m-%d")
            active = active_by_day.get(day_str, set())
            new = len(active - seen_user_ids)
            seen_user_ids |= active
            out.append((day_str, new, len(active)))
        return out
    except Exception as e:
        print(f"⚠ Failed to get weekly activity report: {e}")
        return []


def _empty_week_rows(start_est: datetime) -> List[Tuple[str, int, int]]:
    """Return 7 rows of (date, 0, 0) for the past week."""
    return [((start_est + timedelta(days=i)).strftime("%Y-%m-%d"), 0, 0) for i in range(7)]


# In-memory storage for pending style selections (temporary during selection process)
pending_selections: Dict[int, dict] = {}


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


def apply_random_tag_removal(text: str, removal_chance: float = 0.4, has_link: bool = False) -> str:
    """
    Apply random removal to tag sequences in the text (start, middle, end).
    Tags are identified as consecutive @ or # mentions separated only by whitespace.
    
    Behavior depends on whether the message has a link:
    - If has_link=True: Apply removal_chance to BOTH @ and # in groups of 2+
    - If has_link=False: Apply removal_chance to # only, keep ALL @ mentions
    
    When multiple tags are found together, each has a removal_chance probability
    of being removed. At least one tag per sequence is always kept.
    
    Args:
        text: Full text containing tags like "@user1 @user2 text @user3"
        removal_chance: Probability (0.0-1.0) of removing each tag in a sequence
        has_link: Whether the message contains a link (affects @ mention handling)
    
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
        
        # Separate @ mentions and # hashtags
        mentions = [tag for tag in individual_tags if tag.startswith('@')]
        hashtags = [tag for tag in individual_tags if tag.startswith('#')]
        
        kept_tags = []
        
        # Process hashtags (always apply removal chance if 2+ hashtags)
        if len(hashtags) >= 2:
            kept_hashtags = [tag for tag in hashtags if random.random() > removal_chance]
            # Always keep at least one hashtag
            if not kept_hashtags and hashtags:
                kept_hashtags = [random.choice(hashtags)]
            kept_tags.extend(kept_hashtags)
        else:
            kept_tags.extend(hashtags)
        
        # Process mentions based on has_link
        if has_link:
            # With link: apply removal chance to mentions in groups of 2+
            if len(mentions) >= 2:
                kept_mentions = [tag for tag in mentions if random.random() > removal_chance]
                # Always keep at least one mention
                if not kept_mentions and mentions:
                    kept_mentions = [random.choice(mentions)]
                kept_tags.extend(kept_mentions)
            else:
                # Less than 2 mentions, keep all
                kept_tags.extend(mentions)
        else:
            # Without link: keep ALL mentions
            kept_tags.extend(mentions)
        
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
    
    Returns None if more than one x.com link is found (invalid).
    """
    pattern = r'(?:twitter\.com|x\.com|mobile\.twitter\.com)/(?:\w+)/status/(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if len(matches) > 1:
        return None

    if len(matches) == 1:
        return matches[0]
    
    return None


_ARABIC_SCRIPT_RE = re.compile(
    r"["
    r"\u0600-\u06FF"  # Arabic
    r"\u0750-\u077F"  # Arabic Supplement
    r"\u08A0-\u08FF"  # Arabic Extended-A
    r"\uFB50-\uFDFF"  # Arabic Presentation Forms-A
    r"\uFE70-\uFEFF"  # Arabic Presentation Forms-B
    r"\U0001EE00-\U0001EEFF"  # Arabic Mathematical Alphabetic Symbols
    r"]"
)


INVALID_TWEET_INPUT_MESSAGE = (
    "Please send a valid English Tweet reply for me to rephrase."
)


def contains_arabic_script(text: str) -> bool:
    """Return True if text contains Arabic-script letters (Arabic/Farsi, etc.)."""
    return bool(_ARABIC_SCRIPT_RE.search(text or ""))


def has_forwarded_media(message: dict) -> bool:
    """
    Check if message contains forwarded media (photo, video, document, etc.).
    Returns True if the message is forwarded AND contains media.
    """
    if not is_forwarded(message):
        return False
    
    # Check for various media types
    media_fields = ["photo", "video", "document", "animation", "voice", "video_note", "audio", "sticker"]
    return any(field in message for field in media_fields)


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
        try:
            r = await http.post(url, json=payload)
            r.raise_for_status()
            print(f"✅ Message sent successfully to user {chat_id}")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_detail = e.response.text if e.response.text else "No details"
            
            if status_code == 403:
                # User blocked the bot or bot can't access chat - NORMAL, don't crash
                print(f"⚠️  403 FORBIDDEN: User {chat_id} blocked bot or chat inaccessible")
                print(f"   → This is NORMAL user behavior, not an error")
                
                # Log to database for analytics
                if supabase_client:
                    try:
                        log_activity(
                            user_id=chat_id,
                            action_type="bot_blocked_by_user",
                            error_type="telegram_403_forbidden",
                            error_message=f"User blocked bot or chat inaccessible. Detail: {error_detail}"
                        )
                    except Exception as log_err:
                        print(f"   → Failed to log 403 to database: {log_err}")
                
                # Don't raise - this is expected behavior
                return
            
            elif status_code == 400:
                # Bad request - likely malformed message
                print(f"❌ 400 BAD REQUEST: Failed to send to {chat_id}")
                print(f"   → Error: {error_detail}")
                print(f"   → Payload: {payload}")
                
                # Log to database
                if supabase_client:
                    try:
                        log_activity(
                            user_id=chat_id,
                            action_type="telegram_api_error",
                            error_type="telegram_400_bad_request",
                            error_message=f"Bad request: {error_detail}"
                        )
                    except Exception as log_err:
                        print(f"   → Failed to log 400 to database: {log_err}")
                
                # Don't crash bot for this user's error
                return
            
            elif status_code == 429:
                # Rate limited - already handled elsewhere but log it
                print(f"⚠️  429 RATE LIMITED: Telegram rate limit hit for user {chat_id}")
                print(f"   → Bot will retry automatically")
                raise  # Let retry logic handle it
            
            else:
                # Other errors (401, 500, etc.) - these ARE problems
                print(f"❌ {status_code} ERROR: Failed to send to {chat_id}")
                print(f"   → Error: {error_detail}")
                
                # Log to database
                if supabase_client:
                    try:
                        log_activity(
                            user_id=chat_id,
                            action_type="telegram_api_error",
                            error_type=f"telegram_{status_code}_error",
                            error_message=f"HTTP {status_code}: {error_detail}"
                        )
                    except Exception as log_err:
                        print(f"   → Failed to log {status_code} to database: {log_err}")
                
                # Re-raise - these need attention
                raise
        
        except httpx.TimeoutException as e:
            # Network timeout
            print(f"⏱️  TIMEOUT: Failed to reach Telegram API for user {chat_id}")
            print(f"   → Error: {str(e)}")
            
            # Log to database
            if supabase_client:
                try:
                    log_activity(
                        user_id=chat_id,
                        action_type="telegram_api_error",
                        error_type="telegram_timeout",
                        error_message=f"Timeout: {str(e)}"
                    )
                except Exception as log_err:
                    print(f"   → Failed to log timeout to database: {log_err}")
            
            # Don't crash - retry will happen naturally
            return
        
        except Exception as e:
            # Unexpected error
            print(f"❌ UNEXPECTED ERROR: Failed to send to {chat_id}")
            print(f"   → Error type: {type(e).__name__}")
            print(f"   → Error: {str(e)}")
            
            # Log to database
            if supabase_client:
                try:
                    log_activity(
                        user_id=chat_id,
                        action_type="telegram_api_error",
                        error_type="unexpected_error",
                        error_message=f"{type(e).__name__}: {str(e)}"
                    )
                except Exception as log_err:
                    print(f"   → Failed to log unexpected error to database: {log_err}")
            
            # Re-raise - we need to know about these
            raise


async def show_style_selector(chat_id: int, user_id: int, original_text: str) -> None:
    """Show style selection menu to user."""
    # Initialize pending selection with current preferences or defaults
    prefs = get_user_preferences(user_id)
    print(f"DEBUG: get_user_preferences returned: {prefs}")
    
    # Safely extract preferences with defaults
    tone = "casual"
    length = "medium"
    variation = "moderate"
    
    if prefs:
        tone = prefs.get("tone") or "casual"
        length = prefs.get("length") or "medium"
        variation = prefs.get("variation") or "moderate"
    
    print(f"DEBUG: Using tone={tone}, length={length}, variation={variation}")
    
    pending_selections[user_id] = {
        "tone": tone,
        "length": length,
        "variation": variation,
        "original_text": original_text,
        "chat_id": chat_id
    }
    
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "• Urgent •" if pending_selections[user_id]["tone"] == "urgent" else "Urgent", 
                 "callback_data": "tone_urgent"},
                {"text": "• Casual •" if pending_selections[user_id]["tone"] == "casual" else "Casual", 
                 "callback_data": "tone_casual"},
                {"text": "• Formal •" if pending_selections[user_id]["tone"] == "formal" else "Formal", 
                 "callback_data": "tone_formal"}
            ],
            [
                {"text": "• Short •" if pending_selections[user_id]["length"] == "short" else "Short", 
                 "callback_data": "length_short"},
                {"text": "• Medium •" if pending_selections[user_id]["length"] == "medium" else "Medium", 
                 "callback_data": "length_medium"},
                {"text": "• Long •" if pending_selections[user_id]["length"] == "long" else "Long", 
                 "callback_data": "length_long"}
            ],
            [
                {"text": "• Conservative •" if pending_selections[user_id]["variation"] == "conservative" else "Conservative", 
                 "callback_data": "var_conservative"},
                {"text": "• Moderate •" if pending_selections[user_id]["variation"] == "moderate" else "Moderate", 
                 "callback_data": "var_moderate"},
                {"text": "• Aggressive •" if pending_selections[user_id]["variation"] == "aggressive" else "Aggressive", 
                 "callback_data": "var_aggressive"}
            ],
            [
                {"text": "✅ Generate with these settings", "callback_data": "generate"}
            ]
        ]
    }
    
    tone = pending_selections[user_id]['tone'] or "casual"
    length = pending_selections[user_id]['length'] or "medium"
    variation = pending_selections[user_id]['variation'] or "moderate"
    
    message = (
        "🎨 <b>Choose Your Style</b>\n\n"
        f"<b>Tone:</b> {tone.title()}\n"
        f"<b>Length:</b> {length.title()}\n"
        f"<b>Variation:</b> {variation.title()}\n\n"
        "Tap buttons to change, then Generate."
    )
    
    await telegram_send_message(chat_id, message, reply_markup=keyboard, parse_mode="HTML")


async def update_style_selector(callback_query: dict) -> None:
    """Update the style selector UI after button press."""
    user_id = callback_query["from"]["id"]
    message_id = callback_query["message"]["message_id"]
    chat_id = callback_query["message"]["chat"]["id"]
    
    if user_id not in pending_selections:
        await telegram_send_message(chat_id, "⚠️ Selection expired. Please forward the message again.")
        return
    
    # Update keyboard with new selections
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "• Urgent •" if pending_selections[user_id]["tone"] == "urgent" else "Urgent", 
                 "callback_data": "tone_urgent"},
                {"text": "• Casual •" if pending_selections[user_id]["tone"] == "casual" else "Casual", 
                 "callback_data": "tone_casual"},
                {"text": "• Formal •" if pending_selections[user_id]["tone"] == "formal" else "Formal", 
                 "callback_data": "tone_formal"}
            ],
            [
                {"text": "• Short •" if pending_selections[user_id]["length"] == "short" else "Short", 
                 "callback_data": "length_short"},
                {"text": "• Medium •" if pending_selections[user_id]["length"] == "medium" else "Medium", 
                 "callback_data": "length_medium"},
                {"text": "• Long •" if pending_selections[user_id]["length"] == "long" else "Long", 
                 "callback_data": "length_long"}
            ],
            [
                {"text": "• Conservative •" if pending_selections[user_id]["variation"] == "conservative" else "Conservative", 
                 "callback_data": "var_conservative"},
                {"text": "• Moderate •" if pending_selections[user_id]["variation"] == "moderate" else "Moderate", 
                 "callback_data": "var_moderate"},
                {"text": "• Aggressive •" if pending_selections[user_id]["variation"] == "aggressive" else "Aggressive", 
                 "callback_data": "var_aggressive"}
            ],
            [
                {"text": "✅ Generate with these settings", "callback_data": "generate"}
            ]
        ]
    }
    
    tone = pending_selections[user_id]['tone'] or "casual"
    length = pending_selections[user_id]['length'] or "medium"
    variation = pending_selections[user_id]['variation'] or "moderate"
    
    message = (
        "🎨 <b>Choose Your Style</b>\n\n"
        f"<b>Tone:</b> {tone.title()}\n"
        f"<b>Length:</b> {length.title()}\n"
        f"<b>Variation:</b> {variation.title()}\n\n"
        "Tap buttons to change, then Generate."
    )
    
    # Edit the message
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": message,
        "reply_markup": keyboard,
        "parse_mode": "HTML"
    }
    
    async with httpx.AsyncClient() as http:
        await http.post(url, json=payload)


def build_prompt(masked_text: str, style: str, force_short: bool = False, max_chars: int = 280,
                 user_tone: Optional[str] = None, user_length: Optional[str] = None, 
                 user_variation: Optional[str] = None) -> str:
    # Multi-dimensional randomization for better spam evasion
    # If user preferences provided, use those instead of random
    
    # Structural changes based on variation level
    if user_variation == "conservative":
        structure = "minimal restructure (keep mostly same order)"
    elif user_variation == "aggressive":
        structure = random.choice([
            "significantly restructure (reorder sentences/paragraphs if logical)",
            "move the main point to the beginning (if it doesn't change meaning)",
            "move the main point to the end (if it doesn't change meaning)",
            "reorganize ideas while maintaining logical flow"
        ])
    else:  # moderate or None (random)
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
    
    # Length variation - use user preference if provided
    if user_length == "short":
        length = random.choice([
            "make it concise and brief",
            "compress into fewer sentences",
            f"condense to under {int(max_chars * 0.6)} chars",
            "remove all filler words",
            "trim to essential points only"
        ])
    elif user_length == "long":
        length = random.choice([
            f"make it more detailed and descriptive (BUT UNDER {max_chars} CHARS)",
            f"expand key points naturally (STAY UNDER {max_chars} CHARS)",
            f"add relevant context (MAX {max_chars} CHARACTERS)",
            f"elaborate on concepts (MUST BE UNDER {max_chars} CHARS)"
        ])
    elif user_length == "medium":
        length = "keep similar length but vary sentence structure"
    elif force_short:
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
    
    # Tone - use user preference if provided
    if user_tone == "urgent":
        tone = random.choice([
            "urgent and direct",
            "time-sensitive and action-oriented",
            "immediate and pressing"
        ])
    elif user_tone == "casual":
        tone = random.choice([
            "casual and friendly",
            "conversational and approachable",
            "informal and relaxed"
        ])
    elif user_tone == "formal":
        tone = random.choice([
            "very formal and academic",
            "professional and polished",
            "diplomatic and balanced"
        ])
    else:  # Random
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


@app.on_event("startup")
async def startup_event():
    """Test Supabase connection on startup and start background tasks"""
    # Set bot commands (appears in Telegram UI)
    # Note: /settings is NOT in the command list - only shown as button for Pro users
    commands_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setMyCommands"
    commands = {
        "commands": [
            {"command": "start", "description": "Start the bot"}
        ]
    }
    try:
        async with httpx.AsyncClient() as http:
            await http.post(commands_url, json=commands)
        print("✓ Bot commands registered in Telegram")
    except Exception as e:
        print(f"⚠ Failed to register bot commands: {e}")
    
    if supabase_client:
        try:
            # Test connection by querying the users table
            result = supabase_client.table("users").select("user_id").limit(1).execute()
            print(f"✓ Supabase connected - 'users' table accessible ({len(result.data)} rows)")
            
            # Test activity_logs table
            result2 = supabase_client.table("activity_logs").select("id").limit(1).execute()
            print(f"✓ Supabase connected - 'activity_logs' table accessible ({len(result2.data)} rows)")
            
            print("✓✓ Database tables verified and ready!")
            
            # Start background task to flush logs every 5 minutes
            import asyncio
            async def periodic_flush():
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    flush_activity_logs()
            
            asyncio.create_task(periodic_flush())
            print("✓ Background log flush task started (every 5 minutes)")
        except Exception as e:
            print(f"⚠ Supabase connection error: {e}")
            print("⚠ Make sure you've created the tables in Supabase (see setup instructions)")


@app.on_event("shutdown")
async def shutdown_event():
    """Flush any remaining logs on shutdown"""
    if supabase_client:
        print("Flushing remaining activity logs...")
        flush_activity_logs()
        print("✓ Shutdown complete")


@app.get("/")
def health():
    """Health check endpoint"""
    status = {"ok": True, "supabase_connected": supabase_client is not None}
    
    # Test database tables if connected
    if supabase_client:
        try:
            users_count = supabase_client.table("users").select("user_id", count="exact").limit(0).execute()
            activity_count = supabase_client.table("activity_logs").select("id", count="exact").limit(0).execute()
            status["database"] = {
                "users_table": "ok",
                "activity_logs_table": "ok",
                "total_users": users_count.count,
                "total_activity_logs": activity_count.count
            }
        except Exception as e:
            status["database"] = {"error": str(e)}
    
    return status


@app.post(f"/webhook/{WEBHOOK_SECRET}")
async def webhook(req: Request):
    # Optional hardening: verify Telegram secret header (if you set it during setWebhook)
    if TELEGRAM_WEBHOOK_SECRET_TOKEN:
        header = req.headers.get("x-telegram-bot-api-secret-token")
        if header != TELEGRAM_WEBHOOK_SECRET_TOKEN:
            return {"ok": True}

    update = await req.json()

    # Handle callback queries (button presses)
    if "callback_query" in update:
        callback = update["callback_query"]
        user_id = callback["from"]["id"]
        data = callback["data"]
        
        if user_id not in pending_selections:
            # Selection expired
            chat_id = callback["message"]["chat"]["id"]
            await telegram_send_message(chat_id, "⚠️ Selection expired. Please forward the message again.")
            return {"ok": True}
        
        # Handle button presses
        if data.startswith("tone_"):
            pending_selections[user_id]["tone"] = data.split("_")[1]
            await update_style_selector(callback)
        elif data.startswith("length_"):
            pending_selections[user_id]["length"] = data.split("_")[1]
            await update_style_selector(callback)
        elif data.startswith("var_"):
            pending_selections[user_id]["variation"] = data.split("_")[1]
            await update_style_selector(callback)
        elif data == "generate":
            # User confirmed - save preferences
            # Safety check: Only allow pro users to save preferences (exempt users are NOT pro)
            is_pro = is_pro_user(user_id) if user_id else False
            if not is_pro:
                chat_id = callback["message"]["chat"]["id"]
                await telegram_send_message(chat_id, "⚠️ This feature is only available for Pro users.")
                return {"ok": True}
            
            selection = pending_selections[user_id]
            save_user_preferences(user_id, 
                                tone=selection["tone"],
                                length=selection["length"],
                                variation=selection["variation"])
            
            chat_id = selection["chat_id"]
            user_text = selection["original_text"]
            
            # Delete the selection message
            message_id = callback["message"]["message_id"]
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteMessage"
            async with httpx.AsyncClient() as http:
                await http.post(url, json={"chat_id": chat_id, "message_id": message_id})
            
            # Check if this was from /settings command (placeholder text)
            if user_text == "📝 Send me the text you want to rephrase after choosing your style.":
                # Just confirm and wait for user to send actual text
                await telegram_send_message(chat_id, "✅ Preferences saved! Now send me the text you want to rephrase.")
                # Clean up and return
                del pending_selections[user_id]
                return {"ok": True}
            
            # Continue to rephrasing logic below by creating a synthetic message
            message = {
                "from": callback["from"],
                "chat": callback["message"]["chat"],
                "text": user_text,
                "forward_origin": {"type": "user"},  # Fake forward to bypass check
                "_skip_style_selector": True  # Flag to skip showing selector again
            }
            # Don't return, let it fall through to rephrasing logic
        else:
            return {"ok": True}
        
        # If not "generate", we're done
        if data != "generate":
            return {"ok": True}
    else:
        message = update.get("message") or update.get("edited_message")
    
    if not message:
        return {"ok": True}

    chat_id = (message.get("chat") or {}).get("id")
    if chat_id is None:
        return {"ok": True}

    # Get user_id early to check for exemptions
    user_id = (message.get("from") or {}).get("id")
    is_exempt_user = user_id and user_id in EXEMPT_USER_IDS
    
    print(f"DEBUG: Received message from user_id={user_id}, is_exempt={is_exempt_user}, EXEMPT_USER_IDS={EXEMPT_USER_IDS}")
    
    # Extract user info for tracking
    user_from = message.get("from") or {}
    username = user_from.get("username")
    first_name = user_from.get("first_name", "Unknown")
    last_name = user_from.get("last_name")
    language_code = user_from.get("language_code")
    
    # Track user (only writes if new or info changed)
    if user_id:
        track_user(user_id, username, first_name, last_name, language_code)

    # Prefer text; fall back to caption (forwarded media captions).
    user_text = message.get("text") or message.get("caption") or ""
    print(f"DEBUG: user_text='{user_text}', length={len(user_text)}")
    
    if not user_text.strip():
        print(f"DEBUG: Empty text, sending error message")
        await telegram_send_message(chat_id, "Send a text message (or media caption) to rephrase it.")
        return {"ok": True}

    # Simple commands
    if user_text.strip().lower() in ("/start", "start"):
        # Log /start command
        if user_id:
            log_activity(user_id=user_id, action_type="command_start")
        
        is_pro = is_pro_user(user_id) if user_id else False
        if is_pro:
            # Show persistent keyboard with Settings button for Pro users
            keyboard = {
                "keyboard": [
                    [{"text": "⚙️ Settings"}]
                ],
                "resize_keyboard": True,
                "persistent": True
            }
            await telegram_send_message(chat_id, 
                "🤖 <b>RephraseBot</b>\n\n"
                "Send me any text and I'll rephrase it for you!\n\n"
                "💎 <b>Pro Features:</b>\n"
                "• Custom style preferences (tone, length, variation)\n"
                "• Tap the <b>⚙️ Settings</b> button below to change preferences",
                reply_markup=keyboard,
                parse_mode="HTML")
        elif is_exempt_user:
            # Exception users: show Weekly report button
            keyboard = {
                "keyboard": [[{"text": "📊 Weekly report"}]],
                "resize_keyboard": True,
                "persistent": True,
            }
            await telegram_send_message(
                chat_id,
                "🤖 <b>RephraseBot</b>\n\nSend me any text and I'll rephrase it.\n\n📊 Tap <b>Weekly report</b> below for last 7 days (new / active users).",
                reply_markup=keyboard,
                parse_mode="HTML",
            )
        else:
            await telegram_send_message(chat_id, "Bot is running. Send any message to rephrase it.")
        return {"ok": True}
    
    # /settings or /preferences command - show style selector for exempt/Pro users
    if user_text.strip().lower() in ("/settings", "/preferences", "/style", "⚙️ settings"):
        is_pro = is_pro_user(user_id) if user_id else False
        if is_pro:
            # Show a placeholder message to attach the selector to
            placeholder_text = "📝 Send me the text you want to rephrase after choosing your style."
            await show_style_selector(chat_id, user_id, placeholder_text)
            return {"ok": True}
        else:
            # Don't reveal that this feature exists - just say command not found
            await telegram_send_message(chat_id, "Unknown command. Send me a message to rephrase it.")
            return {"ok": True}

    # /report or "📊 Weekly report" – only for exempt (exception) users
    if user_text.strip().lower() in ("/report", "report") or user_text.strip() == "📊 Weekly report":
        if not is_exempt_user:
            await telegram_send_message(chat_id, "Unknown command. Send me a message to rephrase it.")
            return {"ok": True}
        rows = get_weekly_activity_report()
        if not rows:
            await telegram_send_message(chat_id, "📊 No activity data available for the past week.")
            return {"ok": True}
        # Format: title + table (Date | New | Active), minimal and visual
        start_d = rows[0][0]
        end_d = rows[-1][0]
        table_lines = ["Date       New  Active", "─" * 18] + [
            "{}   {:>3}   {:>6}".format(d, new, active) for d, new, active in rows
        ]
        msg = (
            "📊 <b>Weekly activity</b> ({} – {})\n\n"
            "<code>{}</code>"
        ).format(start_d, end_d, "\n".join(table_lines))
        await telegram_send_message(chat_id, msg, parse_mode="HTML")
        return {"ok": True}

    # Check for forwarded media (reject like Persian messages)
    if has_forwarded_media(message):
        # Log forwarded media error
        if user_id:
            log_activity(
                user_id=user_id,
                action_type="forwarded_media_error",
                error_type="forwarded_media",
                original_length=len(user_text)
            )
        await telegram_send_message(chat_id, INVALID_TWEET_INPUT_MESSAGE)
        return {"ok": True}
    
    # Check if message contains X/Twitter link (early detection)
    tweet_id = extract_tweet_id(user_text)
    
    # Check for multiple x.com links (extract_tweet_id returns None if more than one)
    pattern = r'(?:twitter\.com|x\.com|mobile\.twitter\.com)/(?:\w+)/status/(\d+)'
    link_count = len(re.findall(pattern, user_text, re.IGNORECASE))
    
    if link_count > 1:
        # Reject messages with more than one x.com link
        if user_id:
            log_activity(
                user_id=user_id,
                action_type="multiple_links_error",
                error_type="multiple_x_links",
                original_length=len(user_text)
            )
        await telegram_send_message(chat_id, INVALID_TWEET_INPUT_MESSAGE)
        return {"ok": True}
    
    original_text_with_link = user_text  # Keep original for context
    has_link = tweet_id is not None
    
    # If X link detected, remove it from the text before rephrasing
    if tweet_id:
        # Remove X/Twitter URLs from the text
        pattern = r'https?://(?:twitter\.com|x\.com|mobile\.twitter\.com)/\S+\s*'
        user_text = re.sub(pattern, '', user_text, flags=re.IGNORECASE).strip()
        print(f"DEBUG: Detected X link (tweet_id={tweet_id}), removed URL from text for rephrasing")

    # Input validation:
    # - Must be an English reply (reject Arabic/Farsi script)
    # - If a tweet link is provided, it must include reply text too (not link-only)
    if tweet_id and not user_text.strip():
        await telegram_send_message(chat_id, INVALID_TWEET_INPUT_MESSAGE)
        return {"ok": True}

    if contains_arabic_script(user_text):
        # Log Persian/Arabic character error
        if user_id:
            log_activity(
                user_id=user_id,
                action_type="persian_error",
                error_type="persian_chars",
                original_length=len(user_text)
            )
        await telegram_send_message(chat_id, INVALID_TWEET_INPUT_MESSAGE)
        return {"ok": True}

    # For Pro users only, show style selector ONLY if they don't have saved preferences yet
    # This allows first-time Pro users to set preferences, then uses them automatically
    # Users can change preferences anytime with /settings command
    skip_selector = message.get("_skip_style_selector", False)
    is_pro = is_pro_user(user_id) if user_id else False
    
    # Clear stale pending selections (in case of previous errors)
    if user_id in pending_selections:
        print(f"DEBUG: Clearing stale pending selection for user {user_id}")
        del pending_selections[user_id]
    
    # Check if user has saved preferences (Pro users only)
    has_saved_prefs = False
    if is_pro and user_id:
        prefs = get_user_preferences(user_id)
        has_saved_prefs = prefs is not None and any([prefs.get("tone"), prefs.get("length"), prefs.get("variation")])
    
    print(f"DEBUG: user_id={user_id}, is_exempt={is_exempt_user}, is_pro={is_pro}, has_saved_prefs={has_saved_prefs}, skip_selector={skip_selector}")
    
    # Show selector ONLY for first-time Pro users (no saved preferences yet)
    if is_pro and user_id not in pending_selections and not skip_selector and not has_saved_prefs:
        print(f"DEBUG: First-time user - showing style selector for user {user_id}")
        try:
            # Show style selector menu
            await show_style_selector(chat_id, user_id, user_text)
        except Exception as e:
            print(f"ERROR: Failed to show style selector: {e}")
            # Fall through to normal rephrasing if style selector fails
        else:
            return {"ok": True}
    
    # Forward requirement: Only act on forwarded messages (exempt users and Pro users bypass this)
    if not is_exempt_user and not is_pro:
        if not is_forwarded(message):
            await telegram_send_message(chat_id, "Please forward a message to me, and I'll rephrase it.")
            return {"ok": True}
        
        # Channel restriction: Check if message is from allowed channel (if configured)
        if ALLOWED_FORWARD_CHANNEL:
            if not is_forwarded_from_allowed_channel(message):
                # Log invalid channel error
                if user_id:
                    log_activity(
                        user_id=user_id,
                        action_type="invalid_channel",
                        error_type="wrong_channel"
                    )
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
            # Log rate limit hit
            log_activity(
                user_id=user_id,
                action_type="rate_limited",
                error_type="rate_limit",
                error_message=f"Wait {seconds_remaining} seconds"
            )
            await telegram_send_message(
                chat_id,
                f"⏱️ Please wait {seconds_remaining} second{'s' if seconds_remaining != 1 else ''} before sending another message.\n\n"
                f"Rate limit: 1 message per {RATE_LIMIT_SECONDS} seconds."
            )
            return {"ok": True}
    
    # Load user preferences for rephrasing - ONLY for pro users
    # Non-pro users (including exempt users) should not use saved preferences even if they exist in database
    # Exempt users only bypass rate limiting, they are NOT pro users
    user_tone = None
    user_length = None
    user_variation = None
    if user_id:
        is_pro = is_pro_user(user_id)
        if is_pro:
            user_prefs = get_user_preferences(user_id)
            if user_prefs:
                user_tone = user_prefs.get("tone")
                user_length = user_prefs.get("length")
                user_variation = user_prefs.get("variation")

    # Apply random tag removal to tag sequences (start, middle, end)
    # Behavior depends on whether message has a link:
    # - With link: 40% removal for both # and @ in groups of 2+
    # - Without link: 40% removal for # only, keep ALL @ mentions
    user_text = apply_random_tag_removal(user_text, removal_chance=0.4, has_link=has_link)

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

    # Track request start time for performance logging
    request_start_time = time.time()

    try:
        rewritten_body = ""
        
        # Generate 2 candidates and select the one with lowest similarity score
        all_candidates = []
        
        # Generate 2 candidates with varying styles for maximum diversity
        for candidate_num in range(2):
            # Use different style for each candidate to increase diversity
            style = random.choice(STYLES)
            current_prompt = build_prompt(masked.masked, style=style, force_short=False, max_chars=available_chars,
                                        user_tone=user_tone, user_length=user_length, user_variation=user_variation)
            
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
        
        # Calculate response time
        response_time_ms = int((time.time() - request_start_time) * 1000)
        
        # Log successful rephrase
        if user_id:
            log_activity(
                user_id=user_id,
                action_type="rephrase_success",
                original_length=len(user_text),
                rephrased_length=len(final_message),
                similarity_score=best_similarity if 'best_similarity' in locals() else None,
                response_time_ms=response_time_ms,
                had_x_link=tweet_id is not None,
                was_forwarded=is_forwarded(message)
            )
        
        # Update rate limit timestamp after successful processing
        if user_id:
            update_rate_limit(user_id)
            # Clean up pending selections
            if user_id in pending_selections:
                del pending_selections[user_id]

    except Exception as exc:
        import traceback
        traceback.print_exc()  # Log to console
        
        # Log failed rephrase
        if user_id:
            response_time_ms = int((time.time() - request_start_time) * 1000)
            log_activity(
                user_id=user_id,
                action_type="rephrase_failed",
                error_type="api_error",
                error_message=f"{type(exc).__name__}: {str(exc)[:200]}",
                original_length=len(user_text) if user_text else None,
                response_time_ms=response_time_ms
            )
        
        await telegram_send_message(chat_id, f"Error: {type(exc).__name__} - {str(exc)[:100]}")

    return {"ok": True}