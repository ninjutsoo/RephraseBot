## RephraseBot – Reactivation & Multi‑Tweet Engagement Design

This document is the implementation guide for a **clean, multi‑tweet reactivation flow** that makes it extremely easy for previously active but now inactive users to rephrase replies for several key tweets per day, without forwarding from the channel.

It is written to be used later when you (or another engineer) implement the changes in `main.py` and related scripts.

---

### 1. High‑Level Product Behavior

#### 1.1 Daily Channel Workflow (Already in Place)

- Around **9pm EST**, your forward‑only channel publishes several posts that look like:
  - A message containing an **X / Twitter link**.
  - A **reply text** immediately below it.
- Only these “X link + reply” pairs are saved into Supabase table `public.daily_channel_posts`.

You already have:

- `daily_channel_posts` (per‑day curated tweets + replies).
- `activity_logs` (all bot activity and rephrases).
- `users` (Telegram users + status).

#### 1.2 New Behavior (Target Flow)

For **inactive but previously active** users:

1. Once per day (e.g. shortly after 9pm EST), the bot sends a **single reminder DM**.
2. The reminder shows today’s curated tweets as **inline buttons**:
   - `Tweet 1`, `Tweet 2`, …, `Tweet N` (N usually 2–5).
3. When the user taps `Tweet i`:
   - The bot **generates a rephrase** for that tweet’s `reply_text` using the existing logic.
   - It **edits the same reminder message** to display the rephrased text plus:
     - `💬 Reply on X`
     - `🔁 Quote on X`
   - The `Tweet 1 … Tweet N` buttons remain visible, so tapping another one replaces the rephrase.
4. Optional extra button:
   - `Stop these reminders` → opt‑out from reactivation DMs.

For **active** users (recently used the bot), nothing changes; they keep using the standard forward‑from‑channel flow.

---

### 2. Existing Tables & How We Use Them

#### 2.1 `public.daily_channel_posts`

```sql
create table public.daily_channel_posts (
  id bigserial not null,
  day_est date not null,
  day_index integer not null,
  channel_chat_id bigint not null,
  channel_message_id bigint not null,
  channel_message_date timestamp with time zone not null,
  tweet_url text not null,
  reply_text text not null,
  created_at timestamp with time zone null default now(),
  constraint daily_channel_posts_pkey primary key (id)
) TABLESPACE pg_default;

create unique INDEX IF not exists uq_daily_channel_posts_day_msg
  on public.daily_channel_posts using btree (day_est, channel_chat_id, channel_message_id);

create index IF not exists idx_daily_channel_posts_day
  on public.daily_channel_posts using btree (day_est, day_index);

create index IF not exists idx_daily_channel_posts_message_date
  on public.daily_channel_posts using btree (channel_message_date);
```

**Usage in this design**:

- Source of **curated tweets** for reactivation:
  - Today’s posts: `where day_est = current_date_est`.
  - Ordered by `day_index`.
- Each row provides:
  - `tweet_url`: used to compute `tweet_id` (for X reply/quote buttons).
  - `reply_text`: the “canonical reply” that RephraseBot will rephrase on demand.

No schema changes required.

#### 2.2 `public.activity_logs`

```sql
create table public.activity_logs (
  id bigserial not null,
  user_id bigint not null,
  timestamp timestamp with time zone null default now(),
  action_type text not null,
  original_length integer null,
  rephrased_length integer null,
  similarity_score double precision null,
  response_time_ms integer null,
  error_type text null,
  error_message text null,
  had_x_link boolean null default false,
  was_forwarded boolean null default false,
  created_at timestamp with time zone null default now(),
  bot_type text not null default 'tweet'::text,
  original_tweet_url text null,
  original_tweet_text text null,
  rephrased_text text null,
  constraint activity_logs_pkey primary key (id)
) TABLESPACE pg_default;
```

**Existing indexes** (user, timestamp, action_type, bot_type, tweet_url, etc.) are sufficient.

**New action_types to use** (no schema change, just values):

- `reactivation_reminder_sent`
- `reactivation_button_clicked` (first time user presses any Tweet button in a reminder)
- `reactivation_another_variation` (user presses some Tweet button again to get a new variation)
- Optionally: `reactivation_opt_out` (user taps “Stop these reminders”)

Where useful, populate:

- `bot_type = 'tweet'`
- `original_tweet_url = daily_channel_posts.tweet_url`
- `original_tweet_text = daily_channel_posts.reply_text`
- `rephrased_text = final_message` (for actual rephrases only).

These reuse all your existing indexes for ROI analysis.

#### 2.3 `public.users`

```sql
create table public.users (
  user_id bigint not null,
  username text null,
  first_name text not null,
  last_name text null,
  language_code text null,
  is_blocked boolean null default false,
  created_at timestamp with time zone null default now(),
  is_pro boolean null default false,
  pro_expires_at timestamp with time zone null,
  trial_ends_at timestamp with time zone null,
  preferred_tone text null,
  preferred_length text null,
  preferred_variation text null,
  constraint users_pkey primary key (user_id)
) TABLESPACE pg_default;
```

No change required for this feature.

If you want a hard opt‑out at the DB level, you can **optionally** add:

- `reactivation_opt_out boolean not null default false`

but this is not strictly required if you rely on `activity_logs` to remember opt‑outs.

---

### 3. User Segmentation Logic

We distinguish three groups:

1. **New users** – never had a `rephrase_success`.
2. **Active users** – at least one `rephrase_success` in the last **N days** (e.g. 7 or 14).
3. **Inactive but previously active** – at least one lifetime `rephrase_success`, but none in the last N days.

Only group **3** receives reactivation reminders.

#### 3.1 “Previously active”

SQL idea (Supabase RPC or Python query):

- Any row in `activity_logs` with:
  - `action_type = 'rephrase_success'`
  - `bot_type = 'tweet'`

This gives the set of “ever active” users.

#### 3.2 “Inactive within N days”

For a given `now` in EST:

- Compute `cutoff = now_est - interval 'N days'`.
- For each `user_id` in “ever active”:
  - If **no** `rephrase_success` with `timestamp >= cutoff` → user is inactive.

#### 3.3 Exclusions

Exclude users if:

- `users.is_blocked = true`.
- They generated a `bot_blocked_by_user` in `activity_logs` (you already log that).
- They have:
  - `reactivation_opt_out` in `users` (if you add the column), or
  - a recent `reactivation_opt_out` event in `activity_logs`.
- They already got:
  - a `reactivation_reminder_sent` in the last 24h.
  - more than **D** reminders in the last **K** days (e.g. D=3, K=14).

All of this can be implemented with queries on `activity_logs` + `users` and does not require new tables.

---

### 4. Reminder Message Design (Multi‑Tweet Buttons)

#### 4.1 Which tweets to show

For a given day (EST):

1. Determine `today_est` as date in `America/New_York`.
2. Select from `daily_channel_posts`:

   - `where day_est = today_est`
   - `order by day_index asc`
   - `limit N` (e.g. 3–5 tweets)

3. For each selected row `p`:
   - `p.id` will be used in callback data.
   - `p.tweet_url` is for deriving `tweet_id`.
   - `p.reply_text` is the “base reply” to rephrase.

#### 4.2 Reminder DM layout

**Message text (example):**

> “Here are today’s tweets from the channel. Tap any Tweet button to get an instant rephrased reply ready for X.  
> You can tap different tweets to switch between rephrases.”

**Inline keyboard:**

- Row 1: `Tweet 1` `Tweet 2` `Tweet 3` …
- Row 2: `Stop these reminders` (optional)

Each `Tweet i` button:

- `text`: `"Tweet 1"`, `"Tweet 2"`, …
- `callback_data`: e.g. `"rt:<postId>:0"` where:
  - Prefix `"rt"` = **r**eactivation **t**weet.
  - `<postId>` = `daily_channel_posts.id`.
  - Last segment (`0`) is a variation index (0 for first rephrase, 1/2/... for further variations of the same tweet).

The bot **does not call Gemini** when sending reminders. It only generates a rephrase **after a button press**.

#### 4.3 Behavior on button press

For every `rt:<postId>:<variant>` callback:

1. Load `daily_channel_posts` row by `id = postId`.
2. Use `reply_text` as the **text to rephrase** and `tweet_url` to generate X buttons.
3. Run the existing rephrase pipeline (described in section 5).
4. `editMessageText` on the same reminder message:
   - New text: the rephrased reply.
   - New inline keyboard:
     - Row 1: same `Tweet 1 .. Tweet N` buttons (so user can switch tweets).
     - Row 2:
       - Maybe `Another variation` (if you want).
       - `Stop these reminders` (optional).
5. Log:
   - First time user ever presses any tweet in this reminder:
     - `reactivation_button_clicked` with tweet info.
   - Every subsequent press:
     - `reactivation_another_variation` with tweet info.
   - Normal `rephrase_success` event (with `original_tweet_url`, `original_tweet_text`, `rephrased_text`).

Tapping another tweet button simply repeats steps 1–4 with a different `postId`, overwriting the previous rephrase in the same message.

---

### 5. Code‑Level Design in `main.py`

This section assumes the current structure in `main.py` (FastAPI app, `webhook` handler, Gemini integration, etc.).

#### 5.1 Extract reusable rephrase helper

Right now, the rephrase logic is embedded directly in the `webhook` function, roughly starting around:

```12:2100:C:\Users\mrosh\OneDrive\Documents\GitHub\RephraseBot\main.py
    # Apply random tag removal ...
    user_text = apply_random_tag_removal(user_text, removal_chance=0.4, has_link=has_link)
    # ...
    try:
        rewritten_body = ""
        # Generate 2 candidates ...
        # ...
        await telegram_send_message(chat_id, final_message, reply_markup=reply_markup)
        # ...
```

**Goal**: extract this into a function you can call in two contexts:

1. Normal flow (forwarded message from channel).
2. Reactivation callback flow (using `daily_channel_posts.reply_text` + `tweet_url`).

##### 5.1.1 Proposed helper signature

```python
async def handle_rephrase_request(
    *,
    user_id: int,
    chat_id: int,
    original_text: str,
    original_tweet_url: str | None,
    is_forwarded_message: bool,
    skip_forward_requirement: bool = False,
) -> None:
    ...
```

Behavior:

- Internally:
  - Derive `tweet_id` from `original_tweet_url` (or from `original_text` if provided).
  - Handle:
    - language checks (`contains_arabic_script`)
    - tag processing (`apply_random_tag_removal`, `extract_tag_blocks`, `mask_protected`, etc.)
    - candidate generation via `gemini_generate_candidates` and similarity scoring.
    - X Web Intent buttons.
  - Call `telegram_send_message(chat_id, final_message, reply_markup=reply_markup)`.
  - Log `rephrase_success` in `activity_logs` with:
    - `original_length`
    - `rephrased_length`
    - `similarity_score`
    - `response_time_ms`
    - `had_x_link`
    - `was_forwarded`
    - `bot_type = 'tweet'`
    - `original_tweet_url`
    - `original_tweet_text = original_text`
    - `rephrased_text = final_message`
- For reactivation callbacks, call it with:
  - `skip_forward_requirement=True` and `is_forwarded_message=False`.
- For normal flow, call it with:
  - `skip_forward_requirement=False` and `is_forwarded_message=is_forwarded(message)`.

This refactor is mechanical but important to avoid duplicating complex logic.

#### 5.2 Extend callback_query handling for reactivation

At the top of `webhook`, you already have:

```12:1630:C:\Users\mrosh\OneDrive\Documents\GitHub\RephraseBot\main.py
    if "callback_query" in update:
        callback = update["callback_query"]
        user_id = callback["from"]["id"]
        data = callback["data"]
        ...
        # style selector handling
```

Add a branch **before** the style‑selector logic, something like:

```python
        if data.startswith("rt:"):
            # Reactivation tweet callback
            await handle_reactivation_callback(callback)
            return {"ok": True}
```

Where `handle_reactivation_callback` is a new async function:

```python
async def handle_reactivation_callback(callback: dict) -> None:
    user_id = callback["from"]["id"]
    chat_id = callback["message"]["chat"]["id"]
    message_id = callback["message"]["message_id"]
    data = callback["data"]  # e.g. "rt:12345:0"

    # Parse callback_data
    try:
        _, post_id_str, variant_str = data.split(":")
        post_id = int(post_id_str)
        variant_index = int(variant_str)
    except Exception:
        # Ignore malformed callback
        return

    # TODO: fetch daily_channel_posts row for post_id via supabase_client
    # - tweet_url
    # - reply_text

    # TODO: call handle_rephrase_request with:
    # - original_text = reply_text
    # - original_tweet_url = tweet_url
    # - is_forwarded_message = False
    # - skip_forward_requirement = True

    # TODO: use Telegram editMessageText to update callback["message"]
    #       instead of sending a new message:
    #       - text = final rephrased message (you may need to return it from handle_rephrase_request,
    #         or factor out the "send" step so helper returns text+reply_markup)
    #       - reply_markup: inline keyboard with Tweet1..TweetN, maybe Stop button

    # TODO: log activity:
    # - reactivation_button_clicked or reactivation_another_variation
```

Implementation detail: the current helper pipeline sends the message directly. For editing behavior, you can either:

1. Modify `handle_rephrase_request` to return `(final_message, reply_markup)` instead of sending, and let the caller decide to `sendMessage` or `editMessageText`, **or**
2. Create a thin variant `generate_rephrase_text(...)->(final_message, reply_markup)` reused by both.

Either way, reactivation flow must use `editMessageText` to avoid chat spam.

#### 5.3 Reminder sending script (`reactivation_job.py` or a FastAPI endpoint)

Create a new small module, e.g. `reactivation_job.py`, that:

1. Uses `supabase_client` to query:
   - `daily_channel_posts` for today’s tweets (limit N).
   - `activity_logs` + `users` to build the inactive‑but‑previously‑active user list and apply exclusions.
2. For each target user:
   - Sends a DM with the multi‑tweet reminder message and inline keyboard.
   - Logs `reactivation_reminder_sent` in `activity_logs` including `original_tweet_url` of the primary tweet (or leave it null).

You can run this script:

- As a **separate process** on a cron job (Render cron, GitHub Actions, etc.).
- Or expose a **protected FastAPI endpoint** (e.g. `/admin/run-reactivation`) that you call manually or from a scheduler.

Pseudocode outline:

```python
def get_today_posts(limit: int = 3) -> List[dict]:
    # Query daily_channel_posts by day_est (EST)
    ...

def get_inactive_users(now_est, days_inactive=7, max_reminders_per_14_days=3) -> List[int]:
    # Use activity_logs + users to find target user_ids
    ...

async def send_reactivation_reminders():
    posts = get_today_posts()
    if not posts:
        return

    now_est = datetime.now(EST)
    user_ids = get_inactive_users(now_est)

    for user_id in user_ids:
        chat_id = user_id  # For 1:1 Telegram bots chat_id == user_id in typical case
        # Build text + inline keyboard with 'rt:<postId>:0' per tweet
        await telegram_send_message(chat_id, text, reply_markup=keyboard)
        log_activity(
            user_id=user_id,
            action_type="reactivation_reminder_sent",
            bot_type="tweet",
        )
```

You already have `EST` and weekly analytics functions in `main.py`; you can reuse the same timezone logic.

---

### 6. ROI & Analytics

Because you’re only adding new `action_type` values and reusing existing columns, you can compute all the important metrics purely from `activity_logs` + `daily_channel_posts` + `users`.

#### 6.1 Key metrics

- **Reminder reach**:
  - `count(*) where action_type = 'reactivation_reminder_sent' and bot_type = 'tweet'`.
- **Click‑through rate (CTR)**:
  - `reactivation_button_clicked / reactivation_reminder_sent`.
- **Variation interest**:
  - `reactivation_another_variation / reactivation_button_clicked`.
- **Reactivation conversion**:
  - For users who had `reactivation_reminder_sent` in a window, how many had a `rephrase_success` within the following 7 or 14 days?
- **Lift vs baseline**:
  - Compare `rephrase_success` frequency for:
    - a control group (no reminders),
    - vs a treatment group (got reminders),
  - using `user_id` cohorts.

You can extend `activity_report.ipynb` or create a new notebook to visualize these.

---

### 7. Testing Strategy

This section is meant as a checklist for manual and semi‑automated testing once you implement the code.

#### 7.1 Unit‑ish tests in a dev environment

You can use a local token + a test bot and a test Supabase project.

1. **Helper refactor tests**
   - Call `handle_rephrase_request` (or equivalent) with:
     - A simple English reply without tags.
     - A reply containing multiple tags and a tweet URL.
   - Verify:
     - Returned text (or sent text) always ≤ 280 chars.
     - Tags and mentions preserved as intended.
     - X buttons appear when `original_tweet_url` is set.

2. **Callback parsing**
   - Simulate callbacks with `data = "rt:123:0"`, `"rt:123:1"`, malformed strings.
   - Ensure:
     - Correct behavior with valid strings.
     - Malformed strings are safely ignored without crashing.

3. **Supabase queries**
   - `get_today_posts`:
     - Returns the correct N posts for `day_est`.
   - `get_inactive_users`:
     - Given synthetic `activity_logs`, returns the expected user IDs for different inactivity windows.

4. **Logging**
   - After a reactivation callback, check `activity_logs` contains:
     - `reactivation_button_clicked` or `reactivation_another_variation`.
     - `rephrase_success` with `original_tweet_url`, `original_tweet_text`, `rephrased_text`.

#### 7.2 Staging / Telegram manual tests

Use a staging bot and a staging channel that mirrors your real flow.

1. **Daily posts ingestion**
   - Post a tweet link + reply combination in the staging channel at a known time.
   - Confirm a row appears in `daily_channel_posts` with:
     - Correct `day_est`, `day_index`, `tweet_url`, `reply_text`.

2. **Reminder sending**
   - Manually run `send_reactivation_reminders()` (or hit the admin endpoint).
   - As a test user that has historical `rephrase_success` but none in last N days:
     - Confirm you receive one reminder with `Tweet 1 .. Tweet N` buttons.

3. **First Tweet button press**
   - Tap `Tweet 1`.
   - Ensure:
     - The original reminder message is **edited**, not a new message.
     - The rephrased text is shown.
     - `💬 Reply on X` / `🔁 Quote on X` buttons appear and open X with the correct tweet.
     - The `Tweet 1..N` buttons remain available.
     - `activity_logs` has `reactivation_button_clicked` and `rephrase_success`.

4. **Switching tweets**
   - Tap `Tweet 2`, then `Tweet 3`.
   - Verify each tap:
     - Replaces the text in the same message.
     - Uses the correct base `reply_text` for that tweet.
     - Generates correct X intents.
     - Logs `reactivation_another_variation` appropriately.

5. **Rate limiting & spam checks**
   - Ensure:
     - A user receives at most one reminder per day.
     - After tapping **Stop these reminders** (if implemented), no further reminders are sent to that user.

6. **Reactivation behavior**
   - After a reminder:
     - Use the bot normally (forward from channel or paste links).
     - Confirm:
       - `rephrase_success` events appear.
       - Your query for “inactive users” now excludes this user for the next N days.

---

### 8. Summary of Implementation Tasks

From this doc, the concrete work items are:

1. **Refactor rephrase logic** in `main.py` into a reusable helper that can:
   - Take `original_text` + optional `original_tweet_url`.
   - Return `(final_message, reply_markup)` or send it directly.
2. **Add reactivation callback handling**:
   - Parse `rt:<postId>:<variant>` callback data.
   - Fetch `daily_channel_posts` row from Supabase.
   - Call the rephrase helper.
   - Edit the reminder message with the new text and keyboard.
   - Log `reactivation_button_clicked` / `reactivation_another_variation`.
3. **Implement reminder sending script / endpoint**:
   - Select today’s tweets from `daily_channel_posts`.
   - Compute inactive‑but‑previously‑active user set using `activity_logs` + `users`.
   - Apply reminder caps and exclusions.
   - Send a single DM with `Tweet 1..N` buttons.
   - Log `reactivation_reminder_sent`.
4. **(Optional) Add `reactivation_opt_out` flag** to `users` and a `Stop these reminders` callback.
5. **Add analytics views / notebook cells** to compute ROI from `activity_logs`.
6. **Run the test plan** (dev, staging, then production with small cohorts).

Once this is in place, inactive users can re‑engage by simply tapping `Tweet 1..N`, with zero forwarding friction and minimal chat noise.

