## Future Pro Plan

This document describes a referral- and streak-based Pro system designed to:

- Motivate consistent daily tweeting.
- Encourage inviting friends.
- Make both partners accountable to each other.
- Keep the UX simple, fast, and non-spammy.

The plan is written so it can be implemented incrementally.

---

## 1. Core Concepts

### 1.1 Tweet Day vs Email Day

- **Tweet Day (per user)**: A calendar EST day on which a user:
  - Has at least one valid tweet captured from the channel (non-`#Email`, English, etc.).
  - Taps a `daily_post:` button and receives a successful rephrase (`rephrase_success` in `activity_logs` with `bot_type='tweet'`).
- **Email Day**:
  - A day where posts are tagged `#Email` or the user only uses the email bot.
  - **Does not contribute to streaks** (no progress).
  - **Does not break streaks** (ignored when evaluating “missed tweet days”).

Implementation note:

- We already log successful rephrases in `activity_logs`.
- We can derive `tweet_activity(user_id, date_est)` from that table, filtered by `bot_type='tweet'` and `action_type='rephrase_success'`.

### 1.2 Referral Pair

- Users participate in the Pro streak system as **pairs**.
- A **referral pair** is two users (A,B) who:
  - Are both registered in the `users` table.
  - Have agreed to be each other’s accountability partner.
  - Are connected either via a **referral link** or by manually nominating a partner.

#### 1.2.1 Data Model

Minimal extension to `users`:

- `referred_by_user_id bigint null`
- `referral_partner_user_id bigint null`

Optional dedicated table (flexible but not strictly required for v1):

- `referral_pairs`:
  - `id bigserial`
  - `user_a bigint not null`
  - `user_b bigint not null`
  - `pro_active boolean not null default false`
  - `pro_started_at timestamptz null`
  - `last_evaluated_date date null`
  - `consecutive_missed_pair_tweet_days integer not null default 0`

For v1, we can rely on `referral_partner_user_id` both ways and store Pro state in the existing `users` columns (`is_pro`, `pro_expires_at`, `trial_ends_at`), plus derive streak status from activity logs.

---

## 2. Streak Rules and Pro Eligibility

We want:

- A **clear path to Pro** for non-Pro users.
- A **light but real requirement** to keep Pro.
- Email-only days are neutral: they neither help nor hurt.

### 2.1 Earning Pro

Definition (lenient, tweet-focused):

- Consider only **tweet-eligible days** (days where the channel has tweet content; `#Email` days are ignored).
- A pair (A,B) **earns Pro** when there exists a 3-day EST window \(D1, D2, D3\) of **tweet days** such that:
  - On each of D1, D2, D3:
    - `tweet_activity(A, Dn) == true`
    - `tweet_activity(B, Dn) == true`
- “Tweet days” here are days on which at least one non-`#Email` tweet for that day/channel existed in `daily_channel_posts`.
- Days that are purely `#Email` (no tweets in `daily_channel_posts` for that day) are not part of this evaluation; they are skipped entirely.

User-facing explanation (simplified):

- “You and your partner both need to tweet on 3 tweet-days in a row to unlock Pro. Days with only emails don’t hurt your streak.”

### 2.2 Keeping Pro

After Pro is granted to the pair:

- We walk forward over **tweet days** (same definition, ignoring email-only days).
- On each tweet day:
  - If at least one of A or B has `tweet_activity` → streak continues (no penalty).
  - If both A and B have **no** `tweet_activity` → mark the day as **missed pair tweet day**.
- If we observe **two consecutive missed pair tweet days**, then:
  - Pro is **revoked for both users** (set `is_pro=false` or move `pro_expires_at` to a past value).

This encourages:

- At least one of the pair to “show up” on tweet days.
- Avoiding two back-to-back “no one tweeted” days, without punishing email-only days.

Optional extension (duration-based):

- When they first earn Pro, set `pro_expires_at = now() + interval '30 days'`.
- Each time they complete another 3-day aligned tweet streak:
  - Extend `pro_expires_at` by some number of days (e.g. +7).
- Still apply the “two missed tweet days in a row” rule as an early cancellation.

---

## 3. Referral Flow and Accountability

### 3.1 Referral Links

Goals:

- Make inviting frictionless.
- Handle both new and existing users.

Approach:

- Implement `/invite` command:
  - If the user has an existing partner:
    - Show current partner and streak status.
    - Offer to replace partner (optional, stricter: one partner only).
  - Always generate a deep link:
    - `https://t.me/<bot_username>?start=ref_<user_id_or_token>`
- On `/start ref_<token>`:
  - If the new user has **no account yet**:
    - Register them in `users`.
    - Resolve the referrer from the token.
    - Pair them: set `referral_partner_user_id` both ways.
    - Optionally set `referred_by_user_id` on the invitee.
  - If the user already exists:
    - If they don’t have a partner:
      - Pair them with the referrer.
    - If they have a different partner:
      - Either:
        - ignore the new referral, or
        - ask explicitly if they want to switch partner (more UI).

### 3.2 Daily Accountability Message

Trigger:

- **First valid channel tweet of the day** is saved into `daily_channel_posts` (not `#Email`, passes filters).

Then for each user with a partner (A,B):

- Compute or update that day’s `tweet_activity(A, day)` and `tweet_activity(B, day)`.
- Evaluate:
  - Current streak towards first 3 tweet-days.
  - Whether Pro is active or just unlocked today.
  - Whether they are in danger of losing Pro (1 missed tweet day already).

Send a **status DM** to each partner individually:

- **For non-Pro pair (path to Pro)**:
  - Show:
    - Their own progress: e.g. `You: 2 / 3 tweet-days in a row`.
    - Partner’s progress: e.g. `Partner: 1 / 3 tweet-days in a row`.
  - Copy ideas:
    - “You’re 1 tweet-day away from unlocking Pro together.”
    - “Days with only emails don’t affect your streak.”
    - Short reminder of Pro benefits (see section 4).

- **For Pro pair (maintaining)**:
  - Show:
    - Whether they tweeted today.
    - Whether partner tweeted today.
    - Remaining leeway: e.g. “You’ve both missed 1 tweet-day; don’t miss the next one or you’ll lose Pro.”

Include a button:

- “Remind my partner”
  - Tapping it sends a **preset, friendly ping** to the partner:
    - “Hey, we’re working on our 3-day streak to keep Pro. We still need today’s tweet 🙌”
  - Rate-limited to avoid spam (e.g. one ping per day per pair).

---

## 4. Pro Positioning and Benefits (UX Copy)

We want the Pro pitch to feel like a **productivity unlock**, not just a paywall.

Key points to emphasize in messages:

- **Custom style settings**:
  - “As Pro, you set your preferred **tone**, **length**, and **variation** once.”
  - “Every tweet is rephrased with your style automatically.”
- **10-second rhythm**:
  - “There’s a soft 10-second rhythm between tweets: just tap, wait ~10 seconds, and the next tweet is ready.”
  - “No extra messages, no clutter—just one updated result message under your buttons.”
- **No forwarding needed**:
  - “Pro users don’t need to forward tweets anymore.”
  - “Just tap the buttons for today’s posts and get your replies in place.”
- **Time to complete the day**:
  - “With your style set, you can finish a full day’s tweets in under a minute.”
  - “Aim for ~10 seconds per tweet: 5–6 tweets → less than a minute of work.”

These points should be woven into:

- `/invite` explanation.
- Daily status messages for non-Pro pairs.
- Occasional Pro upsell messages for users without partners.

---

## 5. Mode-Specific Behavior (Global vs Test)

The referral and streak logic should behave consistently in both modes, but some infrastructure around it is mode-dependent.

### 5.1 Global Mode (Production)

- `DAILY_MODE = "global"`.
- **Listens to**: `ALLOWED_FORWARD_CHANNEL`.
- **Buttons sent to**: active Pro users only.
- **Cleanup trigger**:
  - First message of the **next EST day** for that channel cleans:
    - Previous day’s approval messages.
    - Buttons messages and associated result messages.
    - `daily_channel_posts` rows for the previous day.
- **Rate-limit**:
  - Pro users: 10-second silent cooldown.
  - Non-Pro users: 30-second visible cooldown.
  - Exempt users: **skip all rate limits**.

### 5.2 Test Mode

- `DAILY_MODE = "test"`.
- **Listens to**: `TEST_CHANNEL`.
- **Buttons sent to**: exempt users instead of Pro.
- **Cleanup trigger**:
  - Next channel message **after** buttons are sent.
  - Uses `pending_daily_cleanup[chan_id]` to schedule cleanup.
- **Rate-limit**:
  - Exempt users behave like Pro: 10-second silent cooldown.
  - Helpful for testing the 10-second flow without involving Pro states.

Switching modes requires only changing the default of `DAILY_MODE`:

- To use global mode:
  - `DAILY_MODE = os.environ.get("DAILY_MODE", "global")`
- To use test mode:
  - `DAILY_MODE = os.environ.get("DAILY_MODE", "test")`

---

## 6. Implementation Phases

To avoid shipping everything at once, we can implement in phases.

### Phase 1: Infrastructure

- Add `summary` column to `daily_channel_posts` (already done at code + schema level).
- Ensure `#Email` posts are ignored for tweet flow (already done).
- Finalize `DAILY_MODE` handling and cleanup behavior (already done).

### Phase 2: Referral Pair Data

- Add `referred_by_user_id` and `referral_partner_user_id` to `users`.
- Implement `/invite` and `/start ref_<token>` flow.
- Build minimal admin/debug tooling (e.g. a simple log or debug command) to inspect pairs.

### Phase 3: Streak and Pro Upgrade Logic

- Implement a small Python function that, given a pair and a date range, computes:
  - 3-day tweet streak status.
  - Whether Pro should be granted/revoked.
- Hook this into:
  - A daily cron/worker, or
  - The first successful `daily_post:` rephrase per user per day.
- On event:
  - Update `is_pro` and `pro_expires_at` appropriately.

### Phase 4: Daily Status Messages

- On first valid tweet saved for the day:
  - Run streak evaluation for all affected pairs.
  - Send status DMs:
    - For non-Pro pairs: progress toward first Pro unlock.
    - For Pro pairs: maintenance status and warnings.
  - Include “Remind my partner” button, rate-limited.

### Phase 5: Polishing and Experimentation

- Tune numbers:
  - 3 tweet-days to unlock Pro.
  - 2 consecutive missed tweet-days to lose Pro.
  - 30 days initial Pro period (optional).
  - +7 days per additional streak (optional).
- Adjust copy based on user feedback:
  - Make the Pro pitch clearer.
  - Ensure accountability messages feel encouraging, not nagging.

---

## 7. Summary

This Future Pro Plan creates a **clear, social path to Pro**:

- Users invite a friend via referral link and become a pair.
- Pairs unlock Pro by **showing up together** for 3 tweet-days.
- They keep Pro as long as they don’t both totally skip tweeting on two tweet-days in a row.
- Daily status DMs and an easy “Remind my partner” button keep them accountable.
- Pro itself makes the daily tweet routine **fast (≈10 seconds per tweet)** and personalized via tone/length/variation preferences, so once they experience it, they are strongly motivated to maintain the streak.

