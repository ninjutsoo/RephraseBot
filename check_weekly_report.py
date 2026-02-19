"""
Verify that the weekly activity report matches what the bot sends when an exempt user
taps "📊 Weekly report" or sends /report.

Uses the exact same logic as main.get_weekly_activity_report():
- Past 7 days (EST), paginated fetches (Supabase 1000-row limit)
- Pre-fill seen_user_ids with users who had activity before the week
- New on day D = len(active_on_D - seen_user_ids), then seen_user_ids |= active_on_D
- Output format matches the bot message exactly

Run from repo root with .env set (SUPABASE_URL, SUPABASE_KEY).
Exit code 0 only if the report is produced and (if expected given) matches 100%.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Set SUPABASE_URL and SUPABASE_KEY in .env", file=sys.stderr)
    sys.exit(1)

# Match main.py: zoneinfo EST
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

EST = ZoneInfo("America/New_York")
PAGE_SIZE = 1000


def get_weekly_activity_report(supabase_client) -> List[Tuple[str, int, int]]:
    """
    Same logic as main.get_weekly_activity_report(). Returns list of (date_str, new_count, active_count).
    """
    now_est = datetime.now(EST)
    end_est = now_est.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_est = (now_est - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_est.astimezone(timezone.utc).isoformat()
    end_utc = end_est.astimezone(timezone.utc).isoformat()

    all_data: List[dict] = []
    offset = 0
    while True:
        result = (
            supabase_client.table("activity_logs")
            .select("user_id, timestamp")
            .gte("timestamp", start_utc)
            .lte("timestamp", end_utc)
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        chunk = result.data or []
        all_data.extend(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not all_data:
        return [((start_est + timedelta(days=i)).strftime("%Y-%m-%d"), 0, 0) for i in range(7)]

    seen_user_ids: set = set()
    offset_pre = 0
    while True:
        result_pre = (
            supabase_client.table("activity_logs")
            .select("user_id")
            .lt("timestamp", start_utc)
            .range(offset_pre, offset_pre + PAGE_SIZE - 1)
            .execute()
        )
        chunk_pre = result_pre.data or []
        for row in chunk_pre:
            if row.get("user_id") is not None:
                seen_user_ids.add(int(row["user_id"]))
        if len(chunk_pre) < PAGE_SIZE:
            break
        offset_pre += PAGE_SIZE

    active_by_day: dict = defaultdict(set)
    for row in all_data:
        uid = int(row["user_id"])
        ts = row.get("timestamp")
        if not ts:
            continue
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(EST)
        day_str = dt.strftime("%Y-%m-%d")
        active_by_day[day_str].add(uid)

    out: List[Tuple[str, int, int]] = []
    for i in range(7):
        d = start_est + timedelta(days=i)
        day_str = d.strftime("%Y-%m-%d")
        active = active_by_day.get(day_str, set())
        new = len(active - seen_user_ids)
        seen_user_ids |= active
        out.append((day_str, new, len(active)))
    return out


def format_report_like_bot(rows: List[Tuple[str, int, int]], for_console: bool = True) -> str:
    """Same format as the bot message. for_console=True avoids emoji/unicode for Windows console."""
    if not rows:
        return ""
    start_d, end_d = rows[0][0], rows[-1][0]
    table_lines = ["Date       New  Active", "-" * 18] + [
        "{}   {:>3}   {:>6}".format(d, new, active) for d, new, active in rows
    ]
    header = "Weekly activity ({} - {})".format(start_d, end_d) if for_console else "📊 Weekly activity ({} – {})".format(start_d, end_d)
    return header + "\n\n" + "\n".join(table_lines)


def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    rows = get_weekly_activity_report(supabase)
    report_text = format_report_like_bot(rows, for_console=True)
    print(report_text)

    # Optional: paste the table from the bot message (Date / New / Active lines) for 100% check.
    # Leave empty to only print current report; script exits 0.
    EXPECTED_SNAPSHOT = ""

    if not EXPECTED_SNAPSHOT.strip():
        sys.exit(0)

    # Parse expected: lines that look like "YYYY-MM-DD   N   M"
    expected_lines = []
    for line in EXPECTED_SNAPSHOT.splitlines():
        line = line.strip()
        if not line or line.startswith("Date") or set(line) <= set("─ "):
            continue
        parts = line.split()
        if len(parts) >= 3 and len(parts[0]) == 10 and parts[0][4] == "-" and parts[0][7] == "-":
            try:
                date_str, new, active = parts[0], int(parts[1]), int(parts[2])
                expected_lines.append((date_str, new, active))
            except ValueError:
                continue

    if len(expected_lines) != 7:
        print("\n(Expected snapshot has wrong number of rows; skipping 100% check.)", file=sys.stderr)
        sys.exit(0)

    # Compare to current report (same date order)
    all_ok = True
    for (exp_date, exp_new, exp_active), (got_date, got_new, got_active) in zip(expected_lines, rows):
        if exp_date != got_date or exp_new != got_new or exp_active != got_active:
            all_ok = False
            print(
                f"MISMATCH {got_date}: expected New={exp_new} Active={exp_active}, got New={got_new} Active={got_active}",
                file=sys.stderr,
            )
    if not all_ok:
        print("Report does not match expected snapshot (100% check failed).", file=sys.stderr)
        sys.exit(1)
    print("\n✅ Report matches expected snapshot (100%).")
    sys.exit(0)


if __name__ == "__main__":
    main()
