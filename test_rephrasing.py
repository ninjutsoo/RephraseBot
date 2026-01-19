import os
import sys
import random

# Set environment variables for testing (if not already set)
if "TELEGRAM_BOT_TOKEN" not in os.environ:
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "test_key"
if "WEBHOOK_SECRET" not in os.environ:
    os.environ["WEBHOOK_SECRET"] = "test_secret"

# Import from main.py
from main import (
    extract_tag_blocks,
    mask_protected,
    unmask,
    build_prompt,
    gemini_generate_candidates,
    STYLES
)

def test_message(original_text: str, test_num: int):
    print(f"\n{'='*80}")
    print(f"TEST #{test_num}")
    print(f"{'='*80}")
    print(f"ORIGINAL ({len(original_text)} chars):")
    print(f"{original_text}")
    print(f"\n{'-'*80}")
    
    # Step 1: Extract tag blocks
    start_tags, content_body, end_tags = extract_tag_blocks(original_text)
    print(f"\nEXTRACTED TAGS:")
    print(f"  START: '{start_tags}'")
    print(f"  CONTENT: '{content_body}'")
    print(f"  END: '{end_tags}'")
    
    if not content_body.strip():
        print("  [Content is empty - message is only tags]")
        return
    
    # Step 2: Calculate available space
    overhead = len(start_tags) + len(end_tags)
    available_chars = 280 - overhead
    print(f"\nAVAILABLE CHARS: {available_chars} (overhead: {overhead})")
    
    # Step 3: Mask protected elements
    masked = mask_protected(content_body)
    print(f"\nMASKED TEXT ({len(masked.masked)} chars):")
    print(f"{masked.masked}")
    print(f"\nPLACEHOLDERS ({len(masked.placeholders)}):")
    for ph, orig in masked.placeholders:
        print(f"  {ph} -> {orig}")
    
    # Step 4: Build prompt and generate
    style = random.choice(STYLES)
    print(f"\nSTYLE: {style}")
    
    try:
        prompt = build_prompt(masked.masked, style=style, force_short=False, max_chars=available_chars)
        print(f"\nCALLING GEMINI API...")
        candidates = gemini_generate_candidates(prompt)
        
        if not candidates:
            print("  [ERROR: No candidates returned]")
            return
        
        print(f"  Got {len(candidates)} candidate(s)")
        
        # Pick first candidate
        chosen = candidates[0].strip()
        print(f"\nAI OUTPUT (before unmask, {len(chosen)} chars):")
        print(f"{chosen}")
        
        # Step 5: Unmask
        rewritten_body = unmask(chosen, masked.placeholders).strip()
        print(f"\nUNMASKED BODY ({len(rewritten_body)} chars):")
        print(f"{rewritten_body}")
        
        # Step 6: Reassemble
        start_tags_clean = start_tags.rstrip()
        end_tags_clean = end_tags.lstrip()
        rewritten_body_clean = rewritten_body.strip()
        
        final_message = start_tags_clean
        if start_tags_clean and rewritten_body_clean:
            final_message += " "
        final_message += rewritten_body_clean
        
        if end_tags_clean:
            if final_message:
                final_message += " "
            final_message += end_tags_clean
        
        print(f"\nFINAL MESSAGE ({len(final_message)} chars):")
        print(f"{final_message}")
        
        # Validation
        print(f"\n{'='*80}")
        print("VALIDATION:")
        
        # Check all tags are present
        all_tags = [orig for ph, orig in masked.placeholders]
        missing_tags = [tag for tag in all_tags if tag not in final_message]
        
        if missing_tags:
            print(f"  [FAIL] MISSING TAGS: {missing_tags}")
        else:
            print(f"  [OK] All {len(all_tags)} tags preserved")
        
        # Check for corrupted tags (spaces inside mentions/hashtags)
        import re
        corrupted_mentions = re.findall(r'@\w+\s+\w+(?=\s|$|[@#])', final_message)
        corrupted_hashtags = re.findall(r'#\w+\s+\w+(?=\s|$|[@#])', final_message)
        
        if corrupted_mentions or corrupted_hashtags:
            print(f"  [WARNING] POSSIBLE CORRUPTED TAGS:")
            for cm in corrupted_mentions:
                print(f"    Mention: '{cm}'")
            for ch in corrupted_hashtags:
                print(f"    Hashtag: '{ch}'")
        else:
            print(f"  [OK] No corrupted tags detected")
        
        # Check length
        if len(final_message) <= 280:
            print(f"  [OK] Length OK: {len(final_message)}/280 chars")
        else:
            print(f"  [FAIL] TOO LONG: {len(final_message)}/280 chars")
        
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_messages = [
        "@TRobinsonNewEra @FoxNews, thank you for speaking the TRUTH on Iran. The people are rising up empty handed and need help! @POTUS @Keir_Starmer @SecWar please rescue the Iranian people and the world from the Islamic Republic! #EndIslamicRepublicNow#IranRevolution2026",
        "@BretBaier@FoxNews estimates are now 20,000 people. @Keir_Starmer@SecWar During #DigitalBlackoutIran, the people have no voice but you saw fit to give these terrorists one? #IranRevolution2026 #MIGAwithPahlavi"
    ]
    
    test_num = 1
    for msg_idx, message in enumerate(test_messages, 1):
        print(f"\n\n{'#'*80}")
        print(f"MESSAGE {msg_idx}")
        print(f"{'#'*80}")
        
        for i in range(5):
            test_message(message, test_num)
            test_num += 1
    
    print(f"\n\n{'#'*80}")
    print(f"ALL TESTS COMPLETE")
    print(f"{'#'*80}")
