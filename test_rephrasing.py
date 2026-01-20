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
    calculate_text_similarity,
    STYLES
)

def test_message(original_text: str, test_num: int):
    print(f"\n{'='*80}")
    print(f"TEST #{test_num}")
    print(f"{'='*80}")
    print(f"ORIGINAL ({len(original_text)} chars):")
    print(f"{original_text}")
    print(f"\n{'-'*80}")
    
    # Step 1: Extract tag blocks (with random removal applied)
    start_tags, content_body, end_tags = extract_tag_blocks(original_text, apply_random_removal=True)
    print(f"\nEXTRACTED TAGS (with random 40% removal on sequences):")
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
    
    # Step 4: Generate 5 candidates and select best by similarity
    print(f"\nGENERATING 5 CANDIDATES...")
    
    try:
        all_candidates = []
        
        # Generate 5 candidates with different styles
        for candidate_num in range(5):
            style = random.choice(STYLES)
            print(f"\n  Candidate {candidate_num + 1} - Style: {style}")
            
            prompt = build_prompt(masked.masked, style=style, force_short=False, max_chars=available_chars)
            candidates = gemini_generate_candidates(prompt)
            
            if candidates:
                print(f"    Generated {len(candidates)} output(s)")
                all_candidates.extend(candidates)
        
        if not all_candidates:
            print("  [ERROR: No candidates returned]")
            return
        
        print(f"\n  Total candidates: {len(all_candidates)}")
        
        # Unmask and evaluate all candidates
        print(f"\nEVALUATING CANDIDATES:")
        candidate_scores = []
        
        for idx, candidate in enumerate(all_candidates, 1):
            unmasked = unmask(candidate.strip(), masked.placeholders).strip()
            
            if len(unmasked) <= available_chars:
                similarity = calculate_text_similarity(content_body, unmasked)
                candidate_scores.append((unmasked, similarity, idx))
                print(f"  [{idx}] Length: {len(unmasked)}, Similarity: {similarity:.3f}")
            else:
                print(f"  [{idx}] SKIPPED (too long: {len(unmasked)} chars)")
        
        if not candidate_scores:
            print("  [ERROR: No valid candidates under length limit]")
            return
        
        # Select best candidate by LOWEST similarity score (most different from original)
        rewritten_body, best_similarity, best_idx = min(candidate_scores, key=lambda x: x[1])
        
        print(f"\nSELECTED BEST: Candidate #{best_idx} (LOWEST similarity - most different)")
        print(f"  Similarity Score: {best_similarity:.3f} (lower = more different)")
        print(f"  Length: {len(rewritten_body)} chars")
        print(f"\nUNMASKED BODY:")
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
        
        # Calculate final similarity
        final_similarity = calculate_text_similarity(original_text, final_message)
        print(f"  SIMILARITY SCORE: {final_similarity:.3f} (lower = better for spam evasion, 0.0 = completely different, 1.0 = identical)")
        
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
