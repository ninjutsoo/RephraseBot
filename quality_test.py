"""
Quality test: Run rephrasing 20 times and evaluate each result.
Measures: Similarity score, Soundness, Spam Detection Evasion
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

# Set environment variables
if "TELEGRAM_BOT_TOKEN" not in os.environ:
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "AIzaSyAZhMiBS6OrpN3lYrF7DewCGN_IDf_Hgp0"
if "WEBHOOK_SECRET" not in os.environ:
    os.environ["WEBHOOK_SECRET"] = "test_secret"

# Import from main.py
from main import (
    apply_random_tag_removal,
    extract_tag_blocks,
    mask_protected,
    unmask,
    build_prompt,
    gemini_generate_candidates,
    calculate_text_similarity,
    contains_all_placeholders,
)

# Test message
ORIGINAL_MESSAGE = (
    "@TRobinsonNewEra, thank you for speaking the TRUTH on Iran. "
    "The people are rising up empty handed and need help! "
    "@POTUS @Keir_Starmer @SecWar please rescue the Iranian people and the world from the Islamic Republic! "
    "#EndIslamicRepublicNow#IranRevolution2026"
)


def evaluate_soundness(original: str, rephrased: str) -> float:
    """
    Evaluate if the rephrased text makes sense and preserves intent.
    Returns score 0.0-1.0 (1.0 = perfect soundness)
    """
    score = 1.0
    
    # Check for key concepts preserved
    key_concepts = [
        "Iran", "people", "help", "rescue", "Islamic Republic",
        "rising", "empty handed", "thank", "truth"
    ]
    
    original_lower = original.lower()
    rephrased_lower = rephrased.lower()
    
    # Check if key concepts are present (at least in paraphrased form)
    found_concepts = 0
    for concept in key_concepts:
        if concept in original_lower:
            # Check if concept or synonym appears in rephrased
            if concept in rephrased_lower:
                found_concepts += 1
            else:
                # Check for common synonyms
                synonyms = {
                    "people": ["citizens", "population", "individuals", "folk"],
                    "help": ["aid", "assistance", "support", "rescue"],
                    "rescue": ["save", "help", "aid", "liberate"],
                    "rising": ["revolting", "protesting", "standing up", "fighting"],
                    "empty handed": ["unarmed", "without weapons", "defenseless"],
                    "thank": ["grateful", "appreciate", "acknowledge"],
                    "truth": ["reality", "facts", "honesty"]
                }
                if concept in synonyms:
                    found = any(syn in rephrased_lower for syn in synonyms[concept])
                    if found:
                        found_concepts += 1
                    else:
                        score -= 0.1  # Penalty for missing key concept
                else:
                    score -= 0.05  # Minor penalty
    
    # Check for call-to-action preserved
    has_call_to_action = any(word in rephrased_lower for word in ["please", "rescue", "help", "save", "aid", "support"])
    if not has_call_to_action:
        score -= 0.2
    
    # Check for mentions preserved (critical)
    mentions_in_original = ["@TRobinsonNewEra", "@POTUS", "@Keir_Starmer", "@SecWar"]
    mentions_found = sum(1 for mention in mentions_in_original if mention in rephrased)
    if mentions_found < len(mentions_in_original) * 0.5:  # At least 50% should be preserved
        score -= 0.3
    
    # Check for hashtags preserved
    hashtags_in_original = ["#EndIslamicRepublicNow", "#IranRevolution2026"]
    hashtags_found = sum(1 for tag in hashtags_in_original if tag in rephrased)
    if hashtags_found < len(hashtags_in_original) * 0.5:
        score -= 0.2
    
    # Check for grammatical coherence (basic check)
    if len(rephrased.split()) < 5:  # Too short might be incomplete
        score -= 0.2
    
    # Check for obvious errors
    if "ERROR" in rephrased.upper() or "PLACEHOLDER" in rephrased.upper():
        score = 0.0
    
    return max(0.0, min(1.0, score))


def evaluate_spam_evasion(original: str, rephrased: str, similarity: float) -> float:
    """
    Evaluate how well the rephrased text evades spam detection.
    Lower similarity = better evasion.
    Returns score 0.0-1.0 (1.0 = best evasion)
    """
    # Base score from similarity (inverted: lower similarity = higher score)
    base_score = 1.0 - similarity
    
    # Bonus for structural differences
    original_words = original.lower().split()
    rephrased_words = rephrased.lower().split()
    
    # Check word overlap
    overlap_ratio = len(set(original_words) & set(rephrased_words)) / max(len(set(original_words)), 1)
    structural_bonus = (1.0 - overlap_ratio) * 0.3
    
    # Check length variation
    length_diff = abs(len(rephrased) - len(original)) / max(len(original), 1)
    length_bonus = min(length_diff * 0.2, 0.2)
    
    # Check sentence structure variation
    original_sentences = original.count('.') + original.count('!') + original.count('?')
    rephrased_sentences = rephrased.count('.') + rephrased.count('!') + rephrased.count('?')
    if original_sentences != rephrased_sentences:
        sentence_bonus = 0.1
    else:
        sentence_bonus = 0.0
    
    total_score = base_score + structural_bonus + length_bonus + sentence_bonus
    return min(1.0, total_score)


def run_single_test(test_num: int) -> Dict:
    """Run a single rephrasing test and return results."""
    print(f"\n{'='*80}")
    print(f"TEST #{test_num}")
    print(f"{'='*80}")
    
    # Step 1: Apply tag removal
    text_with_removed_tags = apply_random_tag_removal(ORIGINAL_MESSAGE, removal_chance=0.4)
    
    # Step 2: Extract tag blocks
    start_tags, content_body, end_tags = extract_tag_blocks(text_with_removed_tags)
    
    if not content_body.strip():
        return {"error": "Content body is empty"}
    
    # Step 3: Mask protected elements
    masked = mask_protected(content_body)
    
    # Step 4: Generate 2 candidates
    all_candidates = []
    for candidate_num in range(2):
        from main import STYLES
        import random
        style = random.choice(STYLES)
        prompt = build_prompt(masked.masked, style=style, force_short=False, max_chars=280 - len(start_tags) - len(end_tags))
        candidates = gemini_generate_candidates(prompt)
        if candidates:
            all_candidates.extend(candidates)
        time.sleep(0.5)  # Small delay between API calls
    
    if not all_candidates:
        return {"error": "No candidates generated"}
    
    # Step 5: Evaluate candidates
    candidate_results = []
    for idx, candidate in enumerate(all_candidates):
        if contains_all_placeholders(candidate, masked.placeholders):
            unmasked = unmask(candidate.strip(), masked.placeholders).strip()
            
            # Reassemble
            final_message = start_tags.rstrip()
            if start_tags.rstrip() and unmasked:
                final_message += " "
            final_message += unmasked
            if end_tags.lstrip():
                if final_message:
                    final_message += " "
                final_message += end_tags.lstrip()
            
            if len(final_message) <= 280:
                similarity = calculate_text_similarity(ORIGINAL_MESSAGE, final_message)
                soundness = evaluate_soundness(ORIGINAL_MESSAGE, final_message)
                spam_evasion = evaluate_spam_evasion(ORIGINAL_MESSAGE, final_message, similarity)
                
                candidate_results.append({
                    "candidate_num": idx + 1,
                    "rephrased": final_message,
                    "length": len(final_message),
                    "similarity": similarity,
                    "soundness": soundness,
                    "spam_evasion": spam_evasion,
                    "overall_score": (soundness * 0.4 + spam_evasion * 0.4 + (1.0 - similarity) * 0.2)
                })
    
    if not candidate_results:
        return {"error": "No valid candidates"}
    
    # Select best (lowest similarity)
    best = min(candidate_results, key=lambda x: x["similarity"])
    
    return {
        "test_num": test_num,
        "original": ORIGINAL_MESSAGE,
        "original_length": len(ORIGINAL_MESSAGE),
        "tags_removed": text_with_removed_tags != ORIGINAL_MESSAGE,
        "start_tags": start_tags,
        "end_tags": end_tags,
        "best_candidate": best,
        "all_candidates": candidate_results,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run 20 tests and save results."""
    print("="*80)
    print("QUALITY TEST: 20 Rephrasing Iterations")
    print("="*80)
    print(f"\nOriginal Message ({len(ORIGINAL_MESSAGE)} chars):")
    print(ORIGINAL_MESSAGE)
    print(f"\n{'='*80}\n")
    
    all_results = []
    
    for i in range(1, 21):
        try:
            result = run_single_test(i)
            if "error" not in result:
                all_results.append(result)
                print(f"\n✓ Test #{i} Complete")
                print(f"  Best: {result['best_candidate']['rephrased'][:80]}...")
                print(f"  Similarity: {result['best_candidate']['similarity']:.3f}")
                print(f"  Soundness: {result['best_candidate']['soundness']:.3f}")
                print(f"  Spam Evasion: {result['best_candidate']['spam_evasion']:.3f}")
            else:
                print(f"\n✗ Test #{i} Failed: {result['error']}")
            
            # Delay between tests
            if i < 20:
                time.sleep(2)
        except Exception as e:
            print(f"\n✗ Test #{i} Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Calculate statistics
    if all_results:
        similarities = [r['best_candidate']['similarity'] for r in all_results]
        soundnesses = [r['best_candidate']['soundness'] for r in all_results]
        spam_evasions = [r['best_candidate']['spam_evasion'] for r in all_results]
        overall_scores = [r['best_candidate']['overall_score'] for r in all_results]
        lengths = [r['best_candidate']['length'] for r in all_results]
        
        stats = {
            "total_tests": len(all_results),
            "similarity": {
                "mean": sum(similarities) / len(similarities),
                "min": min(similarities),
                "max": max(similarities),
                "std": (sum((x - sum(similarities)/len(similarities))**2 for x in similarities) / len(similarities))**0.5
            },
            "soundness": {
                "mean": sum(soundnesses) / len(soundnesses),
                "min": min(soundnesses),
                "max": max(soundnesses),
                "std": (sum((x - sum(soundnesses)/len(soundnesses))**2 for x in soundnesses) / len(soundnesses))**0.5
            },
            "spam_evasion": {
                "mean": sum(spam_evasions) / len(spam_evasions),
                "min": min(spam_evasions),
                "max": max(spam_evasions),
                "std": (sum((x - sum(spam_evasions)/len(spam_evasions))**2 for x in spam_evasions) / len(spam_evasions))**0.5
            },
            "overall_score": {
                "mean": sum(overall_scores) / len(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores)
            },
            "length": {
                "mean": sum(lengths) / len(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }
        }
        
        # Save results
        output = {
            "test_info": {
                "original_message": ORIGINAL_MESSAGE,
                "original_length": len(ORIGINAL_MESSAGE),
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(all_results)
            },
            "statistics": stats,
            "detailed_results": all_results
        }
        
        filename = f"quality_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"\nTotal Successful Tests: {len(all_results)}/20")
        print(f"\nSimilarity Score:")
        print(f"  Mean: {stats['similarity']['mean']:.3f}")
        print(f"  Min:  {stats['similarity']['min']:.3f} (best evasion)")
        print(f"  Max:  {stats['similarity']['max']:.3f}")
        print(f"  Std:  {stats['similarity']['std']:.3f}")
        
        print(f"\nSoundness Score:")
        print(f"  Mean: {stats['soundness']['mean']:.3f}")
        print(f"  Min:  {stats['soundness']['min']:.3f}")
        print(f"  Max:  {stats['soundness']['max']:.3f}")
        print(f"  Std:  {stats['soundness']['std']:.3f}")
        
        print(f"\nSpam Evasion Score:")
        print(f"  Mean: {stats['spam_evasion']['mean']:.3f}")
        print(f"  Min:  {stats['spam_evasion']['min']:.3f}")
        print(f"  Max:  {stats['spam_evasion']['max']:.3f}")
        print(f"  Std:  {stats['spam_evasion']['std']:.3f}")
        
        print(f"\nOverall Score (Soundness 40% + Evasion 40% + Low Similarity 20%):")
        print(f"  Mean: {stats['overall_score']['mean']:.3f}")
        print(f"  Min:  {stats['overall_score']['min']:.3f}")
        print(f"  Max:  {stats['overall_score']['max']:.3f}")
        
        print(f"\nLength:")
        print(f"  Mean: {stats['length']['mean']:.1f} chars")
        print(f"  Min:  {stats['length']['min']} chars")
        print(f"  Max:  {stats['length']['max']} chars")
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {filename}")
        print(f"{'='*80}\n")
    else:
        print("\n✗ No successful tests completed!")


if __name__ == "__main__":
    main()
