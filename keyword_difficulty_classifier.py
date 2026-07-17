"""
Difficulty classifier for routing datasets.

Classifies each prompt as easy / medium / hard based on:
  - Category baseline difficulty
  - Presence of hard/medium technical keywords
  - Prompt length

Usage:
    python difficulty_classifier.py smart-router-relabeled.csv
    python difficulty_classifier.py smart-router-relabeled.csv --output my-output.csv
"""

import argparse
import csv
import re
from collections import Counter

try:
    from rapidfuzz.distance import Levenshtein
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Category baselines
# ---------------------------------------------------------------------------

CATEGORY_BASE = {
    "Chat":       "easy",
    "Rewrite":    "easy",
    "Summarize":  "easy",
    "Extract":    "easy",
    "Classify":   "easy",
    "Brainstorm": "easy",
    "Open QA":    "easy",
    "Generation": "easy",
    "Closed QA":  "medium",
    "Coding":     "medium",
}

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

HARD_KEYWORDS = [
    "dynamic programming", "recursive", "algorithm", "asymptotic", "big.?o",
    "distributed", "concurrency", "multi.?thread", "race condition",
    "cryptograph", "encryption algorithm",
    "neural network", "machine learning", "deep learning", "backprop", "gradient descent",
    "microservice", "system design", "scalab",
    "implement"
]

MEDIUM_KEYWORDS = [
    "explain", "how does", "why does", "difference between",
    "what is", "summarize", "step by step",
    "code", "function", "class", "script", "program",
    "write.*using", "build.*with", "create.*app"
]

HARD_FUZZY_TARGETS = [
    "algorithm", "recursive", "asymptotic", "distributed", "concurrency",
    "cryptograph", "microservice", "implement", "backprop"
]

_HARD_RE = re.compile("|".join(HARD_KEYWORDS), re.IGNORECASE)
_MEDIUM_RE = re.compile("|".join(MEDIUM_KEYWORDS), re.IGNORECASE)


def _fuzzy_hard_hit(text: str) -> bool:
    """Check if any word in text is a typo of a hard keyword (edit distance <= 2)."""
    if not FUZZY_AVAILABLE:
        return False
    words = re.findall(r"[a-zA-Z]{5,}", text)  # only check words >= 5 chars
    for word in words:
        word_lower = word.lower()
        for target in HARD_FUZZY_TARGETS:
            # Allow edit distance proportional to length: 1 for short, 2 for longer
            max_dist = 1 if len(target) <= 8 else 2
            if Levenshtein.distance(word_lower, target) <= max_dist:
                return True
    return False

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_difficulty(query: str, category: str) -> str:
    base = CATEGORY_BASE.get(category, "easy")
    text = query.strip()
    word_count = len(text.split())
 
    hard_hits = len(_HARD_RE.findall(text))
    medium_hits = len(_MEDIUM_RE.findall(text))
 
    # Fuzzy check only if no exact hard hit found
    if hard_hits == 0 and _fuzzy_hard_hit(text):
        hard_hits = 1
 
    if base == "easy":
        if hard_hits >= 1:
            base = "medium"
        elif medium_hits >= 2 or word_count > 80:
            base = "medium"
 
    if base == "medium":
        if hard_hits >= 1:
            base = "hard"
 
    if base == "easy" and word_count > 120:
        base = "medium"
 
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Classify prompt difficulty as easy/medium/hard.")
    parser.add_argument("input", help="Path to input CSV (requires query, category columns)")
    parser.add_argument("--output", default="smart-router-difficulty.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = []
    with open(args.input, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["difficulty"] = classify_difficulty(row["query"], row["category"])
            rows.append(row)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "category", "label", "difficulty"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["difficulty"] for r in rows)
    total = len(rows)
    print(f"Labeled {total} prompts:")
    for level in ["easy", "medium", "hard"]:
        c = counts[level]
        print(f"  {level:6s}: {c:5d}  ({c/total*100:.1f}%)")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()