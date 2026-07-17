"""
Difficulty-based classifier mapping a user prompt to a routing decision.

V2 design:
  1. Regex rules infer a dataset category (Coding, Open QA, etc.)
  2. Difficulty classifier (Qwen3-0.6B + LoRA) classifies easy/medium/hard
  3. easy/medium -> "small", hard -> "large"

Falls back to regex-only routing if the difficulty classifier fails to load.
"""

import os
import re
import sys
from typing import List, Tuple, Pattern

# Ensure adaptive_router folder is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Category labeler — infers dataset category from prompt text
# ---------------------------------------------------------------------------

_CATEGORY_RULES: List[Tuple[Pattern, str]] = [
    (
    re.compile(
        r"^\s*(implement|code|program|write)\b",
        re.IGNORECASE,
    ),
    "Coding",
    ),
    (
        re.compile(
            r"\b(write|create|generate|implement|build)\s+(?:a |an |the |me )?(?:python|javascript|typescript|java|rust|go|c\+\+|sql|bash|shell)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(write|create|implement|build)\b(?:\s+\w+){0,4}?\s+(function|class|method|script|program|api|endpoint|microservice)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(explain|describe|understand|walk me through|what does)\b.*\b(code|function|method|class|algorithm|snippet)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(debug|fix|why (?:is|does|isn't)|what.s wrong|trace)\b.*\b(error|bug|exception|stacktrace|stack trace|traceback)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(review|critique)\s+(?:this |my |the )?(?:code|pr|pull request|diff|patch)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(design|architect|plan|architecture)\b.*\b(system|service|api|database|schema|module|microservice)\b",
            re.IGNORECASE,
        ),
        "Coding",
    ),
    (
        re.compile(
            r"\b(solve|compute|calculate|prove|derive)\b.*\b(equation|integral|derivative|theorem|proof|problem)\b",
            re.IGNORECASE,
        ),
        "Closed QA",
    ),
    (
        re.compile(
            r"\b(probability|statistics|combinatorics|optimization problem)\b",
            re.IGNORECASE,
        ),
        "Closed QA",
    ),
    (
        re.compile(
            r"\b(write|draft|compose|rewrite|edit|proofread|polish)\b.*\b(email|essay|blog|post|article|letter|memo|copy|paragraph)\b",
            re.IGNORECASE,
        ),
        "Generation",
    ),
    (
        re.compile(
            r"\b(brainstorm|ideas? for|suggest|come up with)\b",
            re.IGNORECASE,
        ),
        "Brainstorm",
    ),
    (
        re.compile(
            r"\b(summarize|summary|tldr|tl;dr)\b",
            re.IGNORECASE,
        ),
        "Summarize",
    ),
    (
        re.compile(
            r"^\s*(who|what|when|where|which|why|how)\s+(?:is|was|were|are|do|does|did)\b",
            re.IGNORECASE,
        ),
        "Open QA",
    ),
    (
        re.compile(r"^\s*(define|definition of|meaning of|explain)\b", re.IGNORECASE),
        "Open QA",
    ),
]

def _infer_category(text: str) -> str:
    for pattern, category in _CATEGORY_RULES:
        if pattern.search(text):
            return category
    return "Open QA"


# ---------------------------------------------------------------------------
# Difficulty classifier — loaded once at module level
# ---------------------------------------------------------------------------

_difficulty_clf = None

def _load_difficulty_classifier() -> bool:
    global _difficulty_clf
    try:
        from difficulty_classifier_inference import DifficultyClassifier
        _difficulty_clf = DifficultyClassifier()
        return True
    except Exception as e:
        print(f"[classifier] Failed to load difficulty classifier: {e}. Falling back to regex.")
        return False

_clf_loaded = _load_difficulty_classifier()


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

_HARD_RE = re.compile(
    r"\b(dynamic programming|memoization|distributed system|consensus|raft|paxos|"
    r"concurren|thread.safe|race condition|cryptograph|machine learning|neural network|"
    r"backprop|gradient descent|implement.{0,30}from scratch|system design|scalab)\b",
    re.IGNORECASE,
)

def _classify_regex_difficulty(text: str) -> str:
    if _HARD_RE.search(text):
        return "hard"
    return "easy"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_prompt(text: str) -> str:
    """
    Classify a user prompt and return a routing decision.

    Returns:
        "small" for easy/medium prompts
        "large" for hard prompts
    """
    if not text or not text.strip():
        return "small"

    truncated = text[:2000]

    if _clf_loaded and _difficulty_clf is not None:
        try:
            category = _infer_category(truncated)
            difficulty = _difficulty_clf.classify(truncated, category)
            print(f"[classifier] category={category} difficulty={difficulty}")
            return "large" if difficulty == "hard" else "small"
        except Exception as e:
            print(f"[classifier] Error: {e}. Falling back to regex.")

    # Regex fallback
    difficulty = _classify_regex_difficulty(truncated)
    return "large" if difficulty == "hard" else "small"