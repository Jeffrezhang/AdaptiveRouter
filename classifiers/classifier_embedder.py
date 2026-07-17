"""
Semantic classifier mapping a user prompt to a RequestType.

V1 design: uses Qwen3-Embedding-0.6B via sentence-transformers to compute
cosine similarity between the prompt and example sentences for each RequestType.
Falls back to regex-based classification if the model fails to load.

Embedding model is loaded once at module level and reused across requests.
"""

import re
from typing import Dict, List, Optional, Pattern

import numpy as np
import psycopg2
from litellm.types.router import RequestType

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
import os
_DB_CONFIG = {
    "host": os.environ.get("PGVECTOR_HOST", "127.0.0.1"),
    "port": int(os.environ.get("PGVECTOR_PORT", 5432)),
    "dbname": os.environ.get("PGVECTOR_DB", "routing"),
    "user": os.environ.get("PGVECTOR_USER", "litellm"),
    "password": os.environ.get("PGVECTOR_PASSWORD", ""),
}
_conn = None

def _get_conn():
    global _conn
    try:
        if _conn is None or _conn.closed:
            _conn = psycopg2.connect(**_DB_CONFIG)
        return _conn
    except Exception as e:
        print(f"[classifier] DB connection failed: {e}")
        return None

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
_embedder = None

def _load_model() -> bool:
    global _embedder
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import os
        _embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=device)
        return True
    except Exception as e:
        print(f"[classifier] Failed to load Qwen3-Embedding: {e}. Falling back to regex.")
        return False

_model_loaded = _load_model()


# ---------------------------------------------------------------------------
# Regex fallback 
# ---------------------------------------------------------------------------
_RULES: List[tuple[Pattern[str], RequestType]] = [
    (
        re.compile(
            r"\b(write|create|generate|implement|build)\s+(?:a |an |the |me )?(?:python|javascript|typescript|java|rust|go|c\+\+|sql|bash|shell)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_GENERATION,
    ),
    (
        re.compile(
            r"\b(write|create|implement|build)\b(?:\s+\w+){0,4}?\s+(function|class|method|script|program|api|endpoint|microservice)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_GENERATION,
    ),
    (
        re.compile(
            r"\b(explain|describe|understand|walk me through|what does)\b.*\b(code|function|method|class|algorithm|snippet)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(debug|fix|why (?:is|does|isn't)|what.s wrong|trace)\b.*\b(error|bug|exception|stacktrace|stack trace|traceback)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(review|critique)\s+(?:this |my |the )?(?:code|pr|pull request|diff|patch)\b",
            re.IGNORECASE,
        ),
        RequestType.CODE_UNDERSTANDING,
    ),
    (
        re.compile(
            r"\b(design|architect|plan|architecture)\b.*\b(system|service|api|database|schema|module|microservice)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\b(should i (?:use|choose|pick)|tradeoffs? between|compare)\b.*\b(library|framework|language|database|protocol|postgres|postgresql|mongodb|dynamodb|mysql|redis|kafka|sql|nosql)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\bhow (?:should|do) i (?:design|structure|organize|model)\b",
            re.IGNORECASE,
        ),
        RequestType.TECHNICAL_DESIGN,
    ),
    (
        re.compile(
            r"\b(solve|compute|calculate|prove|derive)\b.*\b(equation|integral|derivative|theorem|proof|problem)\b",
            re.IGNORECASE,
        ),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(r"\b(if .+ then|given .+ find|suppose|assume)\b", re.IGNORECASE),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(
            r"\b(probability|statistics|combinatorics|optimization problem)\b",
            re.IGNORECASE,
        ),
        RequestType.ANALYTICAL_REASONING,
    ),
    (
        re.compile(
            r"\b(write|draft|compose|rewrite|edit|proofread|polish)\b.*\b(email|essay|blog|post|article|letter|memo|copy|paragraph|sentence)\b",
            re.IGNORECASE,
        ),
        RequestType.WRITING,
    ),
    (
        re.compile(
            r"\b(make (?:this|it)|help me)\s+(?:more |less )?(?:concise|formal|casual|professional|persuasive)\b",
            re.IGNORECASE,
        ),
        RequestType.WRITING,
    ),
    (
        re.compile(
            r"^\s*(who|what|when|where|which)\s+(?:is|was|were|are)\b", re.IGNORECASE
        ),
        RequestType.FACTUAL_LOOKUP,
    ),
    (
        re.compile(r"^\s*(define|definition of|meaning of)\b", re.IGNORECASE),
        RequestType.FACTUAL_LOOKUP,
    ),
    (
        re.compile(
            r"^\s*how (?:do you spell|to spell|many .* are there|tall is)\b",
            re.IGNORECASE,
        ),
        RequestType.FACTUAL_LOOKUP,
    ),
]

def _classify_regex(text: str) -> RequestType:
    for pattern, request_type in _RULES:
        if pattern.search(text):
            return request_type
    return RequestType.GENERAL

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_prompt(text: str) -> RequestType:
    """
    Classify a single user prompt into a RequestType.

    Embeds the prompt with Qwen3-Embedding-0.6B, queries pgvector for
    the 5 nearest neighbors, and majority-votes the result.
    Falls back to regex if the DB or model is unavailable.
    """
    if not text or not text.strip():
        return RequestType.GENERAL

    truncated = text[:2000]

    if not _model_loaded or _embedder is None:
        return _classify_regex(truncated)

    try:
        # Embed the prompt
        embedding = _embedder.encode(
            [truncated], normalize_embeddings=True
        )[0].tolist()

        # Query pgvector for top 7 nearest neighbors
        conn = _get_conn()
        if conn is None:
            return _classify_regex(truncated)

        cur = conn.cursor()
        cur.execute("""
            SELECT request_type
            FROM request_examples
            ORDER BY embedding <=> %s::vector
            LIMIT 7
        """, (embedding,))

        rows = cur.fetchall()
        cur.close()

        if not rows:
            return _classify_regex(truncated)

        # Majority vote across top 5
        types = [row[0] for row in rows]
        winner = max(set(types), key=types.count)
        return RequestType(winner)

    except Exception as e:
        try:
            if _conn:
                _conn.close()
        except Exception:
            pass
        return _classify_regex(truncated)