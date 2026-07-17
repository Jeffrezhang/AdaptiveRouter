"""
Quick Targeted Benchmark
=========================
Tests only the weak categories from the full benchmark to quickly
evaluate if fine-tuning improved accuracy without running all 3,696 queries.

Weak categories from previous run:
  Coding:    63.8% accuracy
  Closed QA: 42.6% accuracy
  Extract:   70.6% accuracy
  Classify:  64.3% accuracy

Usage:
    python quick_test.py
    python quick_test.py --categories Coding "Closed QA"
    python quick_test.py --output quick_results.csv
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import httpx

LITELLM_BASE = "http://localhost:4000"
ROUTER_MODEL = "qwen-semantic-router"

MODEL_LABEL = {
    "small-model": 0,
    "large-model": 1,
}

# Default weak categories to test
DEFAULT_CATEGORIES = ["Coding", "Closed QA", "Extract", "Classify"]


def route_query(query: str, timeout: int = 30) -> dict:
    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{LITELLM_BASE}/v1/chat/completions",
            headers={
                "Authorization": "Bearer anything",
                "Content-Type": "application/json",
            },
            json={
                "model": ROUTER_MODEL,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 1,
            },
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        routed = resp.headers.get("x-litellm-adaptive-router-model", "large-model")
        return {
            "routed":  routed,
            "label":   MODEL_LABEL.get(routed, 1),
            "elapsed": elapsed,
            "error":   None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "routed":  "large-model",
            "label":   1,
            "elapsed": elapsed,
            "error":   str(e),
        }


def load_csv(path: str, categories: list):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in categories:
                rows.append({
                    "query":        row["query"],
                    "category":     row["category"],
                    "ground_truth": int(row["label"]),
                })
    return rows


def run_test(rows, output_path):
    total    = len(rows)
    results  = []
    errors   = 0

    print(f"\nTesting {total} queries...")
    print()

    start_all = time.perf_counter()

    with open(output_path, "w", newline="", encoding="utf-8-sig") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=["query", "category", "ground_truth",
                        "routed_to", "routed_label", "correct", "elapsed_ms", "error"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()

        for i, row in enumerate(rows):
            r = route_query(row["query"])
            results.append(r)
            if r["error"]:
                errors += 1

            writer.writerow({
                "query":        row["query"],
                "category":     row["category"],
                "ground_truth": row["ground_truth"],
                "routed_to":    r["routed"],
                "routed_label": r["label"],
                "correct":      1 if row["ground_truth"] == r["label"] else 0,
                "elapsed_ms":   round(r["elapsed"] * 1000, 1),
                "error":        r["error"] or "",
            })
            out_f.flush()

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.perf_counter() - start_all
                avg     = elapsed / (i + 1)
                remain  = avg * (total - i - 1)
                print(
                    f"  [{i+1:>3}/{total}]  "
                    f"elapsed {elapsed:5.1f}s  "
                    f"avg {avg*1000:.0f}ms/query  "
                    f"~{remain:.0f}s remaining"
                )

    return results, time.perf_counter() - start_all, errors


def print_report(rows, results, total_elapsed, errors):
    total    = len(rows)
    correct  = sum(1 for row, r in zip(rows, results) if row["ground_truth"] == r["label"])
    accuracy = correct / total * 100

    cat_stats = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0
    })
    for row, r in zip(rows, results):
        cat = row["category"]
        gt  = row["ground_truth"]
        pr  = r["label"]
        cat_stats[cat]["total"]   += 1
        cat_stats[cat]["correct"] += 1 if gt == pr else 0
        if gt == 1 and pr == 1: cat_stats[cat]["tp"] += 1
        if gt == 0 and pr == 1: cat_stats[cat]["fp"] += 1
        if gt == 0 and pr == 0: cat_stats[cat]["tn"] += 1
        if gt == 1 and pr == 0: cat_stats[cat]["fn"] += 1

    tp = sum(1 for row, r in zip(rows, results) if row["ground_truth"] == 1 and r["label"] == 1)
    fp = sum(1 for row, r in zip(rows, results) if row["ground_truth"] == 0 and r["label"] == 1)
    tn = sum(1 for row, r in zip(rows, results) if row["ground_truth"] == 0 and r["label"] == 0)
    fn = sum(1 for row, r in zip(rows, results) if row["ground_truth"] == 1 and r["label"] == 0)

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_ms    = (total_elapsed / total) * 1000

    print()
    print("=" * 62)
    print("  QUICK TEST RESULTS")
    print("=" * 62)
    print(f"\n  Queries tested:    {total}")
    print(f"  Total time:        {total_elapsed:.1f}s")
    print(f"  Avg per query:     {avg_ms:.0f}ms")
    print(f"  Errors:            {errors}")

    print()
    print("-" * 62)
    print("  OVERALL ACCURACY")
    print("-" * 62)
    print(f"  Accuracy:   {correct}/{total}  ({accuracy:.1f}%)")
    print(f"  Precision:  {precision:.1f}%")
    print(f"  Recall:     {recall:.1f}%")
    print(f"  F1 score:   {f1:.1f}%")

    print()
    print("-" * 62)
    print("  ACCURACY BY CATEGORY  (vs previous benchmark)")
    print("-" * 62)

    # Previous benchmark scores for comparison
    previous = {
        "Coding":    63.8,
        "Closed QA": 42.6,
        "Extract":   70.6,
        "Classify":  64.3,
    }

    print(f"  {'Category':<15} {'Total':>6} {'Correct':>8} {'Acc%':>6} {'Prev%':>6} {'Change':>7}")
    print("." * 62)
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
        acc  = s["correct"] / s["total"] * 100
        prev = previous.get(cat, 0)
        diff = acc - prev
        change = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        print(
            f"  {cat:<15} {s['total']:>6,} {s['correct']:>8,} "
            f"{acc:>5.1f}% {prev:>5.1f}% {change:>7}"
        )

    print()
    print("=" * 62)
    print(f"  BOTTOM LINE: {accuracy:.1f}% accuracy on weak categories")
    print(f"  F1 score: {f1:.1f}%")
    print("=" * 62)
    print()


def main():
    parser = argparse.ArgumentParser(description="Quick targeted benchmark for weak categories")
    parser.add_argument("--csv",        default=None,                help="Path to smart-router-relabeled.csv")
    parser.add_argument("--output",     default="quick_results.csv", help="Output CSV")
    parser.add_argument("--categories", nargs="+",                   default=DEFAULT_CATEGORIES,
                        help=f"Categories to test (default: {DEFAULT_CATEGORIES})")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path:
        candidates = [
            "smart-router-relabeled.csv",
            os.path.join(os.path.dirname(__file__), "smart-router-relabeled.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
    if not csv_path or not os.path.exists(csv_path):
        print("ERROR: Could not find smart-router-relabeled.csv")
        print("Pass the path with --csv path/to/file.csv")
        return

    print(f"Loading categories: {args.categories}")
    rows = load_csv(csv_path, args.categories)
    print(f"Found {len(rows)} queries in selected categories")

    if not rows:
        print("No queries found for selected categories.")
        return

    # Show breakdown
    cat_counts = defaultdict(int)
    for r in rows:
        cat_counts[r["category"]] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<15} {count} queries")

    results, total_elapsed, errors = run_test(rows, args.output)
    print_report(rows, results, total_elapsed, errors)
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()