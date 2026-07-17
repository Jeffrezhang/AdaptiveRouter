"""
Compare keyword-based vs LLM-based difficulty classifications.

Usage:
    python compare_difficulty.py smart-router-difficulty.csv llm-difficulty.csv
"""

import argparse
import csv
from collections import Counter

def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword_csv", help="Keyword-based difficulty CSV")
    parser.add_argument("llm_csv", help="LLM-based difficulty CSV")
    args = parser.parse_args()

    kw_rows = load_csv(args.keyword_csv)
    llm_rows = load_csv(args.llm_csv)

    print(f"Keyword CSV: {len(kw_rows)} rows")
    print(f"LLM CSV:     {len(llm_rows)} rows")

    # Match by query
    llm_by_query = {r["query"]: r["llm_difficulty"] for r in llm_rows}

    matched = []
    for row in kw_rows:
        q = row["query"]
        kw_diff = row["difficulty"]
        llm_diff = llm_by_query.get(q)
        if llm_diff:
            matched.append({
                "query": q,
                "category": row["category"],
                "label": row["label"],
                "kw_difficulty": kw_diff,
                "llm_difficulty": llm_diff,
            })

    print(f"Matched:     {len(matched)} rows\n")

    # --- Distribution comparison ---
    kw_counts = Counter(r["kw_difficulty"] for r in matched)
    llm_counts = Counter(r["llm_difficulty"] for r in matched)
    total = len(matched)

    print("=" * 50)
    print(f"{'Level':<10} {'Keyword':>12} {'LLM':>12}")
    print("=" * 50)
    for level in ["easy", "medium", "hard"]:
        kc = kw_counts[level]
        lc = llm_counts[level]
        print(f"{level:<10} {kc:>6} ({kc/total*100:.1f}%)  {lc:>6} ({lc/total*100:.1f}%)")
    print("=" * 50)

    # --- Agreement rate ---
    agree = sum(1 for r in matched if r["kw_difficulty"] == r["llm_difficulty"])
    print(f"\nAgreement: {agree}/{total} ({agree/total*100:.1f}%)")

    # --- Hard hits that differ ---
    kw_hard_not_llm = [r for r in matched if r["kw_difficulty"] == "hard" and r["llm_difficulty"] != "hard"]
    llm_hard_not_kw = [r for r in matched if r["llm_difficulty"] == "hard" and r["kw_difficulty"] != "hard"]

    print(f"\n{'=' * 50}")
    print(f"Keyword=hard but LLM disagrees: {len(kw_hard_not_llm)}")
    print(f"{'=' * 50}")
    for r in kw_hard_not_llm:
        print(f"  [{r['category']}] LLM={r['llm_difficulty']} | {r['query'][:80]}")

    print(f"\n{'=' * 50}")
    print(f"LLM=hard but Keyword disagrees: {len(llm_hard_not_kw)}")
    print(f"{'=' * 50}")
    for r in llm_hard_not_kw:
        print(f"  [{r['category']}] KW={r['kw_difficulty']} | {r['query'][:80]}")

    # --- Save combined output ---
    out_path = "difficulty_comparison.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "category", "label", "kw_difficulty", "llm_difficulty", "agree"])
        writer.writeheader()
        for r in matched:
            r["agree"] = "yes" if r["kw_difficulty"] == r["llm_difficulty"] else "no"
            writer.writerow(r)
    print(f"\nSaved combined comparison to {out_path}")

if __name__ == "__main__":
    main()
