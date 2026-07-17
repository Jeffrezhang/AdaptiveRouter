"""
Embedding Router Cost Analysis with Error Accounting
=====================================================
Calculates actual cost savings using real benchmark results,
accounting for routing errors (FP and FN) that incur extra costs.

Usage:
    python cost_analysis_with_errors.py
    python cost_analysis_with_errors.py --results router_results.csv
    python cost_analysis_with_errors.py --results router_results.csv --scale 1000000
"""

import argparse
import csv
import os
from collections import defaultdict

# Pricing
LARGE_INPUT_PRICE  = 0.20 / 1_000_000
LARGE_OUTPUT_PRICE = 0.20 / 1_000_000
SMALL_INPUT_PRICE  = 0.10 / 1_000_000
SMALL_OUTPUT_PRICE = 0.10 / 1_000_000

DEFAULT_INPUT_TOKENS  = 90
DEFAULT_OUTPUT_TOKENS = 180
ROUTING_OVERHEAD_TOKENS = 1

DIV  = "=" * 62
DIV2 = "-" * 62
DIV3 = "." * 62


def load_results(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "query":         row["query"],
                "category":      row["category"],
                "ground_truth":  int(row["ground_truth"]),
                "routed_to":     row["routed_to"],
                "routed_label":  int(row["routed_label"]),
                "correct":       int(row["correct"]),
                "elapsed_ms":    float(row["elapsed_ms"]),
                "error":         row["error"],
            })
    return rows


def compute_costs(rows, input_tokens, output_tokens, scale):
    total = len(rows)
    sf    = scale / total

    large_cq   = input_tokens * LARGE_INPUT_PRICE  + output_tokens * LARGE_OUTPUT_PRICE
    small_cq   = input_tokens * SMALL_INPUT_PRICE  + output_tokens * SMALL_OUTPUT_PRICE
    routing_cq = input_tokens * SMALL_INPUT_PRICE  + ROUTING_OVERHEAD_TOKENS * SMALL_OUTPUT_PRICE

    # --- Baseline: all large ---
    cost_all_large = total * large_cq

    # --- Perfect router (ground truth labels, no errors) ---
    small_gt = sum(1 for r in rows if r["ground_truth"] == 0)
    large_gt = total - small_gt
    cost_perfect_raw      = small_gt * small_cq + large_gt * large_cq
    cost_perfect_overhead = total * routing_cq
    cost_perfect_net      = cost_perfect_raw + cost_perfect_overhead

    # --- Actual router (with errors) ---
    # TP: should be large, sent to large -> large cost (correct)
    # TN: should be small, sent to small -> small cost (correct)
    # FP: should be small, sent to large -> paid large, wasted money
    # FN: should be large, sent to small -> paid small but got wrong quality
    #     (may need to re-run on large — model as additional large cost)
    tp = sum(1 for r in rows if r["ground_truth"] == 1 and r["routed_label"] == 1)
    tn = sum(1 for r in rows if r["ground_truth"] == 0 and r["routed_label"] == 0)
    fp = sum(1 for r in rows if r["ground_truth"] == 0 and r["routed_label"] == 1)
    fn = sum(1 for r in rows if r["ground_truth"] == 1 and r["routed_label"] == 0)

    # Basic actual cost (what we actually paid per routing decision)
    cost_actual_raw      = (tp + fn) * large_cq + (tn + fp) * small_cq
    # Wait — actual routing is: routed_label=1 -> large, routed_label=0 -> small
    small_routed = sum(1 for r in rows if r["routed_label"] == 0)
    large_routed = total - small_routed
    cost_actual_raw      = large_routed * large_cq + small_routed * small_cq
    cost_actual_overhead = total * routing_cq
    cost_actual_net      = cost_actual_raw + cost_actual_overhead

    # FP cost penalty: sent small queries to large model unnecessarily
    fp_penalty = fp * (large_cq - small_cq)

    # FN cost penalty: sent large queries to small model
    # Best case: small model handles it (no extra cost, but quality loss)
    # Worst case: need to re-run on large (pay twice)
    fn_penalty_worst = fn * large_cq  # had to re-run on large

    # Errors (timeouts/failures) — defaulted to large model
    error_count = sum(1 for r in rows if r["error"])

    savings_perfect = cost_all_large - cost_perfect_net
    savings_actual  = cost_all_large - cost_actual_net
    savings_pct_perfect = savings_perfect / cost_all_large * 100
    savings_pct_actual  = savings_actual  / cost_all_large * 100

    accuracy = sum(r["correct"] for r in rows) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total, "sf": sf,
        "small_gt": small_gt, "large_gt": large_gt,
        "small_routed": small_routed, "large_routed": large_routed,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "error_count": error_count,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "cost_all_large": cost_all_large,
        "cost_perfect_raw": cost_perfect_raw,
        "cost_perfect_net": cost_perfect_net,
        "cost_actual_raw": cost_actual_raw,
        "cost_actual_net": cost_actual_net,
        "cost_perfect_overhead": cost_perfect_overhead,
        "cost_actual_overhead": cost_actual_overhead,
        "fp_penalty": fp_penalty,
        "fn_penalty_worst": fn_penalty_worst,
        "savings_perfect": savings_perfect,
        "savings_actual": savings_actual,
        "savings_pct_perfect": savings_pct_perfect,
        "savings_pct_actual": savings_pct_actual,
        "routing_cq": routing_cq,
        "large_cq": large_cq,
        "small_cq": small_cq,
    }


def category_breakdown(rows, input_tokens, output_tokens, scale):
    large_cq = input_tokens * LARGE_INPUT_PRICE + output_tokens * LARGE_OUTPUT_PRICE
    small_cq = input_tokens * SMALL_INPUT_PRICE + output_tokens * SMALL_OUTPUT_PRICE

    stats = defaultdict(lambda: {
        "total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0
    })
    for r in rows:
        cat = r["category"]
        stats[cat]["total"] += 1
        gt = r["ground_truth"]
        pr = r["routed_label"]
        if gt == 1 and pr == 1: stats[cat]["tp"] += 1
        if gt == 0 and pr == 0: stats[cat]["tn"] += 1
        if gt == 0 and pr == 1: stats[cat]["fp"] += 1
        if gt == 1 and pr == 0: stats[cat]["fn"] += 1

    sf = scale / len(rows)
    results = []
    for cat, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        cost_all_large  = s["total"] * large_cq
        cost_actual     = (s["tp"] + s["fn"]) * large_cq + (s["tn"] + s["fp"]) * small_cq
        # Recalculate based on actual routing
        actual_large    = s["tp"] + s["fp"]  # queries sent to large
        actual_small    = s["tn"] + s["fn"]  # queries sent to small
        cost_actual     = actual_large * large_cq + actual_small * small_cq
        savings         = cost_all_large - cost_actual
        savings_pct     = savings / cost_all_large * 100 if cost_all_large > 0 else 0
        acc             = (s["tp"] + s["tn"]) / s["total"] * 100
        results.append({
            "category": cat, "total": s["total"],
            "tp": s["tp"], "tn": s["tn"],
            "fp": s["fp"], "fn": s["fn"],
            "acc": acc,
            "savings": savings * sf,
            "savings_pct": savings_pct,
        })
    return results


def print_report(c, cats, input_tokens, output_tokens):
    sf = c["sf"]

    print()
    print(DIV)
    print("  ROUTER COST ANALYSIS WITH ERROR ACCOUNTING")
    print(DIV)
    print(f"\n  Queries in benchmark: {c['total']:,}")
    print(f"  Avg input tokens:     {input_tokens}")
    print(f"  Avg output tokens:    {output_tokens}")
    print(f"\n  Pricing:")
    print(f"    Qwen3-8B  input/output  ${LARGE_INPUT_PRICE*1_000_000:.2f}/1M tokens")
    print(f"    MiniCPM5  input/output  ${SMALL_INPUT_PRICE*1_000_000:.2f}/1M tokens")

    print()
    print(DIV2)
    print("  ROUTING ACCURACY")
    print(DIV2)
    print(f"  Overall accuracy:  {c['accuracy']:.1f}%")
    print(f"  Precision:         {c['precision']:.1f}%")
    print(f"  Recall:            {c['recall']:.1f}%")
    print(f"  F1 score:          {c['f1']:.1f}%")
    print(f"  Errors (timeouts): {c['error_count']}")

    print()
    print(DIV2)
    print("  ROUTING SPLIT")
    print(DIV2)
    print(f"  Ground truth small: {c['small_gt']:>5,}  ({c['small_gt']/c['total']*100:.1f}%)")
    print(f"  Ground truth large: {c['large_gt']:>5,}  ({c['large_gt']/c['total']*100:.1f}%)")
    print(f"  Actually routed small: {c['small_routed']:>5,}  ({c['small_routed']/c['total']*100:.1f}%)")
    print(f"  Actually routed large: {c['large_routed']:>5,}  ({c['large_routed']/c['total']*100:.1f}%)")

    print()
    print(DIV2)
    print("  CONFUSION MATRIX")
    print(DIV2)
    print(f"  True  Large (TP): {c['tp']:>5,}  -- correctly sent to large")
    print(f"  False Large (FP): {c['fp']:>5,}  -- should have been small (wasted money)")
    print(f"  True  Small (TN): {c['tn']:>5,}  -- correctly sent to small")
    print(f"  False Small (FN): {c['fn']:>5,}  -- should have been large (quality loss)")

    print()
    print(DIV2)
    print(f"  COST COMPARISON  (scaled to {int(c['total']*c['sf']):,} queries)")
    print(DIV2)
    print(f"  All Large (baseline):           ${c['cost_all_large']*sf:>10.4f}")
    print(f"  Perfect Router (no errors):     ${c['cost_perfect_net']*sf:>10.4f}")
    print(f"  Actual Router (with errors):    ${c['cost_actual_net']*sf:>10.4f}")
    print(f"  Routing overhead (both):        ${c['cost_actual_overhead']*sf:>10.4f}")

    print()
    print(DIV2)
    print("  ERROR COST IMPACT")
    print(DIV2)
    print(f"  FP penalty (overpaid for large): ${c['fp_penalty']*sf:>10.4f}")
    print(f"  FN queries (sent to small):      {c['fn']:>5,}  (quality risk, no extra cost)")
    print(f"  FN worst case (re-run on large): ${c['fn_penalty_worst']*sf:>10.4f}")
    print(f"  Gap (perfect vs actual router):  ${(c['cost_actual_net']-c['cost_perfect_net'])*sf:>10.4f}")

    print()
    print(DIV2)
    print("  SAVINGS vs ALL-LARGE BASELINE")
    print(DIV2)
    print(f"  Perfect router savings: ${c['savings_perfect']*sf:>10.4f}  ({c['savings_pct_perfect']:.1f}%)")
    print(f"  Actual router savings:  ${c['savings_actual']*sf:>10.4f}  ({c['savings_pct_actual']:.1f}%)")
    print(f"  Accuracy cost:          ${(c['savings_perfect']-c['savings_actual'])*sf:>10.4f}  (lost due to errors)")

    print()
    print(DIV2)
    print("  SAVINGS AT DIFFERENT SCALES")
    print(DIV2)
    for scale in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        s_perfect = c["savings_perfect"] * (scale / c["total"])
        s_actual  = c["savings_actual"]  * (scale / c["total"])
        print(f"  {scale:>12,} queries  ->  perfect ${s_perfect:.2f}  |  actual ${s_actual:.2f}")

    print()
    print(DIV2)
    print("  BREAKDOWN BY CATEGORY")
    print(DIV2)
    print(f"  {'Category':<15} {'Total':>6} {'Acc%':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Saves':>10} {'Saves%':>7}")
    print(DIV3)
    for cat in cats:
        print(
            f"  {cat['category']:<15} "
            f"{cat['total']:>6,} "
            f"{cat['acc']:>5.1f}% "
            f"{cat['tp']:>5} "
            f"{cat['fp']:>5} "
            f"{cat['tn']:>5} "
            f"{cat['fn']:>5} "
            f"${cat['savings']:>8.4f} "
            f"{cat['savings_pct']:>6.1f}%"
        )

    print()
    print(DIV)
    print(f"  BOTTOM LINE:")
    print(f"  Perfect router saves {c['savings_pct_perfect']:.1f}% vs all-large")
    print(f"  Actual router saves  {c['savings_pct_actual']:.1f}% vs all-large")
    print(f"  Routing errors cost  {c['savings_pct_perfect']-c['savings_pct_actual']:.1f}% in lost savings")
    print(f"  At 1M queries: perfect ${c['savings_perfect']*(1_000_000/c['total']):.2f} | actual ${c['savings_actual']*(1_000_000/c['total']):.2f}")
    print(DIV)
    print()


def main():
    parser = argparse.ArgumentParser(description="Router Cost Analysis with Error Accounting")
    parser.add_argument("--results",       default=None,               help="Path to router_results.csv")
    parser.add_argument("--scale",         type=int, default=1_000_000, help="Scale target (default 1,000,000)")
    parser.add_argument("--input-tokens",  type=int, default=DEFAULT_INPUT_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    args = parser.parse_args()

    results_path = args.results
    if not results_path:
        candidates = [
            "router_results.csv",
            os.path.join(os.path.dirname(__file__), "router_results.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                results_path = c
                break
    if not results_path or not os.path.exists(results_path):
        print("ERROR: Could not find router_results.csv")
        print("Pass the path with --results path/to/file.csv")
        return

    print(f"\nLoading: {results_path}")
    rows  = load_results(results_path)
    costs = compute_costs(rows, args.input_tokens, args.output_tokens, args.scale)
    cats  = category_breakdown(rows, args.input_tokens, args.output_tokens, args.scale)
    print_report(costs, cats, args.input_tokens, args.output_tokens)


if __name__ == "__main__":
    main()
