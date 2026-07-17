"""
Embedding Router Cost Savings Calculator
=========================================
Shows how much money you save by using the semantic embedding router
to route queries between MiniCPM5 (small) and Qwen3-8B (large).

Usage:
    python cost_analysis.py
    python cost_analysis.py --csv path/to/your.csv
    python cost_analysis.py --scale 1000000
    python cost_analysis.py --input-tokens 120 --output-tokens 300

Label convention in CSV:
    0 = small-model (MiniCPM5)
    1 = large-model (Qwen3-8B)
"""

import argparse
import csv
import os
from collections import defaultdict

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


def load_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "query":    row["query"],
                "category": row["category"],
                "label":    int(row["label"]),
            })
    return rows


def compute_costs(rows, input_tokens, output_tokens, scale):
    total       = len(rows)
    small_count = sum(1 for r in rows if r["label"] == 0)
    large_count = total - small_count

    large_cost_per_query = input_tokens * LARGE_INPUT_PRICE + output_tokens * LARGE_OUTPUT_PRICE
    small_cost_per_query = input_tokens * SMALL_INPUT_PRICE + output_tokens * SMALL_OUTPUT_PRICE
    routing_overhead_per_query = (
        input_tokens * SMALL_INPUT_PRICE +
        ROUTING_OVERHEAD_TOKENS * SMALL_OUTPUT_PRICE
    )

    cost_all_large  = total * large_cost_per_query
    cost_all_small  = total * small_cost_per_query
    cost_router_raw = large_count * large_cost_per_query + small_count * small_cost_per_query
    routing_overhead_total    = total * routing_overhead_per_query
    cost_router_with_overhead = cost_router_raw + routing_overhead_total

    savings_raw     = cost_all_large - cost_router_raw
    savings_net     = cost_all_large - cost_router_with_overhead
    savings_pct_raw = savings_raw / cost_all_large * 100
    savings_pct_net = savings_net / cost_all_large * 100
    sf = scale / total

    return {
        "total": total, "small_count": small_count, "large_count": large_count,
        "small_pct": small_count / total * 100, "large_pct": large_count / total * 100,
        "cost_all_large": cost_all_large, "cost_all_small": cost_all_small,
        "cost_router_raw": cost_router_raw, "cost_router_with_overhead": cost_router_with_overhead,
        "routing_overhead_total": routing_overhead_total,
        "savings_raw": savings_raw, "savings_net": savings_net,
        "savings_pct_raw": savings_pct_raw, "savings_pct_net": savings_pct_net,
        "scale": scale, "sf": sf,
    }


def category_breakdown(rows, input_tokens, output_tokens):
    stats = defaultdict(lambda: {"total": 0, "small": 0, "large": 0})
    for r in rows:
        cat = r["category"]
        stats[cat]["total"] += 1
        if r["label"] == 0:
            stats[cat]["small"] += 1
        else:
            stats[cat]["large"] += 1

    large_cq = input_tokens * LARGE_INPUT_PRICE + output_tokens * LARGE_OUTPUT_PRICE
    small_cq = input_tokens * SMALL_INPUT_PRICE + output_tokens * SMALL_OUTPUT_PRICE

    results = []
    for cat, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
        cost_all_large = s["total"] * large_cq
        cost_router    = s["large"] * large_cq + s["small"] * small_cq
        savings        = cost_all_large - cost_router
        savings_pct    = savings / cost_all_large * 100 if cost_all_large > 0 else 0
        results.append({
            "category": cat, "total": s["total"],
            "small": s["small"], "large": s["large"],
            "small_pct": s["small"] / s["total"] * 100,
            "savings": savings, "savings_pct": savings_pct,
        })
    return results


def print_report(rows, c, cats, input_tokens, output_tokens):
    sf = c["sf"]

    print()
    print(DIV)
    print("  EMBEDDING ROUTER - COST SAVINGS REPORT")
    print(DIV)
    print(f"\n  Dataset:           {c['total']:,} queries")
    print(f"  Scale target:      {int(c['scale']):,} queries")
    print(f"  Avg input tokens:  {input_tokens}")
    print(f"  Avg output tokens: {output_tokens}")
    print(f"\n  Pricing:")
    print(f"    Qwen3-8B   input  ${LARGE_INPUT_PRICE*1_000_000:.2f}/1M tokens")
    print(f"    Qwen3-8B   output ${LARGE_OUTPUT_PRICE*1_000_000:.2f}/1M tokens")
    print(f"    MiniCPM5   input  ${SMALL_INPUT_PRICE*1_000_000:.2f}/1M tokens")
    print(f"    MiniCPM5   output ${SMALL_OUTPUT_PRICE*1_000_000:.2f}/1M tokens")

    print()
    print(DIV2)
    print("  ROUTING SPLIT")
    print(DIV2)
    print(f"  -> MiniCPM5  (small): {c['small_count']:>5,} queries  ({c['small_pct']:.1f}%)")
    print(f"  -> Qwen3-8B  (large): {c['large_count']:>5,} queries  ({c['large_pct']:.1f}%)")

    print()
    print(DIV2)
    print("  COST COMPARISON  (scaled to {:,} queries)".format(int(c["scale"])))
    print(DIV2)
    print(f"  All Large (no router):         ${c['cost_all_large']*sf:>10.4f}")
    print(f"  All Small (max savings):        ${c['cost_all_small']*sf:>10.4f}")
    print(f"  Router (excl. overhead):        ${c['cost_router_raw']*sf:>10.4f}")
    print(f"  Router overhead (probe calls):  ${c['routing_overhead_total']*sf:>10.4f}")
    print(f"  Router (incl. overhead):        ${c['cost_router_with_overhead']*sf:>10.4f}")

    print()
    print(DIV2)
    print("  SAVINGS")
    print(DIV2)
    print(f"  Gross savings (no overhead):    ${c['savings_raw']*sf:>10.4f}  ({c['savings_pct_raw']:.1f}%)")
    print(f"  Net savings   (incl overhead):  ${c['savings_net']*sf:>10.4f}  ({c['savings_pct_net']:.1f}%)")

    print()
    print(DIV2)
    print("  SAVINGS AT DIFFERENT SCALES")
    print(DIV2)
    for scale in [1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        s = c["savings_net"] * (scale / c["total"])
        print(f"  {scale:>12,} queries  ->  saves ${s:.2f}")

    print()
    print(DIV2)
    print("  BREAKDOWN BY CATEGORY")
    print(DIV2)
    print(f"  {'Category':<15} {'Total':>6} {'->Small':>8} {'->Large':>8} {'Small%':>7} {'Saves':>10} {'Saves%':>7}")
    print(DIV3)
    for cat in cats:
        print(
            f"  {cat['category']:<15} "
            f"{cat['total']:>6,} "
            f"{cat['small']:>8,} "
            f"{cat['large']:>8,} "
            f"{cat['small_pct']:>6.0f}% "
            f"${cat['savings']*sf:>8.4f} "
            f"{cat['savings_pct']:>6.1f}%"
        )

    print()
    print(DIV)
    print(f"  BOTTOM LINE: Using the router saves {c['savings_pct_net']:.1f}% vs all-large.")
    print(f"  At 1M queries that's ${c['savings_net']*(1_000_000/c['total']):.2f} saved.")
    print(DIV)
    print()


def main():
    parser = argparse.ArgumentParser(description="Embedding Router Cost Savings Calculator")
    parser.add_argument("--csv",           default=None,                    help="Path to smart-router-relabeled.csv")
    parser.add_argument("--scale",         type=int, default=1_000_000,     help="Scale target (default 1,000,000)")
    parser.add_argument("--input-tokens",  type=int, default=DEFAULT_INPUT_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
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

    print(f"\nLoading: {csv_path}")
    rows  = load_csv(csv_path)
    costs = compute_costs(rows, args.input_tokens, args.output_tokens, args.scale)
    cats  = category_breakdown(rows, args.input_tokens, args.output_tokens)
    print_report(rows, costs, cats, args.input_tokens, args.output_tokens)


if __name__ == "__main__":
    main()