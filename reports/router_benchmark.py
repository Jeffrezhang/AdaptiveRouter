"""
Router Benchmark
================
Sends every query in the CSV through the LiteLLM routing proxy (max_tokens=1)
and records which model was picked. Compares against the ground-truth labels
to produce an accuracy report.

Saves results incrementally — if the process is killed, resume from where
it left off by running the same command again.

Usage:
    python router_benchmark.py
    python router_benchmark.py --csv path/to/smart-router-relabeled.csv
    python router_benchmark.py --output results.csv
    python router_benchmark.py --workers 4
    python router_benchmark.py --report   (just print report from existing CSV)

Label convention:
    0 = small-model (MiniCPM5)
    1 = large-model (Qwen3-8B)
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

LITELLM_BASE = "http://localhost:4000"
ROUTER_MODEL = "qwen-semantic-router"

MODEL_LABEL = {
    "small-model": 0,
    "large-model": 1,
}

FIELDNAMES = [
    "query", "category", "ground_truth",
    "routed_to", "routed_label", "correct", "elapsed_ms", "error"
]


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


def load_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "query":        row["query"],
                "category":     row["category"],
                "ground_truth": int(row["label"]),
            })
    return rows


def count_completed(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    with open(output_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return sum(1 for _ in reader) - 1  # subtract header


def make_result_row(row: dict, r: dict) -> dict:
    return {
        "query":        row["query"],
        "category":     row["category"],
        "ground_truth": row["ground_truth"],
        "routed_to":    r["routed"],
        "routed_label": r["label"],
        "correct":      1 if row["ground_truth"] == r["label"] else 0,
        "elapsed_ms":   round(r["elapsed"] * 1000, 1),
        "error":        r["error"] or "",
    }


def run_benchmark(rows, output_path, workers=1, completed_so_far=0):
    results = []
    errors  = 0
    total   = len(rows) + completed_so_far

    file_mode = "a" if completed_so_far > 0 else "w"

    print(f"\nRouting {len(rows):,} queries with {workers} worker(s)...")
    if completed_so_far > 0:
        print(f"Resuming from query {completed_so_far + 1} of {total}")
    print()

    start_all = time.perf_counter()

    with open(output_path, file_mode, newline="", encoding="utf-8-sig") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)

        if completed_so_far == 0:
            writer.writeheader()

        if workers == 1:
            for i, row in enumerate(rows):
                r = route_query(row["query"])
                results.append(r)
                if r["error"]:
                    errors += 1

                writer.writerow(make_result_row(row, r))
                out_f.flush()

                global_i = i + completed_so_far
                if (global_i + 1) % 100 == 0 or (i + 1) == len(rows):
                    elapsed = time.perf_counter() - start_all
                    avg     = elapsed / (i + 1)
                    remain  = avg * (len(rows) - i - 1)
                    print(
                        f"  [{global_i+1:>4}/{total}]  "
                        f"elapsed {elapsed:5.1f}s  "
                        f"avg {avg*1000:.0f}ms/query  "
                        f"~{remain:.0f}s remaining"
                    )
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i, row in enumerate(rows):
                    futures[pool.submit(route_query, row["query"])] = i

                done            = 0
                pending_results = {}
                next_to_write   = 0

                for future in as_completed(futures):
                    i = futures[future]
                    pending_results[i] = future.result()
                    if pending_results[i]["error"]:
                        errors += 1
                    done += 1

                    while next_to_write in pending_results:
                        r   = pending_results.pop(next_to_write)
                        row = rows[next_to_write]
                        results.append(r)
                        writer.writerow(make_result_row(row, r))
                        out_f.flush()
                        next_to_write += 1

                    global_i = done + completed_so_far - 1
                    if done % 100 == 0 or done == len(rows):
                        elapsed = time.perf_counter() - start_all
                        avg     = elapsed / done
                        remain  = avg * (len(rows) - done)
                        print(
                            f"  [{global_i+1:>4}/{total}]  "
                            f"elapsed {elapsed:5.1f}s  "
                            f"avg {avg*1000:.0f}ms/query  "
                            f"~{remain:.0f}s remaining"
                        )

    total_elapsed = time.perf_counter() - start_all
    return results, total_elapsed, errors


def load_completed_results(output_path: str):
    results = []
    rows    = []
    with open(output_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "query":        row["query"],
                "category":     row["category"],
                "ground_truth": int(row["ground_truth"]),
            })
            results.append({
                "routed":  row["routed_to"],
                "label":   int(row["routed_label"]),
                "elapsed": float(row["elapsed_ms"]) / 1000,
                "error":   row["error"] or None,
            })
    return rows, results


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

    avg_ms       = (total_elapsed / total) * 1000
    small_routed = sum(1 for r in results if r["label"] == 0)
    large_routed = sum(1 for r in results if r["label"] == 1)

    print()
    print("=" * 62)
    print("  ROUTER BENCHMARK RESULTS")
    print("=" * 62)
    print(f"\n  Queries processed: {total:,}")
    print(f"  Total time:        {total_elapsed:.1f}s")
    print(f"  Avg per query:     {avg_ms:.0f}ms")
    print(f"  Errors:            {errors}")

    print()
    print("-" * 62)
    print("  ROUTING SPLIT")
    print("-" * 62)
    print(f"  -> MiniCPM5  (small): {small_routed:>5,}  ({small_routed/total*100:.1f}%)")
    print(f"  -> Qwen3-8B  (large): {large_routed:>5,}  ({large_routed/total*100:.1f}%)")

    print()
    print("-" * 62)
    print("  ACCURACY vs GROUND TRUTH LABELS")
    print("-" * 62)
    print(f"  Overall accuracy:  {correct:,}/{total:,}  ({accuracy:.1f}%)")
    print(f"  Precision:         {precision:.1f}%")
    print(f"  Recall:            {recall:.1f}%")
    print(f"  F1 score:          {f1:.1f}%")

    print()
    print("-" * 62)
    print("  CONFUSION MATRIX")
    print("-" * 62)
    print(f"  True  Large (TP):  {tp:>5,}  -- correctly sent to large")
    print(f"  False Large (FP):  {fp:>5,}  -- should have been small")
    print(f"  True  Small (TN):  {tn:>5,}  -- correctly sent to small")
    print(f"  False Small (FN):  {fn:>5,}  -- should have been large")

    print()
    print("-" * 62)
    print("  ACCURACY BY CATEGORY")
    print("-" * 62)
    print(f"  {'Category':<15} {'Total':>6} {'Correct':>8} {'Acc%':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("." * 62)
    for cat, s in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
        acc = s["correct"] / s["total"] * 100
        print(
            f"  {cat:<15} {s['total']:>6,} {s['correct']:>8,} {acc:>5.1f}%"
            f" {s['tp']:>5} {s['fp']:>5} {s['tn']:>5} {s['fn']:>5}"
        )

    print()
    print("=" * 62)
    print(f"  BOTTOM LINE: Router accuracy {accuracy:.1f}% on {total:,} queries")
    print(f"  Average routing latency: {avg_ms:.0f}ms per query")
    print("=" * 62)
    print()


def main():
    parser = argparse.ArgumentParser(description="Router Benchmark")
    parser.add_argument("--csv",     default=None,                 help="Path to smart-router-relabeled.csv")
    parser.add_argument("--output",  default="router_results.csv", help="Output CSV (default: router_results.csv)")
    parser.add_argument("--workers", type=int, default=1,          help="Parallel workers (default 1)")
    parser.add_argument("--limit",   type=int, default=None,       help="Only run first N queries")
    parser.add_argument("--report",  action="store_true",          help="Print report from existing output CSV")
    args = parser.parse_args()

    # Just print report from existing results
    if args.report:
        if not os.path.exists(args.output):
            print(f"ERROR: {args.output} not found")
            return
        rows, results = load_completed_results(args.output)
        elapsed = sum(r["elapsed"] for r in results)
        print_report(rows, results, elapsed, 0)
        return

    # Find input CSV
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

    all_rows = load_csv(csv_path)
    if args.limit:
        all_rows = all_rows[:args.limit]

    # Check for existing partial results and resume
    completed = count_completed(args.output)
    if completed > 0 and completed < len(all_rows):
        print(f"Found {completed} completed results in {args.output} -- resuming.")
        rows_to_run = all_rows[completed:]
    elif completed >= len(all_rows):
        print(f"All {completed} queries already completed. Printing report.")
        rows, results = load_completed_results(args.output)
        elapsed = sum(r["elapsed"] for r in results)
        print_report(rows, results, elapsed, 0)
        return
    else:
        rows_to_run = all_rows

    run_benchmark(
        rows_to_run, args.output,
        workers=args.workers,
        completed_so_far=completed
    )

    # Load all results for the final report
    all_rows_done, all_results = load_completed_results(args.output)
    total_time = sum(r["elapsed"] for r in all_results)
    errors     = sum(1 for r in all_results if r["error"])
    print_report(all_rows_done, all_results, total_time, errors)
    print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()