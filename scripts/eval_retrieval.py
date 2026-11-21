"""Run BM25 vs Dense vs Hybrid retrieval benchmark over golden or blind holdout queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evals.retrieval_benchmark import run_benchmark
from t_edu_agent.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate BM25 vs Dense vs Hybrid retrieval.")
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Run against evals/retrieval_holdout.json instead of evals/retrieval_golden.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query retrieved documents for Hybrid mode.",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.holdout:
        dataset_path = cfg.project_root / "evals" / "retrieval_holdout.json"
        report_path = cfg.project_root / "evals" / "retrieval_holdout_report.json"
        label = "Blind Holdout"
    else:
        dataset_path = cfg.project_root / "evals" / "retrieval_golden.json"
        report_path = cfg.project_root / "evals" / "retrieval_report.json"
        label = "Golden"

    report = run_benchmark(cfg=cfg, golden_path=dataset_path, report_path=report_path)
    print(
        f"[{label}] Backend: {report['backend_used']} | Model: {report['model_used']} | Queries: {report['num_queries']}\n"
    )
    print("| Mode | Doc Hit@1 | Doc Hit@3 | Doc Hit@5 | Doc Recall@8 | Full-Set Hit@8 | Section Hit@5 | MRR | Avg Latency (ms) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for mode in ("bm25", "dense", "hybrid"):
        m = report["summary"][mode]
        print(
            f"| {mode} | {m['doc_hit_at_1']:.2%} | {m['doc_hit_at_3']:.2%} | "
            f"{m['doc_hit_at_5']:.2%} | {m['doc_recall_at_8']:.2%} | "
            f"{m['full_set_hit_at_8']:.2%} | {m['section_hit_at_5']:.2%} | "
            f"{m['mrr']:.4f} | {m['avg_latency_ms']:.2f} |"
        )

    if args.verbose:
        print("\nPer-query Hybrid results:")
        for row in report["per_query"]["hybrid"]:
            print(
                f"  - {row['id']}: h1={int(row['hit_at_1'])} h5={int(row['hit_at_5'])} "
                f"full@8={int(row['full_set_hit_at_8'])} sec@5={row['section_hit_at_5']} | "
                f"got={row['top_docs']} | expected={row['expected_docs']}"
            )

    hybrid_misses = report["misses"].get("hybrid", [])
    print(f"\nHybrid partial/full misses: {len(hybrid_misses)}")
    for miss in hybrid_misses:
        print(
            f"  - {miss['id']}: hit@5={miss['hit_at_5']} full@8={miss['full_set_hit_at_8']} "
            f"sec@5={miss['section_hit_at_5']} | got={miss['top_docs']} | expected={miss['expected_docs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
