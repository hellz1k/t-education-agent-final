"""Benchmark suite comparing BM25, dense embeddings, and hybrid RRF retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from t_edu_agent.config import AppConfig, load_config
from t_edu_agent.retrieval.hybrid import HybridRetriever, RetrievalMode


@dataclass
class QueryEvalResult:
    """Per-query evaluation breakdown for one retrieval mode."""

    query_id: str
    category: str
    query: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_8: float
    full_set_hit_at_8: float
    section_hit_at_5: float | None
    reciprocal_rank: float
    latency_ms: float
    top_docs: list[str]
    expected_docs: list[str]


def _evaluate_query(
    retriever: HybridRetriever,
    item: dict[str, Any],
    mode: RetrievalMode,
    top_k: int = 8,
) -> QueryEvalResult:
    """Evaluate a single golden benchmark entry under a given retrieval mode."""
    query = item["query"]
    all_docs: list[str] = item.get("expected_all_docs", [])
    any_docs: list[str] = item.get("expected_any_docs", [])
    expected_docs = all_docs or any_docs
    expected_set = set(expected_docs)
    expected_sections: set[str] = set(item.get("expected_sections", []))

    res = retriever.search(query, top_k=top_k, mode=mode)
    top_docs = [c.chunk.source_file for c in res.chunks]

    hit_1 = 1.0 if any(d in expected_set for d in top_docs[:1]) else 0.0
    hit_3 = 1.0 if any(d in expected_set for d in top_docs[:3]) else 0.0
    hit_5 = 1.0 if any(d in expected_set for d in top_docs[:5]) else 0.0

    matched_unique = set(top_docs[:top_k]) & expected_set
    recall_8 = len(matched_unique) / len(expected_set) if expected_set else 0.0

    if all_docs:
        full_set_hit = 1.0 if all(d in top_docs[:top_k] for d in all_docs) else 0.0
    else:
        required_hits = min(2, len(any_docs))
        full_set_hit = 1.0 if len(matched_unique) >= required_hits else 0.0

    section_hit: float | None = None
    if expected_sections:
        section_hit = (
            1.0
            if any(
                c.chunk.source_file in expected_set and c.chunk.section in expected_sections
                for c in res.chunks[:5]
            )
            else 0.0
        )

    rr = 0.0
    for rank, doc_name in enumerate(top_docs, start=1):
        if doc_name in expected_set:
            rr = 1.0 / rank
            break

    return QueryEvalResult(
        query_id=item["id"],
        category=item["category"],
        query=query,
        hit_at_1=hit_1,
        hit_at_3=hit_3,
        hit_at_5=hit_5,
        recall_at_8=recall_8,
        full_set_hit_at_8=full_set_hit,
        section_hit_at_5=section_hit,
        reciprocal_rank=rr,
        latency_ms=res.latency_ms,
        top_docs=list(dict.fromkeys(top_docs[:5])),
        expected_docs=expected_docs,
    )


def run_benchmark(
    cfg: AppConfig | None = None,
    golden_path: Path | None = None,
    report_path: Path | None = None,
    retriever: HybridRetriever | None = None,
) -> dict[str, Any]:
    """Run BM25 vs Dense vs Hybrid evaluation over the golden retrieval dataset."""
    app_cfg = cfg or load_config()
    g_path = golden_path or (app_cfg.project_root / "evals" / "retrieval_golden.json")
    r_path = report_path or (app_cfg.project_root / "evals" / "retrieval_report.json")

    golden = json.loads(g_path.read_text(encoding="utf-8"))
    active_retriever = retriever or HybridRetriever(
        app_cfg.paths.processed_dir,
        app_cfg.paths.index_dir,
        app_cfg.retrieval,
    )

    modes: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid")
    summary: dict[str, dict[str, float]] = {}
    misses_by_mode: dict[str, list[dict[str, Any]]] = {}
    per_query_by_mode: dict[str, list[dict[str, Any]]] = {}

    for mode in modes:
        per_query = [_evaluate_query(active_retriever, item, mode=mode) for item in golden]
        n = len(per_query) or 1
        sec_scores = [q.section_hit_at_5 for q in per_query if q.section_hit_at_5 is not None]

        summary[mode] = {
            "doc_hit_at_1": round(sum(q.hit_at_1 for q in per_query) / n, 4),
            "doc_hit_at_3": round(sum(q.hit_at_3 for q in per_query) / n, 4),
            "doc_hit_at_5": round(sum(q.hit_at_5 for q in per_query) / n, 4),
            "doc_recall_at_8": round(sum(q.recall_at_8 for q in per_query) / n, 4),
            "full_set_hit_at_8": round(sum(q.full_set_hit_at_8 for q in per_query) / n, 4),
            "section_hit_at_5": round(sum(sec_scores) / len(sec_scores), 4) if sec_scores else 0.0,
            "mrr": round(sum(q.reciprocal_rank for q in per_query) / n, 4),
            "avg_latency_ms": round(sum(q.latency_ms for q in per_query) / n, 2),
        }

        per_query_by_mode[mode] = [
            {
                "id": q.query_id,
                "query": q.query,
                "hit_at_1": q.hit_at_1,
                "hit_at_5": q.hit_at_5,
                "full_set_hit_at_8": q.full_set_hit_at_8,
                "section_hit_at_5": q.section_hit_at_5,
                "mrr": round(q.reciprocal_rank, 4),
                "top_docs": q.top_docs,
                "expected_docs": q.expected_docs,
            }
            for q in per_query
        ]

        misses_by_mode[mode] = [
            {
                "id": q.query_id,
                "query": q.query,
                "hit_at_5": q.hit_at_5,
                "full_set_hit_at_8": q.full_set_hit_at_8,
                "section_hit_at_5": q.section_hit_at_5,
                "top_docs": q.top_docs,
                "expected_docs": q.expected_docs,
            }
            for q in per_query
            if q.hit_at_5 < 1.0 or q.full_set_hit_at_8 < 1.0 or (q.section_hit_at_5 is not None and q.section_hit_at_5 < 1.0)
        ]

    report = {
        "num_queries": len(golden),
        "backend_used": active_retriever.embedder.backend_name,
        "model_used": active_retriever.embedder.model_name,
        "summary": summary,
        "misses": misses_by_mode,
        "per_query": per_query_by_mode,
    }
    r_path.parent.mkdir(parents=True, exist_ok=True)
    r_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
