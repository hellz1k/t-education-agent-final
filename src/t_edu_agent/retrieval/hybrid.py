"""Hybrid retriever combining Okapi BM25, dense embeddings, RRF, and metadata signals."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

import numpy as np

from t_edu_agent.config import RetrievalConfig
from t_edu_agent.retrieval.bm25 import BM25Index
from t_edu_agent.retrieval.embeddings import BaseEmbedder, EmbeddingCache, create_embedder
from t_edu_agent.retrieval.models import (
    ProgramHit,
    RetrievalResult,
    RetrievedChunk,
    SearchFilters,
)
from t_edu_agent.retrieval.text_processing import expand_query_aliases, infer_query_hints
from t_edu_agent.schemas import ChunkRecord, DocumentRecord

RetrievalMode = Literal["hybrid", "bm25", "dense"]

_ENTITY_BOOST_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("intensive_bi_analytics.md", re.compile(r"\bbi[-\s]?аналитик", re.IGNORECASE)),
    ("intensive_data.md", re.compile(r"интенсив\w*\s+(?:по\s+)?data\b", re.IGNORECASE)),
    ("intensive_javascript.md", re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE)),
    ("intensive_product_analytics.md", re.compile(r"продуктов\w+\s+аналитик", re.IGNORECASE)),
    ("intensive_qa_backend.md", re.compile(r"\bqa\s+backend\b|интенсив\w*\s+по\s+qa", re.IGNORECASE)),
    ("intensive_risk_analytics.md", re.compile(r"риск[-\s]?аналитик", re.IGNORECASE)),
    ("intensive_system_analysis.md", re.compile(r"системн\w+\s+анализ", re.IGNORECASE)),
    ("school_android.md", re.compile(r"\bandroid\b", re.IGNORECASE)),
    ("school_bilet_v_leto.md", re.compile(r"билет\w*\s+в\s+лето", re.IGNORECASE)),
    ("school_dano.md", re.compile(r"\bdano\b", re.IGNORECASE)),
    (
        "school_data_analysis.md",
        re.compile(r"анализ\w*\s+данных.*школьник|курс.*для\s+школьников.*анализ", re.IGNORECASE),
    ),
    ("school_generation.md", re.compile(r"т[-\s]?поколен", re.IGNORECASE)),
    ("school_informatics.md", re.compile(r"информатик\w+\s+решает", re.IGNORECASE)),
    ("school_ios.md", re.compile(r"\bios\b|ios\s+для\s+начинающих", re.IGNORECASE)),
    ("school_math.md", re.compile(r"математик\w+\s+решает", re.IGNORECASE)),
    (
        "school_mshp.md",
        re.compile(r"(?:школ\w+\s+программист\w*|\bмшп\b)(?!.*(?:преподава|ассистент|ваканс))", re.IGNORECASE),
    ),
    (
        "start_mshp_teacher.md",
        re.compile(r"(?:преподава\w*|ассистент|ваканс\w*).*(?:мшп|школ\w+\s+программист)", re.IGNORECASE),
    ),
    ("school_partnership.md", re.compile(r"партнерск\w+\s+программ|т[-\s]?класс|совместн\w+\s+проект", re.IGNORECASE)),
    ("school_prod.md", re.compile(r"\bprod\b", re.IGNORECASE)),
    ("school_project_activities.md", re.compile(r"проектн\w+\s+деятельност|баз\w+\s+школьных\s+проектов", re.IGNORECASE)),
    ("school_t_number.md", re.compile(r"число\s+т|тренажер", re.IGNORECASE)),
    ("start_analyst.md", re.compile(r"т[-\s]?старт\s*:?\s*аналитик", re.IGNORECASE)),
    ("start_dwh.md", re.compile(r"т[-\s]?старт\s*:?\s*ит[-\s]?аналитик|\bdwh\b|\bdlh\b", re.IGNORECASE)),
    ("t_academy_algorithms.md", re.compile(r"алгоритм\w*\s+и\s+структур", re.IGNORECASE)),
    ("t_academy_project_workshop.md", re.compile(r"проектн\w+\s+практикум", re.IGNORECASE)),
    ("t_academy_qa.md", re.compile(r"т[-\s]?академ\w*(?:\s*[:—-]?\s*|\s+по\s+|\s+направлени\w+\s+)(?:qa|тестирован)|qa\s+mobile", re.IGNORECASE)),
    ("t_academy_software_development.md", re.compile(r"т[-\s]?академ\w*(?:\s*[:—-]?\s*|\s+по\s+|\s+направлени\w+\s+)разработк", re.IGNORECASE)),
    (
        "company_overview.md",
        re.compile(r"ано\s+дпо|о\s+компании|в\s+целом\s+об?\s+т[-\s]?образован|когда\s+основан", re.IGNORECASE),
    ),
]


def _load_documents(processed_dir: Path) -> list[DocumentRecord]:
    """Read DocumentRecord items from documents.jsonl."""
    path = processed_dir / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed documents not found: {path}")
    docs: list[DocumentRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line_s = line.strip()
            if line_s:
                docs.append(DocumentRecord.model_validate_json(line_s))
    return docs


def _load_chunks(processed_dir: Path) -> list[ChunkRecord]:
    """Read ChunkRecord items from chunks.jsonl."""
    path = processed_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed chunks not found: {path}")
    chunks: list[ChunkRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line_s = line.strip()
            if line_s:
                chunks.append(ChunkRecord.model_validate_json(line_s))
    return chunks


def _merge_filters(explicit: SearchFilters | None, inferred: SearchFilters) -> SearchFilters:
    """Merge caller-supplied SearchFilters over soft inferred query hints."""
    if explicit is None:
        return inferred
    merged = inferred.model_copy(deep=True)
    for field_name in SearchFilters.model_fields:
        val = getattr(explicit, field_name)
        if val is not None:
            setattr(merged, field_name, val)
    return merged


def _detect_entity_matches(query: str) -> set[str]:
    """Return source_file names whose distinctive program name appears in query."""
    q_norm = query.replace("ё", "е")
    matched: set[str] = set()
    for source_file, pattern in _ENTITY_BOOST_RULES:
        if pattern.search(q_norm):
            matched.add(source_file)
    return matched


class HybridRetriever:
    """Two-stage hybrid retriever: Okapi BM25 + Dense Embeddings + RRF + Catalog Signals."""

    def __init__(
        self,
        processed_dir: Path,
        index_dir: Path,
        cfg: RetrievalConfig,
        embedder: BaseEmbedder | None = None,
        force_rebuild: bool = False,
    ) -> None:
        self.processed_dir = processed_dir
        self.index_dir = index_dir
        self.cfg = cfg

        self.documents: list[DocumentRecord] = _load_documents(processed_dir)
        self.docs_by_id: dict[str, DocumentRecord] = {d.doc_id: d for d in self.documents}
        self.chunks: list[ChunkRecord] = _load_chunks(processed_dir)

        self.bm25 = BM25Index(k1=cfg.bm25_k1, b=cfg.bm25_b).fit(
            [c.search_text for c in self.chunks]
        )

        self.embedder: BaseEmbedder = embedder or create_embedder(
            cfg,
            model_cache_dir=index_dir / "model_cache",
        )
        self.cache = EmbeddingCache(index_dir)
        self.embeddings, self.cache_hit = self.cache.get_or_build(
            self.chunks,
            self.embedder,
            force_rebuild=force_rebuild,
        )

    def _compute_metadata_boost(
        self,
        chunk: ChunkRecord,
        doc: DocumentRecord,
        merged_filters: SearchFilters,
        explicit_filters: SearchFilters | None,
        entity_matches: set[str],
    ) -> tuple[float, list[str]]:
        """Compute bounded explainable metadata adjustment scaled to 1/rrf_k."""
        scale = 1.0 / self.cfg.rrf_k
        boost_units = 0.0
        reasons: list[str] = []
        meta = doc.metadata

        if chunk.source_file in entity_matches:
            boost_units += 0.28
            reasons.append("entity_match")

        if merged_filters.preferred_sections:
            if chunk.section in merged_filters.preferred_sections:
                boost_units += 0.22
                reasons.append(f"section:{chunk.section}")
        elif chunk.chunk_type == "overview_card":
            boost_units += 0.04
            reasons.append("overview_tiebreak")

        if merged_filters.categories:
            if meta.category in merged_filters.categories:
                boost_units += 0.25
                reasons.append(f"category:{meta.category}")

        if merged_filters.format_modes:
            if any(fmt in meta.format_modes for fmt in merged_filters.format_modes):
                boost_units += 0.15
                reasons.append("format_match")
            elif explicit_filters and explicit_filters.format_modes and meta.format_modes:
                boost_units -= 0.15
                reasons.append("format_mismatch")

        if merged_filters.offline_city:
            if merged_filters.offline_city in meta.offline_cities:
                boost_units += 0.35
                reasons.append(f"offline_city:{merged_filters.offline_city}")
            elif explicit_filters and explicit_filters.offline_city and meta.offline_cities:
                boost_units -= 0.15

        if merged_filters.has_selection is not None and meta.has_selection is not None:
            if meta.has_selection is merged_filters.has_selection:
                boost_units += 0.22
                reasons.append(f"selection:{meta.has_selection}")
            else:
                boost_units -= 0.22
                reasons.append("selection_mismatch")

        if merged_filters.is_free is not None and meta.is_free is not None:
            if meta.is_free is merged_filters.is_free:
                boost_units += 0.20
                reasons.append(f"is_free:{meta.is_free}")
            else:
                boost_units -= 0.20
                reasons.append("cost_mismatch")

        if merged_filters.audience and meta.audience:
            if any(aud in meta.audience for aud in merged_filters.audience):
                boost_units += 0.25
                reasons.append("audience_match")
            elif "все желающие" in meta.audience:
                boost_units += 0.10
                reasons.append("audience_open_all")
            else:
                boost_units -= 0.30
                reasons.append("audience_mismatch")

        if merged_filters.enrollment_statuses:
            if meta.enrollment_status in merged_filters.enrollment_statuses:
                boost_units += 0.15
                reasons.append(f"status:{meta.enrollment_status}")

        if merged_filters.required_skills_any:
            if any(s in meta.skills_mentioned for s in merged_filters.required_skills_any):
                boost_units += 0.15
                reasons.append("skill_match")

        if merged_filters.exclude_required_skills:
            excluded = set(merged_filters.exclude_required_skills)
            if excluded & set(meta.required_skills):
                boost_units -= 0.50
                reasons.append("excluded_skill_required")
            elif not meta.required_skills or (excluded & set(meta.optional_or_not_required_skills)):
                boost_units += 0.25
                reasons.append("no_excluded_skill_required")

        return boost_units * scale, reasons

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        max_chunks_per_doc: int | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> RetrievalResult:
        """Retrieve top chunks and grouped parent programs for a user query."""
        t0 = time.perf_counter()
        k = top_k or self.cfg.top_k
        per_doc_cap = max_chunks_per_doc or self.cfg.max_chunks_per_doc

        expanded_query = expand_query_aliases(query)
        inferred_hints, _ = infer_query_hints(query)
        merged_filters = _merge_filters(filters, inferred_hints)
        entity_matches = _detect_entity_matches(query)

        if not self.chunks:
            return RetrievalResult(
                query=query,
                expanded_query=expanded_query,
                applied_filters=merged_filters,
                backend_used=self.embedder.backend_name,
                model_used=self.embedder.model_name,
                latency_ms=0.0,
            )

        allowed_doc_ids = set(merged_filters.doc_ids) if merged_filters.doc_ids else None
        if allowed_doc_ids:
            per_doc_cap = max(per_doc_cap, max(3, k // max(1, len(allowed_doc_ids))))

        candidate_indices = [
            idx
            for idx, ch in enumerate(self.chunks)
            if allowed_doc_ids is None or ch.doc_id in allowed_doc_ids
        ]
        if not candidate_indices:
            return RetrievalResult(
                query=query,
                expanded_query=expanded_query,
                applied_filters=merged_filters,
                backend_used=self.embedder.backend_name,
                model_used=self.embedder.model_name,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

        bm25_scores = self.bm25.score_all(expanded_query)
        if mode == "bm25":
            dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
        else:
            q_vec = self.embedder.encode_query(query)
            dense_scores = (self.embeddings @ q_vec).astype(np.float32)

        pool_size = min(len(candidate_indices), self.cfg.candidate_pool_size)

        bm25_sorted = [
            idx
            for idx in sorted(candidate_indices, key=lambda i: float(bm25_scores[i]), reverse=True)
            if float(bm25_scores[idx]) > 0.0
        ]
        dense_sorted = sorted(
            candidate_indices,
            key=lambda i: float(dense_scores[i]),
            reverse=True,
        )

        bm25_rank_map: dict[int, int] = {
            idx: rank for rank, idx in enumerate(bm25_sorted[:pool_size], start=1)
        }
        dense_rank_map: dict[int, int] = {
            idx: rank for rank, idx in enumerate(dense_sorted[:pool_size], start=1)
        }

        scored: list[RetrievedChunk] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            doc = self.docs_by_id[chunk.doc_id]
            b_score = float(bm25_scores[idx])
            d_score = float(dense_scores[idx])
            b_rank = bm25_rank_map.get(idx)
            d_rank = dense_rank_map.get(idx)

            if mode == "bm25":
                if b_score <= 0.0:
                    continue
                total_score = b_score
                meta_boost = 0.0
                reasons = ["bm25_only"]
            elif mode == "dense":
                total_score = d_score
                meta_boost = 0.0
                reasons = ["dense_only"]
            else:
                if b_rank is None and d_rank is None and allowed_doc_ids is None:
                    continue
                rrf_score = 0.0
                if b_rank is not None:
                    rrf_score += self.cfg.bm25_weight / (self.cfg.rrf_k + b_rank)
                if d_rank is not None:
                    rrf_score += self.cfg.dense_weight / (self.cfg.rrf_k + d_rank)
                meta_boost, reasons = self._compute_metadata_boost(
                    chunk,
                    doc,
                    merged_filters,
                    filters,
                    entity_matches,
                )
                total_score = rrf_score + meta_boost

            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=round(total_score, 6),
                    bm25_score=round(b_score, 4),
                    dense_score=round(d_score, 4),
                    bm25_rank=b_rank,
                    dense_rank=d_rank,
                    metadata_boost=round(meta_boost, 6),
                    match_reasons=reasons,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)

        selected: list[RetrievedChunk] = []
        doc_counts: dict[str, int] = {}
        for item in scored:
            did = item.chunk.doc_id
            if doc_counts.get(did, 0) >= per_doc_cap:
                continue
            selected.append(item)
            doc_counts[did] = doc_counts.get(did, 0) + 1
            if len(selected) >= k:
                break

        # Group retrieved chunks into program-level hits
        grouped: dict[str, list[RetrievedChunk]] = {}
        for item in selected:
            grouped.setdefault(item.chunk.doc_id, []).append(item)

        program_hits: list[ProgramHit] = []
        for did, hit_chunks in grouped.items():
            best_score = max(c.score for c in hit_chunks)
            multi_bonus = 0.05 * best_score * max(0, len(hit_chunks) - 1)
            program_hits.append(
                ProgramHit(
                    doc=self.docs_by_id[did],
                    score=round(best_score + multi_bonus, 6),
                    matched_chunks=hit_chunks,
                )
            )
        program_hits.sort(key=lambda p: p.score, reverse=True)
        program_hits = program_hits[: self.cfg.expand_top_programs]

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RetrievalResult(
            query=query,
            expanded_query=expanded_query,
            applied_filters=merged_filters,
            backend_used=self.embedder.backend_name,
            model_used=self.embedder.model_name,
            chunks=selected,
            programs=program_hits,
            latency_ms=latency_ms,
        )
