"""Pydantic v2 data contracts for hybrid retrieval filters, results, and program hits."""

from __future__ import annotations

from pydantic import BaseModel, Field

from t_edu_agent.schemas import ChunkRecord, DocumentRecord


class SearchFilters(BaseModel):
    """Structured metadata filters and soft preference hints for retrieval."""

    categories: list[str] | None = None
    audience: list[str] | None = None
    format_modes: list[str] | None = None
    offline_city: str | None = None
    has_selection: bool | None = None
    is_free: bool | None = None
    enrollment_statuses: list[str] | None = None
    required_skills_any: list[str] | None = None
    exclude_required_skills: list[str] | None = None
    doc_ids: list[str] | None = None
    preferred_sections: list[str] | None = None


class RetrievedChunk(BaseModel):
    """Ranked chunk with transparent BM25, dense, and metadata scoring breakdown."""

    chunk: ChunkRecord
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    bm25_rank: int | None = None
    dense_rank: int | None = None
    metadata_boost: float = 0.0
    match_reasons: list[str] = Field(default_factory=list)


class ProgramHit(BaseModel):
    """Program-level aggregation linking top retrieved chunks to their parent document."""

    doc: DocumentRecord
    score: float
    matched_chunks: list[RetrievedChunk] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Complete retrieval output returned to the agent or evaluation harness."""

    query: str
    expanded_query: str
    applied_filters: SearchFilters = Field(default_factory=SearchFilters)
    backend_used: str
    model_used: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    programs: list[ProgramHit] = Field(default_factory=list)
    latency_ms: float = 0.0
