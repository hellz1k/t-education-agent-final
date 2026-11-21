"""Hybrid retrieval package (BM25 + dense embeddings + RRF + catalog metadata signals)."""

from t_edu_agent.retrieval.bm25 import BM25Index
from t_edu_agent.retrieval.embeddings import (
    BaseEmbedder,
    EmbeddingCache,
    FastEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
    create_embedder,
)
from t_edu_agent.retrieval.hybrid import HybridRetriever
from t_edu_agent.retrieval.models import (
    ProgramHit,
    RetrievalResult,
    RetrievedChunk,
    SearchFilters,
)

__all__ = [
    "BM25Index",
    "BaseEmbedder",
    "EmbeddingCache",
    "FastEmbedder",
    "HybridRetriever",
    "ProgramHit",
    "RetrievalResult",
    "RetrievedChunk",
    "SearchFilters",
    "SentenceTransformerEmbedder",
    "TfidfEmbedder",
    "create_embedder",
]
