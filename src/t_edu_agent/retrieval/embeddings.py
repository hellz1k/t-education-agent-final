"""Embedding backends (FastEmbed ONNX, SentenceTransformers, offline TF-IDF) and disk cache."""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from t_edu_agent.config import RetrievalConfig
from t_edu_agent.retrieval.text_processing import tokenize
from t_edu_agent.schemas import ChunkRecord


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a 2D array to unit L2 length as float32."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return (arr / max(norm, 1e-12)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norms, 1e-12)).astype(np.float32)


class BaseEmbedder(ABC):
    """Abstract dense text embedder returning L2-normalized float32 vectors."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short identifier of the embedding backend."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier recorded in index manifests."""

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Encode document passages into a 2D normalized float32 matrix."""

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query into a 1D normalized float32 vector."""


class FastEmbedder(BaseEmbedder):
    """Lightweight ONNX transformer embedder powered by fastembed."""

    def __init__(
        self,
        model_name: str,
        model_cache_dir: Path | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from fastembed import TextEmbedding

        self._model_name = model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

        cache_str: str | None = None
        has_local_cache = False
        if model_cache_dir is not None:
            model_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_str = str(model_cache_dir)
            has_local_cache = any(model_cache_dir.iterdir())

        if has_local_cache:
            try:
                self._model = TextEmbedding(
                    model_name=model_name,
                    cache_dir=cache_str,
                    local_files_only=True,
                )
                return
            except Exception:
                pass

        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=cache_str,
            local_files_only=False,
        )

    @property
    def backend_name(self) -> str:
        return "fastembed"

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefixed = [f"{self._passage_prefix}{t}" if self._passage_prefix else t for t in texts]
        vectors = list(self._model.embed(prefixed, batch_size=32))
        return _l2_normalize(np.vstack(vectors))

    def encode_query(self, query: str) -> np.ndarray:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query
        vectors = list(self._model.embed([text], batch_size=1))
        return _l2_normalize(np.asarray(vectors[0], dtype=np.float32))


class SentenceTransformerEmbedder(BaseEmbedder):
    """Optional secondary neural backend using sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model = SentenceTransformer(model_name, device="cpu")

    @property
    def backend_name(self) -> str:
        return "sentence_transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefixed = [f"{self._passage_prefix}{t}" if self._passage_prefix else t for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))

    def encode_query(self, query: str) -> np.ndarray:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return _l2_normalize(np.asarray(vec, dtype=np.float32))


class TfidfEmbedder(BaseEmbedder):
    """Zero-network offline fallback combining stemmed word n-grams and character n-grams."""

    def __init__(self, model_name: str = "tfidf-word-char-hybrid") -> None:
        self._model_name = model_name
        self._word_vec = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
        )
        self._fitted = False

    @property
    def backend_name(self) -> str:
        return "tfidf"

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            self._fitted = False
            return np.zeros((0, 0), dtype=np.float32)
        word_mat = self._word_vec.fit_transform(texts)
        try:
            char_mat = self._char_vec.fit_transform(texts)
            combined = hstack([word_mat, char_mat * 0.6]).toarray()
        except ValueError:
            combined = word_mat.toarray()
        self._fitted = True
        return _l2_normalize(combined)

    def encode_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fitted or loaded from cache before encoding queries")
        word_mat = self._word_vec.transform([query])
        if hasattr(self._char_vec, "vocabulary_"):
            char_mat = self._char_vec.transform([query])
            combined = hstack([word_mat, char_mat * 0.6]).toarray()[0]
        else:
            combined = word_mat.toarray()[0]
        return _l2_normalize(combined)

    def save_state(self, path: Path) -> None:
        """Persist fitted vectorizers for offline cache hits."""
        with path.open("wb") as fh:
            pickle.dump({"word_vec": self._word_vec, "char_vec": self._char_vec}, fh)

    def load_state(self, path: Path) -> None:
        """Restore fitted vectorizers from disk."""
        with path.open("rb") as fh:
            state = pickle.load(fh)
        self._word_vec = state["word_vec"]
        self._char_vec = state["char_vec"]
        self._fitted = True


def create_embedder(cfg: RetrievalConfig, model_cache_dir: Path | None = None) -> BaseEmbedder:
    """Instantiate the configured embedding backend with safe auto-detection."""
    backend = cfg.embedding_backend

    if backend == "fastembed":
        return FastEmbedder(
            model_name=cfg.embedding_model,
            model_cache_dir=model_cache_dir,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
        )
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=cfg.embedding_model,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
        )
    if backend == "tfidf":
        return TfidfEmbedder()

    # Auto mode: prefer fastembed ONNX, then sentence-transformers, then offline TF-IDF
    try:
        return FastEmbedder(
            model_name=cfg.embedding_model,
            model_cache_dir=model_cache_dir,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
        )
    except Exception:
        pass

    try:
        return SentenceTransformerEmbedder(
            model_name=cfg.embedding_model,
            query_prefix=cfg.query_prefix,
            passage_prefix=cfg.passage_prefix,
        )
    except Exception:
        pass

    return TfidfEmbedder()


def compute_chunks_fingerprint(chunks: list[ChunkRecord]) -> str:
    """Compute deterministic SHA-256 fingerprint across chunk IDs and search texts."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(chunk.search_text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


class EmbeddingCache:
    """Disk cache storing dense chunk vectors and manifest validation metadata."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.embeddings_path = index_dir / "embeddings.npy"
        self.manifest_path = index_dir / "index_manifest.json"
        self.tfidf_state_path = index_dir / "tfidf_state.pkl"

    def get_or_build(
        self,
        chunks: list[ChunkRecord],
        embedder: BaseEmbedder,
        force_rebuild: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Return cached embedding matrix if fingerprint matches, otherwise rebuild and save."""
        fingerprint = compute_chunks_fingerprint(chunks)

        if (
            not force_rebuild
            and self.embeddings_path.exists()
            and self.manifest_path.exists()
        ):
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                matches = (
                    manifest.get("backend") == embedder.backend_name
                    and manifest.get("model_name") == embedder.model_name
                    and manifest.get("chunks_fingerprint") == fingerprint
                    and manifest.get("num_chunks") == len(chunks)
                )
                if matches:
                    if isinstance(embedder, TfidfEmbedder):
                        if not self.tfidf_state_path.exists():
                            matches = False
                        else:
                            embedder.load_state(self.tfidf_state_path)
                    if matches:
                        matrix = np.load(self.embeddings_path)
                        if matrix.shape[0] == len(chunks):
                            return matrix.astype(np.float32), True
            except Exception:
                pass

        self.index_dir.mkdir(parents=True, exist_ok=True)
        passages = [c.search_text for c in chunks]
        matrix = embedder.encode_passages(passages)

        np.save(self.embeddings_path, matrix)
        if isinstance(embedder, TfidfEmbedder):
            embedder.save_state(self.tfidf_state_path)

        manifest_data = {
            "backend": embedder.backend_name,
            "model_name": embedder.model_name,
            "chunks_fingerprint": fingerprint,
            "num_chunks": len(chunks),
            "embedding_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.manifest_path.write_text(
            json.dumps(manifest_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return matrix, False
