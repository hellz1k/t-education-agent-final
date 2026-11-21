"""Configuration models and YAML loader for T-Education AI Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "settings.yaml"


def _resolve_path(value: str | Path, root: Path) -> Path:
    """Resolve a path relative to project root when not already absolute."""
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


class PathsConfig(BaseModel):
    """Filesystem directories used across ingestion, indexing, and logging."""

    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    index_dir: Path = Field(default=Path("data/index"))
    logs_dir: Path = Field(default=Path("logs"))


class IngestionConfig(BaseModel):
    """Settings controlling document loading, parsing, and chunking."""

    supported_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".txt", ".docx", ".html", ".htm"]
    )
    max_chunk_chars: int = Field(default=1000, ge=200)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    min_chunk_chars: int = Field(default=100, ge=0)
    include_overview_card_chunk: bool = True


class RetrievalConfig(BaseModel):
    """Settings for hybrid BM25 + dense embedding retrieval (Stage 2+)."""

    embedding_backend: Literal["auto", "fastembed", "sentence_transformers", "tfidf"] = "auto"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_max_tokens: int = Field(default=512, ge=64)
    query_prefix: str = ""
    passage_prefix: str = ""
    bm25_k1: float = Field(default=1.5, ge=0.1)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    candidate_pool_size: int = Field(default=60, ge=5)
    top_k: int = Field(default=8, ge=1)
    max_chunks_per_doc: int = Field(default=2, ge=1)
    expand_top_programs: int = Field(default=4, ge=1)


class LLMConfig(BaseModel):
    """LLM provider and generation settings (secrets stay in environment vars)."""

    provider: str = "groq"
    planner_model: str = "qwen/qwen3.8-27b"
    primary_model: str = "openai/gpt-oss-120b"
    fallback_models: list[str] = Field(default_factory=lambda: ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"])
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2400, ge=64)
    timeout_sec: int = Field(default=30, ge=1)
    max_retries: int = Field(default=1, ge=0)


class LoggingConfig(BaseModel):
    """Application and diagnostic logging configuration."""

    level: str = "INFO"
    diagnostic_log_file: Path = Field(default=Path("logs/diagnostics.jsonl"))


class AppConfig(BaseModel):
    """Top-level application configuration container."""

    project_root: Path = PROJECT_ROOT
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Path | None = None, project_root: Path | None = None) -> AppConfig:
    """Load YAML configuration and resolve relative paths against project root."""
    root = (project_root or PROJECT_ROOT).resolve()
    target = config_path or (root / "configs" / "settings.yaml")
    raw_data: dict[str, Any] = {}
    if target.exists():
        with target.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected YAML mapping in {target}, got {type(loaded).__name__}")
                raw_data = loaded

    cfg = AppConfig(**raw_data, project_root=root)
    cfg.paths = PathsConfig(
        raw_dir=_resolve_path(cfg.paths.raw_dir, root),
        processed_dir=_resolve_path(cfg.paths.processed_dir, root),
        index_dir=_resolve_path(cfg.paths.index_dir, root),
        logs_dir=_resolve_path(cfg.paths.logs_dir, root),
    )
    cfg.logging = LoggingConfig(
        level=cfg.logging.level,
        diagnostic_log_file=_resolve_path(cfg.logging.diagnostic_log_file, root),
    )
    return cfg
