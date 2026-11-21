"""Pydantic v2 data contracts for documents, chunks, source registry, and KB stats."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProgramCategory = Literal[
    "intensive",
    "school_course",
    "academy",
    "course",
    "internship",
    "olympiad",
    "announcement",
    "company_overview",
    "general",
]

EnrollmentStatus = Literal["open", "closed", "rolling", "unknown"]
RegistryStatus = Literal["active", "archived_older_version", "duplicate_skipped"]
ChunkType = Literal["overview_card", "section"]


class SectionRecord(BaseModel):
    """Parsed semantic section of a knowledge base document."""

    section_id: str
    heading: str
    level: int = Field(ge=1, le=6)
    text: str
    char_count: int = Field(ge=0)
    has_program_rule: bool = False
    has_general_rule: bool = False
    has_missing_info_note: bool = False


class ProgramMetadata(BaseModel):
    """Structured metadata extracted deterministically from a program page or brief."""

    category: ProgramCategory = "general"
    audience: list[str] = Field(default_factory=list)
    format_modes: list[str] = Field(default_factory=list)
    offline_cities: list[str] = Field(default_factory=list)
    priority_cities: list[str] = Field(default_factory=list)
    has_selection: bool | None = None
    is_free: bool | None = None
    cost_note: str | None = None
    enrollment_status: EnrollmentStatus = "unknown"
    enrollment_note: str | None = None
    workload_note: str | None = None
    duration_note: str | None = None
    skills_mentioned: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    optional_or_not_required_skills: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """Normalized active knowledge base document with sections and metadata."""

    doc_id: str
    title: str
    source_file: str
    file_format: str
    source_url: str | None = None
    additional_urls: list[str] = Field(default_factory=list)
    verified_at: str | None = None
    published_at: str | None = None
    doc_version: str | None = None
    content_hash: str
    char_count: int = Field(ge=0)
    metadata: ProgramMetadata = Field(default_factory=ProgramMetadata)
    summary_card: str
    sections: list[SectionRecord] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """Searchable contextualized chunk linked back to its source document."""

    chunk_id: str
    doc_id: str
    chunk_type: ChunkType
    title: str
    section: str
    part_index: int = Field(ge=1)
    total_parts: int = Field(ge=1)
    source_url: str | None = None
    source_file: str
    verified_at: str | None = None
    category: str
    audience: list[str] = Field(default_factory=list)
    format_modes: list[str] = Field(default_factory=list)
    has_selection: bool | None = None
    enrollment_status: str = "unknown"
    text: str
    search_text: str
    char_count: int = Field(ge=0)
    search_char_count: int = Field(ge=0)


class SourceRegistryEntry(BaseModel):
    """Audit entry for every file discovered during ingestion."""

    doc_id: str
    title: str
    source_file: str
    file_format: str
    source_url: str | None = None
    additional_urls: list[str] = Field(default_factory=list)
    category: str
    verified_at: str | None = None
    published_at: str | None = None
    doc_version: str | None = None
    content_hash: str
    status: RegistryStatus
    chunk_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    note: str | None = None


class IngestionError(BaseModel):
    """Non-fatal file loading or parsing error captured during ingestion."""

    source_file: str
    error_type: str
    message: str


class VersionConflict(BaseModel):
    """Record of two documents sharing a source URL with different content."""

    source_url: str
    kept_file: str
    archived_file: str
    kept_verified_at: str | None = None
    archived_verified_at: str | None = None
    reason: str


class KBStats(BaseModel):
    """Aggregate knowledge base metrics and audit report metadata."""

    generated_at: str
    total_raw_files: int = Field(ge=0)
    unique_documents: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    total_clean_chars: int = Field(ge=0)
    max_chunk_search_chars: int = Field(ge=0)
    avg_chunk_search_chars: float = Field(ge=0.0)
    categories_count: dict[str, int] = Field(default_factory=dict)
    sections_count: dict[str, int] = Field(default_factory=dict)
    formats_processed: dict[str, int] = Field(default_factory=dict)
    duplicates_skipped: int = Field(ge=0)
    version_conflicts: list[VersionConflict] = Field(default_factory=list)
    errors: list[IngestionError] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
