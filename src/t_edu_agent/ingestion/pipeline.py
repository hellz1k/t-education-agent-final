"""End-to-end ingestion orchestration, deduplication, versioning, and KB audit reports."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from t_edu_agent.config import IngestionConfig
from t_edu_agent.ingestion.chunker import build_chunks
from t_edu_agent.ingestion.loader import EmptyDocumentError, UnsupportedFormatError, load_raw_document
from t_edu_agent.ingestion.parser import category_label_ru, parse_document, slugify
from t_edu_agent.schemas import (
    ChunkRecord,
    DocumentRecord,
    IngestionError,
    KBStats,
    SourceRegistryEntry,
    VersionConflict,
)


def _normalize_url_key(url: str | None) -> str | None:
    """Normalize a URL for version-conflict grouping."""
    if not url:
        return None
    return url.strip().rstrip("/").lower()


def _is_catalog_root_url(url: str) -> bool:
    """Return True when a URL points to a top-level section/catalog root rather than a program page."""
    parsed = urlparse(url.strip())
    segments = [seg for seg in parsed.path.split("/") if seg]
    return len(segments) <= 1


def _discover_raw_files(raw_dir: Path) -> list[Path]:
    """List candidate raw files deterministically, ignoring hidden/temp files."""
    if not raw_dir.exists():
        return []
    files: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(raw_dir).parts
        if any(part.startswith(".") or part.startswith("~$") for part in rel_parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(raw_dir).as_posix())


def _resolve_url_versions(
    candidates: list[DocumentRecord],
    mtime_by_file: dict[str, str],
) -> tuple[list[DocumentRecord], list[DocumentRecord], list[VersionConflict]]:
    """Keep the newest document per source_url and archive older versions."""
    by_key: dict[tuple[str, str | None], list[DocumentRecord]] = {}
    without_url: list[DocumentRecord] = []

    for doc in candidates:
        url_key = _normalize_url_key(doc.source_url)
        if url_key is None:
            without_url.append(doc)
        elif doc.source_url and _is_catalog_root_url(doc.source_url):
            by_key.setdefault((url_key, slugify(doc.title)), []).append(doc)
        else:
            by_key.setdefault((url_key, None), []).append(doc)

    active_docs: list[DocumentRecord] = list(without_url)
    archived_docs: list[DocumentRecord] = []
    conflicts: list[VersionConflict] = []

    for (url_key, _), docs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        if len(docs) == 1:
            active_docs.append(docs[0])
            continue

        ordered = sorted(
            docs,
            key=lambda d: (
                d.verified_at or "",
                mtime_by_file.get(d.source_file, ""),
                d.source_file,
            ),
            reverse=True,
        )
        winner = ordered[0]
        active_docs.append(winner)

        for loser in ordered[1:]:
            archived_docs.append(loser)
            reason = (
                f"Выбрана более актуальная дата проверки ({winner.verified_at} >= {loser.verified_at})"
                if winner.verified_at != loser.verified_at
                else f"Одинаковая дата проверки ({winner.verified_at or 'не указана'}); выбран более новый файл"
            )
            conflicts.append(
                VersionConflict(
                    source_url=winner.source_url or url_key,
                    kept_file=winner.source_file,
                    archived_file=loser.source_file,
                    kept_verified_at=winner.verified_at,
                    archived_verified_at=loser.verified_at,
                    reason=reason,
                )
            )

    active_docs.sort(key=lambda d: d.doc_id)
    archived_docs.sort(key=lambda d: d.source_file)
    return active_docs, archived_docs, conflicts


def _render_kb_report(
    stats: KBStats,
    registry: list[SourceRegistryEntry],
) -> str:
    """Render a clean human-readable KB audit report in Markdown."""
    lines: list[str] = [
        "# Реестр источников и статистика базы знаний (Т-Образование)",
        "",
        f"Сформировано: `{stats.generated_at}`",
        "",
        "## Сводные показатели",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
        f"| Всего файлов в `data/raw` | {stats.total_raw_files} |",
        f"| Уникальных активных документов | {stats.unique_documents} |",
        f"| Всего чанков | {stats.total_chunks} |",
        f"| Объём очищенного текста (символов) | {stats.total_clean_chars} |",
        f"| Средняя длина чанка для поиска (символов) | {stats.avg_chunk_search_chars} |",
        f"| Максимальная длина чанка для поиска (символов) | {stats.max_chunk_search_chars} |",
        f"| Пропущено точных дублей | {stats.duplicates_skipped} |",
        f"| Конфликтов версий (архивировано) | {len(stats.version_conflicts)} |",
        f"| Ошибок обработки | {len(stats.errors)} |",
        "",
        "## Тематические категории",
        "",
        "| Категория | Количество документов |",
        "| --- | --- |",
    ]
    for cat, count in stats.categories_count.items():
        lines.append(f"| {category_label_ru(cat)} (`{cat}`) | {count} |")

    lines.extend(
        [
            "",
            "## Реестр источников",
            "",
            "| Файл | Документ | Категория | Формат | Статус | Проверено | Чанки | URL |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in registry:
        url_cell = entry.source_url or "—"
        ver_cell = entry.verified_at or "—"
        title_clean = entry.title.replace("|", "\\|")
        lines.append(
            f"| `{entry.source_file}` | {title_clean} | `{entry.category}` | `{entry.file_format}` | "
            f"`{entry.status}` | {ver_cell} | {entry.chunk_count} | {url_cell} |"
        )

    lines.extend(["", "## Разделы в активных документах", "", "| Раздел | Встречается раз |", "| --- | --- |"])
    for sec_name, count in stats.sections_count.items():
        lines.append(f"| {sec_name} | {count} |")

    lines.extend(["", "## Конфликты версий", ""])
    if stats.version_conflicts:
        for conflict in stats.version_conflicts:
            lines.append(
                f"- `{conflict.source_url}`: оставлен `{conflict.kept_file}` "
                f"({conflict.kept_verified_at or 'без даты'}), архивирован `{conflict.archived_file}` "
                f"({conflict.archived_verified_at or 'без даты'}) — {conflict.reason}"
            )
    else:
        lines.append("Конфликты версий не обнаружены.")

    lines.extend(["", "## Ошибки обработки", ""])
    if stats.errors:
        for err in stats.errors:
            lines.append(f"- `{err.source_file}` [{err.error_type}]: {err.message}")
    else:
        lines.append("Ошибок обработки не обнаружено.")

    lines.extend(["", "## Поддерживаемые форматы и ограничения", ""])
    for item in stats.known_limitations:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _write_outputs(
    out_dir: Path,
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    registry: list[SourceRegistryEntry],
    stats: KBStats,
) -> None:
    """Persist processed documents, chunks, source registry, stats JSON, and Markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc.model_dump_json() + "\n")

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")

    with (out_dir / "source_registry.json").open("w", encoding="utf-8") as fh:
        json.dump(
            [entry.model_dump() for entry in registry],
            fh,
            ensure_ascii=False,
            indent=2,
        )

    with (out_dir / "kb_stats.json").open("w", encoding="utf-8") as fh:
        fh.write(stats.model_dump_json(indent=2) + "\n")

    (out_dir / "kb_report.md").write_text(
        _render_kb_report(stats, registry),
        encoding="utf-8",
    )


def run_ingestion(
    raw_dir: Path,
    out_dir: Path,
    cfg: IngestionConfig,
) -> tuple[list[DocumentRecord], list[ChunkRecord], list[SourceRegistryEntry], KBStats]:
    """Run full deterministic ingestion from raw_dir into out_dir."""
    raw_files = _discover_raw_files(raw_dir)
    errors: list[IngestionError] = []
    formats_counter: Counter[str] = Counter()

    seen_hashes: dict[str, str] = {}
    unique_candidates: list[DocumentRecord] = []
    mtime_by_file: dict[str, str] = {}
    registry: list[SourceRegistryEntry] = []
    duplicates_skipped = 0

    for path in raw_files:
        rel_path = path.relative_to(raw_dir).as_posix()
        ext_label = path.suffix.lower() or "(no_ext)"
        try:
            loaded = load_raw_document(path, raw_dir, cfg.supported_extensions)
            doc = parse_document(loaded)
        except (UnsupportedFormatError, EmptyDocumentError, OSError, ValueError) as exc:
            errors.append(
                IngestionError(
                    source_file=rel_path,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        formats_counter[ext_label] += 1
        mtime_by_file[doc.source_file] = loaded.mtime_iso

        if doc.content_hash in seen_hashes:
            duplicates_skipped += 1
            registry.append(
                SourceRegistryEntry(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_file=doc.source_file,
                    file_format=doc.file_format,
                    source_url=doc.source_url,
                    additional_urls=doc.additional_urls,
                    category=doc.metadata.category,
                    verified_at=doc.verified_at,
                    published_at=doc.published_at,
                    doc_version=doc.doc_version,
                    content_hash=doc.content_hash,
                    status="duplicate_skipped",
                    chunk_count=0,
                    char_count=doc.char_count,
                    note=f"Точный дубль файла {seen_hashes[doc.content_hash]}",
                )
            )
            continue

        seen_hashes[doc.content_hash] = doc.source_file
        unique_candidates.append(doc)

    active_docs, archived_docs, version_conflicts = _resolve_url_versions(
        unique_candidates,
        mtime_by_file,
    )

    chunks: list[ChunkRecord] = []
    categories_counter: Counter[str] = Counter()
    sections_counter: Counter[str] = Counter()

    for doc in active_docs:
        doc_chunks = build_chunks(doc, cfg)
        chunks.extend(doc_chunks)
        categories_counter[doc.metadata.category] += 1
        for sec in doc.sections:
            sections_counter[sec.heading] += 1

        registry.append(
            SourceRegistryEntry(
                doc_id=doc.doc_id,
                title=doc.title,
                source_file=doc.source_file,
                file_format=doc.file_format,
                source_url=doc.source_url,
                additional_urls=doc.additional_urls,
                category=doc.metadata.category,
                verified_at=doc.verified_at,
                published_at=doc.published_at,
                doc_version=doc.doc_version,
                content_hash=doc.content_hash,
                status="active",
                chunk_count=len(doc_chunks),
                char_count=doc.char_count,
            )
        )

    for archived in archived_docs:
        registry.append(
            SourceRegistryEntry(
                doc_id=archived.doc_id,
                title=archived.title,
                source_file=archived.source_file,
                file_format=archived.file_format,
                source_url=archived.source_url,
                additional_urls=archived.additional_urls,
                category=archived.metadata.category,
                verified_at=archived.verified_at,
                published_at=archived.published_at,
                doc_version=archived.doc_version,
                content_hash=archived.content_hash,
                status="archived_older_version",
                chunk_count=0,
                char_count=archived.char_count,
                note="Заменён более актуальной версией того же URL",
            )
        )

    registry.sort(key=lambda entry: entry.source_file)

    search_lengths = [c.search_char_count for c in chunks]
    max_search_len = max(search_lengths) if search_lengths else 0
    avg_search_len = round(sum(search_lengths) / len(search_lengths), 2) if search_lengths else 0.0

    known_limitations = [
        "Реально поддерживаемые форматы на текущем этапе: .md, .txt, .docx и .html/.htm (с очисткой меню, скриптов и футеров).",
        "Файлы .pdf (включая сканы без текстового слоя) не индексируются и фиксируются как ошибка обработки в kb_stats.json.",
        "База отражает сохранённый локальный снимок страниц (дата указана в поле verified_at) и не выполняет живой скрапинг при запросах.",
        "Незаполненные на официальных страницах поля (проходной балл, число мест, адреса офисов, неуказанная стоимость/год) сохраняются как отсутствующие без догадок.",
    ]

    stats = KBStats(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        total_raw_files=len(raw_files),
        unique_documents=len(active_docs),
        total_chunks=len(chunks),
        total_clean_chars=sum(doc.char_count for doc in active_docs),
        max_chunk_search_chars=max_search_len,
        avg_chunk_search_chars=avg_search_len,
        categories_count=dict(sorted(categories_counter.items())),
        sections_count=dict(sorted(sections_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        formats_processed=dict(sorted(formats_counter.items())),
        duplicates_skipped=duplicates_skipped,
        version_conflicts=version_conflicts,
        errors=errors,
        known_limitations=known_limitations,
    )

    _write_outputs(out_dir, active_docs, chunks, registry, stats)
    return active_docs, chunks, registry, stats


def add_file_and_reingest(
    file_path: Path,
    raw_dir: Path,
    out_dir: Path,
    cfg: IngestionConfig,
) -> tuple[list[DocumentRecord], list[ChunkRecord], list[SourceRegistryEntry], KBStats]:
    """Copy a new or updated document into raw_dir and rerun full ingestion."""
    resolved_src = file_path.resolve()
    if not resolved_src.exists() or not resolved_src.is_file():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    target = (raw_dir / resolved_src.name).resolve()
    if resolved_src != target:
        shutil.copy2(resolved_src, target)

    return run_ingestion(raw_dir, out_dir, cfg)
