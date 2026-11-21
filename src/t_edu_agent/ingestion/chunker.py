"""Contextualized section-aware chunker respecting embedding token budgets."""

from __future__ import annotations

import re

from t_edu_agent.config import IngestionConfig
from t_edu_agent.ingestion.parser import category_label_ru, slugify, status_label_ru
from t_edu_agent.schemas import ChunkRecord, DocumentRecord, SectionRecord


def _build_contextual_header(doc: DocumentRecord, section_heading: str) -> str:
    """Create a concise metadata prefix so every chunk remains self-contained."""
    cat_ru = category_label_ru(doc.metadata.category)
    if doc.metadata.category in {"company_overview", "general"}:
        return (
            f"Документ: {doc.title}\n"
            f"Категория: {cat_ru}\n"
            f"Раздел: {section_heading}"
        )

    audience_str = ", ".join(doc.metadata.audience) if doc.metadata.audience else "не уточнена"
    format_str = ", ".join(doc.metadata.format_modes) if doc.metadata.format_modes else "не указан"
    if doc.metadata.has_selection is True:
        selection_str = "есть"
    elif doc.metadata.has_selection is False:
        selection_str = "без отбора"
    else:
        selection_str = "не указано"
    status_str = status_label_ru(doc.metadata.enrollment_status)

    return (
        f"Документ: {doc.title}\n"
        f"Категория: {cat_ru} | Аудитория: {audience_str} | Формат: {format_str} | "
        f"Отбор: {selection_str} | Статус набора: {status_str}\n"
        f"Раздел: {section_heading}"
    )


def _is_markdown_table(block: str) -> bool:
    """Return True if a block looks like a Markdown pipe table."""
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    return len(lines) >= 2 and lines[0].startswith("|") and "---" in lines[1]


def _split_markdown_table(table_block: str, max_chars: int) -> list[str]:
    """Split an oversized Markdown table by rows while repeating header + separator."""
    lines = [ln.strip() for ln in table_block.split("\n") if ln.strip()]
    if len(lines) <= 2 or len(table_block) <= max_chars:
        return [table_block]

    header_block = "\n".join(lines[:2])
    chunks: list[str] = []
    current_lines = [lines[0], lines[1]]
    current_len = len(header_block)

    for row in lines[2:]:
        candidate_len = current_len + 1 + len(row)
        if candidate_len > max_chars and len(current_lines) > 2:
            chunks.append("\n".join(current_lines))
            current_lines = [lines[0], lines[1], row]
            current_len = len(header_block) + 1 + len(row)
        else:
            current_lines.append(row)
            current_len = candidate_len

    if len(current_lines) > 2:
        chunks.append("\n".join(current_lines))
    return chunks or [table_block]


def _split_prose_by_sentences(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph along sentence or word boundaries without cutting words."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)
    parts: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if current:
                parts.append(current)
                current = ""
            words = sent.split()
            word_buf = ""
            for word in words:
                candidate = f"{word_buf} {word}".strip() if word_buf else word
                if len(candidate) <= max_chars:
                    word_buf = candidate
                else:
                    if word_buf:
                        parts.append(word_buf)
                    word_buf = word[:max_chars] if len(word) > max_chars else word
            if word_buf:
                parts.append(word_buf)
            continue

        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sent

    if current:
        parts.append(current)
    return parts


def _split_block_units(text: str, max_chars: int) -> list[str]:
    """Break section text into atomic semantic blocks (paragraphs, list items, tables)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []

    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
            continue

        if _is_markdown_table(para):
            units.extend(_split_markdown_table(para, max_chars))
            continue

        lines = para.split("\n")
        if len(lines) > 1 and any(re.match(r"^(?:[-*]|\d+\.)\s+", ln.strip()) for ln in lines):
            for line in lines:
                line_s = line.strip()
                if not line_s:
                    continue
                if len(line_s) <= max_chars:
                    units.append(line_s)
                else:
                    units.extend(_split_prose_by_sentences(line_s, max_chars))
        else:
            units.extend(_split_prose_by_sentences(para, max_chars))

    return units


def _tail_overlap(text: str, overlap_chars: int) -> str:
    """Take a clean word-aligned tail snippet up to overlap_chars."""
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text
    snippet = text[-overlap_chars:]
    space_idx = snippet.find(" ")
    if 0 < space_idx < len(snippet) - 1:
        snippet = snippet[space_idx + 1 :]
    return snippet.strip()


def _split_section_text(section: SectionRecord, max_chars: int, cfg: IngestionConfig) -> list[str]:
    """Split a section into balanced body chunks <= max_chars with optional overlap."""
    text = section.text.strip()
    if len(text) <= max_chars:
        return [text]

    units = _split_block_units(text, max_chars)
    raw_groups: list[list[str]] = []
    current_group: list[str] = []
    current_len = 0

    for unit in units:
        sep_len = 2 if current_group else 0
        if current_len + sep_len + len(unit) <= max_chars:
            current_group.append(unit)
            current_len += sep_len + len(unit)
        else:
            if current_group:
                raw_groups.append(current_group)
            current_group = [unit]
            current_len = len(unit)

    if current_group:
        raw_groups.append(current_group)

    # Merge a tiny trailing group into the previous one when it fits cleanly
    if (
        len(raw_groups) >= 2
        and len("\n\n".join(raw_groups[-1])) < cfg.min_chunk_chars
        and len("\n\n".join(raw_groups[-2])) + 2 + len("\n\n".join(raw_groups[-1])) <= max_chars
    ):
        raw_groups[-2].extend(raw_groups[-1])
        raw_groups.pop()

    base_parts = ["\n\n".join(group) for group in raw_groups]
    if cfg.chunk_overlap_chars <= 0 or len(base_parts) <= 1:
        return base_parts

    overlapped: list[str] = [base_parts[0]]
    for idx in range(1, len(base_parts)):
        body = base_parts[idx]
        available_for_overlap = max_chars - len(body) - 2
        if available_for_overlap >= 40:
            prev_tail = _tail_overlap(base_parts[idx - 1], min(cfg.chunk_overlap_chars, available_for_overlap))
            if prev_tail:
                overlapped.append(f"{prev_tail}\n\n{body}")
                continue
        overlapped.append(body)
    return overlapped


def _make_chunk_record(
    *,
    chunk_id: str,
    doc: DocumentRecord,
    chunk_type: str,
    section_heading: str,
    part_index: int,
    total_parts: int,
    body_text: str,
    header: str,
) -> ChunkRecord:
    """Instantiate a validated ChunkRecord from components."""
    search_text = f"{header}\n\n{body_text}"
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        chunk_type=chunk_type,  # type: ignore[arg-type]
        title=doc.title,
        section=section_heading,
        part_index=part_index,
        total_parts=total_parts,
        source_url=doc.source_url,
        source_file=doc.source_file,
        verified_at=doc.verified_at,
        category=doc.metadata.category,
        audience=doc.metadata.audience,
        format_modes=doc.metadata.format_modes,
        has_selection=doc.metadata.has_selection,
        enrollment_status=doc.metadata.enrollment_status,
        text=body_text,
        search_text=search_text,
        char_count=len(body_text),
        search_char_count=len(search_text),
    )


def build_chunks(doc: DocumentRecord, cfg: IngestionConfig) -> list[ChunkRecord]:
    """Generate contextualized overview and section chunks for one active document."""
    chunks: list[ChunkRecord] = []

    if cfg.include_overview_card_chunk and doc.summary_card.strip():
        card_section = (
            "Сводка документа"
            if doc.metadata.category in {"company_overview", "general"}
            else "Сводка программы"
        )
        card_header = _build_contextual_header(doc, card_section)
        chunks.append(
            _make_chunk_record(
                chunk_id=f"{doc.doc_id}__overview_card",
                doc=doc,
                chunk_type="overview_card",
                section_heading=card_section,
                part_index=1,
                total_parts=1,
                body_text=doc.summary_card.strip(),
                header=card_header,
            )
        )

    seen_heading_slugs: dict[str, int] = {}
    for section in doc.sections:
        header = _build_contextual_header(doc, section.heading)
        # Keep search_text well within 512-token multilingual model windows (~1450 chars max)
        max_body_chars = max(250, min(cfg.max_chunk_chars, 1440 - len(header) - 2))
        parts = _split_section_text(section, max_body_chars, cfg)

        base_slug = slugify(section.heading, max_len=36)
        count = seen_heading_slugs.get(base_slug, 0) + 1
        seen_heading_slugs[base_slug] = count
        heading_slug = base_slug if count == 1 else f"{base_slug}_{count}"

        total_parts = len(parts)
        for part_idx, body in enumerate(parts, start=1):
            chunks.append(
                _make_chunk_record(
                    chunk_id=f"{doc.doc_id}__{heading_slug}__p{part_idx:02d}",
                    doc=doc,
                    chunk_type="section",
                    section_heading=section.heading,
                    part_index=part_idx,
                    total_parts=total_parts,
                    body_text=body,
                    header=header,
                )
            )

    return chunks
