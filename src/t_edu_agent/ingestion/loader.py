"""Document loaders and noise cleaners for Markdown, text, DOCX, and HTML files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import html as lxml_html
from lxml.etree import Element


class UnsupportedFormatError(ValueError):
    """Raised when a file extension is not supported by the ingestion pipeline."""


class EmptyDocumentError(ValueError):
    """Raised when a file contains no extractable text after cleaning."""


@dataclass(frozen=True)
class LoadedRawDocument:
    """Normalized text extracted from a raw file together with provenance metadata."""

    source_file: str
    abs_path: Path
    file_format: str
    raw_text: str
    mtime_iso: str


_BOILERPLATE_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "iframe", "form"}
_BOILERPLATE_ATTR_RE = re.compile(
    r"\b(nav|navbar|menu|footer|header|cookie|consent|breadcrumb|sidebar|banner|popup|modal|social|share)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize line endings, spaces, and excessive blank lines."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [line.rstrip() for line in cleaned.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _format_markdown_table(rows: list[list[str]]) -> str:
    """Convert a 2D cell grid into a clean Markdown pipe table."""
    non_empty_rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not non_empty_rows:
        return ""

    col_count = max(len(row) for row in non_empty_rows)
    padded = [
        [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in row] + [""] * (col_count - len(row))
        for row in non_empty_rows
    ]
    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join([header, separator, *body])


def _load_text_file(path: Path) -> str:
    """Load UTF-8 Markdown or plain text file."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251")


def _is_docx_list_paragraph(paragraph: Paragraph) -> bool:
    """Detect whether a DOCX paragraph is formatted as a bullet or numbered list item."""
    style_name = (paragraph.style.name if paragraph.style and paragraph.style.name else "").lower()
    if "list" in style_name or "bullet" in style_name:
        return True
    p_pr = paragraph._p.pPr
    return p_pr is not None and p_pr.numPr is not None


def _load_docx_file(path: Path) -> str:
    """Load DOCX preserving headings, lists, paragraphs, and Markdown tables."""
    doc = DocxDocument(str(path))
    blocks: list[str] = []

    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            para = Paragraph(child, doc)
            text = re.sub(r"\s+", " ", para.text).strip()
            if not text:
                continue
            style_name = (para.style.name if para.style and para.style.name else "").lower()
            if style_name in {"title", "heading 1"} or style_name.startswith("heading 1"):
                blocks.append(f"# {text}")
            elif style_name.startswith("heading 2"):
                blocks.append(f"## {text}")
            elif style_name.startswith("heading 3"):
                blocks.append(f"### {text}")
            elif style_name.startswith("heading 4"):
                blocks.append(f"#### {text}")
            elif _is_docx_list_paragraph(para):
                blocks.append(f"- {text}")
            else:
                blocks.append(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            md_table = _format_markdown_table(rows)
            if md_table:
                blocks.append(md_table)

    return "\n\n".join(blocks)


def _strip_html_boilerplate(root: Element) -> None:
    """Remove navigation, scripts, styles, cookie banners, and footers in-place."""
    to_remove: list[Element] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        if tag in _BOILERPLATE_TAGS:
            to_remove.append(el)
            continue
        if tag in {"html", "body", "main", "article"}:
            continue
        attr_blob = " ".join(
            str(el.attrib.get(attr, "")) for attr in ("role", "class", "id", "aria-label")
        )
        if attr_blob and _BOILERPLATE_ATTR_RE.search(attr_blob):
            to_remove.append(el)

    for el in to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)


def _clean_inline_text(el: Element) -> str:
    """Collapse internal whitespace inside an HTML element."""
    return re.sub(r"\s+", " ", el.text_content() or "").strip()


def _html_node_to_blocks(el: Element, blocks: list[str]) -> None:
    """Recursively convert cleaned HTML DOM nodes into Markdown blocks."""
    if not isinstance(el.tag, str):
        return
    tag = el.tag.lower()

    if tag in {"h1", "h2", "h3", "h4"}:
        text = _clean_inline_text(el)
        if text:
            level = int(tag[1])
            blocks.append(f"{'#' * level} {text}")
        return

    if tag in {"ul", "ol"}:
        items: list[str] = []
        for li in el.findall("./li"):
            text = _clean_inline_text(li)
            if text:
                items.append(f"- {text}")
        if items:
            blocks.append("\n".join(items))
        return

    if tag == "table":
        rows: list[list[str]] = []
        for tr in el.xpath(".//tr"):
            cells = [_clean_inline_text(cell) for cell in tr.xpath("./th|./td")]
            if cells:
                rows.append(cells)
        md_table = _format_markdown_table(rows)
        if md_table:
            blocks.append(md_table)
        return

    if tag in {"p", "blockquote", "pre"}:
        text = _clean_inline_text(el)
        if text:
            blocks.append(text)
        return

    direct_text = re.sub(r"\s+", " ", el.text or "").strip()
    if direct_text and len(el) == 0:
        blocks.append(direct_text)
        return

    for child in el:
        _html_node_to_blocks(child, blocks)
        tail = re.sub(r"\s+", " ", child.tail or "").strip()
        if tail:
            blocks.append(tail)


def _load_html_file(path: Path) -> str:
    """Parse HTML, drop navigation/chrome, and emit clean Markdown blocks."""
    raw_text = _load_text_file(path)
    parser = lxml_html.HTMLParser(encoding="utf-8")
    doc = lxml_html.fromstring(raw_text.encode("utf-8"), parser=parser)
    _strip_html_boilerplate(doc)

    content_roots = doc.xpath("//main | //article")
    target = content_roots[0] if content_roots else (doc.find(".//body") or doc)

    blocks: list[str] = []
    _html_node_to_blocks(target, blocks)
    return "\n\n".join(blocks)


def load_raw_document(
    path: Path,
    raw_dir: Path,
    supported_extensions: list[str],
) -> LoadedRawDocument:
    """Load and clean a single raw document from disk."""
    ext = path.suffix.lower()
    allowed = {item.lower() for item in supported_extensions}
    if ext not in allowed:
        raise UnsupportedFormatError(
            f"Unsupported file format '{ext or '(none)'}'. Supported: {sorted(allowed)}"
        )

    if ext in {".md", ".txt"}:
        text = _load_text_file(path)
    elif ext == ".docx":
        text = _load_docx_file(path)
    elif ext in {".html", ".htm"}:
        text = _load_html_file(path)
    else:
        raise UnsupportedFormatError(f"No loader registered for extension '{ext}'")

    normalized = normalize_text(text)
    if not normalized:
        raise EmptyDocumentError("Document contains no extractable text after cleaning")

    rel_path = path.relative_to(raw_dir).as_posix()
    mtime_iso = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return LoadedRawDocument(
        source_file=rel_path,
        abs_path=path,
        file_format=ext.lstrip("."),
        raw_text=normalized,
        mtime_iso=mtime_iso,
    )
