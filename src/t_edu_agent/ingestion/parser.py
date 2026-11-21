"""Deterministic Markdown parser and metadata extractor for T-Education KB."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from t_edu_agent.ingestion.loader import LoadedRawDocument
from t_edu_agent.schemas import (
    DocumentRecord,
    EnrollmentStatus,
    ProgramCategory,
    ProgramMetadata,
    SectionRecord,
)

_TRANSLIT_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

_URL_RE = re.compile(r"https?://[^\s<>\")]+")
_RU_DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

_CATEGORY_LABELS_RU: dict[str, str] = {
    "intensive": "Интенсив",
    "school_course": "Курс для школьников",
    "academy": "Т-Академия",
    "course": "Курс",
    "internship": "Стажировка",
    "olympiad": "Олимпиада",
    "announcement": "Анонс",
    "company_overview": "Справка о компании",
    "general": "Общий документ",
}

_STATUS_LABELS_RU: dict[str, str] = {
    "open": "набор открыт",
    "closed": "набор закрыт",
    "rolling": "без фиксированных дат (доступ после регистрации)",
    "unknown": "требует уточнения на странице",
}

_SKILL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("python", re.compile(r"\bpython\b|питон", re.IGNORECASE)),
    ("sql", re.compile(r"\bsql\b|select-запрос|селект", re.IGNORECASE)),
    ("ml", re.compile(r"\bml\b|машинн\w+\s+обучени", re.IGNORECASE)),
    ("statistics", re.compile(r"статистик|теори\w+\s+вероятност|\bцпт\b|(?:основн\w*|нормальн\w*|вероятностн\w*)\s+распределени", re.IGNORECASE)),
    ("ab_testing", re.compile(r"a/[bб]-тест|проверк\w+\s+гипотез", re.IGNORECASE)),
    ("bi", re.compile(r"\bbi\b|superset|tableau|power\s*bi|дашборд", re.IGNORECASE)),
    ("proteus", re.compile(r"\bproteus\b", re.IGNORECASE)),
    ("excel", re.compile(r"\bexcel\b", re.IGNORECASE)),
    ("javascript", re.compile(r"\bjavascript\b|\bjs\b", re.IGNORECASE)),
    ("typescript", re.compile(r"\btypescript\b", re.IGNORECASE)),
    ("java", re.compile(r"\bjava\b(?!script)", re.IGNORECASE)),
    ("qa", re.compile(r"\bqa\b|тестировщик|тест-дизайн|(?:основ\w*|теори\w*|автоматизац\w*)\s+тестирован|тестирован\w*\s+(?:по\b|бэкенд|мобильн|приложен|программ|кода|и\s+отладк)", re.IGNORECASE)),
    ("system_analysis", re.compile(r"системн\w*\s+анали|bpmn|uml|archimate", re.IGNORECASE)),
    ("data_engineering", re.compile(r"дата-инженер|\betl\b|\belt\b|\bdwh\b|data\s+lake", re.IGNORECASE)),
    ("pandas", re.compile(r"\bpandas\b", re.IGNORECASE)),
    ("numpy", re.compile(r"\bnumpy\b", re.IGNORECASE)),
    ("greenplum", re.compile(r"\bgreenplum\b", re.IGNORECASE)),
    ("kafka", re.compile(r"\bkafka\b", re.IGNORECASE)),
    ("airflow", re.compile(r"\bairflow\b", re.IGNORECASE)),
    ("algorithms", re.compile(r"алгоритм", re.IGNORECASE)),
    ("git", re.compile(r"\bgit\b|ci/cd", re.IGNORECASE)),
    ("docker", re.compile(r"\bdocker\b", re.IGNORECASE)),
    ("kotlin", re.compile(r"\bkotlin\b", re.IGNORECASE)),
    ("swift", re.compile(r"\bswift\b", re.IGNORECASE)),
    ("cpp", re.compile(r"\bc\+\+", re.IGNORECASE)),
    ("go", re.compile(r"\bgo\b", re.IGNORECASE)),
    ("linux", re.compile(r"\blinux\b", re.IGNORECASE)),
    ("networks", re.compile(r"компьютерн\w*\s+сет", re.IGNORECASE)),
]

_NEGATED_SKILL_RE = re.compile(
    r"не\s*обязательн|не\s+требуется|не\s+нужно|не\s+нужен|не\s+обязательно|будет\s+плюсом|будут?\s+преимуществом|желательно|пригод\w*",
    re.IGNORECASE,
)

_OFFLINE_CITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Москва", re.compile(r"москв|московск", re.IGNORECASE)),
    ("Санкт-Петербург", re.compile(r"санкт-петербург|петербург", re.IGNORECASE)),
    ("Екатеринбург", re.compile(r"екатеринбург", re.IGNORECASE)),
    ("Новосибирск", re.compile(r"новосибирск", re.IGNORECASE)),
    ("Казань", re.compile(r"казан", re.IGNORECASE)),
    ("Нижний Новгород", re.compile(r"нижн\w*\s+новгород", re.IGNORECASE)),
]

_PRIORITY_CITIES: list[tuple[str, re.Pattern[str]]] = [
    ("Москва", re.compile(r"москв", re.IGNORECASE)),
    ("Санкт-Петербург", re.compile(r"санкт-петербург", re.IGNORECASE)),
    ("Ростов-на-Дону", re.compile(r"ростов-на-дону", re.IGNORECASE)),
    ("Нижний Новгород", re.compile(r"нижн\w*\s+новгород", re.IGNORECASE)),
    ("Краснодар", re.compile(r"краснодар", re.IGNORECASE)),
    ("Воронеж", re.compile(r"воронеж", re.IGNORECASE)),
    ("Самара", re.compile(r"самар", re.IGNORECASE)),
    ("Казань", re.compile(r"казан", re.IGNORECASE)),
    ("Екатеринбург", re.compile(r"екатеринбург", re.IGNORECASE)),
    ("Новосибирск", re.compile(r"новосибирск", re.IGNORECASE)),
    ("Томск", re.compile(r"томск", re.IGNORECASE)),
    ("Омск", re.compile(r"\bомск", re.IGNORECASE)),
    ("Пермь", re.compile(r"перм", re.IGNORECASE)),
    ("Челябинск", re.compile(r"челябинск", re.IGNORECASE)),
    ("Красноярск", re.compile(r"красноярск", re.IGNORECASE)),
    ("Рязань", re.compile(r"рязан", re.IGNORECASE)),
    ("Сочи", re.compile(r"\bсочи\b", re.IGNORECASE)),
    ("Таганрог", re.compile(r"таганрог", re.IGNORECASE)),
    ("Иннополис", re.compile(r"иннополис", re.IGNORECASE)),
    ("Уфа", re.compile(r"\bуф[аеуы]\b", re.IGNORECASE)),
    ("Ижевск", re.compile(r"ижевск", re.IGNORECASE)),
    ("Саратов", re.compile(r"саратов", re.IGNORECASE)),
]

_AUDIENCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("школьники", re.compile(r"школьник|ученик\w*\s+\d+[—–-]\d+", re.IGNORECASE)),
    ("студенты", re.compile(r"студент", re.IGNORECASE)),
    ("выпускники", re.compile(r"выпускник", re.IGNORECASE)),
    ("начинающие специалисты", re.compile(r"начинающ\w*\s+специалист|\bджун", re.IGNORECASE)),
    ("учителя", re.compile(r"учител|педагог", re.IGNORECASE)),
    ("все желающие", re.compile(r"всех?\s+желающих", re.IGNORECASE)),
]


def slugify(text: str, max_len: int = 48) -> str:
    """Convert Russian or Latin text into an ASCII snake_case identifier."""
    out: list[str] = []
    for ch in text.lower():
        if ch in _TRANSLIT_MAP:
            out.append(_TRANSLIT_MAP[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = re.sub(r"_+", "_", "".join(out)).strip("_")
    return slug[:max_len].strip("_") or "item"


def category_label_ru(category: str) -> str:
    """Return human-readable Russian label for a program category."""
    return _CATEGORY_LABELS_RU.get(category, category)


def status_label_ru(status: str) -> str:
    """Return human-readable Russian label for an enrollment status."""
    return _STATUS_LABELS_RU.get(status, status)


def _normalize_date(raw_value: str) -> str | None:
    """Convert DD.MM.YYYY or YYYY-MM-DD text into ISO YYYY-MM-DD."""
    ru_match = _RU_DATE_RE.search(raw_value)
    if ru_match:
        day, month, year = ru_match.groups()
        return f"{year}-{month}-{day}"
    iso_match = _ISO_DATE_RE.search(raw_value)
    if iso_match:
        return iso_match.group(1)
    return None


def _clean_url(url: str) -> str:
    """Strip trailing punctuation attached to extracted URLs."""
    return url.rstrip(".,;:)]>»")


def _extract_urls(text: str) -> list[str]:
    """Return unique URLs in order of appearance."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.findall(text):
        cleaned = _clean_url(match)
        if cleaned not in seen:
            seen.add(cleaned)
            urls.append(cleaned)
    return urls


def _split_title_header_sections(raw_text: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Split Markdown text into title, pre-section header lines, and ## sections."""
    lines = raw_text.split("\n")
    title = ""
    title_idx = -1

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            title = stripped[2:].strip()
            title_idx = idx
            break

    start_idx = title_idx + 1 if title_idx >= 0 else 0
    if not title:
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                title = stripped[:120]
                start_idx = idx + 1
                break

    header_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines[start_idx:]:
        match = re.match(r"^##\s+(.+)$", line.strip())
        if match:
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = match.group(1).strip()
            current_lines = []
        elif current_heading is None:
            header_lines.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    if not sections:
        body = "\n".join(header_lines).strip()
        if body:
            sections.append(("Основное содержание", body))

    return title, header_lines, sections


def _extract_header_provenance(
    header_lines: list[str],
    full_text: str,
) -> tuple[str | None, list[str], str | None, str | None, str | None]:
    """Extract primary URL, additional URLs, verification date, publication date, and version."""
    source_url: str | None = None
    additional_urls: list[str] = []
    verified_at: str | None = None
    published_at: str | None = None
    doc_version: str | None = None

    expect_additional_url = False
    for line in header_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if expect_additional_url:
            urls = _extract_urls(stripped)
            if urls:
                additional_urls.extend(urls)
                expect_additional_url = False
                continue
            expect_additional_url = False

        lower = stripped.lower()
        if lower.startswith("источник:"):
            urls = _extract_urls(stripped)
            if urls and source_url is None:
                source_url = urls[0]
        elif lower.startswith("дополнительный источник") or lower.startswith("официальный сайт"):
            urls = _extract_urls(stripped)
            if urls:
                additional_urls.extend(urls)
            elif lower.startswith("дополнительный источник"):
                expect_additional_url = True
        elif lower.startswith("проверено:"):
            verified_at = _normalize_date(stripped)
        elif lower.startswith("дата публикации:"):
            raw_pub = stripped.split(":", 1)[1].strip()
            published_at = _normalize_date(raw_pub) or (
                None if "отсутствует" in raw_pub.lower() or "не указан" in raw_pub.lower() else raw_pub
            )
        elif lower.startswith("версия:"):
            raw_ver = stripped.split(":", 1)[1].strip()
            if raw_ver and "отсутствует" not in raw_ver.lower() and "не указан" not in raw_ver.lower():
                doc_version = raw_ver

    if source_url is None:
        site_match = re.search(r"(?:^|\n)Сайт:\s*(https?://\S+)", full_text)
        if site_match:
            source_url = _clean_url(site_match.group(1))

    dedup_additional: list[str] = []
    seen = {source_url} if source_url else set()
    for url in additional_urls:
        if url not in seen:
            seen.add(url)
            dedup_additional.append(url)

    return source_url, dedup_additional, verified_at, published_at, doc_version


def _find_section(sections: list[SectionRecord], keyword: str) -> SectionRecord | None:
    """Find the first section whose heading contains the given keyword."""
    kw = keyword.lower()
    for sec in sections:
        if kw in sec.heading.lower():
            return sec
    return None


def _detect_category(
    title: str,
    source_file: str,
    source_url: str | None,
    header_text: str,
) -> ProgramCategory:
    """Classify document category deterministically from filename, URL, and title."""
    title_l = title.lower()
    url_l = (source_url or "").lower()
    file_l = Path(source_file).name.lower()
    header_l = header_text.lower()

    if file_l.startswith("company_overview") or "информация о компании" in title_l or "справка партнёра" in header_l:
        return "company_overview"
    if "олимпиад" in title_l:
        return "olympiad"
    if "/school/" in url_l or "для школьников" in title_l:
        return "school_course"
    if "/intensive/" in url_l or "интенсив" in title_l:
        return "intensive"
    if "/academy/" in url_l or "академи" in title_l:
        return "academy"
    if "/start/" in url_l or "стажировк" in title_l or "т-старт" in title_l:
        return "internship"
    if "t.me/" in url_l:
        return "announcement"
    if "/study/" in url_l or "курс" in title_l:
        return "course"
    return "general"


def _extract_audience(title: str, sections: list[SectionRecord], category: ProgramCategory) -> list[str]:
    """Extract target audience tags."""
    if category == "company_overview":
        parts = [
            s.text for s in sections if any(k in s.heading.lower() for k in ("чем занимается", "основные форматы"))
        ]
        target_text = "\n".join(parts)
    else:
        for_whom = _find_section(sections, "для кого")
        desc = _find_section(sections, "описание")
        target_text = "\n".join([title, for_whom.text if for_whom else "", desc.text if desc else ""])

    found: list[str] = []
    for label, pattern in _AUDIENCE_PATTERNS:
        if pattern.search(target_text) and label not in found:
            found.append(label)
    return found


def _extract_format_and_cities(
    sections: list[SectionRecord],
    full_text: str,
    category: ProgramCategory,
) -> tuple[list[str], list[str], list[str]]:
    """Extract format_modes, offline_cities, and priority_cities without negation false positives."""
    if category == "company_overview":
        return [], [], []
    fmt_sec = _find_section(sections, "формат")
    fmt_text = fmt_sec.text if fmt_sec else full_text
    if fmt_sec and "формат проведения" in fmt_text.lower() and "не указан" in fmt_text.lower():
        return [], [], []

    lines = [ln.strip() for ln in fmt_text.split("\n") if ln.strip()]
    has_online = False
    has_offline = False
    explicit_no_offline = False
    offline_cities: list[str] = []

    for line in lines:
        line_l = line.lower()
        is_general_rule = "общее правило" in line_l
        if "онлайн" in line_l or "удалённ" in line_l or "удаленн" in line_l:
            has_online = True
        if "офлайн-площадки нет" in line_l or "без офлайн-площадки" in line_l or "только онлайн" in line_l:
            explicit_no_offline = True
            continue
        if is_general_rule:
            continue
        if re.search(
            r"\bочно\b|\bофлайн\b|очное\s+участие|в\s+одном\s+из\s+офисов|выездные\s+летние\s+школы",
            line_l,
        ):
            if not re.search(r"(?:нет|без|отсутствует)\s+офлайн", line_l):
                has_offline = True
                for city, pattern in _OFFLINE_CITY_PATTERNS:
                    if pattern.search(line) and city not in offline_cities:
                        offline_cities.append(city)

    if explicit_no_offline and not offline_cities:
        has_offline = False

    format_modes: list[str] = []
    if has_online:
        format_modes.append("онлайн")
    if has_offline:
        format_modes.append("очно")

    # Priority cities come from geographic preference sentences in "Для кого"
    priority_cities: list[str] = []
    for_whom = _find_section(sections, "для кого")
    if for_whom:
        geo_lines = [
            ln for ln in for_whom.text.split("\n")
            if "географи" in ln.lower() or "приоритет" in ln.lower()
        ]
        geo_blob = "\n".join(geo_lines)
        for city, pattern in _PRIORITY_CITIES:
            if pattern.search(geo_blob) and city not in priority_cities:
                priority_cities.append(city)

    return format_modes, offline_cities, priority_cities


def _extract_selection(sections: list[SectionRecord]) -> bool | None:
    """Determine whether competitive selection/exam is required."""
    sel_sec = _find_section(sections, "порядок отбора") or _find_section(sections, "отбор")
    if not sel_sec:
        return None
    text_l = sel_sec.text.lower()
    if "отбора нет" in text_l or "без отбора" in text_l or "конкурсных испытаний" in text_l and "не заявлено" in text_l:
        return False
    if any(kw in text_l for kw in ("контест", "вступительные испытания", "экзамен", "тест", "отбор")):
        return True
    return None


def _extract_cost_and_enrollment(
    sections: list[SectionRecord],
) -> tuple[bool | None, str | None, EnrollmentStatus, str | None]:
    """Extract cost and enrollment status, respecting explicit missing/uncertain notes."""
    dates_sec = _find_section(sections, "даты и стоимость")
    sel_sec = _find_section(sections, "порядок отбора") or _find_section(sections, "отбор")
    if not dates_sec:
        return None, None, "unknown", None

    lines = [ln.strip().lstrip("- ").strip() for ln in dates_sec.text.split("\n") if ln.strip()]

    cost_line = next(
        (
            ln for ln in lines
            if ln.lower().startswith("стоимость")
            or "обучение платное" in ln.lower()
            or "стажировка оплачиваемая" in ln.lower()
            or "оплачиваемая работа" in ln.lower()
            or "стоимость" in ln.lower()
        ),
        None,
    )
    is_free: bool | None = None
    cost_note: str | None = None
    if cost_line:
        cost_note = cost_line.split(":", 1)[1].strip() if ":" in cost_line else cost_line
        cost_l = cost_line.lower()
        if (
            "обучение платное" in cost_l
            or "оплатить выбранный курс" in cost_l
            or "оплата обучения" in cost_l
        ):
            is_free = False
        elif "стажировка оплачиваемая" in cost_l or "оплачиваемая работа" in cost_l:
            is_free = None
        elif cost_l.startswith("стоимость: бесплатно") or cost_l.startswith("стоимость: участие бесплатное"):
            is_free = True
        elif any(
            phrase in cost_l
            for phrase in (
                "на странице программы не указана",
                "на странице курса не указана",
                "стоимость сотрудничества и отдельных программ на странице не указана",
                "стоимость и статус нужно уточнять",
                "требует отдельной проверки",
                "отдельного подтверждения стоимости нет",
                "стоимость участия на странице т-образования не указаны",
                "стоимость участия на странице не указана",
            )
        ):
            is_free = None
        elif "информация отсутствует" in cost_l and "подтверждение" not in cost_l and "общее правило" not in cost_l:
            is_free = None
        elif "бесплатн" in cost_l or "оплачивает путёвку" in cost_l:
            is_free = True
        elif "платн" in cost_l:
            is_free = False
    elif "бесплатн" in dates_sec.text.lower():
        is_free = True

    extra_status_lines: list[str] = []
    if sel_sec:
        for ln in sel_sec.text.split("\n"):
            cleaned = ln.strip().lstrip("- ").strip()
            cleaned_l = cleaned.lower()
            if any(
                kw in cleaned_l
                for kw in ("статус набора", "требует ручной проверки", "набор на программу закрыт")
            ):
                extra_status_lines.append(cleaned)

    candidate_lines = lines + extra_status_lines
    status_line = next(
        (
            ln for ln in candidate_lines
            if ln.lower().startswith("статус")
            or any(
                kw in ln.lower()
                for kw in (
                    "фиксированных дат набора",
                    "дат набора",
                    "статус набора",
                    "требует ручной проверки",
                    "не позволяют однозначно определить",
                    "сроки 2026 года уже прошли",
                    "сроки qa mobile уже прошли",
                    "перечисленные сроки уже прошли",
                    "регистрация закрыта",
                    "открытых городов на странице не показано",
                    "набор закрыт",
                    "набор на программу закрыт",
                    "набор открыт",
                    "старт в любое время",
                    "доступ к базе открывается",
                    "доступ открывается",
                )
            )
        ),
        None,
    )
    enrollment_note = (
        status_line.split(":", 1)[1].strip() if (status_line and ":" in status_line) else status_line
    )
    status_target = "\n".join([dates_sec.text, *extra_status_lines]).lower()

    if any(
        phrase in status_target
        for phrase in (
            "требует ручной проверки",
            "статус набора противоречив",
            "не позволяют однозначно определить актуальный набор",
            "не помечена как «набор закрыт»",
            "статус нужно уточнять отдельно",
            "статус набора на странице не обозначен",
            "дат набора и дедлайнов на общей странице нет",
            "дат набора, дедлайнов и расписания потоков на странице нет",
        )
    ):
        enrollment_status: EnrollmentStatus = "unknown"
    elif any(
        phrase in status_target
        for phrase in (
            "набор закрыт",
            "набор на программу закрыт",
            "регистрация закрыта",
            "открытых городов на странице не показано",
            "сроки 2026 года уже прошли",
            "сроки qa mobile уже прошли",
            "перечисленные сроки уже прошли",
            "сроки уже прошли",
        )
    ):
        enrollment_status = "closed"
    elif any(
        phrase in status_target
        for phrase in (
            "фиксированных дат набора нет",
            "дат набора и дедлайнов нет",
            "старт в любое время",
            "начинается сразу после регистрации",
            "начинается после регистрации",
            "доступ открывается после",
            "доступ к базе открывается после",
        )
    ):
        enrollment_status = "rolling"
    elif "набор открыт" in status_target:
        enrollment_status = "open"
    else:
        enrollment_status = "unknown"

    return is_free, cost_note, enrollment_status, enrollment_note


def _extract_workload_and_duration(sections: list[SectionRecord]) -> tuple[str | None, str | None]:
    """Extract human-readable workload and duration notes from format/requirements sections."""
    fmt_sec = _find_section(sections, "формат")
    req_sec = _find_section(sections, "требован")

    workload_note: str | None = None
    duration_note: str | None = None

    if fmt_sec:
        for raw_line in fmt_sec.text.split("\n"):
            line = raw_line.strip().lstrip("- ").strip()
            line_l = line.lower()
            workload_match = re.match(r"^нагрузка\s*[:—–-]\s*(.+)$", line, re.IGNORECASE)
            duration_match = re.match(r"^длительность\s*[:—–-]\s*(.+)$", line, re.IGNORECASE)
            if workload_match and workload_note is None:
                workload_note = workload_match.group(1).strip()
            elif duration_match and duration_note is None:
                duration_note = duration_match.group(1).strip()
            elif "час" in line_l and "в неделю" in line_l and workload_note is None:
                match = re.search(
                    r"(?:от|до)?\s*\d+(?:[—–-]\d+)?\s+час\w*\s+в\s+неделю",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    workload_note = match.group(0).strip()

    if workload_note is None and req_sec:
        match = re.search(
            r"(?:от|до)?\s*\d+(?:[—–-]\d+)?\s+час\w*\s+в\s+неделю",
            req_sec.text,
            re.IGNORECASE,
        )
        if match:
            workload_note = match.group(0).strip()

    return workload_note, duration_note


def _extract_skills(
    title: str,
    sections: list[SectionRecord],
) -> tuple[list[str], list[str], list[str]]:
    """Extract mentioned skills, required prerequisites, and explicitly optional skills."""
    desc_sec = _find_section(sections, "описание")
    req_sec = _find_section(sections, "требован")

    mentioned_blob = "\n".join(
        [title, desc_sec.text if desc_sec else "", req_sec.text if req_sec else ""]
    )
    skills_mentioned: list[str] = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(mentioned_blob) and skill not in skills_mentioned:
            skills_mentioned.append(skill)

    required_skills: list[str] = []
    optional_skills: list[str] = []

    if req_sec:
        req_text = req_sec.text
        clauses = re.split(r"[\n;.!?]+", req_text)
        for clause in clauses:
            clause_s = clause.strip()
            if not clause_s:
                continue
            clause_l = clause_s.lower()
            if any(
                skip_kw in clause_l
                for skip_kw in (
                    "вступительных требований и предварительной подготовки",
                    "предварительный опыт",
                    "специальная подготовка не заявлена",
                    "можно учиться без опыта",
                    "получите на курсе",
                    "начинается с основ",
                    "не описаны — информация отсутствует",
                    "требования не раскрыты",
                    "требований к предварительной подготовке",
                    "отдельного требования к предварительной подготовке",
                )
            ):
                continue
            is_optional = bool(_NEGATED_SKILL_RE.search(clause_l))
            for skill, pattern in _SKILL_PATTERNS:
                if pattern.search(clause_s):
                    if is_optional:
                        if skill not in optional_skills:
                            optional_skills.append(skill)
                    elif skill not in required_skills:
                        required_skills.append(skill)

    required_skills = [s for s in required_skills if s not in optional_skills]
    for skill in optional_skills:
        if skill not in skills_mentioned:
            skills_mentioned.append(skill)

    return skills_mentioned, required_skills, optional_skills


def _build_summary_card(
    title: str,
    source_url: str | None,
    verified_at: str | None,
    metadata: ProgramMetadata,
    sections: list[SectionRecord],
) -> str:
    """Build a compact factual summary card without inventing unverified claims."""
    cat_ru = category_label_ru(metadata.category)
    lines: list[str] = [f"{title} — {cat_ru}."]

    if metadata.category == "company_overview":
        if metadata.audience:
            lines.append(f"Аудитория: {', '.join(metadata.audience)}.")
        desc = _find_section(sections, "чем занимается") or (sections[0] if sections else None)
        if desc:
            first_para = desc.text.split("\n\n")[0].strip()
            lines.append(first_para[:280])
    else:
        if metadata.audience:
            lines.append(f"Аудитория: {', '.join(metadata.audience)}.")

        fmt_parts: list[str] = []
        if metadata.format_modes:
            fmt_str = ", ".join(metadata.format_modes)
            if metadata.offline_cities:
                fmt_str += f" (очно: {', '.join(metadata.offline_cities)})"
            fmt_parts.append(f"Формат: {fmt_str}")
        if metadata.duration_note:
            fmt_parts.append(f"Длительность: {metadata.duration_note.rstrip('.')}")
        if metadata.workload_note:
            fmt_parts.append(f"Нагрузка: {metadata.workload_note.rstrip('.')}")
        if fmt_parts:
            lines.append(" | ".join(fmt_parts) + ".")

        cond_parts: list[str] = []
        if metadata.has_selection is True:
            cond_parts.append("Отбор: есть (заявка/анкета/экзамен)")
        elif metadata.has_selection is False:
            cond_parts.append("Отбор: не требуется")
        if metadata.is_free is True:
            cond_parts.append("Стоимость: бесплатно")
        elif metadata.is_free is None and metadata.cost_note:
            cond_parts.append("Стоимость: требует уточнения")
        cond_parts.append(f"Статус набора: {status_label_ru(metadata.enrollment_status)}")
        lines.append(" | ".join(cond_parts) + ".")

        if metadata.required_skills:
            lines.append(f"Требования по навыкам: {', '.join(metadata.required_skills)}.")
        if metadata.optional_or_not_required_skills:
            lines.append(
                f"Необязательные навыки: {', '.join(metadata.optional_or_not_required_skills)}."
            )

        desc = _find_section(sections, "описание")
        if desc:
            first_para = desc.text.split("\n\n")[0].strip()
            if len(first_para) > 260:
                first_para = first_para[:257].rsplit(" ", 1)[0] + "..."
            lines.append(f"Суть программы: {first_para}")

    if source_url:
        ver_suffix = f" (проверено: {verified_at})" if verified_at else ""
        lines.append(f"Источник: {source_url}{ver_suffix}")

    return "\n".join(lines)


def parse_document(loaded: LoadedRawDocument) -> DocumentRecord:
    """Parse a loaded document into structured sections, metadata, and provenance."""
    title, header_lines, raw_sections = _split_title_header_sections(loaded.raw_text)
    header_text = "\n".join(header_lines)
    source_url, additional_urls, verified_at, published_at, doc_version = _extract_header_provenance(
        header_lines,
        loaded.raw_text,
    )

    sections: list[SectionRecord] = []
    for idx, (heading, body) in enumerate(raw_sections, start=1):
        clean_body = body.strip()
        if not clean_body:
            continue
        lower_b = clean_body.lower()
        sections.append(
            SectionRecord(
                section_id=f"sec_{idx:02d}_{slugify(heading, max_len=32)}",
                heading=heading,
                level=2,
                text=clean_body,
                char_count=len(clean_body),
                has_program_rule=("условие программы" in lower_b),
                has_general_rule=("общее правило" in lower_b or "общее требование" in lower_b),
                has_missing_info_note=(
                    "информация отсутствует" in lower_b
                    or "не указан" in lower_b
                    or "не описан" in lower_b
                ),
            )
        )

    category = _detect_category(title, loaded.source_file, source_url, header_text)
    audience = _extract_audience(title, sections, category)
    format_modes, offline_cities, priority_cities = _extract_format_and_cities(
        sections, loaded.raw_text, category
    )
    has_selection = _extract_selection(sections)
    is_free, cost_note, enrollment_status, enrollment_note = _extract_cost_and_enrollment(sections)
    workload_note, duration_note = _extract_workload_and_duration(sections)
    skills_mentioned, required_skills, optional_skills = _extract_skills(title, sections)

    metadata = ProgramMetadata(
        category=category,
        audience=audience,
        format_modes=format_modes,
        offline_cities=offline_cities,
        priority_cities=priority_cities,
        has_selection=has_selection,
        is_free=is_free,
        cost_note=cost_note,
        enrollment_status=enrollment_status,
        enrollment_note=enrollment_note,
        workload_note=workload_note,
        duration_note=duration_note,
        skills_mentioned=skills_mentioned,
        required_skills=required_skills,
        optional_or_not_required_skills=optional_skills,
    )

    summary_card = _build_summary_card(title, source_url, verified_at, metadata, sections)
    content_hash = hashlib.sha256(loaded.raw_text.encode("utf-8")).hexdigest()
    doc_id = f"{slugify(Path(loaded.source_file).stem, max_len=40)}_{content_hash[:8]}"

    return DocumentRecord(
        doc_id=doc_id,
        title=title or Path(loaded.source_file).stem,
        source_file=loaded.source_file,
        file_format=loaded.file_format,
        source_url=source_url,
        additional_urls=additional_urls,
        verified_at=verified_at,
        published_at=published_at,
        doc_version=doc_version,
        content_hash=content_hash,
        char_count=sum(sec.char_count for sec in sections),
        metadata=metadata,
        summary_card=summary_card,
        sections=sections,
    )
