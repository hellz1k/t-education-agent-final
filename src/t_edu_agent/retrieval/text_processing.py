"""Domain-aware Russian/English tokenization, stemming, alias expansion, and query hint inference."""

from __future__ import annotations

import re

from t_edu_agent.retrieval.models import SearchFilters

_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "для", "от", "до", "из", "к", "ко",
    "о", "об", "обо", "у", "а", "но", "или", "да", "что", "как", "ли", "же",
    "бы", "то", "это", "этот", "эта", "эти", "при", "за", "под", "над", "про",
    "чем", "чтобы", "если", "ли", "тут", "там", "где", "когда", "какой", "какая",
    "какие", "есть", "ли", "сейчас", "пока", "меня", "мне", "мой", "моя", "ваш",
    "хочу", "могу", "можно", "нужно", "надо", "расскажи", "подскажи", "помоги",
    "the", "and", "for", "in", "on", "of", "to", "a", "an", "is", "are", "or", "with",
}

_SPECIAL_TOKEN_RE = re.compile(
    r"c\+\+|c#|\.net|ci/cd|a/b|а/б|t-id|т-старт\w*|т-академ\w*|т-поколен\w*|"
    r"т-класс\w*|т-образован\w*|т-курс\w*|риск-аналитик\w*|bi-аналитик\w*|"
    r"ит-аналитик\w*|qa-инженер\w*|qa\s+backend|qa\s+mobile|"
    r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*",
    re.IGNORECASE,
)

# Ordered longest-first so compound inflectional endings strip cleanly
_RU_SUFFIXES = (
    "иями", "ями", "ами", "иях", "овать", "евать", "ться",
    "ого", "его", "ому", "ему", "ыми", "ими", "ией", "иям", "ием",
    "ешь", "ишь", "ете", "ите", "ать", "ять", "еть", "ить",
    "ую", "юю", "ая", "яя", "ое", "ее", "ие", "ые", "ой", "ей",
    "ий", "ый", "ом", "ем", "им", "ым", "ов", "ев", "ам", "ям",
    "ах", "ях", "ию", "ья", "ье", "ью", "у", "ю", "а", "я",
    "о", "е", "и", "ы", "ь",
)

_ALIAS_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bпитон\w*", re.IGNORECASE), "python"),
    (re.compile(r"\bджав\w*", re.IGNORECASE), "java"),
    (re.compile(r"\b(?:жс|js)\b", re.IGNORECASE), "javascript"),
    (re.compile(r"тестировщик\w*|\bqa\b", re.IGNORECASE), "qa тестирование"),
    (re.compile(r"дата\s*сайнс|\bds\b|машинн\w*\s+обучен\w*", re.IGNORECASE), "ml анализ данных"),
    (re.compile(r"удаленк\w*|удаленн\w*", re.IGNORECASE), "онлайн удаленно"),
    (re.compile(r"офлайн\w*", re.IGNORECASE), "очно офлайн"),
    (re.compile(r"бэкенд\w*", re.IGNORECASE), "backend бэкенд"),
    (re.compile(r"фронтенд\w*", re.IGNORECASE), "frontend фронтенд"),
    (re.compile(r"\bмшп\b", re.IGNORECASE), "мшп школа программистов"),
    (re.compile(r"\bdwh\b", re.IGNORECASE), "dwh инженер данных хранилище"),
    (re.compile(r"анализ\w*\s+данн\w*", re.IGNORECASE), "анализ данных аналитика data"),
]

_CITY_HINT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Москва", re.compile(r"москв", re.IGNORECASE)),
    ("Санкт-Петербург", re.compile(r"санкт-петербург|петербург|\bспб\b", re.IGNORECASE)),
    ("Екатеринбург", re.compile(r"екатеринбург", re.IGNORECASE)),
    ("Нижний Новгород", re.compile(r"нижн\w*\s+новгород", re.IGNORECASE)),
    ("Казань", re.compile(r"казан", re.IGNORECASE)),
    ("Новосибирск", re.compile(r"новосибирск", re.IGNORECASE)),
]

_NEG_TRIGGER = (
    r"(?:(?:пока\s+)?не\s+зна\w*|без(?:\s+знания)?|нет\s+опыта(?:\s+работы)?|"
    r"не\s+владе\w*|не\s+уме\w*|не\s+изучал\w*)"
)
_NEG_GAP = r"(?:[^,.;!?—–\s]+\s+){0,4}"

_NEG_SKILL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("python", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}(?:python|питон\w*)", re.IGNORECASE)),
    ("sql", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bsql\b", re.IGNORECASE)),
    ("java", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bjava(?!script)\b", re.IGNORECASE)),
    ("javascript", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}(?:javascript|\bjs\b)", re.IGNORECASE)),
    ("ml", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}(?:\bml\b|машинн\w*\s+обучен\w*)", re.IGNORECASE)),
    ("statistics", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}(?:статистик\w*|теори\w*\s+вероятност\w*)", re.IGNORECASE)),
    ("git", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bgit\b", re.IGNORECASE)),
    ("linux", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\blinux\b", re.IGNORECASE)),
    ("kotlin", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bkotlin\b", re.IGNORECASE)),
    ("swift", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bswift\b", re.IGNORECASE)),
    ("cpp", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}c\+\+", re.IGNORECASE)),
    ("go", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bgo\b", re.IGNORECASE)),
    ("algorithms", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}алгоритм\w*", re.IGNORECASE)),
    ("bi", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}(?:\bbi\b|tableau|power\s*bi|superset)", re.IGNORECASE)),
    ("excel", re.compile(rf"{_NEG_TRIGGER}\s+{_NEG_GAP}\bexcel\b", re.IGNORECASE)),
]


def _stem_russian_word(word: str) -> str:
    """Apply lightweight Russian suffix stripping while preserving short stems."""
    if len(word) < 5 or not any("а" <= ch <= "я" for ch in word):
        return word
    for suffix in _RU_SUFFIXES:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if len(stem) >= 3:
                return stem
    return word


def tokenize(text: str) -> list[str]:
    """Tokenize Russian/English text preserving tech symbols and stemmed Russian words."""
    normalized = text.lower().replace("ё", "е")
    tokens: list[str] = []

    for match in _SPECIAL_TOKEN_RE.finditer(normalized):
        raw = match.group(0).strip()
        if not raw or raw in _STOPWORDS:
            continue

        if raw in {"c++", "c#", ".net", "ci/cd", "a/b", "а/б", "t-id"}:
            tokens.append(raw)
            continue

        if " " in raw:
            subparts = raw.split()
            tokens.append("_".join(_stem_russian_word(p) for p in subparts))
            for sub in subparts:
                if sub not in _STOPWORDS:
                    tokens.append(_stem_russian_word(sub))
            continue

        if "-" in raw:
            parts = [p for p in raw.split("-") if p]
            stemmed_parts = [_stem_russian_word(p) for p in parts]
            tokens.append("-".join(stemmed_parts))
            for part, stemmed in zip(parts, stemmed_parts):
                if len(part) >= 2 and part not in _STOPWORDS:
                    tokens.append(stemmed)
            continue

        tokens.append(_stem_russian_word(raw))

    return tokens


def expand_query_aliases(query: str) -> str:
    """Append transparent domain aliases to a user query for lexical recall."""
    normalized = query.replace("ё", "е")
    extras: list[str] = []
    query_l = normalized.lower()

    for pattern, expansion in _ALIAS_RULES:
        if pattern.search(normalized):
            for word in expansion.split():
                if word.lower() not in query_l and word not in extras:
                    extras.append(word)

    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


def infer_query_hints(query: str) -> tuple[SearchFilters, list[str]]:
    """Extract soft non-destructive metadata and section hints from a natural query."""
    q_l = query.lower().replace("ё", "е")
    hints = SearchFilters()
    reasons: list[str] = []

    preferred_sections: list[str] = []
    if (
        any(
            kw in q_l
            for kw in ("стоимост", "цен", "бесплатн", "платн", "оплат", "набор", "открыт", "закрыт", "дат", "дедлайн", "когда", "поток", "срок")
        )
        or re.search(r"\bстоит\b", q_l)
    ):
        preferred_sections.append("Даты и стоимость")
    if any(kw in q_l for kw in ("отбор", "экзамен", "контест", "этап", "поступ", "без отбора", "проходн", "тестирован")):
        preferred_sections.append("Порядок отбора")
    if any(kw in q_l for kw in ("требован", "нужно знать", "навык", "без опыта", "не знаю", "опыт", "знать ")):
        preferred_sections.append("Требования")
    if any(kw in q_l for kw in ("формат", "онлайн", "очно", "офлайн", "нагрузк", "часов", "расписан", "длительн", "сколько длится", "недел")):
        preferred_sections.append("Формат обучения")
    if any(kw in q_l for kw in ("подать заявку", "записат", "регистрац", "куда писать", "контакт")):
        preferred_sections.append("Как подать заявку")
    if any(kw in q_l for kw in ("для кого", "класс", "курс", "возраст", "гражданств", "студент", "школьник")):
        preferred_sections.append("Для кого")
    if (
        any(kw in q_l for kw in ("учат", "изуча", "темы", "содержан", "проект", "что такое", "дает", "основ", "различ", "сравни", "отлича"))
        or re.search(r"\bпрограмм[аыуе]?\b", q_l)
    ):
        preferred_sections.append("Описание")
    if any(kw in q_l for kw in ("сертификат", "совмещ", "одновременно", "нескольких направ")):
        preferred_sections.append("Дополнительные условия")
    if preferred_sections:
        hints.preferred_sections = preferred_sections
        reasons.append(f"sections:{','.join(preferred_sections)}")

    categories: list[str] = []
    if "олимпиад" in q_l or re.search(r"\b(?:prod|dano)\b", q_l):
        categories.append("olympiad")
    if "стажировк" in q_l or "т-старт" in q_l:
        categories.append("internship")
    if "т-академ" in q_l or "академ" in q_l:
        categories.append("academy")
    if "интенсив" in q_l:
        categories.append("intensive")
    if any(kw in q_l for kw in ("школьник", "класс", "егэ", "огэ", "мшп", "т-поколен", "билет в лето", "число т")):
        categories.append("school_course")
    if any(kw in q_l for kw in ("ано дпо", "о компании", "в целом", "когда основан", "сколько сотрудников")):
        categories.append("company_overview")
    if categories:
        hints.categories = categories
        reasons.append(f"categories:{','.join(categories)}")

    format_modes: list[str] = []
    if any(kw in q_l for kw in ("онлайн", "удален")):
        format_modes.append("онлайн")
    if re.search(r"\bочно\b|\bочный\b|\bочная\b|\bочное\b|офлайн|выездн", q_l):
        format_modes.append("очно")
    if format_modes:
        hints.format_modes = format_modes
        reasons.append(f"format:{','.join(format_modes)}")

    if any(kw in q_l for kw in ("очно", "офлайн", "офис", "город", "в москв", "в петербург", "в спб", "в екатеринбург", "в казан", "в нижн", "в новосибирск")):
        for city, pattern in _CITY_HINT_PATTERNS:
            if pattern.search(q_l):
                hints.offline_city = city
                reasons.append(f"city:{city}")
                break

    if "без отбора" in q_l or "без вступительных" in q_l:
        hints.has_selection = False
        reasons.append("has_selection:False")

    if "бесплатн" in q_l:
        hints.is_free = True
        reasons.append("is_free:True")
    elif re.search(r"\bплатн|\bстоит\b|\bстоимость\b|\bцена\b", q_l) and "бесплатн" not in q_l:
        # Note: asking "сколько стоит" should boost cost sections, not exclude free courses;
        # only hint is_free=False when user explicitly asks for paid courses ("платные курсы")
        if re.search(r"\bплатн\w*\s+(?:курс|школ|программ)", q_l):
            hints.is_free = False
            reasons.append("is_free:False")

    audience: list[str] = []
    if any(kw in q_l for kw in ("школьник", "ученик", "старшекласс", "егэ", "огэ", "всош")) or re.search(
        r"\b\d{1,2}[-\s]?(?:й|м|ом|ем)?\s+класс", q_l
    ):
        audience.append("школьники")
    if any(
        kw in q_l
        for kw in (
            "студент",
            "первокурс",
            "второкурс",
            "третьекурс",
            "четверокурс",
            "старшекурс",
            "бакалавр",
            "магистр",
            "учусь в вуз",
        )
    ) or re.search(
        r"(?:на|я)\s+(?:\d+[-\s]?(?:м|ом|ем)|перв\w+|втор\w+|треть\w+|четверт\w+|старш\w+)\s+курс",
        q_l,
    ):
        audience.append("студенты")
    if any(kw in q_l for kw in ("выпускник", "окончил вуз", "закончил вуз")):
        audience.append("выпускники")
    if any(kw in q_l for kw in ("начинающ", "джун", "junior")):
        audience.append("начинающие специалисты")
    if any(kw in q_l for kw in ("учител", "преподавател", "педагог")):
        audience.append("учителя")
    if audience:
        hints.audience = audience
        reasons.append(f"audience:{','.join(audience)}")

    excluded_skills: list[str] = []
    for skill, pattern in _NEG_SKILL_PATTERNS:
        if pattern.search(q_l):
            excluded_skills.append(skill)
    if excluded_skills:
        hints.exclude_required_skills = excluded_skills
        reasons.append(f"exclude_skills:{','.join(excluded_skills)}")

    return hints, reasons
