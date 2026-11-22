"""Defense-in-depth prompt injection guard, delimiter escaping, and output scrubbing."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from t_edu_agent.agent.state import SessionState


class SecurityAssessment(BaseModel):
    """Structured outcome of pre-LLM input inspection."""

    action: Literal["allow", "block"] = "allow"
    category: str | None = None
    flags: list[str] = Field(default_factory=list)
    safe_refusal: str | None = None
    is_suspicious_probe: bool = False


_REFUSAL_BY_CATEGORY: dict[str, str] = {
    "direct_role_override": (
        "Я не могу менять системную роль, отменять правила или выполнять команды "
        "переопределения инструкций. Я консультант по образовательным программам Т-Образования "
        "и готов помочь с подбором курса, требованиями, форматом или сравнением программ."
    ),
    "prompt_or_secret_exfiltration": (
        "Я не могу раскрывать системный промпт, внутренние инструкции или переменные окружения "
        "и секретные ключи. Если вас интересуют курсы Т-Образования (в том числе по Python или "
        "безопасности), задайте вопрос по программам — с радостью помогу."
    ),
    "cross_session_exfiltration": (
        "Доступ к диалогам, профилям и данным других пользователей или сессий закрыт — "
        "каждая сессия изолирована. Я могу работать только с вопросами в рамках вашего текущего диалога."
    ),
    "forced_official_guarantee": (
        "Я не могу выдавать гарантии зачисления, трудоустройства или указывать статус «Набор открыт» "
        "без подтверждения в источниках. Могу подсказать только фактические условия и статус из проверенной базы."
    ),
    "multi_turn_escalation": (
        "Я не могу применять ранее предложенные тестовые или обходные правила для изменения "
        "моих системных ограничений. Готов продолжить консультацию по программам Т-Образования "
        "в стандартном режиме."
    ),
}

# Delimiters used in internal prompt framing and chat-template control markers
_DELIMITER_TAG_RE = re.compile(
    r"<\s*/?\s*(?:retrieved_sources|source|user_query|user_profile|system|instructions?)\b[^>]*>"
    r"|<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)

# Direct override / persona hijack patterns (imperative intent, not isolated topic keywords)
_DIRECT_OVERRIDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:игнорируй|проигнорируй|забудь|отмени|отбрось|сбрось|обойди|нарушь)\b"
        r"[^.!?\n]{0,70}\b(?:инструкц\w*|правил\w*|ограничени\w*|промпт\w*|запрет\w*|указани\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b"
        r"[^.!?\n]{0,70}\b(?:previous|prior|above|all|system|original)\s+"
        r"(?:instructions?|rules?|prompts?|constraints?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:смени|поменяй|измени)\s+(?:свою\s+)?роль\b"
        r"|\b(?:ты\s+теперь|теперь\s+ты|притворись|стань|войди\s+в\s+режим|включи\s+режим)\s+"
        r"(?:админ\w*|разработчик\w*|developer|dan\b|jailbreak|пират\w*|хаке\w*|без\s+ограничен\w*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now|switch\s+to|enable|enter|act\s+as)\s+"
        r"(?:an?\s+)?(?:admin|developer\s+mode|dan\b|jailbreak|unrestricted)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:\[?\s*(?:system|developer)\s*\]?\s*:|системн(?:ая|ый)\s+(?:инструкция|приказ|режим)\s*:)",
        re.IGNORECASE,
    ),
]

# Bounded single-line quote spans in Russian («…», „…“) and English (“…”, "…", '…')
_QUOTED_SPAN_RE = re.compile(
    r"«[^»\n]{3,140}»|„[^“\n]{3,140}“|“[^”\n]{3,140}”|\"[^\"\n]{3,140}\"|(?<!\w)'[^'\n]{3,140}'(?!\w)"
)

_EDUCATIONAL_QUOTE_FRAME_RE = re.compile(
    r"\b(?:изучаем|разбираем|проходим|обсуждаем|объясни|расскажи|что\s+(?:означает|значит|такое)"
    r"|почему\s+(?:это|эта|такая|команда|фраза|при[её]м)|зачем\s+(?:от\s+не[её]|защища\w*)"
    r"|как\s+(?:работает|защища\w*|защитить\w*)|считается\s+(?:атак\w*|уязвимост\w*|инъекц\w*)"
    r"|what\s+does|why\s+(?:is|are)|explain\s+why|considered\s+(?:an?\s+)?(?:attack|injection|vulnerability))\b",
    re.IGNORECASE,
)

_QUOTE_EXECUTION_DIRECTIVE_RE = re.compile(
    r"\b(?:выполн\w*|сделай|примен\w*|активируй|исполн\w*|повтор\w*|следов\w*|следуй|подчин\w*|запуст\w*|проделай|действуй|работай|отвечай|ответь)\b"
    r"[^.!?\n]{0,35}\b(?:это|его|е[её]|их|команд\w*|инструкц\w*|при[её]м\w*|правил\w*|пример\w*|так|выше|без\s+правил|без\s+ограничен\w*)\b"
    r"|\b(?:продемонстрируй|проверь|попробуй|покажи)\b[^.!?\n]{0,30}\b(?:на\s+себе|на\s+деле|в\s+ответе)\b"
    r"|\b(?:execute|run|apply|follow|obey|perform|activate|carry\s+out|do)\b[^.!?\n]{0,25}\b(?:it|that|this|these|those|command|instruction|above|now)\b"
    r"|\bбез\s+(?:правил|ограничений)\b",
    re.IGNORECASE,
)


def _is_benign_quoted_override_discussion(text: str) -> bool:
    """Allow purely analytical discussion of a quoted override example with no execution intent."""
    if not _QUOTED_SPAN_RE.search(text):
        return False
    outer_text = _QUOTED_SPAN_RE.sub(" ", text)
    if len(outer_text.strip()) < 15:
        return False
    if any(pat.search(outer_text) for pat in _DIRECT_OVERRIDE_PATTERNS):
        return False
    if _QUOTE_EXECUTION_DIRECTIVE_RE.search(outer_text):
        return False
    return bool(_EDUCATIONAL_QUOTE_FRAME_RE.search(outer_text))

# Exfiltration of system prompt, hidden instructions, API keys, or .env
_SECRET_EXFIL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:покажи|выведи|раскрой|напечатай|процитируй|повтори|перескажи|скинь|отправь|отдай|распечатай|слей|дамп\w*"
        r"|show|reveal|print|repeat|dump|leak|output|display|expose)\b"
        r"[^.!?\n]{0,75}"
        r"(?:(?:сво[йюеи]|внутренн\w+|скрыт\w+|тво[йюеи])\s+(?:системн\w+\s+)?(?:промпт\w*|инструкц\w*|правил\w*)"
        r"|системн\w+\s+промпт\w*"
        r"|system\s+prompt|hidden\s+instructions?|developer\s+prompt"
        r"|groq_api_key|api[_\s-]?key|переменн\w+\s+окружения|секретн\w+\s+ключ\w*"
        r"|содержим\w+\s+(?:файла\s+)?\.?env\b|\.?env\s+файл)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\bgroq_api_key\b|\b\.env\b)[^.!?\n]{0,45}\b(?:значение|содержимое|выведи|покажи|раскрой|dump|print|show|value)\b"
        r"|\b(?:значение|содержимое|выведи|покажи|раскрой|dump|print|show)\b[^.!?\n]{0,45}\b(?:groq_api_key|\.env)\b",
        re.IGNORECASE,
    ),
]

# Cross-session / other user memory leakage attempts
_CROSS_SESSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:чуж(?:ие|ую|их|ой)|других?\s+(?:пользовател\w*|сесси\w*|участник\w*|студент\w*)"
        r"|предыдущ\w+\s+пользовател\w*|соседн\w+\s+сесси\w*|other\s+users?|all\s+sessions?)\b"
        r"[^.!?\n]{0,65}\b(?:сесси\w*|диалог\w*|истори\w*|сообщени\w*|профил\w*|данн\w*|chat|history|data|context)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:истори\w*|данн\w*|сообщени\w*|диалог\w*|профил\w*|сесси\w*)\b"
        r"[^.!?\n]{0,55}\b(?:других?\s+(?:пользовател\w*|сесси\w*)|чужих\s+(?:сесси\w*|пользовател\w*)"
        r"|предыдущ\w+\s+пользовател\w*|прошл\w+\s+пользовател\w*|соседн\w+\s+сесси\w*"
        r"|other\s+users?|another\s+session)\b",
        re.IGNORECASE,
    ),
]

# Coerced false admission / job guarantee claims
_FORCED_GUARANTEE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:напиши|скажи|подтверди|гарантируй|официально\s+(?:подтверди|заяви)|пообещай|выдай)\b"
        r"[^.!?\n]{0,80}\b(?:гарантированно\s+зачислен|я\s+зачислен|зачислен\w*\s+(?:на\s+\w+\s+)?без\s+отбора"
        r"|без\s+экзамен\w*\s+принят|гарантированное?\s+трудоустройств\w*|оффер\s+гарантирован|принят\s+в\s+штат)\b",
        re.IGNORECASE,
    ),
]

_INTERROGATIVE_OPEN_RE = re.compile(
    r"\bоткрыт\w*\s+ли\b|\bли\b[^.!?\n]{0,30}\bоткрыт\w*"
    r"|\b(?:какие|какой|какая|какую|где|когда)\b[^.!?\n]{0,45}\b(?:набор\w*|открыт\w*)",
    re.IGNORECASE,
)

_FORCED_OPEN_STATUS_RE = re.compile(
    r"\b(?:пиши|напиши|укажи|пометь|подставь|замени|считай|отвечай)\b"
    r"(?:[^.!?\n]{0,60}?\b(?:что|вместо|на|как|статус)\b|[,\s:«\"'-]{1,4})"
    r"[^.!?\n]{0,40}?\b(?:набор\w*\s+открыт\w*|регистраци\w+\s+открыт\w*|при[её]м\s+(?:заявок\s+)?открыт\w*|доступн\w+\s+сейчас)\b"
    r"|\b(?:скажи|подтверди|ответь)\b[^.!?\n]{0,55}\b(?:вместо|даже\s+если|статус|что)\b[^.!?\n]{0,40}\b(?:набор\w*\s+открыт\w*|регистраци\w+\s+открыт\w*)\b",
    re.IGNORECASE,
)


def _is_forced_status_fabrication(text: str) -> bool:
    """Detect imperative commands asking the agent to fabricate an 'open enrollment' status."""
    if _INTERROGATIVE_OPEN_RE.search(text):
        return False
    lowered = text.lower()
    if "?" in text or " или " in f" {lowered} ":
        if not any(tok in lowered for tok in ("вместо", "замени", "подставь", "даже если")):
            return False
    return bool(_FORCED_OPEN_STATUS_RE.search(text))

# Multi-turn framing probes (Turn 1 setup) and follow-up activation (Turn 2+)
_SETUP_PROBE_RE = re.compile(
    r"\b(?:давай\s+поиграем\s+в\s+игру|режим\s+отладки|тестов\w+\s+песочниц\w*"
    r"|запомни\s+(?:новое\s+)?правило|представь\s*,?\s*что\s+(?:нет\s+ограничен\w*|все\s+правила\s+отменены)"
    r"|remember\s+(?:a\s+)?new\s+rule|debug\s+mode|pretend\s+there\s+are\s+no\s+rules)\b",
    re.IGNORECASE,
)

_MULTI_TURN_EXEC_RE = re.compile(
    r"\b(?:выполни\s+(?:новое\s+)?правило|правил\w*\s+из\s+прошл\w+|как\s+мы\s+договорились"
    r"|в\s+этом\s+режиме|теперь\s+без\s+(?:правил|ограничений)|следуй\s+(?:новой\s+)?инструкц\w*\s+выше"
    r"|теперь\s+(?:ты\s+)?как\s+(?:админ|разработчик))\b",
    re.IGNORECASE,
)

# Indirect prompt injections embedded inside retrieved document text
_HTML_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_INDIRECT_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions?"
        r"|игнорируй\s+(?:все\s+)?(?:предыдущие|прошлые|системные)?\s*инструкци\w*"
        r"|забудь\s+(?:все\s+)?(?:правила|инструкци\w*)"
        r"|system\s+override\s*:|new\s+system\s+instruction\s*:|нова(?:я|й)\s+системн\w+\s+инструкци\w*\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:раскрой|выведи|покажи|reveal|print|output|dump)\s+"
        r"(?:свой\s+)?(?:системн\w+\s+промпт\w*|system\s+prompt|groq_api_key|api[_\s-]?key|\.env)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:напиши|сообщи|скажи|tell)\s+(?:пользовател\w+|the\s+user)\s*,?\s*"
        r"(?:что\s+)?(?:он|she|he|they)\s+(?:автоматически\s+|гарантированно\s+)?(?:зачислен|accepted|admitted)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:SYSTEM|INSTRUCTION|ADMIN|OVERRIDE|ПЕРЕОПРЕДЕЛЕНИЕ)\s*:\s*",
        re.IGNORECASE,
    ),
]

# Secret-like tokens and verbatim internal prompt leakage markers in outputs/logs
_SECRET_TOKEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bGROQ_API_KEY\s*[:=]\s*[^\s\"']+", re.IGNORECASE), "GROQ_API_KEY=[REDACTED_SECRET]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b", re.IGNORECASE), "Bearer [REDACTED_SECRET]"),
]

_PROMPT_LEAK_MARKERS: tuple[tuple[str, str], ...] = (
    ("marker_rules_header", "СТРОГИЕ ПРАВИЛА БЕЗОПАСНОСТИ И ДОСТОВЕРНОСТИ"),
    ("marker_planner_role", "Ты модуль анализа запроса для ИИ-консультанта"),
    ("marker_retrieved_sources_tag", "<retrieved_sources>"),
)


def escape_delimiter_tags(text: str) -> str:
    """Neutralize internal XML/control delimiters in untrusted text so it cannot break prompt structure."""
    if not text:
        return ""
    return _DELIMITER_TAG_RE.sub(lambda m: "[" + m.group(0)[1:-1] + "]", text)


def scrub_secrets(text: str) -> str:
    """Mask API key or bearer token shapes from text before emitting logs or model output."""
    cleaned = text
    for pattern, replacement in _SECRET_TOKEN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def assess_user_input(user_message: str, session: SessionState) -> SecurityAssessment:
    """Inspect user input for direct attacks, boundary spoofing, or multi-turn escalations."""
    text = user_message.strip()
    if not text:
        return SecurityAssessment(action="allow")

    flags: list[str] = []
    primary_category: str | None = None

    if _DELIMITER_TAG_RE.search(text):
        flags.append("xml_boundary_spoof")
        primary_category = primary_category or "direct_role_override"

    for pat in _DIRECT_OVERRIDE_PATTERNS:
        if pat.search(text):
            flags.append("direct_role_override")
            primary_category = primary_category or "direct_role_override"
            break

    for pat in _SECRET_EXFIL_PATTERNS:
        if pat.search(text):
            flags.append("prompt_or_secret_exfiltration")
            primary_category = primary_category or "prompt_or_secret_exfiltration"
            break

    for pat in _CROSS_SESSION_PATTERNS:
        if pat.search(text):
            flags.append("cross_session_exfiltration")
            primary_category = primary_category or "cross_session_exfiltration"
            break

    for pat in _FORCED_GUARANTEE_PATTERNS:
        if pat.search(text):
            flags.append("forced_official_guarantee")
            primary_category = primary_category or "forced_official_guarantee"
            break

    if _is_forced_status_fabrication(text):
        flags.append("forced_open_status")
        primary_category = primary_category or "forced_official_guarantee"

    is_probe = bool(_SETUP_PROBE_RE.search(text))
    prior_probe_in_history = any(_SETUP_PROBE_RE.search(turn.content) for turn in session.history)
    prior_strikes = getattr(session, "security_strikes", 0) > 0 or prior_probe_in_history

    if _MULTI_TURN_EXEC_RE.search(text) and prior_strikes:
        flags.append("multi_turn_escalation")
        primary_category = primary_category or "multi_turn_escalation"
    elif is_probe and prior_strikes and any(
        tok in text.lower()
        for tok in ("систем", "промпт", "инструкц", "правил", "ключ", "сесси", "зачислен", "обой")
    ):
        flags.append("multi_turn_escalation")
        primary_category = primary_category or "multi_turn_escalation"

    dedup_flags = list(dict.fromkeys(flags))
    if (
        dedup_flags == ["direct_role_override"]
        and not is_probe
        and not prior_strikes
        and _is_benign_quoted_override_discussion(text)
    ):
        return SecurityAssessment(action="allow")

    if primary_category is not None:
        return SecurityAssessment(
            action="block",
            category=primary_category,
            flags=dedup_flags,
            safe_refusal=_REFUSAL_BY_CATEGORY[primary_category],
            is_suspicious_probe=is_probe,
        )

    return SecurityAssessment(
        action="allow",
        flags=["suspicious_framing_probe"] if is_probe else [],
        is_suspicious_probe=is_probe,
    )


def sanitize_untrusted_chunk(text: str) -> tuple[str, list[str]]:
    """Strip indirect prompt-injection commands inside RAG chunks while keeping program facts."""
    flags: list[str] = []

    def _replace_comment(match: re.Match[str]) -> str:
        body = match.group(1)
        if any(p.search(body) for p in _INDIRECT_LINE_PATTERNS) or _DELIMITER_TAG_RE.search(body):
            if "indirect_comment_injection" not in flags:
                flags.append("indirect_comment_injection")
            return "[удалён подозрительный комментарий]"
        return match.group(0)

    cleaned = _HTML_COMMENT_RE.sub(_replace_comment, text)

    if _DELIMITER_TAG_RE.search(cleaned):
        flags.append("xml_delimiter_escape")
        cleaned = escape_delimiter_tags(cleaned)

    safe_lines: list[str] = []
    line_stripped = False
    for line in cleaned.splitlines():
        if any(p.search(line) for p in _INDIRECT_LINE_PATTERNS):
            safe_lines.append("[удалена подозрительная инструкция из внешнего документа]")
            line_stripped = True
        else:
            safe_lines.append(line)

    if line_stripped:
        flags.append("indirect_instruction_stripped")

    return "\n".join(safe_lines), list(dict.fromkeys(flags))


def inspect_and_sanitize_output(text: str) -> tuple[str, list[str]]:
    """Verify generated model output did not leak internal prompts or API secret shapes."""
    flags: list[str] = []
    matched_marker_ids = [
        marker_id for marker_id, marker in _PROMPT_LEAK_MARKERS if marker in text
    ]
    if matched_marker_ids:
        return (
            _REFUSAL_BY_CATEGORY["prompt_or_secret_exfiltration"],
            [
                "output_prompt_leak_blocked",
                *(f"output_prompt_leak_marker:{mid}" for mid in matched_marker_ids),
            ],
        )

    scrubbed = scrub_secrets(text)
    if scrubbed != text:
        flags.append("output_secret_redacted")

    return scrubbed, flags
