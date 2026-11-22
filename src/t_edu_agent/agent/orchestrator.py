"""Stateful RAG agent orchestrating query rewriting, hybrid retrieval, and grounded Groq answers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from t_edu_agent.agent.llm_client import BaseLLMClient, GroqLLMClient
from t_edu_agent.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_grounded_user_payload,
    build_planner_user_message,
    format_retrieved_context,
)
from t_edu_agent.agent.state import (
    AgentResponse,
    ChatTurn,
    RetrievalPlan,
    SessionState,
    SessionStore,
    SourceRef,
    UserProfile,
)
from t_edu_agent.config import AppConfig, load_config
from t_edu_agent.retrieval.hybrid import HybridRetriever, _detect_entity_matches
from t_edu_agent.retrieval.models import RetrievalResult, SearchFilters
from t_edu_agent.retrieval.text_processing import infer_query_hints
from t_edu_agent.security import (
    DiagnosticLogger,
    assess_user_input,
    inspect_and_sanitize_output,
)

_STATUS_TO_AUDIENCE: dict[str, list[str]] = {
    "студент": ["студенты"],
    "школьник": ["школьники"],
    "выпускник": ["выпускники"],
    "начинающий специалист": ["начинающие специалисты"],
    "учитель": ["учителя"],
}

_SELF_CHECK_RE = re.compile(
    r"\b(?:проверь|перепроверь|проанализируй|сверь)\b[^.!?\n]{0,45}"
    r"\b(?:предыдущ\w+|прошл\w+|последн\w+|сво[йюе])\s+ответ"
    r"|\bсамопроверк\w*|\bпроверь\s+себя\b"
    r"|\bв\s+(?:прошлом|предыдущем)\s+ответе\b[^.!?\n]{0,45}\b(?:проверь|ошибк|источник|верно|подтвержд)",
    re.IGNORECASE,
)

_RECIPIENT_ROLE_TOKEN = (
    r"(?:себя|меня|сестр\w*|брат\w*|сын\w*|доч\w*|реб[её]нк\w*"
    r"|подруг\w*|друг(?:а|у|ом|е)?|школьник\w*|студент\w*|ученик\w*)"
)
_RECIPIENT_CONTRAST_RE = re.compile(
    rf"\bне\s+для\s+(?P<from>{_RECIPIENT_ROLE_TOKEN})\b[,;\s]+(?:а\s+)?для\s+(?P<to>{_RECIPIENT_ROLE_TOKEN})\b"
    rf"|\bвместо\s+(?P<from_alt>{_RECIPIENT_ROLE_TOKEN})\b[^.!?\n]{{0,40}}\bдля\s+(?P<to_alt>{_RECIPIENT_ROLE_TOKEN})\b",
    re.IGNORECASE,
)
_OTHER_RECIPIENT_RE = re.compile(
    r"\bдля\s+друг(?:ого|ой)\s+(?:человек\w*|реб[её]нк\w*|школьник\w*|студент\w*)\b",
    re.IGNORECASE,
)


def _canonical_recipient_role(token: str) -> str:
    """Normalize a recipient noun stem so identical left/right roles do not trigger a switch."""
    t = token.lower().replace("ё", "е")
    if t in {"себя", "меня"}:
        return "self"
    for prefix, canonical in (
        ("сестр", "sister"),
        ("брат", "brother"),
        ("сын", "son"),
        ("доч", "daughter"),
        ("ребенк", "child"),
        ("подруг", "girlfriend"),
        ("друг", "friend"),
        ("школьник", "schooler"),
        ("студент", "student"),
        ("ученик", "pupil"),
    ):
        if t.startswith(prefix):
            return canonical
    return t


def _is_explicit_recipient_switch(user_message: str) -> bool:
    """Detect an unambiguous switch to a different target recipient (e.g. 'не для сестры, а для брата')."""
    for match in _RECIPIENT_CONTRAST_RE.finditer(user_message):
        src = match.group("from") or match.group("from_alt")
        dst = match.group("to") or match.group("to_alt")
        if src and dst and _canonical_recipient_role(src) != _canonical_recipient_role(dst):
            return True
    for match in _OTHER_RECIPIENT_RE.finditer(user_message):
        if not re.search(r"\bне\s+$", user_message[: match.start()], re.IGNORECASE):
            return True
    return False


def _extract_referenced_docs(
    sources: list[SourceRef],
    answer_text: str,
) -> list[tuple[str, str]]:
    """Order unique source documents putting programs actually cited in the answer first."""
    unique_docs: list[tuple[str, str]] = []
    seen_doc_ids: set[str] = set()
    cited_doc_ids: set[str] = set()
    answer_lower = answer_text.lower()

    for src in sources:
        if src.doc_id not in seen_doc_ids:
            seen_doc_ids.add(src.doc_id)
            unique_docs.append((src.doc_id, src.title))
        core_title = re.sub(r"\s*\([^)]*\)\s*", "", src.title).strip().lower()
        if (
            f"[{src.source_id}]" in answer_text
            or f"({src.source_id})" in answer_text
            or (src.source_url and src.source_url in answer_text)
            or (len(core_title) >= 6 and core_title in answer_lower)
        ):
            cited_doc_ids.add(src.doc_id)

    mentioned = [(did, title) for did, title in unique_docs if did in cited_doc_ids]
    others = [(did, title) for did, title in unique_docs if did not in cited_doc_ids]
    return mentioned + others


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Strip markdown fences and parse the outermost JSON object from LLM output."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Planner output is not a JSON object")
    return parsed


def _status_to_audience(status: str | None) -> list[str] | None:
    """Map canonical profile status to document audience tags."""
    if not status:
        return None
    status_l = status.strip().lower()
    for key, audience_tags in _STATUS_TO_AUDIENCE.items():
        if key in status_l:
            return list(audience_tags)
    return None


def _merge_profile(current: UserProfile, updated: UserProfile) -> UserProfile:
    """Merge newly extracted profile fields over persisted session context, preserving corrections."""
    merged = current.model_copy(deep=True)

    for scalar_field in ("status", "grade_or_course", "direction", "format_preference", "city"):
        new_val = getattr(updated, scalar_field)
        if new_val is not None and str(new_val).strip():
            setattr(merged, scalar_field, str(new_val).strip())

    if updated.no_selection_preferred is not None:
        merged.no_selection_preferred = updated.no_selection_preferred

    known = list(dict.fromkeys(merged.known_skills))
    missing = list(dict.fromkeys(merged.missing_skills))

    for skill in updated.missing_skills:
        s_norm = skill.strip().lower()
        if not s_norm:
            continue
        if s_norm not in missing:
            missing.append(s_norm)
        if s_norm in known:
            known.remove(s_norm)

    for skill in updated.known_skills:
        s_norm = skill.strip().lower()
        if not s_norm or s_norm in [m.lower() for m in updated.missing_skills]:
            continue
        if s_norm not in known:
            known.append(s_norm)
        if s_norm in missing:
            missing.remove(s_norm)

    merged.known_skills = known
    merged.missing_skills = missing
    return merged


def _apply_deterministic_hints(plan: RetrievalPlan, user_message: str) -> RetrievalPlan:
    """Reinforce LLM plan with deterministic negated skills and audience hints."""
    inferred_hints, _ = infer_query_hints(user_message)

    if inferred_hints.exclude_required_skills:
        missing = list(dict.fromkeys([s.lower() for s in plan.updated_profile.missing_skills]))
        known = list(dict.fromkeys([s.lower() for s in plan.updated_profile.known_skills]))
        existing_excl = list(dict.fromkeys(plan.search_filters.exclude_required_skills or []))
        for skill in inferred_hints.exclude_required_skills:
            if skill not in missing:
                missing.append(skill)
            if skill in known:
                known.remove(skill)
            if skill not in existing_excl:
                existing_excl.append(skill)
        plan.updated_profile.missing_skills = missing
        plan.updated_profile.known_skills = known
        plan.search_filters.exclude_required_skills = existing_excl

    if inferred_hints.audience and not plan.search_filters.audience:
        plan.search_filters.audience = list(inferred_hints.audience)

    if not plan.updated_profile.status and inferred_hints.audience:
        aud = inferred_hints.audience[0]
        if aud == "студенты":
            plan.updated_profile.status = "студент"
        elif aud == "школьники":
            plan.updated_profile.status = "школьник"
        elif aud == "выпускники":
            plan.updated_profile.status = "выпускник"

    return plan


def _fallback_plan(session: SessionState, user_message: str) -> RetrievalPlan:
    """Deterministic fallback plan if the LLM query rewriter fails or emits invalid JSON."""
    inferred_hints, _ = infer_query_hints(user_message)
    recipient_switch = _is_explicit_recipient_switch(user_message)
    base_profile = UserProfile() if recipient_switch else session.profile

    parts = [user_message]
    if session.history:
        if base_profile.direction and base_profile.direction.lower() not in user_message.lower():
            parts.append(base_profile.direction)
        if base_profile.status and base_profile.status.lower() not in user_message.lower():
            parts.append(base_profile.status)
        if base_profile.known_skills:
            parts.extend(s for s in base_profile.known_skills if s.lower() not in user_message.lower())
        if (
            base_profile.format_preference
            and base_profile.format_preference.lower() in {"онлайн", "очно"}
            and base_profile.format_preference.lower() not in user_message.lower()
        ):
            parts.append(base_profile.format_preference)

    q_l = user_message.lower()
    intent = "recommend"
    if any(kw in q_l for kw in ("сравни", "разниц", "отлича", "лучше выбрать")):
        intent = "compare"
    elif any(kw in q_l for kw in ("запиш", "отбор", "подать заявку", "без отбора")):
        intent = "rules_or_process"

    profile_copy = base_profile.model_copy(deep=True)
    filters = inferred_hints.model_copy(deep=True)
    merged_missing = list(dict.fromkeys([*profile_copy.missing_skills, *(inferred_hints.exclude_required_skills or [])]))
    if merged_missing:
        profile_copy.missing_skills = merged_missing
        profile_copy.known_skills = [s for s in profile_copy.known_skills if s.lower() not in merged_missing]
        filters.exclude_required_skills = merged_missing

    if recipient_switch:
        if inferred_hints.format_modes and not profile_copy.format_preference:
            profile_copy.format_preference = inferred_hints.format_modes[0]
        if inferred_hints.has_selection is False and profile_copy.no_selection_preferred is None:
            profile_copy.no_selection_preferred = True

    if base_profile.format_preference in {"онлайн", "очно"} and not filters.format_modes:
        filters.format_modes = [base_profile.format_preference]
    if base_profile.no_selection_preferred is True and filters.has_selection is None:
        filters.has_selection = False
    if not filters.audience:
        filters.audience = _status_to_audience(profile_copy.status)

    return _apply_deterministic_hints(
        RetrievalPlan(
            standalone_query=" ".join(parts),
            intent=intent,  # type: ignore[arg-type]
            updated_profile=profile_copy,
            search_filters=filters,
        ),
        user_message,
    )


class TEducationAgent:
    """End-to-end stateful RAG agent with query rewriting, session isolation, and Groq generation."""

    def __init__(
        self,
        cfg: AppConfig,
        retriever: HybridRetriever,
        llm_client: BaseLLMClient,
        session_store: SessionStore | None = None,
        diagnostic_logger: DiagnosticLogger | None = None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm_client
        self.sessions = session_store or SessionStore()
        self.diagnostic_logger = diagnostic_logger

    @classmethod
    def from_config(cls, config_path: Path | None = None) -> TEducationAgent:
        """Instantiate retriever, Groq LLM client, and session store from YAML config."""
        cfg = load_config(config_path)
        retriever = HybridRetriever(
            cfg.paths.processed_dir,
            cfg.paths.index_dir,
            cfg.retrieval,
        )
        llm_client = GroqLLMClient(cfg.llm)
        diagnostic_logger = DiagnosticLogger(cfg.logging.diagnostic_log_file)
        return cls(
            cfg=cfg,
            retriever=retriever,
            llm_client=llm_client,
            diagnostic_logger=diagnostic_logger,
        )

    def new_session(self) -> str:
        """Create and return a new isolated session identifier."""
        return self.sessions.create_new_session()

    def reset_session(self, session_id: str) -> None:
        """Reset conversation history and profile for an existing session."""
        self.sessions.reset(session_id)

    def get_session(self, session_id: str) -> SessionState:
        """Inspect current state of a session."""
        return self.sessions.get_or_create(session_id)

    def restore_session(self, state: SessionState) -> SessionState:
        """Restore a saved SessionState snapshot (for example after page reload)."""
        return self.sessions.restore(state)

    def _plan_retrieval(
        self,
        session: SessionState,
        user_message: str,
        warnings: list[str],
    ) -> RetrievalPlan:
        """Rewrite user message into a standalone search query and updated UserProfile."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": build_planner_user_message(session, user_message)},
        ]
        try:
            raw_json, _ = self.llm.complete(
                messages,
                temperature=0.0,
                max_tokens=700,
                json_mode=True,
                preferred_model=self.cfg.llm.planner_model,
            )
            payload = _extract_json_object(raw_json)
            plan = RetrievalPlan.model_validate(payload)
            return _apply_deterministic_hints(plan, user_message)
        except Exception as exc:
            warnings.append(f"planner_fallback:{type(exc).__name__}")
            return _fallback_plan(session, user_message)

    def _retrieve_for_plan(
        self,
        session: SessionState,
        plan: RetrievalPlan,
        user_message: str = "",
    ) -> RetrievalResult:
        """Execute hybrid retrieval, supplementing comparison targets or follow-up doc context."""
        filters = plan.search_filters.model_copy(deep=True)
        user_mentioned_files = _detect_entity_matches(user_message)
        mentioned_files = user_mentioned_files | _detect_entity_matches(
            plan.standalone_query
        )
        # Skip Turn-1 profile penalties when the user explicitly names a program to inspect or compare
        override_turn_constraints = (
            plan.intent in {"compare", "program_details"} and bool(mentioned_files)
        )
        if session.profile.missing_skills and not override_turn_constraints:
            merged_excl = list(
                dict.fromkeys([*(filters.exclude_required_skills or []), *session.profile.missing_skills])
            )
            filters.exclude_required_skills = merged_excl
        if (
            session.profile.format_preference in {"онлайн", "очно"}
            and not filters.format_modes
        ):
            filters.format_modes = [session.profile.format_preference]
        if (
            session.profile.no_selection_preferred is True
            and filters.has_selection is None
            and not override_turn_constraints
        ):
            filters.has_selection = False
        if not filters.audience and not override_turn_constraints:
            filters.audience = _status_to_audience(session.profile.status)

        primary_res = self.retriever.search(plan.standalone_query, filters=filters)
        max_total_chunks = self.cfg.retrieval.top_k + 2

        # When explicitly self-checking the previous answer, keep exact prior cited programs pinned
        if _SELF_CHECK_RE.search(user_message) and session.discussed_doc_ids and not user_mentioned_files:
            pinned_ids = session.discussed_doc_ids[:4]
            self_check_res = self.retriever.search(
                user_message,
                filters=SearchFilters(doc_ids=pinned_ids),
                top_k=max(max_total_chunks, 2 * len(pinned_ids)),
                max_chunks_per_doc=2,
            )
            order_rank = {did: idx for idx, did in enumerate(pinned_ids)}
            progs_sorted = sorted(
                self_check_res.programs,
                key=lambda p: order_rank.get(p.doc.doc_id, len(pinned_ids)),
            )
            interleaved_chunks = []
            seen_chunk_ids = set()
            for rank in range(2):
                for prog in progs_sorted:
                    if rank < len(prog.matched_chunks):
                        chunk = prog.matched_chunks[rank]
                        if chunk.chunk.chunk_id not in seen_chunk_ids:
                            interleaved_chunks.append(chunk)
                            seen_chunk_ids.add(chunk.chunk.chunk_id)
            for chunk in self_check_res.chunks:
                if chunk.chunk.chunk_id not in seen_chunk_ids:
                    interleaved_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk.chunk_id)
            self_check_res.programs = progs_sorted
            self_check_res.chunks = interleaved_chunks[:max_total_chunks]
            return self_check_res

        # Unpenalized retrieval for explicitly named programs so prior filters cannot hide them
        if mentioned_files:
            mentioned_doc_ids = [
                d.doc_id
                for d in self.retriever.documents
                if d.source_file in mentioned_files
            ]
            if mentioned_doc_ids:
                mention_query = user_message.strip() or plan.standalone_query
                mention_top_k = (
                    4
                    if len(mentioned_doc_ids) == 1 and len(user_mentioned_files) == 1
                    else min(6, 2 * len(mentioned_doc_ids))
                )
                mention_res = self.retriever.search(
                    mention_query,
                    filters=SearchFilters(
                        doc_ids=mentioned_doc_ids,
                        preferred_sections=filters.preferred_sections,
                    ),
                    top_k=mention_top_k,
                )
                mention_chunk_ids = {c.chunk.chunk_id for c in mention_res.chunks}
                merged_chunks = list(mention_res.chunks) + [
                    c for c in primary_res.chunks if c.chunk.chunk_id not in mention_chunk_ids
                ]
                programs_by_id = {p.doc.doc_id: p for p in mention_res.programs}
                for p in primary_res.programs:
                    if p.doc.doc_id not in programs_by_id:
                        programs_by_id[p.doc.doc_id] = p
                primary_res.chunks = merged_chunks[:max_total_chunks]
                primary_res.programs = list(programs_by_id.values())[: self.cfg.retrieval.expand_top_programs]

        # For comparisons, prioritize each target program's chunks and Hit FIRST before general primary hits
        if plan.intent == "compare" and len(plan.target_program_queries) >= 2:
            target_chunks = []
            seen_chunk_ids = set()
            target_programs = {}

            for target_q in plan.target_program_queries[:3]:
                sub_res = self.retriever.search(
                    target_q,
                    filters=SearchFilters(preferred_sections=filters.preferred_sections),
                    top_k=3,
                    max_chunks_per_doc=2,
                )
                top_prog = sub_res.programs[0] if sub_res.programs else None
                if top_prog is not None:
                    target_programs.setdefault(top_prog.doc.doc_id, top_prog)
                    candidate_chunks = top_prog.matched_chunks[:2] or sub_res.chunks[:2]
                else:
                    candidate_chunks = sub_res.chunks[:2]
                for chunk in candidate_chunks:
                    if chunk.chunk.chunk_id not in seen_chunk_ids:
                        target_chunks.append(chunk)
                        seen_chunk_ids.add(chunk.chunk.chunk_id)

            for chunk in primary_res.chunks:
                if len(target_chunks) >= max_total_chunks:
                    break
                if chunk.chunk.chunk_id not in seen_chunk_ids:
                    target_chunks.append(chunk)
                    seen_chunk_ids.add(chunk.chunk.chunk_id)
            if len(target_programs) < 2:
                for prog in primary_res.programs:
                    if prog.doc.doc_id not in target_programs:
                        target_programs[prog.doc.doc_id] = prog

            primary_res.chunks = target_chunks[:max_total_chunks]
            primary_res.programs = list(target_programs.values())
            return primary_res

        # For follow-ups referencing previously discussed programs without naming them explicitly
        if (
            plan.intent in {"program_details", "rules_or_process"}
            and session.discussed_doc_ids
            and not mentioned_files
        ):
            follow_res = self.retriever.search(
                plan.standalone_query,
                filters=SearchFilters(
                    doc_ids=session.discussed_doc_ids[:2],
                    preferred_sections=filters.preferred_sections,
                ),
                top_k=4,
            )
            seen_ids = {c.chunk.chunk_id for c in primary_res.chunks}
            merged_chunks = list(follow_res.chunks) + [
                c for c in primary_res.chunks if c.chunk.chunk_id not in {x.chunk.chunk_id for x in follow_res.chunks}
            ]
            programs_by_id = {p.doc.doc_id: p for p in follow_res.programs}
            for p in primary_res.programs:
                if p.doc.doc_id not in programs_by_id:
                    programs_by_id[p.doc.doc_id] = p
            primary_res.chunks = merged_chunks[: max(self.cfg.retrieval.top_k, len(seen_ids))]
            primary_res.programs = list(programs_by_id.values())[: self.cfg.retrieval.expand_top_programs]

        return primary_res

    def answer(self, session_id: str, user_message: str) -> AgentResponse:
        """Process one user turn with isolated state, prompt-injection guards, and Groq generation."""
        cleaned_msg = user_message.strip()
        if not cleaned_msg:
            raise ValueError("Пустой запрос пользователя")

        t_total_start = time.perf_counter()
        session = self.sessions.get_or_create(session_id)
        warnings: list[str] = []

        assessment = assess_user_input(cleaned_msg, session)
        security_flags = list(assessment.flags)
        if assessment.is_suspicious_probe or assessment.action == "block":
            session.security_strikes += 1
        if assessment.is_suspicious_probe:
            warnings.append("security_probe_noted")

        if assessment.action == "block":
            warnings.append(f"security_blocked:{assessment.category}")
            total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            if self.diagnostic_logger is not None:
                self.diagnostic_logger.log_turn(
                    session_id=session.session_id,
                    query=cleaned_msg,
                    intent="out_of_scope",
                    model_used="security_guard",
                    latency_ms=total_latency_ms,
                    retrieval_latency_ms=0.0,
                    llm_latency_ms=0.0,
                    source_ids=[],
                    warnings=warnings,
                    security_action="block",
                    security_flags=security_flags,
                )
            return AgentResponse(
                session_id=session.session_id,
                answer=assessment.safe_refusal or "Запрос отклонён политикой безопасности.",
                sources=[],
                profile=session.profile.model_copy(deep=True),
                intent="out_of_scope",
                standalone_query="",
                model_used="security_guard",
                latency_ms=total_latency_ms,
                retrieval_latency_ms=0.0,
                llm_latency_ms=0.0,
                warnings=warnings,
            )

        working_session = session.model_copy(deep=True)
        if _is_explicit_recipient_switch(cleaned_msg):
            working_session.profile = UserProfile()

        plan = self._plan_retrieval(working_session, cleaned_msg, warnings)
        working_session.profile = _merge_profile(working_session.profile, plan.updated_profile)

        t_ret_start = time.perf_counter()
        retrieval = self._retrieve_for_plan(working_session, plan, cleaned_msg)
        retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

        context_xml, sources = format_retrieved_context(
            retrieval.programs,
            retrieval.chunks,
            intent=plan.intent,
            docs_by_id=self.retriever.docs_by_id,
            sanitization_warnings=warnings,
        )
        grounded_payload = build_grounded_user_payload(working_session, cleaned_msg, context_xml)

        # Keep system instructions strictly separated from untrusted retrieved sources and clean history
        recent_history = [
            {"role": turn.role, "content": turn.content}
            for turn in session.history[-10:]
        ]
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            *recent_history,
            {"role": "user", "content": grounded_payload},
        ]

        t_llm_start = time.perf_counter()
        answer_text, model_used = self.llm.complete(messages)
        llm_latency_ms = round((time.perf_counter() - t_llm_start) * 1000.0, 2)

        answer_text, out_flags = inspect_and_sanitize_output(answer_text)
        for flag in out_flags:
            warnings.append(f"output_guard:{flag}")
            security_flags.append(flag)

        for w in warnings:
            if w.startswith("indirect_injection_sanitized:"):
                security_flags.append(w)

        # Commit updated profile and clean turns only after retrieval and answer generation succeed
        session.profile = working_session.profile
        session.history.append(ChatTurn(role="user", content=cleaned_msg))
        session.history.append(ChatTurn(role="assistant", content=answer_text))
        if len(session.history) > 10:
            session.history = session.history[-10:]

        referenced_docs = _extract_referenced_docs(sources, answer_text)
        if not referenced_docs:
            referenced_docs = [(p.doc.doc_id, p.doc.title) for p in retrieval.programs[:3]]

        ref_ids = list(dict.fromkeys(did for did, _ in referenced_docs))
        ref_titles = list(dict.fromkeys(title for _, title in referenced_docs))
        session.discussed_doc_ids = (
            ref_ids + [did for did in session.discussed_doc_ids if did not in ref_ids]
        )[:5]
        session.discussed_titles = (
            ref_titles + [t for t in session.discussed_titles if t not in ref_titles]
        )[:5]

        total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
        has_sanitization = bool(out_flags) or any(
            w.startswith("indirect_injection_sanitized:") for w in warnings
        )
        if self.diagnostic_logger is not None:
            self.diagnostic_logger.log_turn(
                session_id=session.session_id,
                query=cleaned_msg,
                intent=plan.intent,
                model_used=model_used,
                latency_ms=total_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                source_ids=[s.source_id for s in sources],
                warnings=warnings,
                security_action="sanitized" if has_sanitization else "allow",
                security_flags=list(dict.fromkeys(security_flags)),
            )

        return AgentResponse(
            session_id=session.session_id,
            answer=answer_text,
            sources=sources,
            profile=session.profile.model_copy(deep=True),
            intent=plan.intent,
            standalone_query=plan.standalone_query,
            model_used=model_used,
            latency_ms=total_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            llm_latency_ms=llm_latency_ms,
            warnings=warnings,
        )
