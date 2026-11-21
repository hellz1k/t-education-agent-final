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

    parts = [user_message]
    if session.history:
        if session.profile.direction and session.profile.direction.lower() not in user_message.lower():
            parts.append(session.profile.direction)
        if session.profile.status and session.profile.status.lower() not in user_message.lower():
            parts.append(session.profile.status)
        if session.profile.known_skills:
            parts.extend(s for s in session.profile.known_skills if s.lower() not in user_message.lower())
        if (
            session.profile.format_preference
            and session.profile.format_preference.lower() in {"онлайн", "очно"}
            and session.profile.format_preference.lower() not in user_message.lower()
        ):
            parts.append(session.profile.format_preference)

    q_l = user_message.lower()
    intent = "recommend"
    if any(kw in q_l for kw in ("сравни", "разниц", "отлича", "лучше выбрать")):
        intent = "compare"
    elif any(kw in q_l for kw in ("запиш", "отбор", "подать заявку", "без отбора")):
        intent = "rules_or_process"

    profile_copy = session.profile.model_copy(deep=True)
    filters = inferred_hints.model_copy(deep=True)
    merged_missing = list(dict.fromkeys([*profile_copy.missing_skills, *(inferred_hints.exclude_required_skills or [])]))
    if merged_missing:
        profile_copy.missing_skills = merged_missing
        profile_copy.known_skills = [s for s in profile_copy.known_skills if s.lower() not in merged_missing]
        filters.exclude_required_skills = merged_missing

    if session.profile.format_preference in {"онлайн", "очно"} and not filters.format_modes:
        filters.format_modes = [session.profile.format_preference]
    if session.profile.no_selection_preferred is True and filters.has_selection is None:
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
        mentioned_files = _detect_entity_matches(user_message) | _detect_entity_matches(
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

        # Unpenalized retrieval for explicitly named programs so prior filters cannot hide them
        if mentioned_files:
            mentioned_doc_ids = [
                d.doc_id
                for d in self.retriever.documents
                if d.source_file in mentioned_files
            ]
            if mentioned_doc_ids:
                mention_query = user_message.strip() or plan.standalone_query
                mention_res = self.retriever.search(
                    mention_query,
                    filters=SearchFilters(
                        doc_ids=mentioned_doc_ids,
                        preferred_sections=filters.preferred_sections,
                    ),
                    top_k=min(6, 2 * len(mentioned_doc_ids)),
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

        plan = self._plan_retrieval(session, cleaned_msg, warnings)
        session.profile = _merge_profile(session.profile, plan.updated_profile)

        t_ret_start = time.perf_counter()
        retrieval = self._retrieve_for_plan(session, plan, cleaned_msg)
        retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)

        context_xml, sources = format_retrieved_context(
            retrieval.programs,
            retrieval.chunks,
            intent=plan.intent,
            docs_by_id=self.retriever.docs_by_id,
            sanitization_warnings=warnings,
        )
        grounded_payload = build_grounded_user_payload(session, cleaned_msg, context_xml)

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

        # Persist only clean user/assistant turns (never raw retrieved source XML)
        session.history.append(ChatTurn(role="user", content=cleaned_msg))
        session.history.append(ChatTurn(role="assistant", content=answer_text))
        if len(session.history) > 10:
            session.history = session.history[-10:]

        new_doc_ids = [p.doc.doc_id for p in retrieval.programs[:3]]
        new_titles = [p.doc.title for p in retrieval.programs[:3]]
        for did, title in zip(new_doc_ids, new_titles):
            if did not in session.discussed_doc_ids:
                session.discussed_doc_ids.insert(0, did)
            if title not in session.discussed_titles:
                session.discussed_titles.insert(0, title)
        session.discussed_doc_ids = session.discussed_doc_ids[:5]
        session.discussed_titles = session.discussed_titles[:5]

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
