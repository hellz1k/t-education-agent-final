"""Session state, user profile, retrieval plan, and response contracts."""

from __future__ import annotations

import threading
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from t_edu_agent.retrieval.models import SearchFilters

AgentIntent = Literal[
    "recommend",
    "program_details",
    "compare",
    "rules_or_process",
    "out_of_scope",
]


class ChatTurn(BaseModel):
    """Single clean conversation turn stored in session history."""

    role: Literal["user", "assistant"]
    content: str


class UserProfile(BaseModel):
    """Structured user context tracked across turns in a single session."""

    status: str | None = None
    grade_or_course: str | None = None
    direction: str | None = None
    format_preference: str | None = None
    city: str | None = None
    known_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    no_selection_preferred: bool | None = None


class RetrievalPlan(BaseModel):
    """Structured plan produced by the query rewriter before retrieval."""

    standalone_query: str
    intent: AgentIntent = "recommend"
    updated_profile: UserProfile = Field(default_factory=UserProfile)
    search_filters: SearchFilters = Field(default_factory=SearchFilters)
    target_program_queries: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class SourceRef(BaseModel):
    """Verifiable source citation surfaced alongside an agent answer."""

    source_id: str
    doc_id: str
    title: str
    section: str
    source_url: str | None = None
    additional_urls: list[str] = Field(default_factory=list)
    verified_at: str | None = None
    category: str = "general"


class AgentResponse(BaseModel):
    """Final grounded response returned by TEducationAgent."""

    session_id: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)
    intent: str = "recommend"
    standalone_query: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    """Isolated per-session conversation memory."""

    session_id: str
    history: list[ChatTurn] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)
    discussed_doc_ids: list[str] = Field(default_factory=list)
    discussed_titles: list[str] = Field(default_factory=list)


class SessionStore:
    """Thread-safe in-memory session registry ensuring strict cross-session isolation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_new_session(self) -> str:
        """Create and register a fresh isolated session ID."""
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return session_id

    def get_or_create(self, session_id: str) -> SessionState:
        """Return an existing SessionState or initialize a fresh empty one."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def reset(self, session_id: str) -> SessionState:
        """Clear all history, profile, and discussed programs for a session."""
        with self._lock:
            fresh = SessionState(session_id=session_id)
            self._sessions[session_id] = fresh
            return fresh
