"""Local SQLite persistence for isolated user chats and agent session restoration."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from t_edu_agent.agent.state import ChatTurn, SessionState, SourceRef, UserProfile
from t_edu_agent.security.guard import scrub_secrets


class ChatSummary(BaseModel):
    """Lightweight chat list entry rendered in the Streamlit sidebar."""

    chat_id: str
    title: str
    updated_at: str


class StoredMessage(BaseModel):
    """Single saved user or assistant message along with verified citations."""

    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceRef] = Field(default_factory=list)
    created_at: str = ""


class RestoredChat(BaseModel):
    """Full chat history plus agent memory restored after a page reload."""

    chat_id: str
    title: str
    messages: list[StoredMessage] = Field(default_factory=list)
    profile: UserProfile = Field(default_factory=UserProfile)
    discussed_doc_ids: list[str] = Field(default_factory=list)
    discussed_titles: list[str] = Field(default_factory=list)

    def to_session_state(self) -> SessionState:
        """Convert persisted SQLite chat back into an in-memory SessionState."""
        history = [ChatTurn(role=m.role, content=m.content) for m in self.messages[-10:]]
        return SessionState(
            session_id=self.chat_id,
            history=history,
            profile=self.profile.model_copy(deep=True),
            discussed_doc_ids=list(self.discussed_doc_ids[:5]),
            discussed_titles=list(self.discussed_titles[:5]),
        )


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_chat_title(first_message: str) -> str:
    clean = " ".join(scrub_secrets(first_message).split())
    if not clean:
        return "Новый чат"
    return clean if len(clean) <= 48 else clean[:45].rstrip() + "..."


class SQLiteChatStore:
    """Thread-safe SQLite chat store isolating chats per hashed browser owner token."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @staticmethod
    def generate_token() -> str:
        """Create a random URL-safe identifier (128 bits of entropy)."""
        return secrets.token_urlsafe(16)

    @staticmethod
    def hash_owner(owner_token: str) -> str:
        """Store only a one-way SHA-256 hash of owner_token inside SQLite."""
        return hashlib.sha256(owner_token.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    owner_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    discussed_doc_ids_json TEXT NOT NULL,
                    discussed_titles_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_owner_updated
                    ON chats (owner_hash, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    owner_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat_owner
                    ON messages (chat_id, owner_hash, id ASC);
                """
            )

    def create_chat(self, owner_token: str, title: str = "Новый чат") -> str:
        """Create an empty chat owned strictly by owner_token and return chat_id."""
        chat_id = self.generate_token()
        owner_hash = self.hash_owner(owner_token)
        now = _now_utc_iso()
        empty_profile = UserProfile().model_dump_json()

        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO chats (
                    chat_id, owner_hash, title, profile_json,
                    discussed_doc_ids_json, discussed_titles_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, '[]', '[]', ?, ?)
                """,
                (chat_id, owner_hash, title, empty_profile, now, now),
            )
        return chat_id

    def list_chats(self, owner_token: str, limit: int = 30) -> list[ChatSummary]:
        """List recent chats belonging only to the caller's owner_token."""
        owner_hash = self.hash_owner(owner_token)
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT chat_id, title, updated_at
                FROM chats
                WHERE owner_hash = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (owner_hash, limit),
            ).fetchall()
        return [
            ChatSummary(chat_id=r["chat_id"], title=r["title"], updated_at=r["updated_at"])
            for r in rows
        ]

    def load_chat(self, owner_token: str, chat_id: str) -> RestoredChat | None:
        """Load messages, sources, and agent profile only if chat belongs to owner_token."""
        if not owner_token or not chat_id:
            return None
        owner_hash = self.hash_owner(owner_token)
        with self._lock, closing(self._connect()) as conn:
            chat_row = conn.execute(
                """
                SELECT chat_id, title, profile_json,
                       discussed_doc_ids_json, discussed_titles_json
                FROM chats
                WHERE chat_id = ? AND owner_hash = ?
                """,
                (chat_id, owner_hash),
            ).fetchone()
            if chat_row is None:
                return None

            msg_rows = conn.execute(
                """
                SELECT role, content, sources_json, created_at
                FROM messages
                WHERE chat_id = ? AND owner_hash = ?
                ORDER BY id ASC
                """,
                (chat_id, owner_hash),
            ).fetchall()

        messages: list[StoredMessage] = []
        for row in msg_rows:
            raw_sources = json.loads(row["sources_json"] or "[]")
            sources = [SourceRef.model_validate(item) for item in raw_sources]
            messages.append(
                StoredMessage(
                    role=row["role"],
                    content=row["content"],
                    sources=sources,
                    created_at=row["created_at"],
                )
            )

        return RestoredChat(
            chat_id=chat_row["chat_id"],
            title=chat_row["title"],
            messages=messages,
            profile=UserProfile.model_validate_json(chat_row["profile_json"]),
            discussed_doc_ids=json.loads(chat_row["discussed_doc_ids_json"] or "[]"),
            discussed_titles=json.loads(chat_row["discussed_titles_json"] or "[]"),
        )

    def save_exchange(
        self,
        owner_token: str,
        chat_id: str,
        user_message: str,
        assistant_answer: str,
        sources: list[SourceRef],
        session_state: SessionState,
    ) -> None:
        """Persist a verified Q&A pair and updated SessionState snapshot atomically."""
        owner_hash = self.hash_owner(owner_token)
        safe_user = scrub_secrets(user_message)
        safe_assistant = scrub_secrets(assistant_answer)
        sources_payload = json.dumps(
            [src.model_dump(exclude_none=True) for src in sources],
            ensure_ascii=False,
        )
        now = _now_utc_iso()

        with self._lock, closing(self._connect()) as conn, conn:
            chat_row = conn.execute(
                "SELECT title FROM chats WHERE chat_id = ? AND owner_hash = ?",
                (chat_id, owner_hash),
            ).fetchone()
            if chat_row is None:
                raise PermissionError("Chat not found or access denied for this session.")

            new_title = (
                _make_chat_title(safe_user)
                if chat_row["title"] == "Новый чат"
                else chat_row["title"]
            )
            conn.execute(
                """
                INSERT INTO messages (chat_id, owner_hash, role, content, sources_json, created_at)
                VALUES (?, ?, 'user', ?, '[]', ?),
                       (?, ?, 'assistant', ?, ?, ?)
                """,
                (chat_id, owner_hash, safe_user, now, chat_id, owner_hash, safe_assistant, sources_payload, now),
            )
            conn.execute(
                """
                UPDATE chats
                SET title = ?,
                    profile_json = ?,
                    discussed_doc_ids_json = ?,
                    discussed_titles_json = ?,
                    updated_at = ?
                WHERE chat_id = ? AND owner_hash = ?
                """,
                (
                    new_title,
                    session_state.profile.model_dump_json(),
                    json.dumps(session_state.discussed_doc_ids[:5], ensure_ascii=False),
                    json.dumps(session_state.discussed_titles[:5], ensure_ascii=False),
                    now,
                    chat_id,
                    owner_hash,
                ),
            )

    def clear_chat(self, owner_token: str, chat_id: str) -> None:
        """Clear messages and saved agent memory for the owner's active chat."""
        owner_hash = self.hash_owner(owner_token)
        now = _now_utc_iso()
        empty_profile = UserProfile().model_dump_json()

        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                "DELETE FROM messages WHERE chat_id = ? AND owner_hash = ?",
                (chat_id, owner_hash),
            )
            conn.execute(
                """
                UPDATE chats
                SET title = 'Новый чат',
                    profile_json = ?,
                    discussed_doc_ids_json = '[]',
                    discussed_titles_json = '[]',
                    updated_at = ?
                WHERE chat_id = ? AND owner_hash = ?
                """,
                (empty_profile, now, chat_id, owner_hash),
            )
