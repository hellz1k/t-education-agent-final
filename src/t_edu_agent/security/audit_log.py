"""Thread-safe JSONL diagnostic logger with automatic secret scrubbing and session hashing."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from t_edu_agent.security.guard import scrub_secrets


class DiagnosticLogger:
    """Append compact turn metadata to a JSONL file without writing raw secrets or session IDs."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()

    @staticmethod
    def hash_session_id(session_id: str) -> str:
        """Return non-reversible short session digest for audit correlation."""
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]

    def log_turn(
        self,
        *,
        session_id: str,
        query: str,
        intent: str,
        model_used: str,
        latency_ms: float,
        retrieval_latency_ms: float,
        llm_latency_ms: float,
        source_ids: Iterable[str],
        warnings: Iterable[str],
        security_action: str,
        security_flags: Iterable[str],
    ) -> None:
        """Persist one redacted diagnostic record; swallow I/O errors to protect UX stability."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_hash": self.hash_session_id(session_id),
            "intent": intent,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "source_ids": list(source_ids),
            "warnings": list(warnings),
            "security_action": security_action,
            "security_flags": list(security_flags),
            "query_preview": scrub_secrets(query).replace("\n", " ").strip()[:160],
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            pass
