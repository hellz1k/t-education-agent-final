"""Security guardrails and safe diagnostic logging for Stage 4."""

from t_edu_agent.security.audit_log import DiagnosticLogger
from t_edu_agent.security.guard import (
    SecurityAssessment,
    assess_user_input,
    escape_delimiter_tags,
    inspect_and_sanitize_output,
    sanitize_untrusted_chunk,
    scrub_secrets,
)

__all__ = [
    "DiagnosticLogger",
    "SecurityAssessment",
    "assess_user_input",
    "escape_delimiter_tags",
    "inspect_and_sanitize_output",
    "sanitize_untrusted_chunk",
    "scrub_secrets",
]
