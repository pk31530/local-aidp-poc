"""Provenance helpers for the AiDP v1.2 control-plane lifecycle service.

Deliberately independent of src/processing/pipeline.py, src/ml/train.py and
src/ingestion/consumer.py, for the same reason src/control_plane/config.py
stays independent — Phase 3 will have those modules depend on this one, not
the other way around.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from src.common.config import PROJECT_ROOT
from src.control_plane.config import redact_secret_keys

GIT_SHA_TIMEOUT_SECONDS = 2.0
MAX_ERROR_MESSAGE_LENGTH = 2000

_REDACTED = "***REDACTED***"

# scheme://user:password@host — only fires when actual userinfo credentials
# are present, never on an ordinary scheme://host URL.
_CREDENTIAL_URL_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s/:@]+:[^\s/@]+@")

# Sensitive query-parameter *names* only — deliberately narrower than
# config.redact_secret_keys's key list (no "dsn"/"url": those describe a
# whole field's purpose, not a realistic query-parameter name, and including
# them here would redact ordinary, non-credential URL parameters).
_SENSITIVE_QUERY_PARAM_TERMS = ("token", "key", "password", "secret", "credential")
_SENSITIVE_QUERY_PARAM_PATTERN = re.compile(
    r"(?i)\b(\w*(?:" + "|".join(_SENSITIVE_QUERY_PARAM_TERMS) + r")\w*)=([^&\s]+)"
)


def redact_credentials(text: str) -> str:
    """Redact URL userinfo credentials and sensitive query-parameter values
    from free text. Leaves ordinary, credential-free URLs untouched."""
    text = _CREDENTIAL_URL_PATTERN.sub(lambda m: f"{m.group(1)}{_REDACTED}@", text)
    text = _SENSITIVE_QUERY_PARAM_PATTERN.sub(lambda m: f"{m.group(1)}={_REDACTED}", text)
    return text


def truncate_text(text: Optional[str], max_length: int = MAX_ERROR_MESSAGE_LENGTH) -> Optional[str]:
    """Trim `text` to at most `max_length` characters, preserving a marker
    that truncation occurred."""
    if text is None:
        return None
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def get_git_sha(*, timeout: float = GIT_SHA_TIMEOUT_SECONDS, cwd: Path = PROJECT_ROOT) -> Optional[str]:
    """Current commit SHA, or None if Git is unavailable, times out, or the
    working directory isn't a Git checkout. Never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def summarize_exception(exc: BaseException, *, max_length: int = MAX_ERROR_MESSAGE_LENGTH) -> tuple[str, str]:
    """A safe (redacted, length-limited) (error_type, error_message) pair
    for persisting alongside a FAILED run. Full exception details belong in
    structured application logs, not here."""
    error_type = type(exc).__name__
    message = redact_credentials(str(exc))
    return error_type, truncate_text(message, max_length)


def normalize_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Redact secret-shaped keys and any credential-bearing text in string
    values, for metadata (e.g. artifacts) about to be persisted."""
    redacted = redact_secret_keys(data)
    return {key: (redact_credentials(value) if isinstance(value, str) else value) for key, value in redacted.items()}
