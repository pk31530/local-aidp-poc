import subprocess

import pytest

from src.control_plane.provenance import (
    MAX_ERROR_MESSAGE_LENGTH,
    get_git_sha,
    normalize_metadata,
    redact_credentials,
    summarize_exception,
    truncate_text,
)


# ---- get_git_sha: deterministic, mocked subprocess -----------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_git_sha_success(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(0, "abc123def456abc123def456abc123def456abc\n"),
    )
    assert get_git_sha() == "abc123def456abc123def456abc123def456abc"


def test_git_sha_non_zero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(128, ""))
    assert get_git_sha() is None


def test_git_sha_timeout(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2.0)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert get_git_sha() is None


def test_git_sha_missing_binary(monkeypatch):
    def _raise(*a, **k):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert get_git_sha() is None


# ---- redact_credentials ---------------------------------------------------------


def test_redact_credentials_scrubs_url_userinfo():
    text = "connection failed: postgresql://aidp:hunter2@localhost:5432/aidp"
    redacted = redact_credentials(text)
    assert "hunter2" not in redacted
    assert "postgresql://***REDACTED***@localhost:5432/aidp" in redacted


def test_redact_credentials_scrubs_sensitive_query_params():
    text = "callback failed: https://api.example.com/hook?token=abcd1234&page=2"
    redacted = redact_credentials(text)
    assert "abcd1234" not in redacted
    assert "token=***REDACTED***" in redacted
    assert "page=2" in redacted  # ordinary param untouched


def test_redact_credentials_handles_both_in_one_message():
    text = "postgresql://u:p@host/db and https://x.example.com/a?secret=s3cr3t&sort=asc"
    redacted = redact_credentials(text)
    assert "u:p@" not in redacted
    assert "s3cr3t" not in redacted
    assert "sort=asc" in redacted


def test_redact_credentials_leaves_ordinary_url_untouched():
    text = "see https://example.com/docs?page=2&sort=asc for details"
    assert redact_credentials(text) == text


# ---- truncate_text ---------------------------------------------------------------


def test_truncate_text_leaves_short_text_unchanged():
    assert truncate_text("short message") == "short message"


def test_truncate_text_none_passthrough():
    assert truncate_text(None) is None


def test_truncate_text_truncates_long_text():
    long_text = "x" * (MAX_ERROR_MESSAGE_LENGTH + 500)
    truncated = truncate_text(long_text)
    assert len(truncated) <= MAX_ERROR_MESSAGE_LENGTH
    assert truncated.endswith("…")


# ---- summarize_exception ---------------------------------------------------------


def test_summarize_exception_returns_type_and_redacted_message():
    exc = ConnectionError("could not connect to postgresql://aidp:hunter2@localhost/aidp")
    error_type, message = summarize_exception(exc)
    assert error_type == "ConnectionError"
    assert "hunter2" not in message


def test_summarize_exception_truncates_long_message():
    exc = RuntimeError("x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
    _, message = summarize_exception(exc)
    assert len(message) <= MAX_ERROR_MESSAGE_LENGTH


# ---- normalize_metadata -----------------------------------------------------------


def test_normalize_metadata_redacts_secret_keys_and_credential_urls():
    data = {
        "postgres_dsn": "postgresql://u:p@host/db",
        "note": "webhook is https://x.example.com/hook?token=abcd",
        "records": 42,
    }
    normalized = normalize_metadata(data)
    assert normalized["postgres_dsn"] == "***REDACTED***"
    assert "abcd" not in normalized["note"]
    assert normalized["records"] == 42
