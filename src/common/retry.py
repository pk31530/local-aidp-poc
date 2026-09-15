"""Retry-with-backoff for transient infrastructure failures (fix H3) —
deliberately separate from the DLQ path, which is for messages that are
malformed and will never succeed no matter how many times they're retried.
Config comes from config/settings.yaml's streaming.retry block.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from src.common.config import get_app_settings
from src.common.logging import get_logger

_RETRY_CFG = get_app_settings()["streaming"]["retry"]


def _log_before_sleep(retry_state) -> None:
    log = get_logger("retry")
    log.warning(
        "transient_failure_retrying",
        attempt=retry_state.attempt_number,
        max_attempts=_RETRY_CFG["max_attempts"],
        error=str(retry_state.outcome.exception()),
    )


def transient_retry():
    return retry(
        stop=stop_after_attempt(_RETRY_CFG["max_attempts"]),
        wait=wait_exponential_jitter(
            initial=_RETRY_CFG["base_delay_seconds"],
            max=_RETRY_CFG["max_delay_seconds"],
        ),
        before_sleep=_log_before_sleep,
        reraise=True,
    )
