"""Structured JSON logging with a transaction-id correlation key (fix M1).

Call `configure_logging(service_name)` once per process (generator, consumer,
API, dashboard), then `bind_transaction_id(tx_id)` around the handling of one
transaction so every log line emitted while processing it — including any
that cross into DB/storage helper code — carries the same transaction_id.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import TextIO

import structlog

_transaction_id_var: ContextVar[str | None] = ContextVar("transaction_id", default=None)


def bind_transaction_id(transaction_id: str) -> None:
    _transaction_id_var.set(transaction_id)
    structlog.contextvars.bind_contextvars(transaction_id=transaction_id)


def clear_transaction_id() -> None:
    _transaction_id_var.set(None)
    structlog.contextvars.unbind_contextvars("transaction_id")


def _ensure_transaction_id(logger, method_name, event_dict):
    tx_id = _transaction_id_var.get()
    if tx_id and "transaction_id" not in event_dict:
        event_dict["transaction_id"] = tx_id
    return event_dict


def configure_logging(
    service_name: str,
    level: int = logging.INFO,
    stream: TextIO | None = None,
    *,
    force: bool = False,
) -> None:
    """Logs go to stderr by default (stdout is reserved for a program's own
    output, e.g. the CLI's --json result). `stream` is resolved fresh here
    rather than as a function-default value, and passed explicitly to both
    the stdlib handler and structlog's PrintLoggerFactory — the latter's own
    default binds to a `stdout` reference captured once at structlog's
    import time, which a caller has no way to override after the fact.

    `force` defaults to False, matching `logging.basicConfig()`'s normal
    behaviour: if the root logger already has handlers (e.g. a host
    application configured its own logging before importing this code),
    this call leaves them alone rather than tearing them down. Only a
    caller that owns its whole process outright — the CLI, at startup —
    should pass `force=True`, to guarantee a single, deterministic handler
    bound to this call's stream regardless of how many times it runs in
    that process.
    """
    resolved_stream = sys.stderr if stream is None else stream
    logging.basicConfig(format="%(message)s", stream=resolved_stream, level=level, force=force)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _ensure_transaction_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=resolved_stream),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service_name)


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
