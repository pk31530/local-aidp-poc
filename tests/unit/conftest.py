"""Shared fixtures for the unit suite.

Root cause of a real, order-dependent test-isolation defect (Phase 7A
corrective pass): `src.common.logging.configure_logging(..., force=True)`
-- called once per process by `src.cli.__main__.main()`, i.e. by any test
that exercises the CLI end to end -- mutates two pieces of GLOBAL,
process-wide state:

1. The stdlib root logger's handlers/level, via
   `logging.basicConfig(..., force=True)`.
2. structlog's own global configuration, via `structlog.configure(...,
   logger_factory=structlog.PrintLoggerFactory(file=resolved_stream),
   cache_logger_on_first_use=True)`.

Both are bound to whatever `sys.stderr` happens to be active at that
EXACT call -- which, under pytest's own `capsys` fixture, is a stream
object substituted only for the DURATION of one test and then discarded
(closed) once that test ends. A CLI test therefore leaks a structlog
logger factory (and a stdlib logging handler) bound to a now-closed
stream into every later test in the same pytest process -- the first
later test that logs anything at all (via `get_logger(...).error(...)`,
anywhere in the whole codebase, regardless of what that later test is
actually about) then crashes with `ValueError: I/O operation on closed
file`. This is exactly what made the Phase 6/7A "focused test" command
order-dependent: it failed or passed depending on which OTHER test file
happened to run afterward and "heal" the global state by reconfiguring
it again, not on anything about the failing test itself.

This autouse, function-scoped fixture snapshots both pieces of global
state before every single test and restores them after, so no test can
ever leak logging configuration into another -- regardless of test file
order, and without touching monkeypatch (already function-scoped and
self-reverting) or pytest's own `capsys`/`capfd` stdout/stderr capture
machinery, which this fixture deliberately never touches directly.
"""
from __future__ import annotations

import logging

import pytest
import structlog


@pytest.fixture(autouse=True)
def _isolate_logging_state():
    root_logger = logging.root
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_structlog_config = structlog.get_config()

    yield

    root_logger.handlers[:] = original_handlers
    root_logger.setLevel(original_level)
    structlog.configure(**original_structlog_config)
