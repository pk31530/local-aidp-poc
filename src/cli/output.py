"""Human/JSON rendering and error handling shared by every aidp CLI command.

Exit code policy:
  0 - success
  2 - validation/user-input error (bad config, run/alias not found, an
      out-of-bounds --limit, or an argparse-level parsing error)
  3 - operational failure (an unexpected exception from a real workload or
      infrastructure call)
"""
from __future__ import annotations

import json
import sys
from typing import Any

from src.control_plane.provenance import redact_credentials, truncate_text

EXIT_SUCCESS = 0
EXIT_USER_ERROR = 2
EXIT_OPERATIONAL_ERROR = 3


class CLIUserError(Exception):
    """A command handler raises this for any condition that is the
    caller's fault (bad input, not found) — always maps to EXIT_USER_ERROR,
    regardless of what underlying exception it was built from."""


def _safe_message(exc: BaseException) -> str:
    return truncate_text(redact_credentials(str(exc))) or ""


def _format_human(data: Any, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    lines: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_format_human(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        if not data:
            lines.append(f"{prefix}(none)")
        for item in data:
            lines.append(f"{prefix}-")
            lines.extend(_format_human(item, indent + 1))
    else:
        lines.append(f"{prefix}{data}")
    return lines


def emit_result(result: dict, *, json_mode: bool) -> None:
    """Prints a successful command's result to stdout only. In JSON mode
    this is exactly one JSON object with no other stdout output; in human
    mode, indented key/value text."""
    if json_mode:
        print(json.dumps(result))
    else:
        for line in _format_human(result):
            print(line)


def emit_error(exc: BaseException, *, json_mode: bool, exit_code: int) -> None:
    """Prints a safe (redacted, length-limited) error and exits. JSON mode
    emits exactly one JSON error object on stdout; human mode prints to
    stderr. Never raises past this point."""
    message = _safe_message(exc)
    if json_mode:
        print(json.dumps({"error": type(exc).__name__, "message": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)
