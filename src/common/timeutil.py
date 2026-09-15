"""Single pinned timezone, used identically everywhere a "night time" or
"local hour" decision is made (fix M2) — the generator, the shared feature
module, and decisioning all import APP_TZ from here rather than each picking
their own.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.common.config import get_settings

APP_TZ = ZoneInfo(get_settings().app_timezone)


def to_app_tz(dt: datetime) -> datetime:
    """Convert an aware datetime to the pinned app timezone."""
    if dt.tzinfo is None:
        raise ValueError("to_app_tz requires a timezone-aware datetime")
    return dt.astimezone(APP_TZ)
