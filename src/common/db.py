"""Tiny Postgres connection helper shared across the pipeline/API/consumer."""
from __future__ import annotations

import psycopg2

from src.common.config import get_settings


def get_connection(database: str | None = None):
    settings = get_settings()
    dsn = settings.postgres_dsn_for(database) if database else settings.postgres_dsn
    return psycopg2.connect(dsn)
