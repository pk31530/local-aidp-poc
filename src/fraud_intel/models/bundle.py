"""Injectable channel_model_bundles store (guide sections 11, 22). Mirrors
src.control_plane.runs.RunLifecycle's injectable-store pattern exactly --
no database connection is made by any Phase 4 unit test; every test
supplies a `_FakeChannelModelBundleStore` instead.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import psycopg2.extras
from pydantic import BaseModel, ConfigDict

from src.common.db import get_connection

# Components required before a bundle may become OPERATIONAL (guide
# section 22). Phase 4's own candidate registration deliberately leaves
# anomaly_model_version unset -- anomaly training is Phase 5's job -- so a
# Phase 4 bundle is correctly, structurally NOT promotion-eligible yet.
REQUIRED_OPERATIONAL_COMPONENTS = (
    "gbm_model_version",
    "lr_model_version",
    "anomaly_model_version",
    "preprocessing_artifact_version",
    "feature_schema_version",
)


class IncompleteBundleError(ValueError):
    """A bundle is missing a component required for OPERATIONAL status.
    Raised by promotion-time validation (Phase 6/22) -- never by Phase 4's
    own register_candidate(), which may legitimately write an
    anomaly-incomplete candidate bundle."""


class ChannelModelBundleRecord(BaseModel):
    """Immutable once constructed (guide section 22/Phase 4 decision 8) --
    a bundle row is never mutated in place; a later training run
    registers a new row with a new bundle_version instead."""

    model_config = ConfigDict(protected_namespaces=(), frozen=True)

    bundle_id: int
    channel: str
    bundle_version: int
    gbm_model_version: Optional[str] = None
    lr_model_version: Optional[str] = None
    anomaly_model_version: Optional[str] = None
    preprocessing_artifact_version: Optional[str] = None
    feature_schema_version: Optional[str] = None
    training_run_id: Optional[int] = None
    dataset_version: Optional[str] = None
    evaluation_report_ref: Optional[str] = None
    status: str = "CANDIDATE"
    created_at: datetime
    promoted_at: Optional[datetime] = None
    promoted_by: Optional[str] = None

    def is_promotion_eligible(self) -> bool:
        return all(getattr(self, field_name) is not None for field_name in REQUIRED_OPERATIONAL_COMPONENTS)


def validate_promotion_eligible(bundle: ChannelModelBundleRecord) -> None:
    """Guide section 22: "promotion validation must later reject any
    bundle missing required Phase 5 components." Phase 6's real promotion
    code calls this (or an equivalent check); it is defined here now so
    the incomplete-bundle safeguard has a concrete, testable form in this
    phase rather than being only a documented intention."""
    if not bundle.is_promotion_eligible():
        missing = [name for name in REQUIRED_OPERATIONAL_COMPONENTS if getattr(bundle, name) is None]
        raise IncompleteBundleError(
            f"bundle {bundle.bundle_id} (channel={bundle.channel!r}, version={bundle.bundle_version}) "
            f"is missing required component(s) for promotion: {missing}"
        )


class ChannelModelBundleStore(Protocol):
    def register_candidate(self, **fields: Any) -> ChannelModelBundleRecord: ...


class _PostgresChannelModelBundleStore:
    """The real, Postgres-backed store. Not exercised by any Phase 4 unit
    test (see module docstring) -- reviewed as SQL instead, exactly like
    src.control_plane.runs._PostgresRunStore in v1.2 Phase 2. Not used
    anywhere until Phase 6/7B applies migration 003 and wires real
    training runs to it.

    Concurrency safety: `SELECT ... FOR UPDATE` locks every existing row
    for this channel for the duration of the transaction, so a second,
    concurrent registration for the same channel cannot compute the same
    "next" version while the first is still in flight. For the very first
    bundle of a channel (no existing rows to lock), the table's
    UNIQUE(channel, bundle_version) constraint (Phase 1 schema) is the
    backstop -- at most one of two racing first-inserts can succeed; the
    loser gets a real, visible constraint-violation error, never a silent
    duplicate version.
    """

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def register_candidate(self, **fields: Any) -> ChannelModelBundleRecord:
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT COALESCE(MAX(bundle_version), 0) AS max_version "
                        "FROM channel_model_bundles WHERE channel = %s FOR UPDATE",
                        (fields["channel"],),
                    )
                    next_version = cur.fetchone()["max_version"] + 1

                    columns = list(fields.keys()) + ["bundle_version", "status"]
                    values = [fields[c] for c in fields] + [next_version, "CANDIDATE"]
                    placeholders = ", ".join(["%s"] * len(columns))
                    cur.execute(
                        f"""
                        INSERT INTO channel_model_bundles ({", ".join(columns)})
                        VALUES ({placeholders})
                        RETURNING bundle_id, channel, bundle_version, gbm_model_version, lr_model_version,
                                  anomaly_model_version, preprocessing_artifact_version, feature_schema_version,
                                  training_run_id, dataset_version, evaluation_report_ref, status, created_at,
                                  promoted_at, promoted_by
                        """,
                        values,
                    )
                    return ChannelModelBundleRecord(**cur.fetchone())
        finally:
            conn.close()


def create_default_bundle_store(database: Optional[str] = None) -> ChannelModelBundleStore:
    return _PostgresChannelModelBundleStore(database)


class _FakeChannelModelBundleStore:
    """In-memory stand-in -- no database touched. Mirrors the real store's
    per-channel next-version computation and its UNIQUE(channel,
    bundle_version) backstop (raises ValueError on an attempted duplicate
    version -- the same failure class a real unique-constraint violation
    would surface as)."""

    def __init__(self) -> None:
        self.rows: list[ChannelModelBundleRecord] = []
        self._next_id = 1

    def register_candidate(self, **fields: Any) -> ChannelModelBundleRecord:
        channel = fields["channel"]
        existing_versions = [row.bundle_version for row in self.rows if row.channel == channel]
        next_version = (max(existing_versions) if existing_versions else 0) + 1
        if any(row.channel == channel and row.bundle_version == next_version for row in self.rows):
            raise ValueError(f"bundle_version {next_version} already exists for channel {channel!r}")

        record = ChannelModelBundleRecord(
            bundle_id=self._next_id,
            bundle_version=next_version,
            status="CANDIDATE",
            created_at=datetime.now(timezone.utc),
            **{key: value for key, value in fields.items()},
        )
        self._next_id += 1
        self.rows.append(record)
        return record
