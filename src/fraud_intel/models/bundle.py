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
# section 22). Phase 4's own candidate registration deliberately left
# anomaly_model_version unset -- anomaly training was Phase 5's job -- so
# that bundle is correctly, structurally NOT promotion-eligible.
#
# Phase 5 decision 3 adds four more required fields -- rule_set_version,
# graph_policy_version, ensemble_policy_version, reason_code_version --
# recording which versioned policies a bundle was registered against.
# These are kept OPTIONAL on ChannelModelBundleRecord itself (so Phase 4's
# already-registered, immutable bundle stays valid and unchanged) but are
# now part of REQUIRED_OPERATIONAL_COMPONENTS, so Phase 5's own complete
# bundle must supply all nine before it can ever be promotion-eligible.
#
# channel_model_bundles (Phase 1 schema.sql / migration 003) does NOT yet
# have columns for these four -- Phase 6's migration 004 and schema.sql
# must add rule_set_version, graph_policy_version, ensemble_policy_version,
# and reason_code_version before any real (Postgres-backed) bundle
# persistence can occur. Not implemented in Phase 5 -- no migration is
# edited or applied here; every Phase 5 test uses the in-memory fake store.
REQUIRED_OPERATIONAL_COMPONENTS = (
    "gbm_model_version",
    "lr_model_version",
    "anomaly_model_version",
    "preprocessing_artifact_version",
    "feature_schema_version",
    "rule_set_version",
    "graph_policy_version",
    "ensemble_policy_version",
    "reason_code_version",
)

# Phase 7B Stage 3 corrective pass: the fixed, AiDP-specific namespace half
# of the two-int-key `pg_advisory_xact_lock(int, int)` call
# _PostgresChannelModelBundleStore.register_candidate() uses to serialize
# candidate-bundle-version allocation per channel. Arbitrary but MUST stay
# fixed forever once assigned -- same convention as
# src.fraud_intel.generator._shared.CHANNEL_SALTS. Must fit Postgres' int4
# range (0 .. 2^31-1); derived once as the low 31 bits of
# sha256(b"aidp_fraud_intel_channel_model_bundles") purely so the value is
# reproducible and documented, never recomputed at runtime.
_BUNDLE_VERSION_LOCK_NAMESPACE = 0x22CE32AA


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
    rule_set_version: Optional[str] = None
    graph_policy_version: Optional[str] = None
    ensemble_policy_version: Optional[str] = None
    reason_code_version: Optional[str] = None
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
    src.control_plane.runs._PostgresRunStore in v1.2 Phase 2.

    Concurrency safety (Phase 7B Stage 3 corrective pass -- the original
    `SELECT ... FOR UPDATE` here turned out to be rejected outright by
    real Postgres: `FeatureNotSupported: FOR UPDATE is not allowed with
    aggregate functions`, first surfaced on the very first real training
    run, since this method had never before been exercised against a
    real database): a session-level problem needs a session-level lock,
    not a row lock -- and a row lock cannot serialize a channel's very
    FIRST bundle anyway, since no row yet exists to lock. A transaction-
    scoped Postgres advisory lock (`pg_advisory_xact_lock`, auto-released
    at COMMIT/ROLLBACK -- never an explicit unlock call) has neither
    problem: it locks a (namespace, channel-hash) KEY, not a row, so it
    serializes concurrent registrations for the SAME channel (including
    a channel's first-ever bundle) while leaving every other channel free
    to register independently. `_BUNDLE_VERSION_LOCK_NAMESPACE` is an
    arbitrary but fixed-forever-once-assigned constant (same convention
    as src.fraud_intel.generator._shared.CHANNEL_SALTS), combined with
    `hashtext(channel)` (computed server-side from the parameterized
    channel value, never string-interpolated) via the two-int-key
    `pg_advisory_xact_lock(int, int)` overload. Two DIFFERENT channels
    whose `hashtext()` values happened to collide would serialize
    together unnecessarily -- an extremely unlikely, purely-performance,
    never-a-correctness cost, since correctness comes from the lock
    PLUS the MAX-then-INSERT happening in the same transaction, not from
    perfect key uniqueness. The table's UNIQUE(channel, bundle_version)
    constraint (Phase 1 schema) remains an independent backstop.
    """

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def register_candidate(self, **fields: Any) -> ChannelModelBundleRecord:
        # NOTE (Phase 5): the INSERT's column list is built dynamically
        # from `fields`, so it already accepts rule_set_version/
        # graph_policy_version/ensemble_policy_version/reason_code_version
        # once a caller passes them -- but the RETURNING clause below is
        # NOT yet updated to select those four columns, because they do
        # not exist in channel_model_bundles until Phase 6's migration 004
        # adds them. Passing those four kwargs to this method today would
        # fail with a real "column does not exist" error, which is
        # correct and intentional: this method is not exercised by any
        # Phase 5 test (every test uses _FakeChannelModelBundleStore), and
        # must not silently pretend to support columns that do not exist
        # yet. Update the RETURNING clause in the same commit that adds
        # migration 004's four new columns.
        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Transaction-scoped advisory lock FIRST -- serializes
                    # any concurrent registration for this exact channel;
                    # released automatically when this `with conn:` block
                    # commits or rolls back, never by an explicit unlock.
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
                        (_BUNDLE_VERSION_LOCK_NAMESPACE, fields["channel"]),
                    )

                    # Same connection, same transaction, same cursor as the
                    # lock above and the INSERT below -- no aggregate
                    # function combined with FOR UPDATE anywhere.
                    cur.execute(
                        "SELECT COALESCE(MAX(bundle_version), 0) AS max_version "
                        "FROM channel_model_bundles WHERE channel = %s",
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
