"""Bundle promotion (guide section 22, Phase 6): a real component
verifier, deterministic full-channel row locking, and an atomic
retire-and-promote transaction. No real MLflow/PostgreSQL contact in this
module's own code -- every unit test supplies a fake `ModelVersionVerifier`
and a fake `BundlePromotionStore`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol

from src.fraud_intel.ensemble.policy import load_ensemble_policy
from src.fraud_intel.graph.entity_graph import load_graph_policy
from src.fraud_intel.models.bundle import ChannelModelBundleRecord, validate_promotion_eligible
from src.fraud_intel.reason_codes.builder import REASON_CODE_VERSION
from src.fraud_intel.rules.provider import _load_rule_set_config


class BundleVerificationFailedError(ValueError):
    """One or more of a candidate bundle's referenced components could not
    be verified -- promotion is refused before any lock is taken. The CLI
    must refuse promotion whenever this is raised, or when verification
    itself cannot run at all (guide/Phase 6 decision 9)."""


class BundlePromotionRaceError(ValueError):
    """The candidate bundle changed, or is no longer CANDIDATE, between
    verification and the locking transaction -- promotion aborted (guide/
    Phase 6 decision 7)."""


class ModelVersionVerifier(Protocol):
    def verify(self, model_name: str, version: str) -> bool: ...


def verify_bundle_components(bundle: ChannelModelBundleRecord, *, model_version_verifier: ModelVersionVerifier) -> list[str]:
    """Returns a list of problems (empty list = fully verified). Metadata
    only -- never loads model weights, mirrors
    src.common.mlflow_setup.get_model_alias_info()'s existing pattern.
    Verifies every component the guide/Phase 6 decision 7 requires: GBM/
    LR/anomaly model versions (via the injected verifier), preprocessing
    artifact/feature-schema presence, and that rule-set/graph-policy/
    ensemble-policy/reason-code versions match the CURRENT local policy
    files/constants -- catching a bundle that is stale relative to a
    policy that has since changed.
    """
    problems: list[str] = []

    for label, version in (
        ("gbm", bundle.gbm_model_version),
        ("lr", bundle.lr_model_version),
        ("anomaly", bundle.anomaly_model_version),
    ):
        if not version:
            problems.append(f"{label}_model_version is missing")
            continue
        try:
            if not model_version_verifier.verify(f"{bundle.channel}-{label}", version):
                problems.append(f"{label} model version {version!r} could not be verified")
        except Exception as exc:
            problems.append(f"{label} model version {version!r} verification raised {type(exc).__name__}")

    if not bundle.preprocessing_artifact_version:
        problems.append("preprocessing_artifact_version is missing")
    if not bundle.feature_schema_version:
        problems.append("feature_schema_version is missing")

    try:
        rule_config = _load_rule_set_config(bundle.channel)
        if bundle.rule_set_version != rule_config.rule_set_version:
            problems.append(
                f"rule_set_version {bundle.rule_set_version!r} does not match the current "
                f"rules_{bundle.channel}.yaml ({rule_config.rule_set_version!r})"
            )
    except Exception as exc:
        problems.append(f"rule_set_version could not be verified: {type(exc).__name__}")

    try:
        graph_policy = load_graph_policy(bundle.channel)
        if bundle.graph_policy_version != graph_policy.graph_policy_version:
            problems.append(
                f"graph_policy_version {bundle.graph_policy_version!r} does not match the current "
                f"graph_policy_{bundle.channel}.yaml ({graph_policy.graph_policy_version!r})"
            )
    except Exception as exc:
        problems.append(f"graph_policy_version could not be verified: {type(exc).__name__}")

    try:
        ensemble_policy = load_ensemble_policy(bundle.channel)
        if bundle.ensemble_policy_version != ensemble_policy.policy_version:
            problems.append(
                f"ensemble_policy_version {bundle.ensemble_policy_version!r} does not match the current "
                f"ensemble_policy_{bundle.channel}.yaml ({ensemble_policy.policy_version!r})"
            )
    except Exception as exc:
        problems.append(f"ensemble_policy_version could not be verified: {type(exc).__name__}")

    if bundle.reason_code_version != REASON_CODE_VERSION:
        problems.append(
            f"reason_code_version {bundle.reason_code_version!r} does not match the current "
            f"REASON_CODE_VERSION ({REASON_CODE_VERSION!r})"
        )

    return problems


class BundlePromotionStore(Protocol):
    def get_bundle(self, channel: str, bundle_version: int) -> ChannelModelBundleRecord: ...

    def promote_locked(
        self, *, channel: str, candidate_bundle_id: int, promoted_by: str, expected_candidate: ChannelModelBundleRecord
    ) -> ChannelModelBundleRecord: ...


def promote_bundle(
    *,
    channel: str,
    bundle_version: int,
    promoted_by: str,
    model_version_verifier: ModelVersionVerifier,
    store: BundlePromotionStore,
) -> ChannelModelBundleRecord:
    candidate = store.get_bundle(channel, bundle_version)
    validate_promotion_eligible(candidate)  # Phase 5 -- all 9 required fields present

    problems = verify_bundle_components(candidate, model_version_verifier=model_version_verifier)
    if problems:
        raise BundleVerificationFailedError(f"bundle {candidate.bundle_id} failed verification: {problems}")

    return store.promote_locked(
        channel=channel, candidate_bundle_id=candidate.bundle_id, promoted_by=promoted_by, expected_candidate=candidate
    )


# ---- real, Postgres-backed store (reviewed as SQL, not exercised) ----------------------


class _PostgresBundlePromotionStore:
    """Not exercised by any Phase 6 unit test -- reviewed as SQL instead.
    Locks EVERY row for the channel, in deterministic bundle_version
    order, before deciding which is currently OPERATIONAL and re-reading
    the candidate -- eliminating the empty-current-row race and the
    two-concurrent-candidates race a narrower lock would leave open (guide/
    Phase 6 decision 7). The partial unique index
    (uq_one_operational_bundle_per_channel, migration 003) remains the
    final database-level backstop.
    """

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def get_bundle(self, channel: str, bundle_version: int) -> ChannelModelBundleRecord:
        import psycopg2.extras

        from src.common.db import get_connection

        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM channel_model_bundles WHERE channel = %s AND bundle_version = %s",
                        (channel, bundle_version),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise LookupError(f"no bundle version {bundle_version} for channel {channel!r}")
                    return ChannelModelBundleRecord(**row)
        finally:
            conn.close()

    def promote_locked(self, *, channel, candidate_bundle_id, promoted_by, expected_candidate) -> ChannelModelBundleRecord:
        import psycopg2.extras

        from src.common.db import get_connection

        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM channel_model_bundles WHERE channel = %s ORDER BY bundle_version FOR UPDATE",
                        (channel,),
                    )
                    rows = [ChannelModelBundleRecord(**r) for r in cur.fetchall()]
                    current_candidate = next((r for r in rows if r.bundle_id == candidate_bundle_id), None)
                    if current_candidate is None or current_candidate != expected_candidate:
                        raise BundlePromotionRaceError(
                            f"candidate bundle {candidate_bundle_id} changed or disappeared before promotion"
                        )
                    if current_candidate.status != "CANDIDATE":
                        raise BundlePromotionRaceError(
                            f"candidate bundle {candidate_bundle_id} is no longer CANDIDATE "
                            f"(status={current_candidate.status!r})"
                        )

                    operational = next((r for r in rows if r.status == "OPERATIONAL"), None)
                    if operational is not None:
                        cur.execute(
                            "UPDATE channel_model_bundles SET status = 'RETIRED' WHERE bundle_id = %s",
                            (operational.bundle_id,),
                        )
                    cur.execute(
                        "UPDATE channel_model_bundles SET status = 'OPERATIONAL', promoted_at = now(), promoted_by = %s "
                        "WHERE bundle_id = %s RETURNING *",
                        (promoted_by, candidate_bundle_id),
                    )
                    return ChannelModelBundleRecord(**cur.fetchone())
        finally:
            conn.close()


def create_default_bundle_promotion_store(database: Optional[str] = None) -> BundlePromotionStore:
    return _PostgresBundlePromotionStore(database)


class _FakeBundlePromotionStore:
    """In-memory stand-in built on top of a `_FakeChannelModelBundleStore`
    (src.fraud_intel.models.bundle). `promote_locked` re-reads and
    confirms the candidate is unchanged before promoting -- a test can
    mutate the underlying bundle_store's rows between `get_bundle()` and
    `promote_locked()` to directly exercise the race-abort path."""

    def __init__(self, bundle_store) -> None:
        self._bundle_store = bundle_store

    def get_bundle(self, channel: str, bundle_version: int) -> ChannelModelBundleRecord:
        for row in self._bundle_store.rows:
            if row.channel == channel and row.bundle_version == bundle_version:
                return row
        raise LookupError(f"no bundle version {bundle_version} for channel {channel!r}")

    def promote_locked(self, *, channel, candidate_bundle_id, promoted_by, expected_candidate) -> ChannelModelBundleRecord:
        rows = sorted((r for r in self._bundle_store.rows if r.channel == channel), key=lambda r: r.bundle_version)
        current_candidate = next((r for r in rows if r.bundle_id == candidate_bundle_id), None)
        if current_candidate is None or current_candidate != expected_candidate:
            raise BundlePromotionRaceError(f"candidate bundle {candidate_bundle_id} changed or disappeared before promotion")
        if current_candidate.status != "CANDIDATE":
            raise BundlePromotionRaceError(
                f"candidate bundle {candidate_bundle_id} is no longer CANDIDATE (status={current_candidate.status!r})"
            )

        operational = next((r for r in rows if r.status == "OPERATIONAL"), None)
        new_rows = []
        for row in self._bundle_store.rows:
            if operational is not None and row.bundle_id == operational.bundle_id:
                new_rows.append(row.model_copy(update={"status": "RETIRED"}))
            elif row.bundle_id == candidate_bundle_id:
                new_rows.append(
                    row.model_copy(update={"status": "OPERATIONAL", "promoted_at": datetime.now(timezone.utc), "promoted_by": promoted_by})
                )
            else:
                new_rows.append(row)
        self._bundle_store.rows = new_rows
        return next(r for r in new_rows if r.bundle_id == candidate_bundle_id)
