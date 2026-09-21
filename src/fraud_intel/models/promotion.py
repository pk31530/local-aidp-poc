"""Bundle promotion (guide section 22, Phase 6; Phase 7B Stage 5
corrective pass): a real component verifier, deterministic full-channel
row locking, an atomic retire-and-promote transaction, and -- for a
channel's first-ever promotion (no OPERATIONAL bundle yet) -- mandatory,
freshly-recomputed cold-start promotion-gate enforcement plus a real
`model_promotion` audit trail. No real MLflow/PostgreSQL contact in this
module's own code -- every unit test supplies a fake `ModelVersionVerifier`
and a fake `BundlePromotionStore`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from src.control_plane.runs import RunLifecycle
from src.fraud_intel.ensemble.policy import load_ensemble_policy
from src.fraud_intel.evaluation.cold_start import evaluate_cold_start_promotion_gate, load_and_validate_cold_start_report
from src.fraud_intel.graph.entity_graph import load_graph_policy
from src.fraud_intel.models.bundle import ChannelModelBundleRecord, validate_promotion_eligible
from src.fraud_intel.models.mlflow_naming import registered_model_name
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


class ColdStartPromotionGateFailedError(ValueError):
    """Phase 7B Stage 5: a channel with no current OPERATIONAL bundle must
    clear its cold-start promotion gate (freshly recomputed from the
    candidate's own evaluation_report_ref, never a cached/remembered
    value) before promotion -- refused here, before any lock is taken."""


class ModelVersionVerifier(Protocol):
    def verify(self, model_name: str, version: str, *, expected_run_id: Optional[str] = None) -> bool: ...


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

    # Phase 7B Stage 5: expected MLflow run ids, when available from the
    # candidate's own (immutable, training-time) evaluation_report_ref --
    # lets the verifier confirm each registered version points at the
    # exact run training actually produced, not merely that a version
    # number exists under the right name.
    expected_run_ids: dict[str, Optional[str]] = {"gbm": None, "lr-shadow": None, "anomaly": None}
    if bundle.evaluation_report_ref:
        try:
            report = json.loads(bundle.evaluation_report_ref)
            expected_run_ids["gbm"] = report.get("gbm_mlflow_run_id")
            expected_run_ids["lr-shadow"] = report.get("lr_mlflow_run_id")
            expected_run_ids["anomaly"] = report.get("anomaly_mlflow_run_id")
        except json.JSONDecodeError:
            pass  # malformed report -- the run-id cross-check is simply skipped; other checks still run

    for component, version in (
        ("gbm", bundle.gbm_model_version),
        ("lr-shadow", bundle.lr_model_version),
        ("anomaly", bundle.anomaly_model_version),
    ):
        if not version:
            problems.append(f"{component}_model_version is missing")
            continue
        model_name = registered_model_name(bundle.channel, component)
        expected_run_id = expected_run_ids[component]
        try:
            if not model_version_verifier.verify(model_name, version, expected_run_id=expected_run_id):
                problems.append(
                    f"{component} model {model_name!r} version {version!r} could not be verified "
                    f"(expected_run_id={expected_run_id!r})"
                )
        except Exception as exc:
            problems.append(f"{component} model {model_name!r} version {version!r} verification raised {type(exc).__name__}")

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

    def get_operational_bundle(self, channel: str) -> Optional[ChannelModelBundleRecord]: ...

    def promote_locked(
        self, *, channel: str, candidate_bundle_id: int, promoted_by: str, expected_candidate: ChannelModelBundleRecord
    ) -> ChannelModelBundleRecord: ...


def promote_bundle(
    *,
    channel: str,
    bundle_version: int,
    promoted_by: str,
    database: str,
    model_version_verifier: ModelVersionVerifier,
    store: BundlePromotionStore,
) -> ChannelModelBundleRecord:
    """Phase 6 decision 7/9 verification and locking, PLUS (Phase 7B
    Stage 5 corrective pass) mandatory cold-start promotion-gate
    enforcement for a channel's first-ever promotion, and a real
    `model_promotion` RunLifecycle audit trail.

    `database` is required, with no default -- mirrors
    src.fraud_intel.models.training.train_channel_configured()'s own
    Stage 3 fix: this function must never silently fall back to `.env`'s
    POSTGRES_DB default. `promoted_by` (manual operator approval) is
    ALWAYS required, even when the cold-start gate passes -- passing the
    gate only makes a candidate ELIGIBLE, it never self-promotes.

    Cross-transaction limitation (documented, not hidden): `store.
    promote_locked()` below runs its own, separate Postgres transaction
    (a different connection than this function's RunLifecycle). By the
    time this function reaches `lifecycle.succeed(...)`, the bundle is
    ALREADY durably OPERATIONAL -- if the lifecycle's own SUCCESS write
    then fails (e.g. a transient database error), `channel_model_bundles`
    remains the authoritative promotion record (status/promoted_at/
    promoted_by are already correct there), but the `model_promotion`
    pipeline_runs row may be missing or left at RUNNING and would need
    manual audit reconciliation against the bundle row. This function
    does not attempt to hide or paper over that gap.
    """
    if not database:
        raise ValueError("database is required and must not be empty -- promote_bundle() never assumes a default")
    if not promoted_by:
        raise ValueError("promoted_by is required -- manual approval is mandatory even when the cold-start gate passes")

    lifecycle = RunLifecycle(database=database)
    run = lifecycle.begin("model_promotion", trigger_source="cli")

    gate_evidence: dict[str, Any] = {
        "evaluation_mode": None,
        "promotion_gate_passed": None,
        "promotion_gate_reasons": None,
        "test_fraud_count": None,
        "test_legitimate_count": None,
        "gbm_pr_auc": None,
        "test_fraud_prevalence": None,
        "source_generation_run_id": None,
        "source_dataset_version": None,
        "supervised_population_hash": None,
    }

    try:
        candidate = store.get_bundle(channel, bundle_version)
        validate_promotion_eligible(candidate)  # Phase 5 -- all 9 required fields present

        problems = verify_bundle_components(candidate, model_version_verifier=model_version_verifier)
        if problems:
            raise BundleVerificationFailedError(f"bundle {candidate.bundle_id} failed verification: {problems}")

        operational = store.get_operational_bundle(channel)
        if operational is None:
            # This would be the channel's first-ever OPERATIONAL bundle --
            # the cold-start gate is mandatory here and ALWAYS freshly
            # recomputed, never a cached/remembered value from an earlier
            # `evaluate` invocation or conversation history.
            report = load_and_validate_cold_start_report(candidate)
            gate = evaluate_cold_start_promotion_gate(report)
            test_counts = report.split_class_counts.get("test", {})
            gate_evidence.update(
                evaluation_mode="candidate_training_holdout",
                promotion_gate_passed=gate["passed"],
                promotion_gate_reasons=gate["reasons"],
                test_fraud_count=test_counts.get("fraud"),
                test_legitimate_count=test_counts.get("legitimate"),
                gbm_pr_auc=gate["gbm_pr_auc"],
                test_fraud_prevalence=gate["test_fraud_prevalence"],
                source_generation_run_id=report.source_generation_run_id,
                source_dataset_version=report.source_dataset_version,
                supervised_population_hash=report.supervised_population_hash,
            )
            if not gate["passed"]:
                raise ColdStartPromotionGateFailedError(
                    f"bundle {candidate.bundle_id} failed the cold-start promotion gate: {gate['reasons']}"
                )

        promoted = store.promote_locked(
            channel=channel, candidate_bundle_id=candidate.bundle_id, promoted_by=promoted_by, expected_candidate=candidate
        )
    except Exception as exc:
        lifecycle.fail_from_exception(run.run_id, exc)
        raise

    lifecycle.succeed(
        run.run_id,
        artifacts={
            "channel": channel,
            "bundle_id": promoted.bundle_id,
            "bundle_version": promoted.bundle_version,
            "promoted_by": promoted_by,
            "previous_status": "CANDIDATE",
            "resulting_status": promoted.status,
            **gate_evidence,
            "gbm_model_version": promoted.gbm_model_version,
            "lr_model_version": promoted.lr_model_version,
            "anomaly_model_version": promoted.anomaly_model_version,
            "preprocessing_artifact_version": promoted.preprocessing_artifact_version,
            "feature_schema_version": promoted.feature_schema_version,
            "rule_set_version": promoted.rule_set_version,
            "graph_policy_version": promoted.graph_policy_version,
            "ensemble_policy_version": promoted.ensemble_policy_version,
            "reason_code_version": promoted.reason_code_version,
        },
    )
    return promoted


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

    def get_operational_bundle(self, channel: str) -> Optional[ChannelModelBundleRecord]:
        import psycopg2.extras

        from src.common.db import get_connection

        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT * FROM channel_model_bundles WHERE channel = %s AND status = 'OPERATIONAL'",
                        (channel,),
                    )
                    row = cur.fetchone()
                    return ChannelModelBundleRecord(**row) if row else None
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

    def get_operational_bundle(self, channel: str) -> Optional[ChannelModelBundleRecord]:
        for row in self._bundle_store.rows:
            if row.channel == channel and row.status == "OPERATIONAL":
                return row
        return None

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
