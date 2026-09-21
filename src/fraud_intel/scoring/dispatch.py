"""Reference-channel (online_banking) real scoring dispatch (Phase 6
corrective pass). Wires the CLI's `aidp fraud-intel score` command to the
already-built, already-tested pieces (RunLifecycle, the scoring
orchestrator, the alert queue) rather than leaving it a permanent
NotImplementedError stub.

No database, MLflow, or network contact in `score_channel()` itself -- it
only calls its injected `data_access`/`get_operational_bundle`/
`artifact_loader`/`alert_queue_store` collaborators. Every unit test
supplies fakes for all four; the real, Postgres/MLflow-backed
implementations below (`_PostgresScoringDataAccess`,
`_MlflowBundleArtifactLoader`) are reviewed as code, never exercised by
any unit test -- same precedent as every other "_Postgres*Store"/
"_Mlflow*" class in this codebase.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

from src.common.logging import get_logger
from src.control_plane.provenance import get_git_sha
from src.control_plane.runs import RunLifecycle
from src.fraud_intel.alerts.queue import AlertQueueStore, score_and_record_alert
from src.fraud_intel.ensemble.policy import EnsemblePolicy, load_ensemble_policy
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import GraphPolicy, ResolvedFraudEntityEvidence, load_graph_policy
from src.fraud_intel.models.bundle import ChannelModelBundleRecord
from src.fraud_intel.models.mlflow_naming import registered_model_name
from src.fraud_intel.reason_codes.builder import REASON_CODE_VERSION
from src.fraud_intel.rules.provider import LocalYamlRuleProvider, RuleProvider, _load_rule_set_config
from src.fraud_intel.scoring.orchestrator import LoadedChannelBundle

log = get_logger(__name__)

_MAX_PENDING_ALERTS_PER_RUN = 500
_MAX_HISTORICAL_EVENTS_PER_ALERT = 1000
_MAX_SOURCE_ALERT_HISTORY_PER_ALERT = 200
_MAX_RESOLVED_FRAUD_EVIDENCE_ROWS = 500


class NoOperationalBundleError(LookupError):
    """No OPERATIONAL channel_model_bundles row exists for the requested
    channel -- scoring cannot proceed (the CLI maps this to a
    CLIUserError, not an operational failure: it means "promote a bundle
    first," not "the system is broken")."""


class BundlePolicyMismatchError(ValueError):
    """The OPERATIONAL bundle's pinned rule/graph/ensemble/reason-code
    version no longer matches the currently loaded config/constant --
    scoring must refuse rather than silently score against a policy the
    bundle was never validated against (same principle as
    src.fraud_intel.models.promotion.verify_bundle_components, applied at
    scoring time instead of promotion time)."""


class BundleArtifactLoadError(RuntimeError):
    """The OPERATIONAL bundle's pinned model/preprocessing artifacts could
    not be loaded (missing evaluation_report_ref, a malformed run
    reference, or the real MLflow/artifact-store call itself failing) --
    an operational failure (exit 3), not a user input error."""


# ---- injectable collaborators -----------------------------------------------------


@dataclass(frozen=True)
class PendingScoringItem:
    """One source alert selected for scoring, with everything
    `score_and_record_alert()` needs already assembled and as-of-time
    validated."""

    event: FraudEvent
    source_alert: SourceAlertContext
    context: FeatureComputationContext
    resolved_fraud_evidence: tuple[ResolvedFraudEntityEvidence, ...]


class ScoringDataAccess(Protocol):
    def list_pending(self, channel: str) -> Sequence[PendingScoringItem]: ...


class BundleArtifactLoader(Protocol):
    def load(self, bundle: ChannelModelBundleRecord) -> LoadedChannelBundle: ...


GetOperationalBundle = Callable[[str], Optional[ChannelModelBundleRecord]]


def load_and_validate_pinned_policies(bundle: ChannelModelBundleRecord) -> tuple[RuleProvider, GraphPolicy, EnsemblePolicy]:
    """Loads the CURRENT local policy config/constants and validates each
    against the bundle's own PINNED version -- refuses (rather than
    silently scoring against drifted policy) exactly when
    src.fraud_intel.models.promotion.verify_bundle_components() would have
    refused promotion. Local YAML/constant reads only -- no database,
    MLflow, or network I/O, same as every existing caller of these three
    loaders (e.g. tests/unit/test_fraud_intel_promotion.py, unmocked)."""
    rule_config = _load_rule_set_config(bundle.channel)
    if bundle.rule_set_version != rule_config.rule_set_version:
        raise BundlePolicyMismatchError(
            f"bundle {bundle.bundle_id} rule_set_version {bundle.rule_set_version!r} does not match "
            f"the current rules_{bundle.channel}.yaml ({rule_config.rule_set_version!r})"
        )

    graph_policy = load_graph_policy(bundle.channel)
    if bundle.graph_policy_version != graph_policy.graph_policy_version:
        raise BundlePolicyMismatchError(
            f"bundle {bundle.bundle_id} graph_policy_version {bundle.graph_policy_version!r} does not "
            f"match the current graph_policy_{bundle.channel}.yaml ({graph_policy.graph_policy_version!r})"
        )

    ensemble_policy = load_ensemble_policy(bundle.channel)
    if bundle.ensemble_policy_version != ensemble_policy.policy_version:
        raise BundlePolicyMismatchError(
            f"bundle {bundle.bundle_id} ensemble_policy_version {bundle.ensemble_policy_version!r} does "
            f"not match the current ensemble_policy_{bundle.channel}.yaml ({ensemble_policy.policy_version!r})"
        )

    if bundle.reason_code_version != REASON_CODE_VERSION:
        raise BundlePolicyMismatchError(
            f"bundle {bundle.bundle_id} reason_code_version {bundle.reason_code_version!r} does not "
            f"match the current REASON_CODE_VERSION ({REASON_CODE_VERSION!r})"
        )

    return LocalYamlRuleProvider(), graph_policy, ensemble_policy


def _bundle_config_hash(bundle: ChannelModelBundleRecord) -> str:
    canonical = json.dumps(
        {
            "bundle_id": bundle.bundle_id,
            "bundle_version": bundle.bundle_version,
            "gbm_model_version": bundle.gbm_model_version,
            "lr_model_version": bundle.lr_model_version,
            "anomaly_model_version": bundle.anomaly_model_version,
            "preprocessing_artifact_version": bundle.preprocessing_artifact_version,
            "rule_set_version": bundle.rule_set_version,
            "graph_policy_version": bundle.graph_policy_version,
            "ensemble_policy_version": bundle.ensemble_policy_version,
            "reason_code_version": bundle.reason_code_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- real, MLflow-backed artifact loader (reviewed as code, not exercised) --------


class _MlflowBundleArtifactLoader:
    """Loads the OPERATIONAL bundle's actual GBM/LR/anomaly model objects
    and its preprocessing/anomaly-normalization artifacts from MLflow,
    using exactly the registered-model names and run-scoped artifact paths
    src.fraud_intel.models.training.train_channel_configured() writes them
    under. Not exercised by any unit test -- reviewed as code, same
    precedent as every other "_Mlflow*"/"_Postgres*Store" in this
    codebase."""

    def load(self, bundle: ChannelModelBundleRecord) -> LoadedChannelBundle:
        import mlflow

        from src.common.mlflow_setup import configure_mlflow
        from src.fraud_intel.models.anomaly import AnomalyNormalization
        from src.fraud_intel.models.preprocessing import ChannelPreprocessor

        if not bundle.evaluation_report_ref:
            raise BundleArtifactLoadError(
                f"bundle {bundle.bundle_id} has no evaluation_report_ref -- cannot locate its "
                "preprocessor.json/anomaly_normalization.json MLflow run artifacts"
            )
        try:
            report = json.loads(bundle.evaluation_report_ref)
            gbm_run_id = report["gbm_mlflow_run_id"]
            anomaly_run_id = report["anomaly_mlflow_run_id"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise BundleArtifactLoadError(
                f"bundle {bundle.bundle_id}'s evaluation_report_ref is missing the MLflow run "
                "id(s) needed to locate its preprocessor/anomaly-normalization artifacts"
            ) from exc

        configure_mlflow()
        gbm_model = mlflow.xgboost.load_model(f"models:/{registered_model_name(bundle.channel, 'gbm')}/{bundle.gbm_model_version}")
        lr_model = mlflow.sklearn.load_model(f"models:/{registered_model_name(bundle.channel, 'lr-shadow')}/{bundle.lr_model_version}")
        anomaly_model = mlflow.sklearn.load_model(
            f"models:/{registered_model_name(bundle.channel, 'anomaly')}/{bundle.anomaly_model_version}"
        )

        preprocessor = ChannelPreprocessor.from_json_dict(mlflow.artifacts.load_dict(f"runs:/{gbm_run_id}/preprocessor.json"))
        anomaly_normalization = AnomalyNormalization.from_json_dict(
            mlflow.artifacts.load_dict(f"runs:/{anomaly_run_id}/anomaly_normalization.json")
        )

        return LoadedChannelBundle(
            channel=bundle.channel,
            bundle_id=bundle.bundle_id,
            bundle_version=bundle.bundle_version,
            gbm_model=gbm_model,
            lr_model=lr_model,
            anomaly_model=anomaly_model,
            anomaly_normalization=anomaly_normalization,
            preprocessor=preprocessor,
            gbm_model_version=bundle.gbm_model_version,
            lr_model_version=bundle.lr_model_version,
            anomaly_model_version=bundle.anomaly_model_version,
            preprocessing_artifact_version=bundle.preprocessing_artifact_version,
            feature_schema_version=bundle.feature_schema_version,
            rule_set_version=bundle.rule_set_version,
            graph_policy_version=bundle.graph_policy_version,
            ensemble_policy_version=bundle.ensemble_policy_version,
            reason_code_version=bundle.reason_code_version,
        )


def create_default_bundle_artifact_loader() -> BundleArtifactLoader:
    return _MlflowBundleArtifactLoader()


# ---- real, Postgres-backed data access (reviewed as code, not exercised) ----------


class _PostgresScoringDataAccess:
    """Selects source alerts for `channel` that have no fraud_alerts row
    yet (idempotent-creation-aware: an alert already scored, even once, is
    never re-selected here -- a deliberate rescore is a separate,
    not-yet-built operator action, out of this corrective pass's scope),
    and assembles each one's FeatureComputationContext and resolved-fraud
    graph evidence from real history. Bounded batch size and lookback
    windows (module-level `_MAX_*` constants) keep a single dispatch run's
    query cost predictable; full incremental/paginated batching is later
    phase scope. Not exercised by any unit test -- reviewed as SQL."""

    def __init__(self, database: Optional[str] = None):
        self._database = database

    def list_pending(self, channel: str) -> Sequence[PendingScoringItem]:
        import psycopg2.extras

        from src.common.db import get_connection
        from src.fraud_intel.registry import get_channel_adapter

        conn = get_connection(self._database)
        try:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT sa.*, ce.event_id AS ce_event_id, ce.channel AS ce_channel,
                               ce.customer_id AS ce_customer_id, ce.account_id AS ce_account_id,
                               ce.event_timestamp AS ce_event_timestamp, ce.amount_minor_units AS ce_amount_minor_units,
                               ce.direction AS ce_direction, ce.device_id AS ce_device_id, ce.ip_address AS ce_ip_address,
                               ce.channel_payload AS ce_channel_payload, ce.scenario_id AS ce_scenario_id,
                               ce.schema_version AS ce_schema_version
                        FROM source_alerts sa
                        JOIN channel_events ce ON ce.event_id = sa.event_id
                        WHERE ce.channel = %s
                          AND NOT EXISTS (
                              SELECT 1 FROM fraud_alerts fa
                              WHERE fa.source_system = sa.source_system AND fa.source_alert_id = sa.source_alert_id
                          )
                        ORDER BY sa.source_alert_created_at ASC
                        LIMIT %s
                        """,
                        (channel, _MAX_PENDING_ALERTS_PER_RUN),
                    )
                    pending_rows = cur.fetchall()
                    payload_class = get_channel_adapter(channel).payload_class

                    items: list[PendingScoringItem] = []
                    for row in pending_rows:
                        event = FraudEvent(
                            event_id=row["ce_event_id"], channel=row["ce_channel"], customer_id=row["ce_customer_id"],
                            account_id=row["ce_account_id"], event_timestamp=row["ce_event_timestamp"],
                            amount_minor_units=row["ce_amount_minor_units"], direction=row["ce_direction"],
                            device_id=row["ce_device_id"], ip_address=row["ce_ip_address"],
                            scenario_id=row["ce_scenario_id"], schema_version=row["ce_schema_version"],
                            channel_payload=payload_class(**row["ce_channel_payload"]),
                        )
                        source_alert = SourceAlertContext(
                            source_alert_id=row["source_alert_id"], source_system=row["source_system"],
                            event_id=row["event_id"], source_alert_created_at=row["source_alert_created_at"],
                            source_rule_ids=row["source_rule_ids"], source_rule_version=row["source_rule_version"],
                            source_alert_score=row["source_alert_score"],
                            source_alert_reason_codes=row["source_alert_reason_codes"],
                            generation_run_id=row["generation_run_id"], dataset_version=row["dataset_version"],
                            created_at=row["created_at"],
                        )

                        cur.execute(
                            "SELECT * FROM channel_events WHERE customer_id = %s AND event_timestamp < %s "
                            "ORDER BY event_timestamp DESC LIMIT %s",
                            (event.customer_id, event.event_timestamp, _MAX_HISTORICAL_EVENTS_PER_ALERT),
                        )
                        # A customer's cross-channel history can include events from
                        # channels other than `channel` itself -- each row's OWN
                        # channel picks its own payload class, never the outer one.
                        historical_events = tuple(
                            FraudEvent(
                                event_id=h["event_id"], channel=h["channel"], customer_id=h["customer_id"],
                                account_id=h["account_id"], event_timestamp=h["event_timestamp"],
                                amount_minor_units=h["amount_minor_units"], direction=h["direction"],
                                device_id=h["device_id"], ip_address=h["ip_address"], scenario_id=h["scenario_id"],
                                schema_version=h["schema_version"],
                                channel_payload=get_channel_adapter(h["channel"]).payload_class(**h["channel_payload"]),
                            )
                            for h in cur.fetchall()
                        )

                        cur.execute(
                            "SELECT sa2.* FROM source_alerts sa2 JOIN channel_events ce2 ON ce2.event_id = sa2.event_id "
                            "WHERE ce2.customer_id = %s AND sa2.source_alert_created_at < %s "
                            "ORDER BY sa2.source_alert_created_at DESC LIMIT %s",
                            (event.customer_id, event.event_timestamp, _MAX_SOURCE_ALERT_HISTORY_PER_ALERT),
                        )
                        source_alert_history = tuple(
                            SourceAlertContext(
                                source_alert_id=h["source_alert_id"], source_system=h["source_system"],
                                event_id=h["event_id"], source_alert_created_at=h["source_alert_created_at"],
                                source_rule_ids=h["source_rule_ids"], source_rule_version=h["source_rule_version"],
                                source_alert_score=h["source_alert_score"],
                                source_alert_reason_codes=h["source_alert_reason_codes"],
                                generation_run_id=h["generation_run_id"], dataset_version=h["dataset_version"],
                                created_at=h["created_at"],
                            )
                            for h in cur.fetchall()
                        )

                        context = FeatureComputationContext(
                            current_event=event, historical_events=historical_events,
                            source_alert_history=source_alert_history, as_of_time=event.event_timestamp,
                        )

                        cur.execute(
                            "SELECT la.assessment_id, la.policy_version, la.basis_timestamp, la.resolved_label_source, "
                            "fa.customer_id, fa.account_id FROM label_assessments la "
                            "JOIN fraud_alerts fa ON fa.alert_id = la.alert_id "
                            "WHERE la.resolved_label = 'RESOLVED_FRAUD' AND la.eligibility_result = true "
                            "AND la.basis_timestamp < %s ORDER BY la.evaluated_at DESC LIMIT %s",
                            (event.event_timestamp, _MAX_RESOLVED_FRAUD_EVIDENCE_ROWS),
                        )
                        resolved_fraud_evidence: list[ResolvedFraudEntityEvidence] = []
                        for fraud_row in cur.fetchall():
                            common = dict(
                                label_assessment_id=str(fraud_row["assessment_id"]),
                                resolved_fraud_at=fraud_row["basis_timestamp"],
                                eligibility_policy_version=fraud_row["policy_version"],
                                label_source=fraud_row["resolved_label_source"],
                            )
                            resolved_fraud_evidence.append(
                                ResolvedFraudEntityEvidence(entity_type="customer", entity_id=fraud_row["customer_id"], **common)
                            )
                            resolved_fraud_evidence.append(
                                ResolvedFraudEntityEvidence(entity_type="account", entity_id=fraud_row["account_id"], **common)
                            )

                        items.append(
                            PendingScoringItem(
                                event=event, source_alert=source_alert, context=context,
                                resolved_fraud_evidence=tuple(resolved_fraud_evidence),
                            )
                        )
                    return items
        finally:
            conn.close()


def create_default_scoring_data_access(database: Optional[str] = None) -> ScoringDataAccess:
    return _PostgresScoringDataAccess(database)


# ---- dispatch orchestration (fully unit-testable with injected fakes) -------------


def score_channel(
    *,
    channel: str,
    lifecycle: RunLifecycle,
    data_access: ScoringDataAccess,
    get_operational_bundle: GetOperationalBundle,
    artifact_loader: BundleArtifactLoader,
    alert_queue_store: AlertQueueStore,
) -> dict:
    """One `fraud_score` pipeline run: finds the OPERATIONAL bundle for
    `channel`, loads and validates its pinned artifacts/policies, scores
    every currently-pending source alert through the existing
    score_and_record_alert() (idempotent alert/evidence persistence,
    catastrophic-failure handling all already built -- Phase 6), and
    records the run's outcome.

    A single alert's scoring failure is recorded (score_and_record_alert()
    already best-effort-persists a catastrophic evidence row for it) and
    counted as rejected, but does not abort the run -- one bad alert must
    not block scoring the rest of a batch. A STRUCTURAL failure (no
    OPERATIONAL bundle, a stale/mismatched policy, or an artifact-loading
    failure) aborts the whole run: fail_from_exception() records it and
    the original exception is always re-raised, never swallowed."""
    run = lifecycle.begin("fraud_score", trigger_source="cli")
    try:
        bundle_record = get_operational_bundle(channel)
        if bundle_record is None:
            raise NoOperationalBundleError(f"no OPERATIONAL bundle for channel {channel!r}")

        rule_provider, graph_policy, ensemble_policy = load_and_validate_pinned_policies(bundle_record)
        loaded_bundle = artifact_loader.load(bundle_record)
        config_hash = _bundle_config_hash(bundle_record)
        git_sha = get_git_sha()

        pending_items = data_access.list_pending(channel)
        processed = 0
        rejected = 0
        alert_summaries: list[dict] = []
        for item in pending_items:
            try:
                alert, evidence = score_and_record_alert(
                    event=item.event,
                    source_alert=item.source_alert,
                    context=item.context,
                    bundle=loaded_bundle,
                    rule_provider=rule_provider,
                    ensemble_policy=ensemble_policy,
                    graph_policy=graph_policy,
                    resolved_fraud_evidence=item.resolved_fraud_evidence,
                    config_hash=config_hash,
                    git_sha=git_sha,
                    store=alert_queue_store,
                )
                processed += 1
                alert_summaries.append(
                    {"alert_id": alert.alert_id, "status": alert.status, "priority_band": alert.initial_priority_band}
                )
            except Exception:
                rejected += 1
                log.error(
                    "fraud_score_alert_failed",
                    source_alert_id=str(item.source_alert.source_alert_id),
                    exc_info=True,
                )
    except Exception as exc:
        lifecycle.fail_from_exception(run.run_id, exc)
        raise

    lifecycle.succeed(
        run.run_id,
        records_processed=processed,
        records_rejected=rejected,
        model_version=loaded_bundle.gbm_model_version,
        artifacts={
            "bundle_id": bundle_record.bundle_id,
            "bundle_version": bundle_record.bundle_version,
            "rule_set_version": bundle_record.rule_set_version,
            "graph_policy_version": bundle_record.graph_policy_version,
            "ensemble_policy_version": bundle_record.ensemble_policy_version,
            "reason_code_version": bundle_record.reason_code_version,
        },
    )
    return {
        "run_id": run.run_id,
        "channel": channel,
        "bundle_id": bundle_record.bundle_id,
        "bundle_version": bundle_record.bundle_version,
        "records_processed": processed,
        "records_rejected": rejected,
        "alerts": alert_summaries,
    }
