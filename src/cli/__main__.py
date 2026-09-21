"""Unified aidp operator CLI.

Every command either delegates to a Phase 3 configured workload entry point
(run_pipeline_configured / train_configured / run_configured, each with
trigger_source="cli"), or to RunLifecycle's existing get/list, or to
mlflow_setup's model-registry metadata lookup — no pipeline, training,
streaming, model-registry, or SQL logic is duplicated here.

Usage: python -m src.cli <command> ...
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from src.cli.output import (
    EXIT_OPERATIONAL_ERROR,
    EXIT_USER_ERROR,
    CLIUserError,
    emit_error,
    emit_result,
)
from src.common.config import get_app_settings, get_fraud_rules, get_settings
from src.common.logging import configure_logging, get_logger
from src.common.mlflow_setup import ModelAliasNotFoundError, get_model_alias_info
from src.control_plane.config import BatchRunConfig, StreamRunConfig, TrainingRunConfig
from src.control_plane.runs import PIPELINE_NAMES, STATUSES, RunLifecycle, RunNotFoundError
from src.fraud_intel.alerts.queue import (
    InvalidAlertTransitionError,
    create_default_alert_queue_store,
    record_disposition,
)
from src.fraud_intel.cli_data_access import generate_and_write
from src.fraud_intel.ensemble.policy import load_ensemble_policy
from src.fraud_intel.models.bundle import IncompleteBundleError, create_default_bundle_store
from src.fraud_intel.models.promotion import (
    BundlePromotionRaceError,
    BundleVerificationFailedError,
    create_default_bundle_promotion_store,
)
from src.ingestion.consumer import run_configured
from src.ml.train import DEFAULT_FEATURES_PATH, train_configured
from src.processing.pipeline import DEFAULT_INPUT, run_pipeline_configured

FRAUD_INTEL_ALL_CHANNELS = {"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"}
# Phase 7A: every channel now has a registered adapter
# (src.fraud_intel.registry) -- generate/train/score all route through it,
# so this widens from {"online_banking"} to every channel the registry
# actually knows about (kept as its own name/copy rather than a bare
# alias of FRAUD_INTEL_ALL_CHANNELS, so a future channel that is added to
# the Channel Literal but not yet registered can still be distinguished).
FRAUD_INTEL_IMPLEMENTED_CHANNELS = set(FRAUD_INTEL_ALL_CHANNELS)

# Not created at module level: structlog's cache_logger_on_first_use binds a
# logger proxy to whatever stream was active on its first use. main() calls
# configure_logging() itself, so the logger must be fetched fresh inside
# main() (after that call) each time — a module-level logger created once at
# import time would stay bound to the first configure_logging() call's
# stream for the rest of the process.


# ---- command handlers --------------------------------------------------------------
# Each returns a JSON-serialisable dict on success, or raises CLIUserError for a
# validation/not-found condition (mapped to exit code 2 by main()).


def _handle_config_validate(args: argparse.Namespace) -> dict:
    checks: dict[str, str] = {}
    try:
        get_settings()
        checks["settings"] = "ok"
        get_app_settings()
        checks["app_settings"] = "ok"
        get_fraud_rules()
        checks["fraud_rules"] = "ok"
        BatchRunConfig()
        checks["batch_run_config_defaults"] = "ok"
        TrainingRunConfig()
        checks["training_run_config_defaults"] = "ok"
        StreamRunConfig()
        checks["stream_run_config_defaults"] = "ok"
    except Exception as exc:
        raise CLIUserError(str(exc)) from exc
    return {"status": "valid", "checks": checks}


def _handle_pipeline_run_batch(args: argparse.Namespace) -> dict:
    try:
        config = BatchRunConfig(input_path=args.input, upload_to_minio=not args.no_upload)
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc
    return run_pipeline_configured(config, trigger_source="cli")


def _handle_train_run(args: argparse.Namespace) -> dict:
    try:
        config = TrainingRunConfig(features_path=args.features_path)
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc
    return train_configured(config, trigger_source="cli")


def _handle_stream_run(args: argparse.Namespace) -> dict:
    try:
        config = StreamRunConfig(duration=args.duration, from_beginning=args.from_beginning)
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc
    return run_configured(config, trigger_source="cli")


def _handle_run_show(args: argparse.Namespace) -> dict:
    lifecycle = RunLifecycle()
    try:
        record = lifecycle.get(args.run_id)
    except RunNotFoundError as exc:
        raise CLIUserError(f"run {args.run_id} not found") from exc
    return record.model_dump(mode="json")


def _handle_run_list(args: argparse.Namespace) -> dict:
    lifecycle = RunLifecycle()
    try:
        records = lifecycle.list(pipeline_name=args.pipeline_name, status=args.status, limit=args.limit)
    except ValueError as exc:
        raise CLIUserError(str(exc)) from exc
    return {"runs": [record.model_dump(mode="json") for record in records], "count": len(records)}


def _handle_model_show(args: argparse.Namespace) -> dict:
    settings = get_settings()
    try:
        return get_model_alias_info(settings.mlflow_model_name, args.alias)
    except ModelAliasNotFoundError as exc:
        raise CLIUserError(str(exc)) from exc
    # Any other MlflowException (connection failure, auth failure, server
    # error) is intentionally not caught here — it propagates to main()'s
    # generic handler as an operational failure (exit 3).


# ---- v1.3 fraud-intel / alerts handlers (Phase 6) -------------------------------------
#
# Every database-touching command below requires an explicit --database
# argument (guide section 25/Phase 6) — never a silent default to the
# production "aidp" database. Each handler checks this itself (rather than
# relying on argparse's own required=True) so a missing --database produces
# the exact same CLIUserError / emit_error() JSON-formatted response as
# every other CLI validation error, in both --json and human modes.


def _require_database(args: argparse.Namespace) -> str:
    if not getattr(args, "database", None):
        raise CLIUserError("--database is required for this command")
    return args.database


def _require_implemented_channel(channel: str) -> None:
    if channel not in FRAUD_INTEL_IMPLEMENTED_CHANNELS:
        raise CLIUserError(f"channel {channel!r} is not yet implemented (Phase 7A)")


def _handle_fraud_intel_generate(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    if args.reference_date is not None:
        try:
            reference_date = date.fromisoformat(args.reference_date)
        except ValueError as exc:
            raise CLIUserError(f"--reference-date must be an ISO YYYY-MM-DD date: {exc}") from exc
    else:
        reference_date = date.today()

    try:
        return generate_and_write(
            channel=args.channel, count=args.count, seed=args.seed, database=database, reference_date=reference_date
        )
    except ValueError as exc:
        raise CLIUserError(str(exc)) from exc


def _handle_fraud_intel_train(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    _require_implemented_channel(args.channel)
    if not args.generation_run_id:
        raise CLIUserError(
            "--generation-run-id is required for training -- training must never silently load every row ever "
            "generated for a channel"
        )

    from src.fraud_intel.cli_data_access import (
        GenerationRunChannelMismatchError,
        GenerationRunDatasetVersionError,
        UnknownGenerationRunError,
        load_channel_population,
    )
    from src.fraud_intel.config import ChannelTrainingRunConfig
    from src.fraud_intel.ensemble.policy import load_ensemble_policy
    from src.fraud_intel.graph.entity_graph import load_graph_policy
    from src.fraud_intel.models.training import train_channel_configured
    from src.fraud_intel.reason_codes.builder import REASON_CODE_VERSION
    from src.fraud_intel.rules.provider import _load_rule_set_config

    try:
        events, source_alerts, labels = load_channel_population(
            args.channel, database, generation_run_id=args.generation_run_id
        )
    except (UnknownGenerationRunError, GenerationRunChannelMismatchError, GenerationRunDatasetVersionError) as exc:
        raise CLIUserError(str(exc)) from exc

    try:
        config = ChannelTrainingRunConfig(channel=args.channel)
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc

    # Phase 7B Stage 3 corrective pass: pin the channel's CURRENT rule/
    # graph/ensemble/reason-code policy versions into the new candidate --
    # train_channel_configured() itself never loads or evaluates any of
    # these (Phase 5 decision 3, unchanged); wiring them here is what
    # makes the resulting bundle promotion-eligible (validate_promotion_
    # eligible()) instead of the Phase-4-style deliberately-incomplete
    # bundle every prior CLI train invocation produced.
    try:
        rule_set_version = _load_rule_set_config(args.channel).rule_set_version
        graph_policy_version = load_graph_policy(args.channel).graph_policy_version
        ensemble_policy_version = load_ensemble_policy(args.channel).policy_version
    except (FileNotFoundError, ValidationError) as exc:
        raise CLIUserError(f"failed to load current policy versions for channel {args.channel!r}: {exc}") from exc

    return train_channel_configured(
        config,
        database=database,
        trigger_source="cli",
        channel_events=events,
        source_alerts=source_alerts,
        synthetic_labels=labels,
        bundle_store=create_default_bundle_store(database),
        rule_set_version=rule_set_version,
        graph_policy_version=graph_policy_version,
        ensemble_policy_version=ensemble_policy_version,
        reason_code_version=REASON_CODE_VERSION,
    )


def _handle_fraud_intel_score(args: argparse.Namespace) -> dict:
    """Real dispatch (Phase 6 corrective pass; Phase 7B Stage 6 corrective
    pass adds required --generation-run-id scoping): finds the
    OPERATIONAL bundle, loads/validates its pinned artifacts and
    policies, scores every pending source alert -- scoped to the one
    explicit generation and relative to the current operational bundle,
    never silently channel-wide -- through the existing, already-tested
    alert queue, and records a real `fraud_score` RunLifecycle run. See
    src.fraud_intel.scoring.dispatch.score_channel for the full
    orchestration -- this handler only wires up its collaborators. A
    channel other than one of the seven registered channels is rejected
    by `_require_implemented_channel` below with a CLIUserError, not a
    NotImplementedError -- it is a supported-later, not a broken, state."""
    database = _require_database(args)
    _require_implemented_channel(args.channel)
    if not args.generation_run_id:
        raise CLIUserError(
            "--generation-run-id is required for scoring -- scoring must never silently score every pending "
            "alert for a channel across every generation ever run"
        )

    from src.fraud_intel.cli_data_access import (
        GenerationRunChannelMismatchError,
        GenerationRunDatasetVersionError,
        UnknownGenerationRunError,
    )
    from src.fraud_intel.scoring.dispatch import (
        BundlePolicyMismatchError,
        NoOperationalBundleError,
        create_default_bundle_artifact_loader,
        create_default_scoring_data_access,
        score_channel,
    )

    try:
        return score_channel(
            channel=args.channel,
            generation_run_id=args.generation_run_id,
            lifecycle=RunLifecycle(database),
            data_access=create_default_scoring_data_access(database),
            get_operational_bundle=lambda channel: _get_operational_bundle(channel, database),
            artifact_loader=create_default_bundle_artifact_loader(),
            alert_queue_store=create_default_alert_queue_store(database),
        )
    except (
        NoOperationalBundleError,
        BundlePolicyMismatchError,
        UnknownGenerationRunError,
        GenerationRunChannelMismatchError,
        GenerationRunDatasetVersionError,
    ) as exc:
        raise CLIUserError(str(exc)) from exc


def _evaluate_candidate_cold_start(args: argparse.Namespace, database: str) -> dict:
    """No OPERATIONAL bundle exists yet for this channel -- there is
    structurally no live scored population to evaluate against
    (fraud_alerts/alert_evidence/label_assessments are only ever created
    by a real scoring run, which itself requires an OPERATIONAL bundle).
    Evaluates the CANDIDATE using its own immutable, already-written
    training-time held-out test-split report instead -- real evidence,
    never fabricated, but explicitly labeled as such (never implied to be
    live operational evaluation). If the report is absent, malformed, or
    does not match the bundle it is attached to, evaluation fails and
    `promotion_gate_result` is never produced -- promotion remains
    blocked (promote is a separate command requiring its own separate
    approval regardless, but this function also never calls it)."""
    from src.fraud_intel.evaluation.cold_start import ColdStartReportError, evaluate_cold_start_promotion_gate, load_and_validate_cold_start_report

    promotion_store = create_default_bundle_promotion_store(database)
    try:
        candidate_bundle = promotion_store.get_bundle(args.channel, args.candidate_bundle_version)
    except LookupError as exc:
        raise CLIUserError(str(exc)) from exc

    try:
        report = load_and_validate_cold_start_report(candidate_bundle)
    except ColdStartReportError as exc:
        raise CLIUserError(str(exc)) from exc

    gate = evaluate_cold_start_promotion_gate(report)

    return {
        "channel": args.channel,
        "evaluation_mode": "candidate_training_holdout",
        "operational_bundle": None,
        "operational_evaluation": None,
        "candidate_bundle": {"bundle_id": candidate_bundle.bundle_id, "bundle_version": candidate_bundle.bundle_version},
        "candidate_evaluation": {
            "gbm_evaluation": report.gbm_evaluation,
            "lr_shadow_evaluation": report.lr_shadow_evaluation,
            "realized_split_fractions": report.realized_split_fractions,
            "split_class_counts": report.split_class_counts,
            "training_run_id": report.training_run_id,
            "dataset_version": report.dataset_version,
        },
        "rules_only_baseline": {
            "test_fraud_prevalence": gate["test_fraud_prevalence"],
            "note": (
                "Cold-start rules-only baseline is the held-out test split's own fraud prevalence -- no "
                "live rules-only re-scoring is possible before any real scoring run exists for this "
                "channel. A GBM pr_auc at or below this value indicates no ranking value over random "
                "guessing."
            ),
        },
        "shadow_candidate_comparison": None,
        "candidate_scoring_errors": [],
        "promotion_gate_result": gate,
        "disclaimer": (
            "Synthetic, held-out test-window evidence from the training run -- NOT live operational "
            "evaluation. Do not interpret as proof of real-world (Citizens Bank) production accuracy."
        ),
    }


def _evaluate_live(args: argparse.Namespace, database: str, operational_bundle, capacity) -> dict:
    """An OPERATIONAL bundle exists -- evaluates the real, live,
    source-alerted, RESOLVED population (src.fraud_intel.evaluation.
    cross_channel), with an OPTIONAL shadow-candidate comparison when
    `--candidate-bundle-version` is also given. Shadow comparison re-
    scores the exact same resolved population against the CANDIDATE
    bundle IN MEMORY, via the pure score_source_alert() path
    (src.fraud_intel.evaluation.shadow_candidate.score_candidate_shadow())
    -- never score_and_record_alert(), never a fraud_alerts/alert_evidence
    write, never a promotion."""
    from src.fraud_intel.cli_data_access import load_resolved_alert_outcomes
    from src.fraud_intel.evaluation.cross_channel import evaluate_channel

    outcomes = load_resolved_alert_outcomes(args.channel, database)
    try:
        operational_result = evaluate_channel(args.channel, outcomes, capacity=capacity, recall_target=args.recall_target)
    except ValueError as exc:
        raise CLIUserError(str(exc)) from exc

    operational_dump = operational_result.model_dump(mode="json")
    rules_only_baseline = operational_dump.pop("rules_only_baseline")

    result: dict = {
        "channel": args.channel,
        "evaluation_mode": "live_resolved_alerts",
        "operational_bundle": {"bundle_id": operational_bundle.bundle_id, "bundle_version": operational_bundle.bundle_version},
        "operational_evaluation": operational_dump,
        "candidate_bundle": None,
        "candidate_evaluation": None,
        "rules_only_baseline": rules_only_baseline,
        "shadow_candidate_comparison": None,
        "candidate_scoring_errors": [],
        "promotion_gate_result": None,
        "disclaimer": (
            "Live evaluation against real, scored, resolved alerts in this database -- still a local POC "
            "demonstration, not Citizens Bank production or regulatory evidence."
        ),
    }

    if args.candidate_bundle_version is not None:
        from src.fraud_intel.cli_data_access import load_resolved_alert_scoring_contexts
        from src.fraud_intel.evaluation.shadow_candidate import compare_operational_vs_candidate, score_candidate_shadow
        from src.fraud_intel.scoring.dispatch import (
            BundlePolicyMismatchError,
            create_default_bundle_artifact_loader,
            load_and_validate_pinned_policies,
        )

        promotion_store = create_default_bundle_promotion_store(database)
        try:
            candidate_bundle = promotion_store.get_bundle(args.channel, args.candidate_bundle_version)
        except LookupError as exc:
            raise CLIUserError(str(exc)) from exc
        result["candidate_bundle"] = {"bundle_id": candidate_bundle.bundle_id, "bundle_version": candidate_bundle.bundle_version}

        try:
            rule_provider, graph_policy, ensemble_policy = load_and_validate_pinned_policies(candidate_bundle)
        except BundlePolicyMismatchError as exc:
            raise CLIUserError(str(exc)) from exc
        loaded_candidate_bundle = create_default_bundle_artifact_loader().load(candidate_bundle)

        scoring_inputs = load_resolved_alert_scoring_contexts(args.channel, database)
        candidate_scores, candidate_errors = score_candidate_shadow(
            scoring_inputs, bundle=loaded_candidate_bundle, rule_provider=rule_provider,
            ensemble_policy=ensemble_policy, graph_policy=graph_policy,
        )

        try:
            comparison = compare_operational_vs_candidate(
                args.channel, outcomes, candidate_scores,
                operational_bundle_id=operational_bundle.bundle_id, operational_bundle_version=operational_bundle.bundle_version,
                candidate_bundle_id=candidate_bundle.bundle_id, candidate_bundle_version=candidate_bundle.bundle_version,
                capacity=capacity, recall_target=args.recall_target,
            )
        except ValueError as exc:
            raise CLIUserError(str(exc)) from exc
        result["shadow_candidate_comparison"] = comparison.model_dump(mode="json")
        result["candidate_scoring_errors"] = [e.model_dump(mode="json") for e in candidate_errors]

    return result


def _handle_fraud_intel_evaluate(args: argparse.Namespace) -> dict:
    """Phase 7A corrective pass: two distinct, clearly-labeled evaluation
    modes (`evaluation_mode` in the output) -- `candidate_training_holdout`
    (cold start: no OPERATIONAL bundle exists yet, so a candidate is
    evaluated from its own training-time held-out report instead) and
    `live_resolved_alerts` (an OPERATIONAL bundle exists; a candidate, if
    given, is additionally shadow-compared against it). `--capacity-mode`/
    `--capacity-value`/`--recall-target` are all required regardless of
    mode -- no implicit default anywhere (Phase 7A decision 6); the live
    mode uses them directly, the cold-start mode ignores them (its gate
    has no capacity/recall-target concept -- there is no live population
    to rank)."""
    database = _require_database(args)
    _require_implemented_channel(args.channel)

    from src.fraud_intel.evaluation.capacity import CountCapacity, FractionCapacity

    try:
        capacity = (
            CountCapacity(value=args.capacity_value)
            if args.capacity_mode == "count"
            else FractionCapacity(value=args.capacity_value)
        )
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc

    operational_bundle = _get_operational_bundle(args.channel, database)

    if operational_bundle is None and args.candidate_bundle_version is None:
        raise CLIUserError(
            f"no OPERATIONAL bundle for channel {args.channel!r} and no --candidate-bundle-version given -- "
            "nothing to evaluate"
        )

    if operational_bundle is None:
        return _evaluate_candidate_cold_start(args, database)
    return _evaluate_live(args, database, operational_bundle, capacity)


class _MlflowModelVersionVerifier:
    """Metadata-only -- never loads model weights, never registers/logs/
    aliases/tags anything. Phase 7B Stage 5 corrective pass: this
    verifier must be independently correct -- it calls configure_mlflow()
    itself, on every single call, rather than relying on some other
    command in the same process having already done so. Without this,
    MlflowClient() targets MLflow's own default tracking URI, not this
    project's configured one -- exactly the bug that made every real
    promotion attempt fail with an opaque MlflowException, discovered
    only on the first real promotion verification attempt. Follows the
    same configure_mlflow()-then-MlflowClient() order
    src.fraud_intel.scoring.dispatch._MlflowBundleArtifactLoader.load()
    already uses.

    When the caller supplies `expected_run_id` (from the candidate's own
    evaluation_report_ref), also confirms the registered version points
    at that exact MLflow run -- not merely that the version exists."""

    def verify(self, model_name: str, version: str, *, expected_run_id: str | None = None) -> bool:
        import mlflow

        from src.common.mlflow_setup import configure_mlflow

        configure_mlflow()
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(model_name, version)
        if model_version.status != "READY":
            return False
        if expected_run_id is not None and model_version.run_id != expected_run_id:
            return False
        return True


def _handle_fraud_intel_promote(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    from src.fraud_intel.evaluation.cold_start import ColdStartReportError
    from src.fraud_intel.models.promotion import ColdStartPromotionGateFailedError, promote_bundle

    store = create_default_bundle_promotion_store(database)
    try:
        result = promote_bundle(
            channel=args.channel,
            bundle_version=args.bundle_version,
            promoted_by=args.promoted_by,
            database=database,
            model_version_verifier=_MlflowModelVersionVerifier(),
            store=store,
        )
    except (
        IncompleteBundleError,
        BundleVerificationFailedError,
        BundlePromotionRaceError,
        ColdStartReportError,
        ColdStartPromotionGateFailedError,
    ) as exc:
        raise CLIUserError(str(exc)) from exc
    return result.model_dump(mode="json")


def _get_operational_bundle(channel: str, database: str):
    import psycopg2.extras

    from src.common.db import get_connection
    from src.fraud_intel.models.bundle import ChannelModelBundleRecord

    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM channel_model_bundles WHERE channel = %s AND status = 'OPERATIONAL'", (channel,)
                )
                row = cur.fetchone()
                return ChannelModelBundleRecord(**row) if row else None
    finally:
        conn.close()


def _handle_fraud_intel_model_show(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    bundle = _get_operational_bundle(args.channel, database)
    if bundle is None:
        raise CLIUserError(f"no OPERATIONAL bundle for channel {args.channel!r}")
    return bundle.model_dump(mode="json")


def _handle_fraud_intel_labels_assess(args: argparse.Namespace) -> dict:
    """Phase 7B Stage 0's explicit label-eligibility command -- scoring
    (`_handle_fraud_intel_score`) must never silently create a
    label_assessments row; this is the only command that does, and only
    when explicitly invoked. Wires the real Postgres label-basis loader
    (src.fraud_intel.cli_data_access.load_alert_label_bases) and the real
    Postgres assessment store into
    src.fraud_intel.labels.eligibility.assess_channel_labels(), which
    owns the actual `label_eligibility` RunLifecycle run, the per-alert
    assess_label() calls, and the idempotent append_if_changed() writes.

    Phase 7B Stage 7 corrective pass: requires --generation-run-id, same
    contract as train/score's own required-but-not-argparse-required flag
    (validated here, after --database, so a missing value is a clean
    CLIUserError/JSON error) -- label assessment must never silently
    consider every alert ever generated for a channel across every
    generation."""
    database = _require_database(args)
    _require_implemented_channel(args.channel)
    if not args.generation_run_id:
        raise CLIUserError(
            "--generation-run-id is required for label assessment -- label assessment must never silently "
            "consider every fraud_alerts row for a channel across every generation ever run"
        )

    from src.fraud_intel.cli_data_access import (
        GenerationRunChannelMismatchError,
        GenerationRunDatasetVersionError,
        UnknownGenerationRunError,
        load_alert_label_bases,
    )
    from src.fraud_intel.labels.eligibility import assess_channel_labels, create_default_label_assessment_store

    try:
        return assess_channel_labels(
            channel=args.channel,
            generation_run_id=args.generation_run_id,
            lifecycle=RunLifecycle(database),
            load_label_bases=lambda channel, generation_run_id: load_alert_label_bases(
                channel, database, generation_run_id=generation_run_id
            ),
            store=create_default_label_assessment_store(database),
        )
    except (UnknownGenerationRunError, GenerationRunChannelMismatchError, GenerationRunDatasetVersionError) as exc:
        raise CLIUserError(str(exc)) from exc


def _list_alerts(database: str, *, channel=None, status=None, priority_band=None):
    import psycopg2.extras

    from src.common.db import get_connection
    from src.fraud_intel.alerts.queue import FraudAlertRecord

    conn = get_connection(database)
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                clauses, params = [], []
                if channel:
                    clauses.append("channel = %s")
                    params.append(channel)
                if status:
                    clauses.append("status = %s")
                    params.append(status)
                if priority_band:
                    clauses.append("initial_priority_band = %s")
                    params.append(priority_band)
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                cur.execute(f"SELECT * FROM fraud_alerts {where} ORDER BY created_at DESC LIMIT 100", params)
                return [FraudAlertRecord(**row) for row in cur.fetchall()]
    finally:
        conn.close()


def _handle_alerts_list(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    alerts = _list_alerts(database, channel=args.channel, status=args.status, priority_band=args.priority_band)
    # Analyst-facing surface: FraudAlertRecord has no scenario_id/synthetic-
    # label field at all -- structurally, not just conventionally, safe.
    return {"alerts": [a.model_dump(mode="json") for a in alerts], "count": len(alerts)}


def _handle_alerts_show(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    store = create_default_alert_queue_store(database)
    try:
        alert = store.get_alert(args.alert_id)
    except Exception as exc:
        raise CLIUserError(f"alert {args.alert_id} not found") from exc
    latest_evidence = store.get_latest_evidence(args.alert_id)
    return {
        # initial_* fields (immutable, from first scoring) are clearly
        # distinguished from latest_evidence (current/latest scoring
        # result) -- Phase 6 decision 2.
        "alert": alert.model_dump(mode="json"),
        "latest_evidence": latest_evidence.model_dump(mode="json") if latest_evidence else None,
    }


def _handle_alerts_disposition(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    store = create_default_alert_queue_store(database)
    try:
        record = record_disposition(
            alert_id=args.alert_id, analyst_id=args.analyst_id, disposition=args.disposition, notes=args.notes, store=store
        )
    except InvalidAlertTransitionError as exc:
        raise CLIUserError(str(exc)) from exc
    return record.model_dump(mode="json")


# ---- argparse tree -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")

    # v1.3 (Phase 6): --database defaults to None, never to "aidp" -- each
    # handler calls _require_database(args) itself, which raises the same
    # CLIUserError every other validation failure raises (rather than
    # relying on argparse's own required=True, which would bypass
    # emit_error()'s JSON-mode formatting).
    database_parent = argparse.ArgumentParser(add_help=False)
    database_parent.add_argument("--database", default=None, help="explicit target database name (required)")

    parser = argparse.ArgumentParser(prog="aidp", description="Unified AiDP operator CLI.")
    top = parser.add_subparsers(dest="command", required=True)

    config_p = top.add_parser("config", help="configuration commands")
    config_sub = config_p.add_subparsers(dest="config_action", required=True)
    validate_p = config_sub.add_parser(
        "validate", parents=[json_parent], help="validate configuration without starting infrastructure"
    )
    validate_p.set_defaults(handler=_handle_config_validate)

    pipeline_p = top.add_parser("pipeline", help="batch pipeline commands")
    pipeline_sub = pipeline_p.add_subparsers(dest="pipeline_action", required=True)
    pipeline_run_p = pipeline_sub.add_parser("run", help="run a pipeline")
    pipeline_run_sub = pipeline_run_p.add_subparsers(dest="pipeline_type", required=True)
    batch_p = pipeline_run_sub.add_parser(
        "batch", parents=[json_parent], help="run the batch RAW->CLEAN->CURATED->FEATURES pipeline"
    )
    batch_p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    batch_p.add_argument("--no-upload", action="store_true", help="skip uploading outputs to MinIO")
    batch_p.set_defaults(handler=_handle_pipeline_run_batch)

    train_p = top.add_parser("train", help="model training commands")
    train_sub = train_p.add_subparsers(dest="train_action", required=True)
    train_run_p = train_sub.add_parser(
        "run", parents=[json_parent], help="train and register the fraud-detection model"
    )
    train_run_p.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH, dest="features_path")
    train_run_p.set_defaults(handler=_handle_train_run)

    stream_p = top.add_parser("stream", help="streaming consumer commands")
    stream_sub = stream_p.add_subparsers(dest="stream_action", required=True)
    stream_run_p = stream_sub.add_parser("run", parents=[json_parent], help="consume and score live transactions")
    stream_run_p.add_argument("--duration", type=float, default=None, help="seconds to run; omit to run until Ctrl+C")
    stream_run_p.add_argument(
        "--from-beginning", action="store_true", dest="from_beginning", help="replay the topic from the earliest offset"
    )
    stream_run_p.set_defaults(handler=_handle_stream_run)

    run_p = top.add_parser("run", help="query recorded pipeline/training/stream runs")
    run_sub = run_p.add_subparsers(dest="run_action", required=True)
    show_p = run_sub.add_parser("show", parents=[json_parent], help="show one run by id")
    show_p.add_argument("run_id", type=int)
    show_p.set_defaults(handler=_handle_run_show)
    list_p = run_sub.add_parser("list", parents=[json_parent], help="list recent runs")
    list_p.add_argument("--limit", type=int, default=10)
    list_p.add_argument("--pipeline-name", dest="pipeline_name", default=None, choices=sorted(PIPELINE_NAMES))
    list_p.add_argument("--status", default=None, choices=sorted(STATUSES))
    list_p.set_defaults(handler=_handle_run_list)

    model_p = top.add_parser("model", help="model registry commands")
    model_sub = model_p.add_subparsers(dest="model_action", required=True)
    model_show_p = model_sub.add_parser(
        "show", parents=[json_parent], help="show metadata for a registered model alias"
    )
    model_show_p.add_argument("alias")
    model_show_p.set_defaults(handler=_handle_model_show)

    # ---- v1.3 fraud-intel / alerts (Phase 6) -----------------------------------------

    fraud_intel_p = top.add_parser("fraud-intel", help="v1.3 fraud-intelligence commands")
    fraud_intel_sub = fraud_intel_p.add_subparsers(dest="fraud_intel_action", required=True)

    fi_generate_p = fraud_intel_sub.add_parser(
        "generate", parents=[json_parent, database_parent], help="generate synthetic channel events/source alerts"
    )
    fi_generate_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    fi_generate_p.add_argument("--count", type=int, required=True)
    fi_generate_p.add_argument("--seed", type=int, default=42)
    fi_generate_p.add_argument(
        "--reference-date", dest="reference_date", default=None,
        help="ISO YYYY-MM-DD; omit to default to today (not reproducible across days)",
    )
    fi_generate_p.set_defaults(handler=_handle_fraud_intel_generate)

    fi_train_p = fraud_intel_sub.add_parser(
        "train", parents=[json_parent, database_parent], help="train a candidate channel model bundle"
    )
    fi_train_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_IMPLEMENTED_CHANNELS))
    # Not argparse `required=True` -- validated inside the handler (after
    # --database) so a missing value is a clean CLIUserError/JSON error,
    # the same contract every other required-but-optionally-typed
    # fraud-intel flag follows, not argparse's own unformatted usage exit.
    fi_train_p.add_argument("--generation-run-id", dest="generation_run_id", default=None)
    fi_train_p.set_defaults(handler=_handle_fraud_intel_train)

    fi_score_p = fraud_intel_sub.add_parser(
        "score", parents=[json_parent, database_parent], help="score pending source alerts for a channel"
    )
    # All 7 channels are accepted at the argparse level (unlike train/
    # evaluate) so an unsupported channel reaches _require_implemented_channel
    # and produces a clean, JSON-formatted CLIUserError -- not argparse's own
    # unformatted usage error.
    fi_score_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    # Not argparse `required=True` -- validated inside the handler (after
    # --database), matching train's own --generation-run-id contract: a
    # missing value is a clean CLIUserError/JSON error, not argparse's own
    # unformatted usage exit.
    fi_score_p.add_argument("--generation-run-id", dest="generation_run_id", default=None)
    fi_score_p.set_defaults(handler=_handle_fraud_intel_score)

    fi_evaluate_p = fraud_intel_sub.add_parser(
        "evaluate", parents=[json_parent, database_parent],
        help="real per-channel evaluation (operational + rules-only baseline + optional shadow-candidate comparison)",
    )
    fi_evaluate_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_IMPLEMENTED_CHANNELS))
    # No implicit default anywhere (Phase 7A decision 6) -- capacity mode/
    # value and the recall target must always be explicitly supplied.
    fi_evaluate_p.add_argument("--capacity-mode", required=True, choices=("count", "fraction"), dest="capacity_mode")
    fi_evaluate_p.add_argument("--capacity-value", required=True, type=float, dest="capacity_value")
    fi_evaluate_p.add_argument("--recall-target", required=True, type=float, dest="recall_target")
    fi_evaluate_p.add_argument(
        "--candidate-bundle-version", type=int, default=None, dest="candidate_bundle_version",
        help="optional CANDIDATE bundle_version to shadow-compare against the OPERATIONAL bundle",
    )
    fi_evaluate_p.set_defaults(handler=_handle_fraud_intel_evaluate)

    fi_promote_p = fraud_intel_sub.add_parser(
        "promote", parents=[json_parent, database_parent], help="promote a complete, verified candidate bundle"
    )
    fi_promote_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    fi_promote_p.add_argument("--bundle-version", type=int, required=True, dest="bundle_version")
    fi_promote_p.add_argument("--promoted-by", required=True, dest="promoted_by")
    fi_promote_p.set_defaults(handler=_handle_fraud_intel_promote)

    fi_model_p = fraud_intel_sub.add_parser("model", help="v1.3 channel-bundle model registry commands")
    fi_model_sub = fi_model_p.add_subparsers(dest="fraud_intel_model_action", required=True)
    fi_model_show_p = fi_model_sub.add_parser(
        "show", parents=[json_parent, database_parent], help="show the OPERATIONAL channel model bundle"
    )
    fi_model_show_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    fi_model_show_p.set_defaults(handler=_handle_fraud_intel_model_show)

    fi_labels_p = fraud_intel_sub.add_parser("labels", help="v1.3 explicit label-eligibility commands")
    fi_labels_sub = fi_labels_p.add_subparsers(dest="fraud_intel_labels_action", required=True)
    fi_labels_assess_p = fi_labels_sub.add_parser(
        "assess", parents=[json_parent, database_parent],
        help="run the versioned label-eligibility policy over a channel's fraud_alerts and append label_assessments",
    )
    fi_labels_assess_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    # Not argparse `required=True` -- validated inside the handler (after
    # --database), matching train's/score's own --generation-run-id
    # contract: a missing value is a clean CLIUserError/JSON error, not
    # argparse's own unformatted usage exit.
    fi_labels_assess_p.add_argument("--generation-run-id", dest="generation_run_id", default=None)
    fi_labels_assess_p.set_defaults(handler=_handle_fraud_intel_labels_assess)

    alerts_p = top.add_parser("alerts", help="analyst alert review commands")
    alerts_sub = alerts_p.add_subparsers(dest="alerts_action", required=True)

    alerts_list_p = alerts_sub.add_parser("list", parents=[json_parent, database_parent], help="list alerts")
    alerts_list_p.add_argument("--channel", default=None, choices=sorted(FRAUD_INTEL_ALL_CHANNELS))
    alerts_list_p.add_argument("--status", default=None, choices=("OPEN", "IN_REVIEW", "CLOSED"))
    alerts_list_p.add_argument("--priority-band", dest="priority_band", default=None, choices=("LOW", "MEDIUM", "HIGH"))
    alerts_list_p.set_defaults(handler=_handle_alerts_list)

    alerts_show_p = alerts_sub.add_parser("show", parents=[json_parent, database_parent], help="show one alert with its latest evidence")
    alerts_show_p.add_argument("alert_id", type=int)
    alerts_show_p.set_defaults(handler=_handle_alerts_show)

    alerts_disposition_p = alerts_sub.add_parser(
        "disposition", parents=[json_parent, database_parent], help="record an analyst disposition"
    )
    alerts_disposition_p.add_argument("alert_id", type=int)
    alerts_disposition_p.add_argument("--analyst-id", required=True, dest="analyst_id")
    alerts_disposition_p.add_argument(
        "--disposition", required=True, choices=("CONFIRMED_FRAUD", "CONFIRMED_LEGITIMATE", "NEEDS_MORE_INFO", "ESCALATED")
    )
    alerts_disposition_p.add_argument("--notes", default=None)
    alerts_disposition_p.set_defaults(handler=_handle_alerts_disposition)

    return parser


def main(argv: list[str] | None = None) -> None:
    # force=True: the CLI owns its whole process from startup, so it's the
    # one caller that should guarantee a clean, deterministic single
    # handler bound to stderr, regardless of anything imported before it.
    configure_logging("cli", force=True)
    log = get_logger(__name__)
    parser = build_parser()
    args = parser.parse_args(argv)
    json_mode = getattr(args, "json", False)

    try:
        result = args.handler(args)
    except CLIUserError as exc:
        emit_error(exc, json_mode=json_mode, exit_code=EXIT_USER_ERROR)
        return
    except Exception as exc:
        log.error("cli_command_failed", exc_info=True)
        emit_error(exc, json_mode=json_mode, exit_code=EXIT_OPERATIONAL_ERROR)
        return

    emit_result(result, json_mode=json_mode)


if __name__ == "__main__":
    main()
