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

FRAUD_INTEL_IMPLEMENTED_CHANNELS = {"online_banking"}
FRAUD_INTEL_ALL_CHANNELS = {"ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"}

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
    try:
        return generate_and_write(
            channel=args.channel, count=args.count, seed=args.seed, database=database, reference_date=date.today()
        )
    except ValueError as exc:
        raise CLIUserError(str(exc)) from exc


def _handle_fraud_intel_train(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    _require_implemented_channel(args.channel)
    from src.fraud_intel.cli_data_access import load_online_banking_population
    from src.fraud_intel.config import ChannelTrainingRunConfig
    from src.fraud_intel.models.training import train_channel_configured

    events, source_alerts, labels = load_online_banking_population(database)
    try:
        config = ChannelTrainingRunConfig(channel=args.channel)
    except ValidationError as exc:
        raise CLIUserError(str(exc)) from exc
    return train_channel_configured(
        config,
        trigger_source="cli",
        channel_events=events,
        source_alerts=source_alerts,
        synthetic_labels=labels,
        bundle_store=create_default_bundle_store(database),
    )


def _handle_fraud_intel_score(args: argparse.Namespace) -> dict:
    """Real alert/evidence persistence (Phase 6) is fully built and unit-
    tested (src.fraud_intel.alerts.queue). What is NOT yet built is real
    MLflow model-artifact loading into a LoadedChannelBundle -- that is
    Phase 7B scope, matching every other "_Postgres*Store" in this
    codebase, which is reviewed but never exercised until Phase 7B applies
    the migrations. This raises NotImplementedError (mapped to exit 3 by
    main()'s generic handler, not a user error) rather than pretending to
    score for real."""
    _require_database(args)
    _require_implemented_channel(args.channel)
    raise NotImplementedError(
        "aidp fraud-intel score requires loading real trained model artifacts from MLflow "
        "(Phase 7B scope, not yet implemented) -- the alert queue, evidence persistence, and "
        "scoring orchestrator this command would call are already built and unit-tested (Phase 5/6)"
    )


def _handle_fraud_intel_evaluate(args: argparse.Namespace) -> dict:
    """Guide/Phase 6 decision 11: a thin, explicitly self-identifying
    reference-channel-only surface -- never implies the complete
    cross-channel evaluation report, which is Phase 7A scope."""
    _require_database(args)
    _require_implemented_channel(args.channel)
    return {
        "scope": "reference_channel_phase6_only",
        "note": "This is a Phase 6 reference-channel-only evaluation surface, not the complete "
        "cross-channel evaluation report -- that is Phase 7A scope.",
        "channel": args.channel,
    }


class _MlflowModelVersionVerifier:
    """Metadata-only -- never loads model weights, mirrors
    src.common.mlflow_setup's existing pattern."""

    def verify(self, model_name: str, version: str) -> bool:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        client.get_model_version(model_name, version)
        return True


def _handle_fraud_intel_promote(args: argparse.Namespace) -> dict:
    database = _require_database(args)
    from src.fraud_intel.models.promotion import promote_bundle

    store = create_default_bundle_promotion_store(database)
    try:
        result = promote_bundle(
            channel=args.channel,
            bundle_version=args.bundle_version,
            promoted_by=args.promoted_by,
            model_version_verifier=_MlflowModelVersionVerifier(),
            store=store,
        )
    except (IncompleteBundleError, BundleVerificationFailedError, BundlePromotionRaceError) as exc:
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
    fi_generate_p.set_defaults(handler=_handle_fraud_intel_generate)

    fi_train_p = fraud_intel_sub.add_parser(
        "train", parents=[json_parent, database_parent], help="train a candidate channel model bundle"
    )
    fi_train_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_IMPLEMENTED_CHANNELS))
    fi_train_p.set_defaults(handler=_handle_fraud_intel_train)

    fi_score_p = fraud_intel_sub.add_parser(
        "score", parents=[json_parent, database_parent], help="score pending source alerts for a channel"
    )
    fi_score_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_IMPLEMENTED_CHANNELS))
    fi_score_p.set_defaults(handler=_handle_fraud_intel_score)

    fi_evaluate_p = fraud_intel_sub.add_parser(
        "evaluate", parents=[json_parent, database_parent], help="reference-channel evaluation surface (Phase 6 scope only)"
    )
    fi_evaluate_p.add_argument("--channel", required=True, choices=sorted(FRAUD_INTEL_IMPLEMENTED_CHANNELS))
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
