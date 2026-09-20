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
from src.ingestion.consumer import run_configured
from src.ml.train import DEFAULT_FEATURES_PATH, train_configured
from src.processing.pipeline import DEFAULT_INPUT, run_pipeline_configured

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


# ---- argparse tree -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument("--json", action="store_true", help="emit machine-readable JSON on stdout")

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
