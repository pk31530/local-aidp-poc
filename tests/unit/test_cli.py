import inspect
import json
import subprocess
from datetime import datetime, timezone

import pytest
from mlflow.exceptions import MlflowException

from src.cli import __main__ as cli_main
from src.common.config import PROJECT_ROOT
from src.control_plane.config import BatchRunConfig, StreamRunConfig, TrainingRunConfig
from src.control_plane.runs import RunLifecycle as RealRunLifecycle
from src.control_plane.runs import RunRecord


class _FakeRunStore:
    """In-memory stand-in for _PostgresRunStore — no database touched, same
    contract used by tests/unit/test_control_plane_runs.py."""

    def __init__(self):
        self.rows: dict[int, dict] = {}
        self._next_id = 1

    def insert(self, row):
        run_id = self._next_id
        self._next_id += 1
        full = {
            "run_id": run_id,
            "trigger_source": None,
            "git_sha": None,
            "config_snapshot": None,
            "config_hash": None,
            "dataset_version": None,
            "model_version": None,
            "error_type": None,
            "error_message": None,
            "started_at": datetime.now(timezone.utc),
            "heartbeat_at": None,
            "completed_at": None,
            **row,
        }
        self.rows[run_id] = full
        return RunRecord(**full)

    def compare_and_set(self, run_id, allowed_from, updates):
        current = self.rows.get(run_id)
        if current is None or current["status"] not in allowed_from:
            return None
        current.update(updates)
        return RunRecord(**current)

    def get(self, run_id):
        row = self.rows.get(run_id)
        return RunRecord(**row) if row else None

    def list(self, *, pipeline_name=None, status=None, limit=50):
        rows = list(self.rows.values())
        if pipeline_name is not None:
            rows = [r for r in rows if r["pipeline_name"] == pipeline_name]
        if status is not None:
            rows = [r for r in rows if r["status"] == status]
        return [RunRecord(**r) for r in rows[:limit]]


# ---- help text: top-level and every nested subcommand ------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["config", "--help"],
        ["config", "validate", "--help"],
        ["pipeline", "--help"],
        ["pipeline", "run", "--help"],
        ["pipeline", "run", "batch", "--help"],
        ["train", "--help"],
        ["train", "run", "--help"],
        ["stream", "--help"],
        ["stream", "run", "--help"],
        ["run", "--help"],
        ["run", "show", "--help"],
        ["run", "list", "--help"],
        ["model", "--help"],
        ["model", "show", "--help"],
    ],
)
def test_help_exits_zero_with_usage_text(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)
    assert exc_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_missing_required_subcommand_exits_2():
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["config"])
    assert exc_info.value.code == 2


# ---- config validate -----------------------------------------------------------------


def test_config_validate_success(capsys):
    cli_main.main(["config", "validate", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid"
    assert all(v == "ok" for v in result["checks"].values())


def test_config_validate_never_dumps_raw_settings(capsys):
    """config validate must report pass/fail only, never the Settings
    object itself (which holds real fields like postgres_password)."""
    cli_main.main(["config", "validate", "--json"])
    out = capsys.readouterr().out
    assert "aidp_local_only" not in out  # the default local postgres_password value
    assert "postgres_password" not in out


def test_config_validate_failure_is_user_error(monkeypatch, capsys):
    def _broken():
        raise RuntimeError("bad env")

    monkeypatch.setattr(cli_main, "get_settings", _broken)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["config", "validate", "--json"])

    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"


# ---- dispatch to configured workload functions, trigger_source="cli" --------------


def test_pipeline_run_batch_dispatches_with_trigger_source_cli(monkeypatch, tmp_path):
    captured = {}

    def _spy(config, *, trigger_source):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(cli_main, "run_pipeline_configured", _spy)
    input_path = tmp_path / "x.csv"

    cli_main.main(["pipeline", "run", "batch", "--input", str(input_path), "--no-upload"])

    assert captured["trigger_source"] == "cli"
    assert captured["config"] == BatchRunConfig(input_path=input_path, upload_to_minio=False)


def test_train_run_dispatches_with_trigger_source_cli(monkeypatch, tmp_path):
    captured = {}

    def _spy(config, *, trigger_source):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(cli_main, "train_configured", _spy)
    features_path = tmp_path / "features.parquet"

    cli_main.main(["train", "run", "--features-path", str(features_path)])

    assert captured["trigger_source"] == "cli"
    assert captured["config"] == TrainingRunConfig(features_path=features_path)


def test_stream_run_dispatches_with_trigger_source_cli(monkeypatch):
    captured = {}

    def _spy(config, *, trigger_source):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"ok": True}

    monkeypatch.setattr(cli_main, "run_configured", _spy)

    cli_main.main(["stream", "run", "--duration", "5", "--from-beginning"])

    assert captured["trigger_source"] == "cli"
    assert captured["config"] == StreamRunConfig(duration=5, from_beginning=True)


def test_stream_run_negative_duration_is_validation_error_not_dispatched(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main, "run_configured", lambda *a, **k: pytest.fail("must not dispatch on invalid config")
    )

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["stream", "run", "--duration", "-5", "--json"])

    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"


# ---- JSON output cleanliness and stderr logging ------------------------------------


def test_json_output_is_clean_single_line_and_logs_go_to_stderr(monkeypatch, capsys):
    def _fake_run_pipeline_configured(config, *, trigger_source):
        from src.common.logging import get_logger

        get_logger("fake.pipeline").info("fake_pipeline_started")
        return {"raw_valid": 5, "total_rejected": 0}

    monkeypatch.setattr(cli_main, "run_pipeline_configured", _fake_run_pipeline_configured)

    cli_main.main(["pipeline", "run", "batch", "--json"])

    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1  # exactly one line: the JSON object
    result = json.loads(captured.out)
    assert result == {"raw_valid": 5, "total_rejected": 0}
    assert "fake_pipeline_started" in captured.err


def test_human_output_is_readable_text_not_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main, "run_pipeline_configured", lambda config, *, trigger_source: {"raw_valid": 5, "nested": {"a": 1}}
    )

    cli_main.main(["pipeline", "run", "batch"])

    out = capsys.readouterr().out
    assert "raw_valid: 5" in out
    assert not out.strip().startswith("{")


# ---- operational errors: exit 3, safe message on stdout, traceback on stderr -------


def test_operational_error_exits_3_with_traceback_on_stderr(monkeypatch, capsys):
    def _boom(config, *, trigger_source):
        raise RuntimeError("db connection refused")

    monkeypatch.setattr(cli_main, "run_pipeline_configured", _boom)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["pipeline", "run", "batch", "--json"])

    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["error"] == "RuntimeError"
    assert "cli_command_failed" in captured.err
    assert "Traceback" in captured.err


def test_operational_error_message_is_credential_redacted(monkeypatch, capsys):
    def _boom(config, *, trigger_source):
        raise RuntimeError("could not connect to postgresql://aidp:hunter2@localhost/aidp")

    monkeypatch.setattr(cli_main, "run_pipeline_configured", _boom)

    with pytest.raises(SystemExit):
        cli_main.main(["pipeline", "run", "batch", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert "hunter2" not in result["message"]


def test_human_mode_error_goes_to_stderr_not_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "run_pipeline_configured", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["pipeline", "run", "batch"])

    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "boom" in captured.err


# ---- run show ------------------------------------------------------------------------


def test_run_show_not_found_is_user_error(monkeypatch, capsys):
    store = _FakeRunStore()
    monkeypatch.setattr(cli_main, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["run", "show", "999", "--json"])

    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert "not found" in result["message"]


def test_run_show_found_returns_record(monkeypatch, capsys):
    store = _FakeRunStore()
    lifecycle = RealRunLifecycle(store=store)
    run = lifecycle.begin("batch")
    monkeypatch.setattr(cli_main, "RunLifecycle", lambda *a, **k: lifecycle)

    cli_main.main(["run", "show", str(run.run_id), "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == run.run_id
    assert result["status"] == "RUNNING"


def test_run_show_output_contains_no_raw_secrets(monkeypatch, capsys):
    store = _FakeRunStore()
    lifecycle = RealRunLifecycle(store=store)
    run = lifecycle.begin("batch", config_snapshot={"postgres_dsn": "postgresql://u:p@h/db"})
    monkeypatch.setattr(cli_main, "RunLifecycle", lambda *a, **k: lifecycle)

    cli_main.main(["run", "show", str(run.run_id), "--json"])

    out = capsys.readouterr().out
    assert "postgresql://u:p@h/db" not in out
    assert "REDACTED" in out


# ---- run list: bounded limit --------------------------------------------------------


def test_run_list_out_of_range_limit_is_user_error(monkeypatch, capsys):
    store = _FakeRunStore()
    monkeypatch.setattr(cli_main, "RunLifecycle", lambda *a, **k: RealRunLifecycle(store=store))

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["run", "list", "--limit", "501", "--json"])

    assert exc_info.value.code == 2


def test_run_list_returns_bounded_results(monkeypatch, capsys):
    store = _FakeRunStore()
    lifecycle = RealRunLifecycle(store=store)
    lifecycle.begin("batch")
    lifecycle.begin("stream")
    monkeypatch.setattr(cli_main, "RunLifecycle", lambda *a, **k: lifecycle)

    cli_main.main(["run", "list", "--limit", "1", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 1
    assert len(result["runs"]) == 1


# ---- model show ------------------------------------------------------------------------


def test_model_show_success(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main,
        "get_model_alias_info",
        lambda model_name, alias: {
            "model_name": model_name,
            "alias": alias,
            "version": "7",
            "model_type": "xgboost",
            "run_id": "run-1",
        },
    )

    cli_main.main(["model", "show", "champion", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["version"] == "7"
    assert set(result.keys()) == {"model_name", "alias", "version", "model_type", "run_id"}


def test_model_show_unknown_alias_is_user_error(monkeypatch, capsys):
    def _raise(model_name, alias):
        raise MlflowException("alias not found")

    monkeypatch.setattr(cli_main, "get_model_alias_info", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["model", "show", "nope", "--json"])

    assert exc_info.value.code == 2


# ---- shell wrapper argument forwarding (safe: --help only, no real work) ----------


def test_scripts_aidp_sh_forwards_arguments():
    result = subprocess.run(
        ["./scripts/aidp.sh", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "usage: aidp" in result.stdout


# ---- legacy entry points remain valid ------------------------------------------------


def test_legacy_entry_points_unchanged_by_cli_addition():
    from src.ingestion.consumer import run
    from src.ml.train import train
    from src.processing.pipeline import run_pipeline

    assert list(inspect.signature(run_pipeline).parameters) == ["input_path", "upload_to_minio"]
    assert list(inspect.signature(train).parameters) == ["features_path"]
    assert list(inspect.signature(run).parameters) == [
        "duration",
        "from_beginning",
        "topic",
        "dlq_topic",
        "database",
        "group_id",
        "raw_bucket",
    ]
