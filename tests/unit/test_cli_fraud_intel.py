"""Phase 6: CLI surfaces for fraud-intel/alerts. No database, Docker,
Kafka, MinIO, or real MLflow contact anywhere in this file -- every
handler's downstream store/orchestration call is monkeypatched, same
convention as tests/unit/test_cli.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.cli import __main__ as cli_main
from src.fraud_intel.alerts.queue import (
    AnalystDispositionRecord,
    FraudAlertRecord,
    InvalidAlertTransitionError,
    _FakeAlertQueueStore,
)
from src.fraud_intel.models.bundle import ChannelModelBundleRecord, IncompleteBundleError
from src.fraud_intel.models.promotion import BundlePromotionRaceError, BundleVerificationFailedError

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _alert(**overrides) -> FraudAlertRecord:
    import uuid

    base = dict(
        alert_id=1,
        event_id=uuid.uuid4(),
        source_alert_id=uuid.uuid4(),
        source_system="core_fraud_engine",
        channel="online_banking",
        customer_id="cust-1",
        account_id="acct-1",
        amount_minor_units=1000,
        initial_operational_priority_score=0.9,
        initial_priority_band="HIGH",
        initial_ensemble_policy_version="v1",
        status="OPEN",
        created_at=T0,
    )
    base.update(overrides)
    return FraudAlertRecord(**base)


def _bundle(**overrides) -> ChannelModelBundleRecord:
    base = dict(bundle_id=1, channel="online_banking", bundle_version=1, status="OPERATIONAL", created_at=T0)
    base.update(overrides)
    return ChannelModelBundleRecord(**base)


# ---- --database is required for every DB-touching command --------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--json"],
        ["fraud-intel", "train", "--channel", "online_banking", "--json"],
        ["fraud-intel", "score", "--channel", "online_banking", "--json"],
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "10", "--recall-target", "0.8", "--json"],
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "a1", "--json"],
        ["fraud-intel", "model", "show", "--channel", "online_banking", "--json"],
        ["alerts", "list", "--json"],
        ["alerts", "show", "1", "--json"],
        ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--json"],
    ],
)
def test_missing_database_is_a_clean_cli_user_error(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(argv)
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "--database" in result["message"]


# ---- fraud-intel generate ------------------------------------------------------------


def test_generate_dispatches_to_generate_and_write(monkeypatch, capsys):
    captured = {}

    def _spy(*, channel, count, seed, database, reference_date):
        captured.update(channel=channel, count=count, seed=seed, database=database)
        return {"channel": channel, "count": count, "generation_run_id": "genrun-1", "dataset_version": "dsv-1"}

    monkeypatch.setattr(cli_main, "generate_and_write", _spy)

    cli_main.main(
        ["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--seed", "7", "--database", "aidp_test", "--json"]
    )

    assert captured == {"channel": "online_banking", "count": 5, "seed": 7, "database": "aidp_test"}
    result = json.loads(capsys.readouterr().out)
    assert result["generation_run_id"] == "genrun-1"


def test_generate_value_error_is_a_cli_user_error(monkeypatch, capsys):
    def _raise(**kwargs):
        raise ValueError("unknown channel")

    monkeypatch.setattr(cli_main, "generate_and_write", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "generate", "--channel", "online_banking", "--count", "5", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- fraud-intel train ----------------------------------------------------------------


def test_train_loads_population_and_dispatches_to_train_channel_configured(monkeypatch, capsys):
    captured = {}

    def _fake_load_population(channel, database):
        captured["channel"] = channel
        captured["database"] = database
        return [], [], []

    def _fake_train_channel_configured(config, *, trigger_source, channel_events, source_alerts, synthetic_labels, bundle_store):
        captured["config"] = config
        captured["trigger_source"] = trigger_source
        return {"bundle_id": 9, "status": "CANDIDATE"}

    fake_bundle_store = object()

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", _fake_load_population)
    monkeypatch.setattr("src.fraud_intel.models.training.train_channel_configured", _fake_train_channel_configured)
    monkeypatch.setattr(cli_main, "create_default_bundle_store", lambda database: fake_bundle_store)

    cli_main.main(["fraud-intel", "train", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    assert captured["channel"] == "online_banking"
    assert captured["database"] == "aidp_test"
    assert captured["config"].channel == "online_banking"
    assert captured["trigger_source"] == "cli"
    result = json.loads(capsys.readouterr().out)
    assert result == {"bundle_id": 9, "status": "CANDIDATE"}


@pytest.mark.parametrize("channel", sorted({"ach", "wire", "mobile_deposit", "atm", "debit_card", "p2p"}))
def test_train_accepts_every_phase7a_registered_channel(monkeypatch, capsys, channel):
    """Phase 7A: FRAUD_INTEL_IMPLEMENTED_CHANNELS is widened to all 7 --
    every non-reference channel now reaches the shared training path
    instead of being rejected by _require_implemented_channel."""
    def _fake_load_population(channel, database):
        return [], [], []

    def _fake_train_channel_configured(config, *, trigger_source, channel_events, source_alerts, synthetic_labels, bundle_store):
        return {"bundle_id": 9, "status": "CANDIDATE", "channel": config.channel}

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_channel_population", _fake_load_population)
    monkeypatch.setattr("src.fraud_intel.models.training.train_channel_configured", _fake_train_channel_configured)
    monkeypatch.setattr(cli_main, "create_default_bundle_store", lambda database: object())

    cli_main.main(["fraud-intel", "train", "--channel", channel, "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == channel


# ---- fraud-intel score: real reference-channel dispatch (Phase 6 corrective pass) -----


def test_score_dispatches_to_score_channel(monkeypatch, capsys):
    captured = {}

    def _fake_score_channel(*, channel, lifecycle, data_access, get_operational_bundle, artifact_loader, alert_queue_store):
        captured.update(
            channel=channel, lifecycle=lifecycle, data_access=data_access,
            get_operational_bundle=get_operational_bundle, artifact_loader=artifact_loader,
            alert_queue_store=alert_queue_store,
        )
        return {"run_id": 1, "channel": channel, "bundle_id": 2, "bundle_version": 1, "records_processed": 3, "records_rejected": 0, "alerts": []}

    fake_data_access = object()
    fake_artifact_loader = object()
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: fake_data_access)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: fake_artifact_loader)
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    assert captured["channel"] == "online_banking"
    assert captured["data_access"] is fake_data_access
    assert captured["artifact_loader"] is fake_artifact_loader
    assert captured["get_operational_bundle"]("online_banking") == _bundle()
    result = json.loads(capsys.readouterr().out)
    assert result == {"run_id": 1, "channel": "online_banking", "bundle_id": 2, "bundle_version": 1, "records_processed": 3, "records_rejected": 0, "alerts": []}


def test_score_accepts_a_non_reference_channel_now_that_all_seven_are_registered(monkeypatch, capsys):
    """Phase 7A: 'wire' used to be rejected by _require_implemented_channel
    (Phase 6 corrective pass, reference channel only) -- FRAUD_INTEL_IMPLEMENTED_CHANNELS
    is now widened to all 7 registered channels, so this reaches score_channel()."""
    def _fake_score_channel(*, channel, **kwargs):
        return {"run_id": 1, "channel": channel, "bundle_id": 1, "bundle_version": 1, "records_processed": 0, "records_rejected": 0, "alerts": []}

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(["fraud-intel", "score", "--channel", "wire", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == "wire"


def test_score_unregistered_channel_name_is_rejected_at_the_argparse_level(capsys):
    """A genuinely unsupported/unknown channel name (not one of the 7
    registered channels at all) still fails cleanly -- argparse's own
    --channel choices reject it before any handler runs, same as every
    other fixed-choice CLI argument in this codebase (--status,
    --priority-band, ...)."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "not_a_real_channel", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("exc_cls_path", [
    "src.fraud_intel.scoring.dispatch.NoOperationalBundleError",
    "src.fraud_intel.scoring.dispatch.BundlePolicyMismatchError",
])
def test_score_maps_domain_errors_to_cli_user_error(monkeypatch, capsys, exc_cls_path):
    import importlib

    module_path, cls_name = exc_cls_path.rsplit(".", 1)
    exc_cls = getattr(importlib.import_module(module_path), cls_name)

    def _raise(**kwargs):
        raise exc_cls("refused")

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _raise)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_score_clean_json_stdout_and_logs_go_to_stderr(monkeypatch, capsys):
    def _fake_score_channel(**kwargs):
        from src.common.logging import get_logger

        get_logger("fake.fraud_score").info("fake_fraud_score_started")
        return {"run_id": 1, "channel": "online_banking", "bundle_id": 2, "bundle_version": 1, "records_processed": 0, "records_rejected": 0, "alerts": []}

    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.score_channel", _fake_score_channel)
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_scoring_data_access", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: object())
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(["fraud-intel", "score", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line]
    assert len(stdout_lines) == 1
    json.loads(stdout_lines[0])  # exactly one clean JSON object, nothing else
    assert "fake_fraud_score_started" not in captured.out
    assert "fake_fraud_score_started" in captured.err


# ---- Phase 7B readiness: no permanent-stub handler remains -------------------------------


def test_no_phase7b_required_handler_raises_not_implemented_error():
    """AST-based (not substring-based) so a docstring/comment that merely
    MENTIONS NotImplementedError -- e.g. explaining that a handler no
    longer raises it -- can never produce a false positive (the same class
    of false positive already hit twice elsewhere in Phase 6:
    tests/unit/test_fraud_intel_migration_004.py)."""
    import ast
    import inspect

    for handler in (
        cli_main._handle_fraud_intel_generate,
        cli_main._handle_fraud_intel_train,
        cli_main._handle_fraud_intel_score,
        cli_main._handle_fraud_intel_evaluate,
        cli_main._handle_fraud_intel_promote,
        cli_main._handle_fraud_intel_model_show,
    ):
        tree = ast.parse(inspect.getsource(handler))
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                func = node.exc.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                assert name != "NotImplementedError", f"{handler.__name__} still raises NotImplementedError"


def test_no_phase7b_required_handler_returns_a_phase_scope_placeholder_dict():
    """The old Phase 6 evaluate() stub returned a literal dict containing
    a "scope"/"reference_channel_phaseN_only" placeholder marker instead
    of doing real work -- proves no Phase 7B-required handler still
    contains that pattern (a dict literal with a 'scope' key) anywhere in
    its own source."""
    import ast
    import inspect

    for handler in (
        cli_main._handle_fraud_intel_generate,
        cli_main._handle_fraud_intel_train,
        cli_main._handle_fraud_intel_score,
        cli_main._handle_fraud_intel_evaluate,
        cli_main._handle_fraud_intel_promote,
        cli_main._handle_fraud_intel_model_show,
    ):
        tree = ast.parse(inspect.getsource(handler))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
                assert "scope" not in keys, f"{handler.__name__} still returns a 'scope' placeholder dict: {keys}"


# ---- fraud-intel evaluate: real Phase 7A evaluation (corrective pass) -----------------


def _outcome_fixtures():
    import uuid

    from src.fraud_intel.evaluation.cross_channel import AlertOutcome

    ids = [uuid.uuid4() for _ in range(4)]
    labels = ["RESOLVED_FRAUD", "RESOLVED_FRAUD", "RESOLVED_LEGITIMATE", "RESOLVED_LEGITIMATE"]
    scores = [0.9, 0.7, 0.3, 0.1]
    bands = ["HIGH", "MEDIUM", "LOW", "LOW"]
    return [
        AlertOutcome(
            source_alert_id=ids[i], channel="online_banking", event_timestamp=T0, operational_priority_score=scores[i],
            baseline_priority_score=scores[i], priority_band=bands[i], resolved_label=labels[i],
        )
        for i in range(4)
    ]


def test_evaluate_returns_operational_and_baseline_sections_without_a_candidate(monkeypatch, capsys):
    outcomes = _outcome_fixtures()
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["channel"] == "online_banking"
    assert result["operational_bundle"] == {"bundle_id": _bundle().bundle_id, "bundle_version": _bundle().bundle_version}
    assert "rules_only_baseline" not in result["operational_evaluation"]  # clearly separated out, not nested
    assert result["rules_only_baseline"]["pr_auc"]["status"] in ("ok", "non_computable_single_class", "non_computable_empty")
    assert result["shadow_candidate_comparison"] is None
    assert result["operational_evaluation"]["total_source_alerts"] == 4


def test_evaluate_includes_shadow_candidate_comparison_when_requested(monkeypatch, capsys):
    """CLI-level test: mocks the real I/O boundaries (bundle lookup,
    pinned-policy loading, artifact loading, population loading, and the
    actual in-memory scoring call) and lets the CLI's own orchestration
    and the real compare_operational_vs_candidate() run."""
    from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScore

    outcomes = _outcome_fixtures()
    candidate_scores = [
        CandidateShadowScore(
            source_alert_id=o.source_alert_id, channel="online_banking", event_timestamp=T0,
            candidate_priority_score=o.operational_priority_score, candidate_priority_band=o.priority_band,
        )
        for o in outcomes
    ]
    candidate_bundle = _bundle(bundle_id=5, bundle_version=3)

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            assert channel == "online_banking"
            assert bundle_version == 3
            return candidate_bundle

    class _FakeArtifactLoader:
        def load(self, bundle_record):
            return "fake-loaded-candidate-bundle"

    def _fake_score_candidate_shadow(scoring_inputs, *, bundle, rule_provider, ensemble_policy, graph_policy, **kwargs):
        assert bundle == "fake-loaded-candidate-bundle"
        return candidate_scores, []

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_scoring_contexts", lambda channel, database: [])
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", lambda bundle_record: (None, None, None))
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: _FakeArtifactLoader())
    monkeypatch.setattr("src.fraud_intel.evaluation.shadow_candidate.score_candidate_shadow", _fake_score_candidate_shadow)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--candidate-bundle-version", "3", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert result["shadow_candidate_comparison"] is not None
    assert result["shadow_candidate_comparison"]["candidate_bundle_id"] == 5
    assert result["shadow_candidate_comparison"]["candidate_bundle_version"] == 3
    assert result["shadow_candidate_comparison"]["matched_source_alert_count"] == 4
    assert result["candidate_scoring_errors"] == []


def test_evaluate_surfaces_candidate_scoring_errors_without_failing_the_command(monkeypatch, capsys):
    from src.fraud_intel.evaluation.shadow_candidate import CandidateShadowScoringError

    outcomes = _outcome_fixtures()
    candidate_bundle = _bundle(bundle_id=5, bundle_version=3)
    scoring_error = CandidateShadowScoringError(source_alert_id=outcomes[0].source_alert_id, error_type="ValueError", message="simulated per-alert failure")

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            return candidate_bundle

    class _FakeArtifactLoader:
        def load(self, bundle_record):
            return "fake-loaded-candidate-bundle"

    def _fake_score_candidate_shadow(scoring_inputs, *, bundle, rule_provider, ensemble_policy, graph_policy, **kwargs):
        return [], [scoring_error]

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_scoring_contexts", lambda channel, database: [])
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.load_and_validate_pinned_policies", lambda bundle_record: (None, None, None))
    monkeypatch.setattr("src.fraud_intel.scoring.dispatch.create_default_bundle_artifact_loader", lambda: _FakeArtifactLoader())
    monkeypatch.setattr("src.fraud_intel.evaluation.shadow_candidate.score_candidate_shadow", _fake_score_candidate_shadow)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
         "--recall-target", "0.8", "--candidate-bundle-version", "3", "--database", "aidp_test", "--json"]
    )

    result = json.loads(capsys.readouterr().out)
    assert len(result["candidate_scoring_errors"]) == 1
    assert result["candidate_scoring_errors"][0]["error_type"] == "ValueError"
    # zero candidate scores -> comparison still returns cleanly, non-computable rather than crashing
    assert result["shadow_candidate_comparison"] is not None


def test_evaluate_unknown_candidate_bundle_version_is_a_cli_user_error(monkeypatch, capsys):
    outcomes = _outcome_fixtures()

    class _FakePromotionStore:
        def get_bundle(self, channel, bundle_version):
            raise LookupError(f"no bundle version {bundle_version} for channel {channel!r}")

    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())
    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: _FakePromotionStore())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2",
             "--recall-target", "0.8", "--candidate-bundle-version", "999", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_no_operational_bundle_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_missing_required_capacity_or_recall_flags_fails_at_argparse_level(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "evaluate", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2


def test_evaluate_invalid_capacity_value_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: [])
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "-5",
             "--recall-target", "0.8", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


def test_evaluate_scope_marker_no_longer_present(monkeypatch, capsys):
    """The removed Phase 6 placeholder marker must not reappear anywhere
    in real output."""
    outcomes = _outcome_fixtures()
    monkeypatch.setattr("src.fraud_intel.cli_data_access.load_resolved_alert_outcomes", lambda channel, database: outcomes)
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: _bundle())

    cli_main.main(
        ["fraud-intel", "evaluate", "--channel", "online_banking", "--capacity-mode", "count", "--capacity-value", "2", "--recall-target", "0.8", "--database", "aidp_test", "--json"]
    )
    out = capsys.readouterr().out
    assert "reference_channel_phase6_only" not in out


# ---- fraud-intel promote ---------------------------------------------------------------


def test_promote_dispatches_and_returns_the_promoted_bundle(monkeypatch, capsys):
    fake_store = object()
    promoted = _bundle(bundle_id=2, bundle_version=1, promoted_by="analyst1", promoted_at=T0)
    captured = {}

    def _fake_promote_bundle(*, channel, bundle_version, promoted_by, model_version_verifier, store):
        captured.update(channel=channel, bundle_version=bundle_version, promoted_by=promoted_by, store=store)
        return promoted

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: fake_store)
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _fake_promote_bundle)

    cli_main.main(
        ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
    )

    assert captured["channel"] == "online_banking"
    assert captured["bundle_version"] == 1
    assert captured["promoted_by"] == "analyst1"
    assert captured["store"] is fake_store
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_id"] == 2
    assert result["status"] == "OPERATIONAL"


@pytest.mark.parametrize("exc_cls", [IncompleteBundleError, BundleVerificationFailedError, BundlePromotionRaceError])
def test_promote_maps_domain_errors_to_cli_user_error(monkeypatch, capsys, exc_cls):
    def _raise(**kwargs):
        raise exc_cls("refused")

    monkeypatch.setattr(cli_main, "create_default_bundle_promotion_store", lambda database: object())
    monkeypatch.setattr("src.fraud_intel.models.promotion.promote_bundle", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["fraud-intel", "promote", "--channel", "online_banking", "--bundle-version", "1", "--promoted-by", "analyst1", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- fraud-intel model show -------------------------------------------------------------


def test_model_show_returns_the_operational_bundle(monkeypatch, capsys):
    bundle = _bundle()
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: bundle)

    cli_main.main(["fraud-intel", "model", "show", "--channel", "online_banking", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["bundle_id"] == 1
    assert result["status"] == "OPERATIONAL"


def test_model_show_no_operational_bundle_is_a_cli_user_error(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_get_operational_bundle", lambda channel, database: None)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["fraud-intel", "model", "show", "--channel", "online_banking", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- alerts list: analyst-facing surface structurally excludes synthetic labels --------


def test_alerts_list_output_has_no_scenario_id_or_synthetic_fields(monkeypatch, capsys):
    monkeypatch.setattr(cli_main, "_list_alerts", lambda database, *, channel, status, priority_band: [_alert()])

    cli_main.main(["alerts", "list", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["count"] == 1
    assert "scenario_id" not in result["alerts"][0]
    assert "synthetic_scenario_label" not in result["alerts"][0]


def test_alerts_list_passes_filters_through(monkeypatch, capsys):
    captured = {}

    def _fake_list(database, *, channel, status, priority_band):
        captured.update(database=database, channel=channel, status=status, priority_band=priority_band)
        return []

    monkeypatch.setattr(cli_main, "_list_alerts", _fake_list)

    cli_main.main(
        ["alerts", "list", "--channel", "online_banking", "--status", "OPEN", "--priority-band", "HIGH", "--database", "aidp_test", "--json"]
    )

    assert captured == {"database": "aidp_test", "channel": "online_banking", "status": "OPEN", "priority_band": "HIGH"}


# ---- alerts show ------------------------------------------------------------------------


def test_alerts_show_returns_alert_and_latest_evidence(monkeypatch, capsys):
    store = _FakeAlertQueueStore()
    alert = _alert()
    store.alerts_by_id[1] = alert

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: store)

    cli_main.main(["alerts", "show", "1", "--database", "aidp_test", "--json"])

    result = json.loads(capsys.readouterr().out)
    assert result["alert"]["alert_id"] == 1
    assert result["latest_evidence"] is None


def test_alerts_show_unknown_alert_is_a_cli_user_error(monkeypatch, capsys):
    store = _FakeAlertQueueStore()
    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: store)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["alerts", "show", "999", "--database", "aidp_test", "--json"])
    assert exc_info.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"] == "CLIUserError"


# ---- alerts disposition -------------------------------------------------------------------


def test_alerts_disposition_dispatches_to_record_disposition(monkeypatch, capsys):
    fake_store = object()
    captured = {}
    record = AnalystDispositionRecord(
        disposition_id=1, alert_id=1, analyst_id="a1", disposition="CONFIRMED_FRAUD", notes=None, disposed_at=T0
    )

    def _fake_record_disposition(*, alert_id, analyst_id, disposition, notes, store):
        captured.update(alert_id=alert_id, analyst_id=analyst_id, disposition=disposition, notes=notes, store=store)
        return record

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: fake_store)
    monkeypatch.setattr(cli_main, "record_disposition", _fake_record_disposition)

    cli_main.main(
        ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--notes", "looks fraudulent", "--database", "aidp_test", "--json"]
    )

    assert captured["alert_id"] == 1
    assert captured["analyst_id"] == "a1"
    assert captured["disposition"] == "CONFIRMED_FRAUD"
    assert captured["notes"] == "looks fraudulent"
    assert captured["store"] is fake_store
    result = json.loads(capsys.readouterr().out)
    assert result["disposition"] == "CONFIRMED_FRAUD"


def test_alerts_disposition_against_a_closed_alert_is_a_cli_user_error(monkeypatch, capsys):
    def _raise(*, alert_id, analyst_id, disposition, notes, store):
        raise InvalidAlertTransitionError(f"alert {alert_id} is CLOSED (terminal); disposition rejected")

    monkeypatch.setattr(cli_main, "create_default_alert_queue_store", lambda database: object())
    monkeypatch.setattr(cli_main, "record_disposition", _raise)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            ["alerts", "disposition", "1", "--analyst-id", "a1", "--disposition", "CONFIRMED_FRAUD", "--database", "aidp_test", "--json"]
        )
    assert exc_info.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == "CLIUserError"
    assert "CLOSED" in result["message"]
