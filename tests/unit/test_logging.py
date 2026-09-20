import json
import logging

import pytest

from src.common.logging import configure_logging, get_logger


@pytest.fixture
def preserved_root_handlers():
    """Several tests here deliberately manipulate the root logger's handler
    list to prove configure_logging()'s force behaviour — this restores
    whatever was there before, so no test leaks handler state into another."""
    root = logging.getLogger()
    original = list(root.handlers)
    yield root
    root.handlers = original


def test_default_call_with_no_stream_argument_goes_to_stderr(capsys):
    """Every existing legacy caller (consumer.py, pipeline.py, train.py,
    api/main.py, generator/*) calls configure_logging(service_name) with no
    stream/force argument — this must default to stderr without any caller
    change."""
    configure_logging("test.service.legacy_call_shape", force=True)
    log = get_logger("test.logging.legacy_default_check")
    log.info("legacy_default_goes_to_stderr")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "legacy_default_goes_to_stderr" in captured.err


def test_default_configuration_preserves_an_external_handler(preserved_root_handlers):
    """force defaults to False: a host application's own logging handler
    (set up before this code is imported/called) must survive a normal
    configure_logging() call, matching logging.basicConfig()'s own
    documented no-op-if-handlers-exist behaviour."""
    root = preserved_root_handlers
    external_handler = logging.NullHandler()
    root.handlers = [external_handler]

    configure_logging("test.service.default_preserve")  # force=False (default)

    assert root.handlers == [external_handler]


def test_force_true_resets_to_one_deterministic_stderr_handler(preserved_root_handlers, capsys):
    """The one legitimate force=True caller (the CLI) must always end up
    with exactly one handler bound to the current stream, regardless of
    what was on the root logger beforehand."""
    root = preserved_root_handlers
    root.handlers = [logging.NullHandler(), logging.NullHandler()]  # simulate leftover handlers

    configure_logging("test.service.force_reset", force=True)

    assert len(root.handlers) == 1
    log = get_logger("test.logging.force_reset_check")
    log.info("force_reset_goes_to_stderr")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "force_reset_goes_to_stderr" in captured.err


def test_repeated_normal_configuration_does_not_accumulate_handlers(preserved_root_handlers):
    """With the default force=False, a second configure_logging() call
    (e.g. a second workload invoked in the same process) must not pile up
    additional AiDP-owned handlers on top of the first."""
    root = preserved_root_handlers
    root.handlers = []  # clean slate for this test only

    configure_logging("test.service.repeat_first")
    first_count = len(root.handlers)

    configure_logging("test.service.repeat_second")
    second_count = len(root.handlers)

    assert first_count == 1
    assert second_count == 1


def test_cli_json_stdout_remains_clean_with_force_true(monkeypatch, capsys):
    """End-to-end through the real CLI dispatch path: main() calls
    configure_logging("cli", force=True), so even a workload that logs
    during dispatch must never contaminate --json's stdout."""
    from src.cli import __main__ as cli_main

    def _fake_run_pipeline_configured(config, *, trigger_source):
        get_logger("fake.pipeline.for_logging_test").info("noise_that_must_not_reach_stdout")
        return {"ok": True}

    monkeypatch.setattr(cli_main, "run_pipeline_configured", _fake_run_pipeline_configured)

    cli_main.main(["pipeline", "run", "batch", "--json"])

    captured = capsys.readouterr()
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {"ok": True}
    assert "noise_that_must_not_reach_stdout" in captured.err
