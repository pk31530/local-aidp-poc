import logging

from src.common.logging import configure_logging, get_logger


def test_configure_logging_sends_to_stderr_and_stdout_stays_empty(capsys):
    configure_logging("test.service.stderr")
    log = get_logger("test.logging.stderr_check")
    log.info("hello_from_stderr_test")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello_from_stderr_test" in captured.err


def test_repeated_configuration_does_not_duplicate_handlers():
    configure_logging("test.service.first")
    first_count = len(logging.getLogger().handlers)

    configure_logging("test.service.second")
    second_count = len(logging.getLogger().handlers)

    assert first_count == 1
    assert second_count == 1


def test_default_call_with_no_stream_argument_goes_to_stderr(capsys):
    """Every existing legacy caller (consumer.py, pipeline.py, train.py,
    api/main.py, generator/*) calls configure_logging(service_name) with no
    stream argument — this must default to stderr without any caller
    change."""
    configure_logging("test.service.legacy_call_shape")
    log = get_logger("test.logging.legacy_default_check")
    log.info("legacy_default_goes_to_stderr")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "legacy_default_goes_to_stderr" in captured.err
