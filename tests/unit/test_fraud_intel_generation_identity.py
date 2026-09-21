"""Phase 7B Stage 2 corrective pass: deterministic generation identity
(generation_run_id/dataset_version derived from the complete generation
spec, not a fresh uuid4() per call) and generation-run-scoped training
population loading. No real database, Docker, or network access anywhere
in this file -- generate_and_write()/load_channel_population()'s own
get_connection() is monkeypatched to a small, purpose-built in-memory
fake that models exactly the SQL these two functions issue (including
Postgres' real ON CONFLICT DO NOTHING semantics), so retry/idempotency/
collision behavior is genuinely proven rather than merely asserted.
"""
from __future__ import annotations

import uuid
from datetime import date

import pytest

from src.fraud_intel.cli_data_access import (
    GenerationIdentityConflictError,
    GenerationRunChannelMismatchError,
    GenerationRunDatasetVersionError,
    UnknownGenerationRunError,
    _generation_identity,
    generate_and_write,
    load_channel_population,
)
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.generator.online_banking import generate_online_banking_events


# ---- _generation_identity: pure, no I/O ------------------------------------------------


def test_generation_identity_is_deterministic_for_identical_inputs():
    a = _generation_identity(channel="online_banking", count=5000, seed=42, reference_date=date(2026, 1, 1))
    b = _generation_identity(channel="online_banking", count=5000, seed=42, reference_date=date(2026, 1, 1))
    assert a == b
    assert a[0].startswith("genrun-")
    assert a[1].startswith("dsv-")


@pytest.mark.parametrize(
    "override",
    [
        {"channel": "wire"},
        {"count": 999},
        {"seed": 7},
        {"reference_date": date(2026, 2, 1)},
    ],
)
def test_generation_identity_changes_when_any_spec_field_changes(override):
    base = dict(channel="online_banking", count=5000, seed=42, reference_date=date(2026, 1, 1))
    base_identity = _generation_identity(**base)
    varied_identity = _generation_identity(**{**base, **override})
    assert varied_identity != base_identity


# ---- generate_and_write: fake in-memory Postgres double --------------------------------


class _FakeGenerateConnection:
    """Models channel_events/source_alerts/synthetic_event_labels well
    enough to prove ON CONFLICT DO NOTHING (first write wins, a retry is
    a no-op) and the pre-insert foreign-collision check, without a real
    database."""

    def __init__(self):
        self.channel_events: dict[str, dict] = {}
        self.source_alerts: dict[tuple, dict] = {}
        self.synthetic_event_labels: dict[str, dict] = {}

    def cursor(self, cursor_factory=None):
        return _FakeGenerateCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeGenerateCursor:
    def __init__(self, conn: _FakeGenerateConnection):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT event_id, generation_run_id FROM channel_events"):
            (event_ids,) = params
            self._result = [
                {"event_id": eid, "generation_run_id": self._conn.channel_events[eid]["generation_run_id"]}
                for eid in event_ids
                if eid in self._conn.channel_events
            ]
        elif normalized.startswith("INSERT INTO channel_events"):
            event_id = params[0]
            self._conn.channel_events.setdefault(
                event_id, {"generation_run_id": params[12], "dataset_version": params[13]}
            )
        elif normalized.startswith("INSERT INTO source_alerts"):
            key = (params[1], params[0])
            self._conn.source_alerts.setdefault(key, {"generation_run_id": params[8]})
        elif normalized.startswith("INSERT INTO synthetic_event_labels"):
            event_id = params[0]
            self._conn.synthetic_event_labels.setdefault(event_id, {"generation_run_id": params[4]})
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:100]}")

    def fetchall(self):
        return self._result


def _patch_connection(monkeypatch, conn):
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: conn)


def test_exact_retry_returns_the_persisted_identifiers(monkeypatch):
    conn = _FakeGenerateConnection()
    _patch_connection(monkeypatch, conn)

    first = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    second = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))

    assert first["generation_run_id"] == second["generation_run_id"]
    assert first["dataset_version"] == second["dataset_version"]
    # the returned identifiers actually exist in (fake) channel_events -- not phantom
    assert conn.channel_events
    assert all(row["generation_run_id"] == first["generation_run_id"] for row in conn.channel_events.values())


def test_exact_retry_creates_zero_duplicate_rows(monkeypatch):
    conn = _FakeGenerateConnection()
    _patch_connection(monkeypatch, conn)

    first = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    assert first["inserted_event_count"] == 20
    assert first["existing_event_count"] == 0
    row_count_after_first = len(conn.channel_events)

    second = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    assert second["inserted_event_count"] == 0
    assert second["existing_event_count"] == 20
    assert len(conn.channel_events) == row_count_after_first  # no growth -- zero duplicate rows


def test_changing_reference_date_does_not_silently_reuse_old_rows(monkeypatch):
    conn = _FakeGenerateConnection()
    _patch_connection(monkeypatch, conn)

    first = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    before = {eid: dict(row) for eid, row in conn.channel_events.items()}

    with pytest.raises(GenerationIdentityConflictError):
        generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 2, 1))

    # the rejected call must not have mutated anything already stored
    assert conn.channel_events == before
    assert all(row["generation_run_id"] == first["generation_run_id"] for row in conn.channel_events.values())


def test_generate_and_write_never_trains_scores_promotes_or_assesses_labels():
    """AST-based (Phase 7A convention): generate_and_write's own source
    never calls anything named train_channel_configured, score_channel,
    promote_bundle, or assess_channel_labels."""
    import ast
    import inspect

    from src.fraud_intel import cli_data_access

    tree = ast.parse(inspect.getsource(cli_data_access.generate_and_write))
    forbidden = {"train_channel_configured", "score_channel", "promote_bundle", "assess_channel_labels"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            assert name not in forbidden, f"generate_and_write must never call {name}"


# ---- load_channel_population: generation-run-scoped loading ---------------------------


class _FakePopulationConnection:
    def __init__(self, channel_events_rows, source_alert_rows, label_rows):
        self.channel_events_rows = channel_events_rows
        self.source_alert_rows = source_alert_rows
        self.label_rows = label_rows

    def cursor(self, cursor_factory=None):
        return _FakePopulationCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakePopulationCursor:
    def __init__(self, conn: _FakePopulationConnection):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT DISTINCT channel FROM channel_events WHERE generation_run_id"):
            (run_id,) = params
            channels = {row["channel"] for row in self._conn.channel_events_rows if row["generation_run_id"] == run_id}
            self._result = [{"channel": c} for c in channels]
        elif normalized.startswith("SELECT count(DISTINCT dataset_version) AS n FROM channel_events"):
            channel, run_id = params
            versions = {
                row["dataset_version"]
                for row in self._conn.channel_events_rows
                if row["channel"] == channel and row["generation_run_id"] == run_id
            }
            self._result = [{"n": len(versions)}]
        elif normalized.startswith("SELECT * FROM channel_events WHERE channel"):
            channel, run_id = params
            self._result = [
                row for row in self._conn.channel_events_rows if row["channel"] == channel and row["generation_run_id"] == run_id
            ]
        elif normalized.startswith("SELECT * FROM source_alerts WHERE event_id"):
            (event_ids,) = params
            self._result = [row for row in self._conn.source_alert_rows if str(row["event_id"]) in event_ids]
        elif normalized.startswith("SELECT * FROM synthetic_event_labels WHERE event_id"):
            (event_ids,) = params
            self._result = [row for row in self._conn.label_rows if str(row["event_id"]) in event_ids]
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:100]}")

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


def _results(n=10, seed=1, reference_date=date(2026, 1, 1)):
    customers = generate_customers(n=max(n // 5, 1), seed=seed, reference_date=reference_date)
    return generate_online_banking_events(
        seed=seed, n=n, reference_date=reference_date, customers=customers,
        generation_run_id="genrun-fixture", dataset_version="dsv-fixture",
    )


def _population_rows(results, *, channel="online_banking", generation_run_id="genrun-fixture", dataset_version="dsv-fixture"):
    channel_events_rows, source_alert_rows, label_rows = [], [], []
    for event, source_alert, label in results:
        channel_events_rows.append(
            {
                "event_id": event.event_id, "channel": channel, "customer_id": event.customer_id,
                "account_id": event.account_id, "event_timestamp": event.event_timestamp,
                "amount_minor_units": event.amount_minor_units, "direction": event.direction,
                "device_id": event.device_id, "ip_address": event.ip_address,
                "channel_payload": event.channel_payload.model_dump(mode="json"),
                "scenario_id": event.scenario_id, "schema_version": event.schema_version,
                "generation_run_id": generation_run_id, "dataset_version": dataset_version,
            }
        )
        if source_alert is not None:
            source_alert_rows.append(
                {
                    "source_alert_id": source_alert.source_alert_id, "source_system": source_alert.source_system,
                    "event_id": event.event_id, "source_alert_created_at": source_alert.source_alert_created_at,
                    "source_rule_ids": source_alert.source_rule_ids, "source_rule_version": source_alert.source_rule_version,
                    "source_alert_score": source_alert.source_alert_score,
                    "source_alert_reason_codes": source_alert.source_alert_reason_codes,
                    "generation_run_id": generation_run_id, "dataset_version": dataset_version,
                    "created_at": source_alert.created_at,
                }
            )
        label_rows.append(
            {
                "event_id": event.event_id, "scenario_id": label.scenario_id,
                "synthetic_scenario_label": label.synthetic_scenario_label, "scenario_type": label.scenario_type,
                "generation_run_id": generation_run_id, "dataset_version": dataset_version,
                "generated_at": label.generated_at,
            }
        )
    return channel_events_rows, source_alert_rows, label_rows


def test_load_channel_population_scopes_to_the_exact_generation_run(monkeypatch):
    results = _results()
    channel_events_rows, source_alert_rows, label_rows = _population_rows(results)
    conn = _FakePopulationConnection(channel_events_rows, source_alert_rows, label_rows)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: conn)

    events, source_alerts, labels = load_channel_population("online_banking", "aidp_test", generation_run_id="genrun-fixture")

    assert len(events) == len(results)
    assert len(labels) == len(results)
    assert len(source_alerts) == sum(1 for _, sa, _ in results if sa is not None)


def test_load_channel_population_unknown_generation_run_raises(monkeypatch):
    conn = _FakePopulationConnection([], [], [])
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: conn)

    with pytest.raises(UnknownGenerationRunError):
        load_channel_population("online_banking", "aidp_test", generation_run_id="genrun-does-not-exist")


def test_load_channel_population_wrong_channel_raises(monkeypatch):
    results = _results()
    channel_events_rows, source_alert_rows, label_rows = _population_rows(results, channel="online_banking")
    conn = _FakePopulationConnection(channel_events_rows, source_alert_rows, label_rows)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: conn)

    with pytest.raises(GenerationRunChannelMismatchError):
        load_channel_population("wire", "aidp_test", generation_run_id="genrun-fixture")


def test_load_channel_population_mixed_dataset_version_raises(monkeypatch):
    results = _results()
    channel_events_rows, source_alert_rows, label_rows = _population_rows(results)
    # Corrupt one row's dataset_version to violate the "exactly one
    # dataset_version per generation_run_id" invariant.
    channel_events_rows[0] = {**channel_events_rows[0], "dataset_version": "dsv-DIFFERENT"}
    conn = _FakePopulationConnection(channel_events_rows, source_alert_rows, label_rows)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: conn)

    with pytest.raises(GenerationRunDatasetVersionError):
        load_channel_population("online_banking", "aidp_test", generation_run_id="genrun-fixture")


def test_load_channel_population_requires_generation_run_id_keyword():
    """Structural: the parameter has no default -- every caller must be
    explicit, so training can never silently fall back to loading every
    row ever generated for a channel."""
    import inspect

    sig = inspect.signature(load_channel_population)
    assert sig.parameters["generation_run_id"].default is inspect.Parameter.empty
    assert sig.parameters["generation_run_id"].kind == inspect.Parameter.KEYWORD_ONLY


# ---- UUID-array cast regression -- first real-Postgres run hit ------------------------
# `channel_events.event_id`/etc are real `UUID` columns; ANY(%s) against a
# plain Python string list adapts to text[], and Postgres has no implicit
# uuid = text operator (psycopg2.errors.UndefinedFunction). Every ANY(...)
# query against a uuid column must cast explicitly: ANY(%s::uuid[]).
# register_uuid() was deliberately NOT used to fix this -- it is global,
# process-wide psycopg2 adapter/typecaster state that would change how
# EVERY existing query in src/common/db.py's connections behaves, not
# just these three, which is a far larger blast radius than this fix
# needs.


def test_all_three_uuid_array_queries_use_an_explicit_cast():
    import inspect

    from src.fraud_intel import cli_data_access

    generate_source = inspect.getsource(cli_data_access.generate_and_write)
    population_source = inspect.getsource(cli_data_access.load_channel_population)

    assert generate_source.count("ANY(%s::uuid[])") == 1
    assert population_source.count("ANY(%s::uuid[])") == 2
    # the old, uncast form must be fully gone from both functions
    assert "ANY(%s)" not in generate_source
    assert "ANY(%s)" not in population_source


def test_no_sql_is_built_through_string_interpolation_or_concatenation():
    """AST-based: every .execute(...) call in generate_and_write/
    load_channel_population passes a plain string literal as its SQL
    argument -- never an f-string, %-formatted string, or + concatenation
    built from a variable -- so parameterization is the only path any
    runtime value ever reaches these queries through."""
    import ast
    import inspect

    from src.fraud_intel import cli_data_access

    for fn in (cli_data_access.generate_and_write, cli_data_access.load_channel_population):
        tree = ast.parse(inspect.getsource(fn))
        execute_call_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                execute_call_count += 1
                sql_arg = node.args[0]
                assert isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str), (
                    f"{fn.__name__}: .execute()'s SQL argument must be a plain string literal, not a "
                    f"dynamically constructed value: {ast.dump(sql_arg)}"
                )
        assert execute_call_count > 0


def test_every_parameterized_execute_call_passes_a_separate_params_argument():
    """Every .execute(sql, params) call whose SQL literal contains a %s
    placeholder supplies a second (params) argument -- values are always
    passed as query parameters, never folded into the SQL text itself."""
    import ast
    import inspect

    from src.fraud_intel import cli_data_access

    for fn in (cli_data_access.generate_and_write, cli_data_access.load_channel_population):
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                sql_literal = node.args[0].value
                if "%s" in sql_literal:
                    assert len(node.args) == 2, f"{fn.__name__}: parameterized query {sql_literal!r} must pass a params argument"


def test_event_id_is_always_a_real_uuid_at_the_type_level():
    """Structural: event_ids fed into every ANY(%s::uuid[]) query are
    always str(uuid.UUID) -- FraudEvent.event_id is itself typed
    uuid.UUID (pydantic-validated at construction), so a malformed
    string can never originate from this codebase's own values in the
    first place. Combined with the explicit ::uuid[] cast, even a
    hypothetical malformed value would be cleanly rejected by Postgres'
    own type system rather than silently matching unintended rows."""
    from src.fraud_intel.events.base import FraudEvent

    assert FraudEvent.model_fields["event_id"].annotation is uuid.UUID


def test_generate_and_write_with_zero_count_skips_the_collision_query_and_writes_nothing(monkeypatch):
    """An empty event_ids list (count=0) must not produce an unbounded or
    unsafe query -- generate_and_write's own `if event_ids:` guard skips
    issuing the ANY(...) query at all in that case."""
    conn = _FakeGenerateConnection()
    issued_sql: list[str] = []
    original_execute = _FakeGenerateCursor.execute

    def _tracking_execute(self, sql, params=()):
        issued_sql.append(sql)
        return original_execute(self, sql, params)

    monkeypatch.setattr(_FakeGenerateCursor, "execute", _tracking_execute)
    _patch_connection(monkeypatch, conn)

    result = generate_and_write(channel="online_banking", count=0, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))

    assert result["requested_count"] == 0
    assert result["inserted_event_count"] == 0
    assert result["existing_event_count"] == 0
    assert not any("ANY(%s::uuid[])" in sql for sql in issued_sql)
    assert not conn.channel_events


def test_generation_identities_and_scoped_population_loading_are_unchanged_by_the_cast_fix(monkeypatch):
    """Regression: the ::uuid[] cast is purely an SQL-text change --
    deterministic-identity computation and generation-scoped population
    loading behave identically to before it, proven end-to-end through
    the same fakes as the rest of this file."""
    identity_before_and_after = [
        _generation_identity(channel="online_banking", count=5000, seed=42, reference_date=date(2026, 1, 1))
        for _ in range(2)
    ]
    assert identity_before_and_after[0] == identity_before_and_after[1]

    conn = _FakeGenerateConnection()
    _patch_connection(monkeypatch, conn)
    first = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    second = generate_and_write(channel="online_banking", count=20, seed=1, database="aidp_test", reference_date=date(2026, 1, 1))
    assert first["generation_run_id"] == second["generation_run_id"]
    assert second["existing_event_count"] == 20
    assert second["inserted_event_count"] == 0

    results = _results()
    channel_events_rows, source_alert_rows, label_rows = _population_rows(results)
    pop_conn = _FakePopulationConnection(channel_events_rows, source_alert_rows, label_rows)
    monkeypatch.setattr("src.fraud_intel.cli_data_access.get_connection", lambda database: pop_conn)
    events, source_alerts, labels = load_channel_population("online_banking", "aidp_test", generation_run_id="genrun-fixture")
    assert len(events) == len(results)
    assert len(source_alerts) == sum(1 for _, sa, _ in results if sa is not None)
    assert len(labels) == len(results)
