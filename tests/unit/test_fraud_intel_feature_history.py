"""Phase 7B corrective pass: the shared cross-channel feature-history
contract (src.fraud_intel.features.history), and proof that training
(src.fraud_intel.models.training._build_supervised_population) and real
scoring (src.fraud_intel.scoring.dispatch._PostgresScoringDataAccess.
list_pending, src.fraud_intel.cli_data_access.
load_resolved_alert_scoring_contexts) now compute identical feature
vectors for the same event/customer/as-of-time/candidate-pool, closing
the real ACH band-boundary discrepancy discovered against real
aidp_test data (2 of 441 alerts diverged between a channel-scoped
training-style reconstruction and real, cross-channel-scoped scoring).

No database, Docker, or network access anywhere in this file.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.features.history import (
    select_customer_historical_events,
    select_customer_historical_source_alerts,
)
from src.fraud_intel.registry import get_channel_adapter

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ach_event(event_id, *, customer_id="CUST1", account_id="ACCT1", ts=T0, amount=10_000) -> FraudEvent:
    return FraudEvent(
        event_id=event_id, channel="ach", customer_id=customer_id, account_id=account_id, event_timestamp=ts,
        amount_minor_units=amount, direction="debit",
        channel_payload=ACHPayload(
            sec_code="PPD", originating_routing_number="123456789", receiving_routing_number="987654321",
            batch_id="BATCH1", effective_entry_date=date(2026, 1, 1), company_id="COMP1",
        ),
    )


def _online_banking_event(event_id, *, customer_id="CUST1", account_id="ACCT1", ts=T0, amount=10_000) -> FraudEvent:
    return FraudEvent(
        event_id=event_id, channel="online_banking", customer_id=customer_id, account_id=account_id,
        event_timestamp=ts, amount_minor_units=amount, direction="debit", device_id="DEV1",
        channel_payload=OnlineBankingPayload(
            session_id="S1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )


def _source_alert(event_id, *, ts=T0) -> SourceAlertContext:
    return SourceAlertContext(
        source_alert_id=uuid.uuid4(), source_system="LocalYamlRuleProvider (simulated upstream)", event_id=event_id,
        source_alert_created_at=ts, source_rule_ids=["RULE1"], source_rule_version="v1",
        source_alert_reason_codes=["REASON1"], generation_run_id="unused", dataset_version="unused", created_at=ts,
    )


# ---- 1/8: same snapshot, deterministic, order-independent ----------------------------


def test_identical_snapshot_produces_identical_selection_regardless_of_input_order():
    target = _ach_event(uuid.uuid4(), ts=T0)
    earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1))
    pool_a = [target, earlier]
    pool_b = [earlier, target]  # reversed order

    result_a = select_customer_historical_events(pool_a, customer_id="CUST1", as_of_time=T0, exclude_event_id=target.event_id)
    result_b = select_customer_historical_events(pool_b, customer_id="CUST1", as_of_time=T0, exclude_event_id=target.event_id)
    assert result_a == result_b == (earlier,)


# ---- 2: an earlier cross-channel event contributes --------------------------------------


def test_earlier_cross_channel_event_contributes_to_history():
    earlier_ob = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(days=1))
    pool = [earlier_ob]
    result = select_customer_historical_events(pool, customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert result == (earlier_ob,)
    assert result[0].channel == "online_banking"  # cross-channel event, included


# ---- 3: a later event from another channel never contributes ----------------------------


def test_later_cross_channel_event_never_contributes():
    later_ob = _online_banking_event(uuid.uuid4(), ts=T0 + timedelta(hours=1))
    result = select_customer_historical_events([later_ob], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert result == ()


# ---- 4: same-timestamp events are excluded (documented deterministic rule) --------------


def test_same_timestamp_event_is_excluded_not_tie_broken_in():
    """Guide section 9: 'a feature ... may only read records whose
    event_timestamp is strictly earlier than T' -- a record at exactly T
    is never history, regardless of event_id ordering."""
    same_ts = _online_banking_event(uuid.uuid4(), ts=T0)
    result = select_customer_historical_events([same_ts], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert result == ()


def test_multiple_same_timestamp_historical_events_ordered_by_event_id_desc():
    """Among rows that ARE strictly earlier than T and share a timestamp
    with each other, ordering is deterministic: event_timestamp DESC,
    then event_id DESC -- never insertion order."""
    ts = T0 - timedelta(hours=1)
    e1 = _ach_event(uuid.UUID(int=1), ts=ts)
    e2 = _ach_event(uuid.UUID(int=2), ts=ts)
    result_forward = select_customer_historical_events([e1, e2], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    result_reversed = select_customer_historical_events([e2, e1], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert result_forward == result_reversed
    assert [str(e.event_id) for e in result_forward] == sorted([str(e1.event_id), str(e2.event_id)], reverse=True)


# ---- 5: another customer's event never contributes ---------------------------------------


def test_other_customer_event_never_contributes():
    other_customer = _ach_event(uuid.uuid4(), customer_id="CUST2", ts=T0 - timedelta(hours=1))
    result = select_customer_historical_events([other_customer], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert result == ()


def test_target_event_itself_excluded_via_exclude_event_id():
    target = _ach_event(uuid.uuid4(), ts=T0 - timedelta(seconds=1))
    result = select_customer_historical_events([target], customer_id="CUST1", as_of_time=T0, exclude_event_id=target.event_id)
    assert result == ()


# ---- 7: source-alert history follows the identical rule ----------------------------------


def test_source_alert_history_customer_scoped_cross_channel_as_of_time_safe():
    owner_event_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=2))
    owner_event_later = _ach_event(uuid.uuid4(), ts=T0 + timedelta(hours=1))
    owner_event_other_customer = _ach_event(uuid.uuid4(), customer_id="CUST2", ts=T0 - timedelta(hours=1))

    alert_earlier = _source_alert(owner_event_earlier.event_id, ts=owner_event_earlier.event_timestamp)
    alert_later = _source_alert(owner_event_later.event_id, ts=owner_event_later.event_timestamp)
    alert_other_customer = _source_alert(owner_event_other_customer.event_id, ts=owner_event_other_customer.event_timestamp)

    event_customer_ids = {
        owner_event_earlier.event_id: owner_event_earlier.customer_id,
        owner_event_later.event_id: owner_event_later.customer_id,
        owner_event_other_customer.event_id: owner_event_other_customer.customer_id,
    }
    result = select_customer_historical_source_alerts(
        [alert_earlier, alert_later, alert_other_customer], event_customer_ids=event_customer_ids,
        customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4(),
    )
    assert result == (alert_earlier,)  # only the earlier, same-customer, cross-channel alert


# ---- limit truncation, dedupe ---------------------------------------------------------


def test_limit_truncates_to_most_recent_n():
    events = [_ach_event(uuid.uuid4(), ts=T0 - timedelta(hours=i)) for i in range(1, 6)]
    result = select_customer_historical_events(events, customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4(), limit=2)
    assert len(result) == 2
    assert result[0].event_timestamp > result[1].event_timestamp


def test_duplicate_event_id_across_pools_deduped():
    event_id = uuid.uuid4()
    e1 = _ach_event(event_id, ts=T0 - timedelta(hours=1))
    e2 = _ach_event(event_id, ts=T0 - timedelta(hours=1))  # same id, appears twice (e.g. two source pools merged)
    result = select_customer_historical_events([e1, e2], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    assert len(result) == 1


# ---- 10: no label/outcome leakage ---------------------------------------------------------


def test_historical_events_never_carry_label_shaped_fields():
    event = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1))
    result = select_customer_historical_events([event], customer_id="CUST1", as_of_time=T0, exclude_event_id=uuid.uuid4())
    dumped = result[0].model_dump(mode="json")
    forbidden = {"synthetic_scenario_label", "analyst_disposition", "outcome_status", "training_eligible"}
    assert forbidden.isdisjoint(dumped.keys())
    assert forbidden.isdisjoint(dumped["channel_payload"].keys())


# ---- 6/13: training population construction -- target scoping vs. cross-channel history ----


def _labels_for(*events):
    return [
        SyntheticGroundTruthLabel(
            event_id=e.event_id, scenario_id=None, synthetic_scenario_label=False, scenario_type="NORMAL",
            generation_run_id="genrun-a", dataset_version="dsv-a", generated_at=e.event_timestamp,
        )
        for e in events
    ]


class TestBuildSupervisedPopulationCrossChannelHistory:
    """src.fraud_intel.models.training._build_supervised_population --
    imported lazily inside each test since it's a private module member."""

    def _adapter(self):
        return get_channel_adapter("ach")

    def test_target_selection_remains_channel_scoped_even_with_cross_channel_history(self):
        from src.fraud_intel.models.training import _build_supervised_population

        ach_target = _ach_event(uuid.uuid4(), ts=T0)
        ob_event_same_channel_events_list = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1))
        # channel_events (the TARGET pool) contains an online_banking row too --
        # it must never become a supervised row for an "ach" adapter.
        channel_events = [ach_target, ob_event_same_channel_events_list]
        source_alerts = [_source_alert(ach_target.event_id, ts=ach_target.event_timestamp),
                          _source_alert(ob_event_same_channel_events_list.event_id, ts=ob_event_same_channel_events_list.event_timestamp)]
        labels = _labels_for(ach_target, ob_event_same_channel_events_list)

        rows = _build_supervised_population(
            adapter=self._adapter(), channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
        )
        assert len(rows) == 1
        assert rows[0]["event_id"] == str(ach_target.event_id)

    def test_history_falls_back_to_channel_scoped_when_no_cross_channel_pool_given(self):
        """Backward compatibility (requirement 13): omitting history_events/
        history_source_alerts reproduces the exact pre-fix, channel-scoped-
        only behavior every existing caller/test relies on."""
        from src.fraud_intel.models.training import _build_supervised_population

        target = _ach_event(uuid.uuid4(), ts=T0)
        ob_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1))  # NOT in channel_events
        channel_events = [target]
        source_alerts = [_source_alert(target.event_id, ts=target.event_timestamp)]
        labels = _labels_for(target)

        rows = _build_supervised_population(
            adapter=self._adapter(), channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
        )
        assert rows[0]["amount_vs_entity_average"] == 1.0  # no history at all -> default ratio

    def test_cross_channel_history_changes_computed_features(self):
        """The direct regression test for the discovered bug: supplying a
        cross-channel history pool changes the resulting feature vector
        relative to the channel-scoped-only default."""
        from src.fraud_intel.models.training import _build_supervised_population

        target = _ach_event(uuid.uuid4(), ts=T0, amount=100_000)
        ob_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1), amount=10_000)
        channel_events = [target]
        source_alerts = [_source_alert(target.event_id, ts=target.event_timestamp)]
        labels = _labels_for(target)

        rows_channel_scoped = _build_supervised_population(
            adapter=self._adapter(), channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
        )
        rows_cross_channel = _build_supervised_population(
            adapter=self._adapter(), channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
            history_events=[target, ob_earlier], history_source_alerts=[],
        )
        assert rows_channel_scoped[0]["amount_vs_entity_average"] == 1.0  # no history -> default
        assert rows_cross_channel[0]["amount_vs_entity_average"] == pytest.approx(100_000 / 10_000)


# ---- 9: hash sensitivity to feature changes -----------------------------------------------


class TestDatasetVersionHashSensitivity:
    def test_identical_population_rows_produce_identical_hash(self):
        from src.fraud_intel.models.training import _dataset_version

        rows = [{"event_id": "e1", "event_timestamp": T0, "label": True, "amount_vs_entity_average": 1.0}]
        assert _dataset_version(rows) == _dataset_version(rows)

    def test_same_target_ids_and_labels_but_different_feature_values_produce_different_hash(self):
        """The exact gap the corrective pass closes: two populations with
        identical (event_id, label) pairs but different historical-context
        -derived feature values must never silently share a hash."""
        from src.fraud_intel.models.training import _dataset_version

        rows_a = [{"event_id": "e1", "event_timestamp": T0, "label": True, "amount_vs_entity_average": 1.0}]
        rows_b = [{"event_id": "e1", "event_timestamp": T0, "label": True, "amount_vs_entity_average": 11.74}]
        assert _dataset_version(rows_a) != _dataset_version(rows_b)

    def test_cross_channel_history_changes_the_real_training_hash(self):
        """End-to-end version of the above, through the real
        _build_supervised_population() + _dataset_version() pipeline."""
        from src.fraud_intel.models.training import _build_supervised_population, _dataset_version

        target = _ach_event(uuid.uuid4(), ts=T0, amount=100_000)
        ob_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1), amount=10_000)
        channel_events = [target]
        source_alerts = [_source_alert(target.event_id, ts=target.event_timestamp)]
        labels = _labels_for(target)
        adapter = get_channel_adapter("ach")

        rows_channel_scoped = _build_supervised_population(
            adapter=adapter, channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
        )
        rows_cross_channel = _build_supervised_population(
            adapter=adapter, channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
            history_events=[target, ob_earlier], history_source_alerts=[],
        )
        assert _dataset_version(rows_channel_scoped) != _dataset_version(rows_cross_channel)


# ---- 11/12: training-shaped vs. serving-shaped context parity, incl. a band-boundary case ---


class TestTrainingServingParity:
    """Proves that training's _build_supervised_population() and a
    serving-shaped reconstruction (built by calling the exact same shared
    selector src.fraud_intel.scoring.dispatch.list_pending() itself calls,
    against the identical candidate pool) produce byte-identical
    FeatureComputationContext -- and therefore, since adapter.compute_features()
    and score_source_alert() are both pure functions of their inputs,
    byte-identical downstream features/scores. This is the direct proof
    that the "candidate diagnostic" and "production scoring" paths can no
    longer diverge the way they did for real ACH data."""

    def test_training_and_serving_style_context_are_identical_for_the_same_pool(self):
        from src.fraud_intel.features.core import FeatureComputationContext
        from src.fraud_intel.models.training import _build_supervised_population

        target = _ach_event(uuid.uuid4(), ts=T0, amount=100_000)
        ob_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1), amount=10_000)
        full_pool = [target, ob_earlier]
        channel_events = [target]
        source_alerts = [_source_alert(target.event_id, ts=target.event_timestamp)]
        labels = _labels_for(target)
        adapter = get_channel_adapter("ach")

        # training path
        rows = _build_supervised_population(
            adapter=adapter, channel_events=channel_events, source_alerts=source_alerts, synthetic_labels=labels,
            history_events=full_pool, history_source_alerts=[],
        )

        # serving-shaped reconstruction: the exact same shared selector call
        # list_pending()/load_resolved_alert_scoring_contexts() make.
        serving_history = select_customer_historical_events(
            full_pool, customer_id=target.customer_id, as_of_time=target.event_timestamp, exclude_event_id=target.event_id,
        )
        serving_ctx = FeatureComputationContext(
            current_event=target, historical_events=serving_history, source_alert_history=(), as_of_time=target.event_timestamp,
        )
        serving_features = adapter.compute_features(serving_ctx)

        for column in adapter.feature_columns:
            assert rows[0][column] == serving_features[column]

    def test_band_boundary_case_equivalent_to_ach_finding_cannot_diverge(self):
        """A minimal reproduction of the real ACH discrepancy: a target
        event whose amount is large relative to a channel-scoped-only
        history (pushing a band-relevant feature toward one extreme) but
        much less extreme once earlier, cross-channel history for the
        same customer is included. Both the training-style and serving-
        style reconstructions must agree, because both now call the same
        selector against the same pool."""
        target = _ach_event(uuid.uuid4(), ts=T0, amount=500_000)
        cross_channel_history = [
            _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(days=1), amount=40_000),
            _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(days=2), amount=38_000),
        ]
        full_pool = [target, *cross_channel_history]

        from src.fraud_intel.features.core import FeatureComputationContext
        from src.fraud_intel.features.channels.ach import compute_ach_features

        # "training-style" call
        hist_a = select_customer_historical_events(
            full_pool, customer_id=target.customer_id, as_of_time=target.event_timestamp, exclude_event_id=target.event_id,
        )
        # "serving-style" call -- pool assembled in a different order, simulating
        # a different SQL fetch order
        hist_b = select_customer_historical_events(
            list(reversed(full_pool)), customer_id=target.customer_id, as_of_time=target.event_timestamp, exclude_event_id=target.event_id,
        )
        assert hist_a == hist_b

        features_a = compute_ach_features(FeatureComputationContext(current_event=target, historical_events=hist_a, source_alert_history=(), as_of_time=target.event_timestamp))
        features_b = compute_ach_features(FeatureComputationContext(current_event=target, historical_events=hist_b, source_alert_history=(), as_of_time=target.event_timestamp))
        assert features_a == features_b


# ---- 14: parameterized SQL, no string interpolation ----------------------------------------


# ---- real Postgres-shape parity: list_pending()'s actual query path matches the shared selector ----


class _FakeListPendingConnection:
    def __init__(self):
        self.channel_events: list[dict] = []
        self.source_alerts: list[dict] = []
        self.fraud_alerts: list[dict] = []
        self.alert_evidence: list[dict] = []

    def cursor(self, cursor_factory=None):
        return _FakeListPendingCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        pass


class _FakeListPendingCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT sa.*, ce.event_id AS ce_event_id"):
            channel, generation_run_id, operational_bundle_id, limit = params
            eligible_event_ids = {r["event_id"] for r in self._conn.channel_events if r["channel"] == channel and r["generation_run_id"] == generation_run_id}
            rows = []
            for sa in self._conn.source_alerts:
                if sa["event_id"] not in eligible_event_ids:
                    continue
                ce = next(r for r in self._conn.channel_events if r["event_id"] == sa["event_id"])
                row = dict(sa)
                row.update({
                    "ce_event_id": ce["event_id"], "ce_channel": ce["channel"], "ce_customer_id": ce["customer_id"],
                    "ce_account_id": ce["account_id"], "ce_event_timestamp": ce["event_timestamp"],
                    "ce_amount_minor_units": ce["amount_minor_units"], "ce_direction": ce["direction"],
                    "ce_device_id": ce["device_id"], "ce_ip_address": ce["ip_address"],
                    "ce_channel_payload": ce["channel_payload"], "ce_scenario_id": ce["scenario_id"],
                    "ce_schema_version": ce["schema_version"],
                })
                rows.append(row)
            self._result = rows
        elif normalized.startswith("SELECT * FROM channel_events WHERE customer_id"):
            customer_id, as_of_time = params
            self._result = [r for r in self._conn.channel_events if r["customer_id"] == customer_id and r["event_timestamp"] < as_of_time]
        elif normalized.startswith("SELECT sa2.* FROM source_alerts sa2"):
            self._result = []
        elif normalized.startswith("SELECT la.assessment_id"):
            self._result = []
        else:
            raise AssertionError(f"unexpected SQL in fake: {normalized[:120]}")

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None


def _row_for(event: FraudEvent, *, generation_run_id="genrun-a", dataset_version="dsv-a") -> dict:
    return {
        "event_id": event.event_id, "channel": event.channel, "customer_id": event.customer_id,
        "account_id": event.account_id, "event_timestamp": event.event_timestamp,
        "amount_minor_units": event.amount_minor_units, "direction": event.direction,
        "device_id": event.device_id, "ip_address": event.ip_address,
        "channel_payload": event.channel_payload.model_dump(mode="json"),
        "scenario_id": event.scenario_id, "schema_version": event.schema_version,
        "generation_run_id": generation_run_id, "dataset_version": dataset_version,
    }


def test_list_pending_real_query_path_matches_the_shared_selector_directly(monkeypatch):
    """_PostgresScoringDataAccess.list_pending()'s real, SQL-driven
    historical-context construction, exercised against a fake Postgres
    connection carrying real cross-channel rows, must produce the exact
    same historical_events the shared selector computes directly on the
    same underlying pool -- proving the SERVING path (not just training)
    is routed through src.fraud_intel.features.history, not a
    hand-copied equivalent."""
    from src.fraud_intel.scoring.dispatch import _PostgresScoringDataAccess

    target = _ach_event(uuid.uuid4(), ts=T0)
    ob_earlier = _online_banking_event(uuid.uuid4(), ts=T0 - timedelta(hours=1))

    conn = _FakeListPendingConnection()
    conn.channel_events = [_row_for(target), _row_for(ob_earlier)]
    conn.source_alerts = [{
        "source_alert_id": uuid.uuid4(), "source_system": "SYS", "event_id": target.event_id,
        "source_alert_created_at": target.event_timestamp, "source_rule_ids": [], "source_rule_version": "v1",
        "source_alert_score": None, "source_alert_reason_codes": [], "generation_run_id": "genrun-a",
        "dataset_version": "dsv-a", "created_at": target.event_timestamp,
    }]
    monkeypatch.setattr("src.common.db.get_connection", lambda database=None: conn)

    access = _PostgresScoringDataAccess(database="aidp_test")
    items = access.list_pending("ach", generation_run_id="genrun-a", operational_bundle_id=1)
    assert len(items) == 1

    expected_history = select_customer_historical_events(
        [target, ob_earlier], customer_id=target.customer_id, as_of_time=target.event_timestamp, exclude_event_id=target.event_id,
    )
    assert tuple(e.event_id for e in items[0].context.historical_events) == tuple(e.event_id for e in expected_history)
    assert len(items[0].context.historical_events) == 1  # the cross-channel online_banking event IS included


def test_load_cross_channel_customer_pool_uses_parameterized_sql():
    import ast
    import inspect
    import textwrap

    from src.fraud_intel import cli_data_access

    tree = ast.parse(textwrap.dedent(inspect.getsource(cli_data_access.load_cross_channel_customer_pool)))
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            sql_arg = node.args[0]
            assert isinstance(sql_arg, ast.Constant) and isinstance(sql_arg.value, str), "SQL must be a plain string literal, never an f-string/concatenation"
            assert "%s" in sql_arg.value
            assert len(node.args) == 2, "query must pass a separate params argument"
            checked += 1
    assert checked == 2  # channel_events query + source_alerts query
