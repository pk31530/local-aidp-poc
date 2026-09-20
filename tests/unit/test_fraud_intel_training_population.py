"""Phase 4: the supervised training population (guide section 11) --
source_alerts -> channel_events -> Phase 2 feature vectors -> Phase-4-
interim-eligible synthetic labels. Non-alerted events must feed history
only, never become a supervised row. No database, Docker, or network
access anywhere in this file.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.source_alert_context import SourceAlertContext, SyntheticGroundTruthLabel
from src.fraud_intel.features.channels.online_banking import ONLINE_BANKING_FEATURE_COLUMNS
from src.fraud_intel.models.training import (
    PHASE4_INTERIM_ELIGIBILITY_POLICY_VERSION,
    _build_supervised_population,
    _phase4_interim_training_eligible,
)

CUSTOMER = "FIC1000"
ACCOUNT = "FIA100000"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _event(*, event_timestamp: datetime, event_id: uuid.UUID | None = None, channel: str = "online_banking") -> FraudEvent:
    return FraudEvent(
        event_id=event_id or uuid.uuid4(),
        channel=channel,
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=event_timestamp,
        amount_minor_units=10_000,
        direction="debit",
        device_id="DEV1",
        channel_payload=OnlineBankingPayload(
            session_id="SESS1", login_method="password", mfa_used_flag=True, transaction_type="transfer", target_account="TGT1"
        ),
    )


def _alert(*, event_id: uuid.UUID, created_at: datetime) -> SourceAlertContext:
    return SourceAlertContext(
        source_system="LocalYamlRuleProvider (simulated upstream)",
        event_id=event_id,
        source_alert_created_at=created_at,
        source_rule_ids=["RULE1"],
        source_rule_version="v1",
        source_alert_reason_codes=["REASON1"],
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        created_at=created_at,
    )


def _label(*, event_id: uuid.UUID, synthetic_scenario_label: bool, label_source: str = "SYNTHETIC_GENERATOR") -> SyntheticGroundTruthLabel:
    return SyntheticGroundTruthLabel(
        event_id=event_id,
        scenario_id="scenario-1",
        synthetic_scenario_label=synthetic_scenario_label,
        scenario_type="ONLINE_BANKING_CREDENTIAL_STUFFING_TRANSFER" if synthetic_scenario_label else "NORMAL",
        generation_run_id="genrun-1",
        dataset_version="dsv-1",
        generated_at=T0,
    )


# ---- eligibility policy ----------------------------------------------------------


def test_interim_eligibility_policy_version_is_recorded_and_stable():
    assert PHASE4_INTERIM_ELIGIBILITY_POLICY_VERSION == "synthetic_label_eligibility_v1"


def test_synthetic_generator_label_is_eligible():
    label = _label(event_id=uuid.uuid4(), synthetic_scenario_label=True)
    assert _phase4_interim_training_eligible(label) is True


def test_eligibility_function_signature_only_accepts_synthetic_ground_truth_label():
    import inspect

    sig = inspect.signature(_phase4_interim_training_eligible)
    (only_param,) = sig.parameters.values()
    assert "SyntheticGroundTruthLabel" in str(only_param.annotation)


# ---- source-alert-only population -------------------------------------------------


def test_population_includes_only_source_alerted_events():
    alerted_event = _event(event_timestamp=T0)
    non_alerted_event = _event(event_timestamp=T0 - timedelta(hours=1))

    rows = _build_supervised_population(
        channel_events=[alerted_event, non_alerted_event],
        source_alerts=[_alert(event_id=alerted_event.event_id, created_at=alerted_event.event_timestamp)],
        synthetic_labels=[
            _label(event_id=alerted_event.event_id, synthetic_scenario_label=True),
            _label(event_id=non_alerted_event.event_id, synthetic_scenario_label=False),
        ],
    )

    assert len(rows) == 1
    assert rows[0]["event_id"] == str(alerted_event.event_id)


def test_non_alerted_event_with_a_label_is_excluded_even_though_labeled():
    """The exact guide-required proof: a fixture event with NO
    source_alerts row is correctly excluded from the training population
    even though it has a synthetic_event_labels row."""
    alerted_event = _event(event_timestamp=T0)
    non_alerted_but_labeled_event = _event(event_timestamp=T0 - timedelta(hours=2))

    rows = _build_supervised_population(
        channel_events=[alerted_event, non_alerted_but_labeled_event],
        source_alerts=[_alert(event_id=alerted_event.event_id, created_at=alerted_event.event_timestamp)],
        synthetic_labels=[
            _label(event_id=alerted_event.event_id, synthetic_scenario_label=True),
            _label(event_id=non_alerted_but_labeled_event.event_id, synthetic_scenario_label=True),
        ],
    )

    result_event_ids = {row["event_id"] for row in rows}
    assert str(alerted_event.event_id) in result_event_ids
    assert str(non_alerted_but_labeled_event.event_id) not in result_event_ids


def test_every_population_row_has_a_corresponding_source_alerts_entry():
    events = [_event(event_timestamp=T0 - timedelta(hours=h)) for h in range(5)]
    alerted = events[:3]
    alerts = [_alert(event_id=e.event_id, created_at=e.event_timestamp) for e in alerted]
    labels = [_label(event_id=e.event_id, synthetic_scenario_label=(i % 2 == 0)) for i, e in enumerate(events)]

    rows = _build_supervised_population(channel_events=events, source_alerts=alerts, synthetic_labels=labels)

    alerted_event_ids = {str(e.event_id) for e in alerted}
    assert {row["event_id"] for row in rows} == alerted_event_ids


def test_event_missing_a_label_is_excluded_not_crashed():
    event = _event(event_timestamp=T0)
    rows = _build_supervised_population(
        channel_events=[event],
        source_alerts=[_alert(event_id=event.event_id, created_at=event.event_timestamp)],
        synthetic_labels=[],  # no label at all
    )
    assert rows == []


def test_non_online_banking_events_are_excluded_in_phase_4():
    from src.fraud_intel.events.ach import ACHPayload
    from datetime import date

    ach_event = FraudEvent(
        channel="ach",
        customer_id=CUSTOMER,
        account_id=ACCOUNT,
        event_timestamp=T0,
        amount_minor_units=10_000,
        direction="debit",
        channel_payload=ACHPayload(
            sec_code="PPD", originating_routing_number="123456789", receiving_routing_number="987654321",
            batch_id="BATCH1", effective_entry_date=date(2026, 1, 1), company_id="COMP1",
        ),
    )
    rows = _build_supervised_population(
        channel_events=[ach_event],
        source_alerts=[_alert(event_id=ach_event.event_id, created_at=ach_event.event_timestamp)],
        synthetic_labels=[_label(event_id=ach_event.event_id, synthetic_scenario_label=True)],
    )
    assert rows == []


# ---- non-alerted events as history only -------------------------------------------


def test_non_alerted_prior_events_feed_history_but_never_become_a_row():
    prior_non_alerted = _event(event_timestamp=T0 - timedelta(hours=3))
    current = _event(event_timestamp=T0)

    rows = _build_supervised_population(
        channel_events=[prior_non_alerted, current],
        source_alerts=[_alert(event_id=current.event_id, created_at=current.event_timestamp)],
        synthetic_labels=[
            _label(event_id=current.event_id, synthetic_scenario_label=False),
        ],
    )
    assert len(rows) == 1
    assert rows[0]["event_id"] == str(current.event_id)
    # events_last_24h (a shared feature) must reflect the non-alerted prior
    # event's presence in HISTORY, proving it was used as context.
    assert rows[0]["events_last_24h"] == 1


# ---- current event's own alert never counted as prior ------------------------------


def test_current_events_own_alert_never_counted_as_a_prior_alert_in_features():
    event = _event(event_timestamp=T0)
    own_alert = _alert(event_id=event.event_id, created_at=event.event_timestamp)
    genuinely_prior_alert_event = _event(event_timestamp=T0 - timedelta(hours=1))
    genuinely_prior_alert = _alert(event_id=genuinely_prior_alert_event.event_id, created_at=genuinely_prior_alert_event.event_timestamp)

    rows = _build_supervised_population(
        channel_events=[event, genuinely_prior_alert_event],
        source_alerts=[own_alert, genuinely_prior_alert],
        synthetic_labels=[
            _label(event_id=event.event_id, synthetic_scenario_label=False),
            _label(event_id=genuinely_prior_alert_event.event_id, synthetic_scenario_label=False),
        ],
    )
    row = next(r for r in rows if r["event_id"] == str(event.event_id))
    assert row["prior_alert_count"] == 1  # only the genuinely prior alert, never its own


# ---- output shape ------------------------------------------------------------------


def test_population_rows_carry_the_full_ordered_online_banking_feature_set():
    event = _event(event_timestamp=T0)
    rows = _build_supervised_population(
        channel_events=[event],
        source_alerts=[_alert(event_id=event.event_id, created_at=event.event_timestamp)],
        synthetic_labels=[_label(event_id=event.event_id, synthetic_scenario_label=True)],
    )
    row = rows[0]
    assert set(ONLINE_BANKING_FEATURE_COLUMNS) <= set(row.keys())
    assert row["label"] is True


def test_population_is_sorted_by_event_timestamp_then_event_id():
    events = [_event(event_timestamp=T0 - timedelta(hours=h)) for h in [3, 1, 2]]
    alerts = [_alert(event_id=e.event_id, created_at=e.event_timestamp) for e in events]
    labels = [_label(event_id=e.event_id, synthetic_scenario_label=False) for e in events]

    rows = _build_supervised_population(channel_events=events, source_alerts=alerts, synthetic_labels=labels)
    timestamps = [row["event_timestamp"] for row in rows]
    assert timestamps == sorted(timestamps)
