"""Phase 1: generator determinism, prevalence, provenance tagging, and
static (no-database) schema/migration-003 consistency. Generator tests use
models/fixtures only -- no database, Docker, or network access anywhere.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from src.common.config import PROJECT_ROOT
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import (
    SourceAlertContext,
    SyntheticGroundTruthLabel,
)
from src.fraud_intel.generator.ach import generate_ach_events
from src.fraud_intel.generator.atm import generate_atm_events
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.generator.debit_card import generate_debit_card_events
from src.fraud_intel.generator.mobile_deposit import generate_mobile_deposit_events
from src.fraud_intel.generator.online_banking import generate_online_banking_events
from src.fraud_intel.generator.p2p import generate_p2p_events
from src.fraud_intel.generator.wire import generate_wire_events

REFERENCE_DATE = date(2026, 1, 1)
SEED = 42
GENERATION_RUN_ID = "genrun-test-1"
DATASET_VERSION = "dsv-test-1"

GENERATORS = {
    "ach": generate_ach_events,
    "wire": generate_wire_events,
    "mobile_deposit": generate_mobile_deposit_events,
    "online_banking": generate_online_banking_events,
    "atm": generate_atm_events,
    "debit_card": generate_debit_card_events,
    "p2p": generate_p2p_events,
}


def _run(channel: str, n: int = 200, seed: int = SEED):
    customers = generate_customers(n=50, seed=seed, reference_date=REFERENCE_DATE)
    return GENERATORS[channel](
        seed=seed,
        n=n,
        reference_date=REFERENCE_DATE,
        customers=customers,
        generation_run_id=GENERATION_RUN_ID,
        dataset_version=DATASET_VERSION,
    )


@pytest.mark.parametrize("channel", GENERATORS.keys())
def test_generator_produces_correct_tuple_shape(channel):
    results = _run(channel, n=20)
    assert len(results) == 20
    for event, source_alert, label in results:
        assert isinstance(event, FraudEvent)
        assert source_alert is None or isinstance(source_alert, SourceAlertContext)
        assert isinstance(label, SyntheticGroundTruthLabel)
        assert event.channel == channel
        assert event.channel_payload.channel == channel
        assert label.event_id == event.event_id
        if source_alert is not None:
            assert source_alert.event_id == event.event_id


@pytest.mark.parametrize("channel", GENERATORS.keys())
def test_generator_is_deterministic_for_a_fixed_seed(channel):
    first = _run(channel, n=100)
    second = _run(channel, n=100)
    assert len(first) == len(second)
    for (e1, s1, l1), (e2, s2, l2) in zip(first, second):
        assert e1.model_dump(mode="json") == e2.model_dump(mode="json")
        assert l1.model_dump(mode="json") == l2.model_dump(mode="json")
        s1_dump = s1.model_dump(mode="json") if s1 is not None else None
        s2_dump = s2.model_dump(mode="json") if s2 is not None else None
        assert s1_dump == s2_dump


def test_generator_is_deterministic_across_channels_combined():
    """Regenerating the full seven-channel batch twice with the same seed
    must produce byte-for-byte identical output, not just per-channel."""
    customers = generate_customers(n=50, seed=SEED, reference_date=REFERENCE_DATE)

    def full_batch():
        out = []
        for channel, fn in GENERATORS.items():
            out.extend(
                fn(
                    seed=SEED,
                    n=100,
                    reference_date=REFERENCE_DATE,
                    customers=customers,
                    generation_run_id=GENERATION_RUN_ID,
                    dataset_version=DATASET_VERSION,
                )
            )
        return out

    first = full_batch()
    second = full_batch()
    assert [e.model_dump(mode="json") for e, _, _ in first] == [e.model_dump(mode="json") for e, _, _ in second]


@pytest.mark.parametrize("channel", GENERATORS.keys())
def test_fraud_prevalence_is_within_configured_low_range(channel):
    """Configured default prevalence is 3% -- below the guide's 5%
    threshold (section 7: "including scenarios below 5%")."""
    results = _run(channel, n=3000)
    labels = [label for _, _, label in results]
    fraud_count = sum(1 for label in labels if label.synthetic_scenario_label)
    prevalence = fraud_count / len(labels)
    assert 0.0 < prevalence < 0.06


def test_hard_false_positives_are_deliberately_alert_firing():
    """Hard false positives -- legitimate-but-unusual events -- must fire a
    source alert often (guide section 7), not be indistinguishable from
    ordinary normal events."""
    results = _run("ach", n=3000)
    hard_fp_fire_count = 0
    hard_fp_count = 0
    for event, source_alert, label in results:
        if label.scenario_type == "ACH_LEGITIMATE_LARGE_BATCH":
            hard_fp_count += 1
            if source_alert is not None:
                hard_fp_fire_count += 1
    assert hard_fp_count > 0
    assert hard_fp_fire_count / hard_fp_count > 0.5


@pytest.mark.parametrize("channel", GENERATORS.keys())
def test_provenance_fields_present_on_every_generated_record(channel):
    results = _run(channel, n=50)
    for event, source_alert, label in results:
        assert label.generation_run_id == GENERATION_RUN_ID
        assert label.dataset_version == DATASET_VERSION
        if source_alert is not None:
            assert source_alert.generation_run_id == GENERATION_RUN_ID
            assert source_alert.dataset_version == DATASET_VERSION


def test_source_alert_identity_is_unique_across_a_combined_batch():
    """Every fired source alert across all seven channels in one generation
    run must have a globally unique source_alert_id (guide section 6/18's
    idempotency-key requirement depends on this holding at generation
    time)."""
    customers = generate_customers(n=50, seed=SEED, reference_date=REFERENCE_DATE)
    all_source_alert_ids = []
    for channel, fn in GENERATORS.items():
        results = fn(
            seed=SEED,
            n=300,
            reference_date=REFERENCE_DATE,
            customers=customers,
            generation_run_id=GENERATION_RUN_ID,
            dataset_version=DATASET_VERSION,
        )
        for _, source_alert, _ in results:
            if source_alert is not None:
                all_source_alert_ids.append(source_alert.source_alert_id)

    assert len(all_source_alert_ids) > 0
    assert len(set(all_source_alert_ids)) == len(all_source_alert_ids)


def test_an_event_may_have_zero_or_one_source_alert_at_generation_time():
    """Phase 1's generators produce at most one SourceAlertContext per
    event (multi-alert-per-event is a Phase 6/7B persistence-layer
    scenario, e.g. re-evaluation under a new rule_set_version -- not
    something a single deterministic generation pass needs to construct)."""
    results = _run("ach", n=200)
    event_ids = [event.event_id for event, _, _ in results]
    assert len(event_ids) == len(set(event_ids))


# ---- Static schema.sql / migration 003 consistency (no database) --------------

SCHEMA_SQL_PATH = PROJECT_ROOT / "infrastructure" / "postgres" / "lib" / "schema.sql"
MIGRATION_003_PATH = PROJECT_ROOT / "infrastructure" / "postgres" / "migrations" / "003_channel_events_and_labels.sql"

EXPECTED_NEW_TABLES = {"channel_events", "source_alerts", "synthetic_event_labels", "channel_model_bundles"}


def _table_names(sql_text: str) -> set[str]:
    return set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", sql_text))


def test_migration_003_file_exists_and_is_not_applied_anywhere():
    assert MIGRATION_003_PATH.is_file()
    text = MIGRATION_003_PATH.read_text()
    assert "BEGIN;" in text
    assert "COMMIT;" in text
    assert "docker exec" not in text.split("--")[0]  # no live command outside comments


def test_schema_sql_and_migration_003_declare_the_same_new_tables():
    schema_text = SCHEMA_SQL_PATH.read_text()
    migration_text = MIGRATION_003_PATH.read_text()

    schema_tables = _table_names(schema_text)
    migration_tables = _table_names(migration_text)

    assert EXPECTED_NEW_TABLES <= schema_tables
    assert EXPECTED_NEW_TABLES == migration_tables


def test_schema_sql_and_migration_003_both_declare_the_partial_unique_index():
    expected = "CREATE UNIQUE INDEX IF NOT EXISTS uq_one_operational_bundle_per_channel"
    schema_text = SCHEMA_SQL_PATH.read_text()
    migration_text = MIGRATION_003_PATH.read_text()

    assert expected in schema_text
    assert expected in migration_text
    assert "WHERE status = 'OPERATIONAL'" in schema_text
    assert "WHERE status = 'OPERATIONAL'" in migration_text


def test_channel_model_bundles_status_allows_retired_in_both_files():
    schema_text = SCHEMA_SQL_PATH.read_text()
    migration_text = MIGRATION_003_PATH.read_text()
    expected = "CHECK (status IN ('CANDIDATE', 'OPERATIONAL', 'RETIRED'))"
    assert expected in schema_text
    assert expected in migration_text


def test_source_alerts_event_id_is_not_the_primary_key_or_unique_in_either_file():
    """event_id on source_alerts must remain a plain (non-unique) foreign
    key -- one event can have more than one source alert (guide section 6).
    """
    for path in (SCHEMA_SQL_PATH, MIGRATION_003_PATH):
        text = path.read_text()
        match = re.search(r"CREATE TABLE IF NOT EXISTS source_alerts \((.*?)\n\);", text, re.DOTALL)
        assert match is not None, f"source_alerts table not found in {path}"
        body = match.group(1)
        assert "event_id" in body
        for line in body.splitlines():
            if "event_id" in line and "UUID" in line:
                assert "PRIMARY KEY" not in line
                assert "UNIQUE" not in line
