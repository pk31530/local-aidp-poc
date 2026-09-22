"""Synthetic-label-leakage regression suite for the Debit Card channel.

Root cause this file exists to pin down: the original generator assigned
four DebitCardPayload fields deterministically from `is_fraud` --
`card_present_flag = not is_fraud` (which the feature adapter turns into
the `card_not_present_flag` MODEL FEATURE, making it an exact copy of the
label), `cross_border_flag = is_hard_fp`, a `card_token` whose digit COUNT
differed by label, and disjoint-ish `amount_minor_units` ranges that left a
pure-fraud band. Held-out metrics were consequently a perfect 1.0 and said
nothing about detection quality.

These tests prove STRUCTURALLY that the direct proxies are gone -- they
deliberately never assert "model metrics must be below 1.0", which would
be a symptom test rather than a cause test.

Pure and offline: the generator, the feature adapter and the split are all
pure functions. No database, MLflow, MinIO, Docker or network access
anywhere in this file.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.common.splits import assign_chronological_split, realized_split_fractions
from src.fraud_intel.cli_data_access import (
    CHANNEL_GENERATOR_VERSION,
    GENERATION_SPEC_VERSION,
    _generation_identity,
)
from src.fraud_intel.config import ChannelTrainingRunConfig
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.features.core import SHARED_FEATURE_COLUMNS, ordered_feature_vector
from src.fraud_intel.generator.customers import generate_customers
from src.fraud_intel.generator.debit_card import (
    CARD_NOT_PRESENT_PROBABILITY,
    CROSS_BORDER_PROBABILITY,
    SMALL_AMOUNT_PROBABILITY,
    generate_debit_card_events,
)
from src.fraud_intel.models.training import (
    _build_supervised_population,
    _class_counts,
    _validate_partition,
)
from src.fraud_intel.registry import get_channel_adapter

CHANNEL = "debit_card"

# The approved deterministic demonstration population
# (docs/DEBIT_CARD_FULL_LIFECYCLE_DEMO.md).
COUNT, SEED, REFERENCE_DATE = 5000, 99, date(2026, 9, 22)

# The identity the PRE-FIX generator issued for exactly that spec. It must
# never be re-issued: the pre-fix and post-fix populations are
# methodologically different data.
RETIRED_PRE_FIX_GENERATION_RUN_ID = "genrun-95ad70f19e5ec9fc"
RETIRED_PRE_FIX_DATASET_VERSION = "dsv-95ad70f19e5ec9fc"


def _generate(count=COUNT, seed=SEED, reference_date=REFERENCE_DATE):
    run_id, dataset_version = _generation_identity(
        channel=CHANNEL, count=count, seed=seed, reference_date=reference_date
    )
    customers = generate_customers(n=max(count // 5, 1), seed=seed, reference_date=reference_date)
    return generate_debit_card_events(
        seed=seed, n=count, reference_date=reference_date, customers=customers,
        generation_run_id=run_id, dataset_version=dataset_version,
    )


@pytest.fixture(scope="module")
def population():
    """The approved deterministic population, generated once."""
    return _generate()


@pytest.fixture(scope="module")
def supervised_rows(population):
    """The real supervised training population, built by the REAL
    production builder (_build_supervised_population) through the REAL
    shared historical feature selectors -- not a hand-made fixture."""
    events = [e for e, _, _ in population]
    alerts = [sa for _, sa, _ in population if sa is not None]
    labels = [l for _, _, l in population]
    return _build_supervised_population(
        adapter=get_channel_adapter(CHANNEL),
        channel_events=events, source_alerts=alerts, synthetic_labels=labels,
        history_events=events, history_source_alerts=alerts,
    )


def _binary_columns(rows, columns):
    """Columns whose observed values are exactly {True, False} or a single
    boolean -- i.e. every column a "equals/inverts the label" test can
    meaningfully apply to."""
    out = []
    for column in columns:
        values = {row[column] for row in rows}
        if values and all(isinstance(v, bool) for v in values):
            out.append(column)
    return out


# ============================================================================
# A. Determinism
# ============================================================================


def test_identical_inputs_produce_byte_identical_events():
    first, second = _generate(count=400), _generate(count=400)
    assert len(first) == len(second) == 400
    for (e1, a1, l1), (e2, a2, l2) in zip(first, second):
        assert e1.model_dump(mode="json") == e2.model_dump(mode="json")
        assert (a1 is None) == (a2 is None)
        if a1 is not None:
            assert a1.model_dump(mode="json") == a2.model_dump(mode="json")
        assert l1.model_dump(mode="json") == l2.model_dump(mode="json")


def test_repeated_identity_calculation_is_stable():
    a = _generation_identity(channel=CHANNEL, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE)
    b = _generation_identity(channel=CHANNEL, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE)
    assert a == b


def test_a_different_seed_still_produces_a_different_population():
    """Determinism must not have become constancy."""
    assert [e.event_id for e, _, _ in _generate(count=200, seed=SEED)] != [
        e.event_id for e, _, _ in _generate(count=200, seed=SEED + 1)
    ]


# ============================================================================
# B. Identity safety
# ============================================================================


def test_corrected_debit_card_identity_differs_from_the_retired_pre_fix_identity():
    run_id, dataset_version = _generation_identity(
        channel=CHANNEL, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE
    )
    assert run_id != RETIRED_PRE_FIX_GENERATION_RUN_ID
    assert dataset_version != RETIRED_PRE_FIX_DATASET_VERSION


def test_debit_card_has_a_registered_generator_version():
    assert CHANNEL_GENERATOR_VERSION[CHANNEL] == "v2"


@pytest.mark.parametrize(
    "channel", ["ach", "wire", "mobile_deposit", "online_banking", "atm", "p2p"]
)
def test_other_channel_identities_are_unchanged_by_the_channel_scoped_version(channel):
    """Channel-scoped versioning: a channel ABSENT from
    CHANNEL_GENERATOR_VERSION must hash exactly as it did before that map
    existed -- recomputed here from the pre-map canonical form rather than
    trusted."""
    import hashlib
    import json

    assert channel not in CHANNEL_GENERATOR_VERSION
    canonical = json.dumps(
        {
            "channel": channel, "count": COUNT, "seed": SEED,
            "reference_date": REFERENCE_DATE.isoformat(),
            "generation_spec_version": GENERATION_SPEC_VERSION,
        },
        sort_keys=True, separators=(",", ":"),
    )
    expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    assert _generation_identity(
        channel=channel, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE
    ) == (f"genrun-{expected_hash}", f"dsv-{expected_hash}")


def test_registering_a_new_channel_version_would_change_only_that_channel(monkeypatch):
    """The mechanism itself, exercised: registering a version for a channel
    changes that channel's identity and no other's."""
    before = {
        c: _generation_identity(channel=c, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE)
        for c in ("ach", "wire", CHANNEL)
    }
    monkeypatch.setitem(CHANNEL_GENERATOR_VERSION, "wire", "v2")
    after = {
        c: _generation_identity(channel=c, count=COUNT, seed=SEED, reference_date=REFERENCE_DATE)
        for c in ("ach", "wire", CHANNEL)
    }
    assert after["wire"] != before["wire"]
    assert after["ach"] == before["ach"]
    assert after[CHANNEL] == before[CHANNEL]


# ============================================================================
# C. Leakage correction
# ============================================================================


def test_card_not_present_flag_is_not_a_copy_or_inverse_of_the_label(supervised_rows):
    equals = [row for row in supervised_rows if bool(row["card_not_present_flag"]) == bool(row["label"])]
    inverts = [row for row in supervised_rows if bool(row["card_not_present_flag"]) != bool(row["label"])]
    # Both groups must be non-empty: if either were empty the feature would
    # be an exact copy or an exact inverse of the label.
    assert equals, "card_not_present_flag never agrees with the label -- it is an exact INVERSE"
    assert inverts, "card_not_present_flag always agrees with the label -- it is an exact COPY"


def test_both_card_present_values_occur_among_fraud_rows(supervised_rows):
    values = {bool(row["card_not_present_flag"]) for row in supervised_rows if row["label"]}
    assert values == {True, False}, f"fraud rows only ever have card_not_present_flag={values}"


def test_both_card_present_values_occur_among_legitimate_rows(supervised_rows):
    values = {bool(row["card_not_present_flag"]) for row in supervised_rows if not row["label"]}
    assert values == {True, False}, f"legitimate rows only ever have card_not_present_flag={values}"


def test_no_single_binary_feature_equals_or_inverts_the_label(supervised_rows):
    """The general form of the defect, applied to EVERY binary Debit Card
    model feature -- so a future generator change that reintroduces the
    same class of leak through a different field is caught here."""
    adapter = get_channel_adapter(CHANNEL)
    labels = [bool(row["label"]) for row in supervised_rows]
    offenders = []
    for column in _binary_columns(supervised_rows, adapter.feature_columns):
        values = [bool(row[column]) for row in supervised_rows]
        if len(set(values)) == 1:
            continue  # constant column carries no label information at all
        if all(v == y for v, y in zip(values, labels)):
            offenders.append(f"{column} is an exact COPY of the label")
        if all(v != y for v, y in zip(values, labels)):
            offenders.append(f"{column} is an exact INVERSE of the label")
    assert not offenders, offenders


def test_card_token_shape_does_not_encode_the_label(population):
    """The token is a graph entity (extract_entities), so a label-dependent
    token shape leaked into the graph component of the operational score."""
    fraud_lengths = {len(e.channel_payload.card_token) for e, _, l in population if l.synthetic_scenario_label}
    legit_lengths = {len(e.channel_payload.card_token) for e, _, l in population if not l.synthetic_scenario_label}
    assert fraud_lengths == legit_lengths == {len("CARDTOK") + 6}


def test_cross_border_flag_does_not_imply_a_legitimate_row(population):
    """cross_border_flag = is_hard_fp made cross_border_flag=True a
    certainty of legitimacy. It feeds both the
    DEBIT_CARD_CROSS_BORDER_NEW_DEVICE rule and the
    cross_border_new_device_combo_flag feature."""
    fraud_cross_border = sum(
        1 for e, _, l in population if e.channel_payload.cross_border_flag and l.synthetic_scenario_label
    )
    legit_cross_border = sum(
        1 for e, _, l in population if e.channel_payload.cross_border_flag and not l.synthetic_scenario_label
    )
    assert fraud_cross_border > 0 and legit_cross_border > 0


def test_amount_range_has_no_pure_fraud_or_pure_legitimate_band(population):
    """Both populations must reach both ends of one SHARED amount support;
    a label-specific range leaves a band that identifies the label with
    certainty."""
    fraud = [e.amount_minor_units for e, _, l in population if l.synthetic_scenario_label]
    legit = [e.amount_minor_units for e, _, l in population if not l.synthetic_scenario_label]
    assert min(fraud) == min(legit), "the amount floor differs by label -- the lower band is pure"
    assert not (max(fraud) < min(legit) or max(legit) < min(fraud)), "amount is threshold-separable"


@pytest.mark.parametrize(
    "probabilities",
    [CARD_NOT_PRESENT_PROBABILITY, SMALL_AMOUNT_PROBABILITY],
    ids=["card_not_present", "small_amount"],
)
def test_payload_probabilities_are_never_certain_for_any_scenario_kind(probabilities):
    """A probability of exactly 0 or 1 for any scenario kind reintroduces a
    one-sided certainty. CROSS_BORDER_PROBABILITY is excluded deliberately:
    its HARD_FALSE_POSITIVE entry is 1.0 by definition of that typology,
    and is made safe by the fraud/normal entries being non-zero, which
    test_cross_border_flag_does_not_imply_a_legitimate_row proves."""
    for kind, probability in probabilities.items():
        assert 0.0 < probability < 1.0, f"{kind} has a certain probability {probability}"


def test_cross_border_probability_is_non_zero_for_fraud_and_normal():
    assert CROSS_BORDER_PROBABILITY["FRAUD_SCENARIO"] > 0
    assert CROSS_BORDER_PROBABILITY["NORMAL"] > 0


# ============================================================================
# D. Chronological safety
# ============================================================================


def test_no_label_field_reaches_the_feature_vector(supervised_rows):
    """Structural: a supervised row's feature columns are exactly the
    adapter's declared columns -- `label`, `event_id` and
    `event_timestamp` are carried alongside, never inside, the vector."""
    adapter = get_channel_adapter(CHANNEL)
    row = supervised_rows[0]
    assert set(row) == set(adapter.feature_columns) | {"event_id", "event_timestamp", "label"}
    vector = ordered_feature_vector(
        {c: row[c] for c in adapter.feature_columns}, adapter.feature_columns
    )
    assert len(vector) == len(adapter.feature_columns)


def test_feature_adapter_never_references_a_label_type():
    """AST/source-level: the Debit Card feature module cannot read
    synthetic_scenario_label, scenario_id, analyst dispositions or any
    outcome field -- they are not in its import graph at all."""
    import inspect

    from src.fraud_intel.features.channels import debit_card as feature_module

    source = inspect.getsource(feature_module)
    for forbidden in (
        "SyntheticGroundTruthLabel", "synthetic_scenario_label", "scenario_id",
        "analyst_disposition", "outcome_status", "resolved_label", "training_eligible",
    ):
        assert forbidden not in source, f"{forbidden} must never appear in the feature adapter"


def test_historical_features_use_only_strictly_earlier_events(population):
    """FeatureComputationContext fails fast on a boundary or future
    historical event. Building every supervised row through the real
    builder therefore proves the whole corrected population is as-of-time
    safe -- if any row's history included its own or a later event, the
    fixture would have raised."""
    from src.fraud_intel.features.core import FeatureComputationContext

    events = [e for e, _, _ in population]
    target = max(events, key=lambda e: e.event_timestamp)
    later = [r for r in events if r.event_timestamp >= target.event_timestamp and r.event_id != target.event_id]
    with pytest.raises(Exception):
        FeatureComputationContext(
            current_event=target,
            historical_events=tuple([target] if not later else later[:1]),
            source_alert_history=(),
            as_of_time=target.event_timestamp,
        )


def test_supervised_population_is_ordered_by_event_time(supervised_rows):
    timestamps = [(row["event_timestamp"], row["event_id"]) for row in supervised_rows]
    assert timestamps == sorted(timestamps)


def test_chronological_split_of_the_corrected_population_is_leakage_safe(supervised_rows):
    adapter = get_channel_adapter(CHANNEL)
    config = ChannelTrainingRunConfig(channel=CHANNEL)
    df = pl.DataFrame(
        [
            {"event_id": r["event_id"], "event_timestamp": r["event_timestamp"], "label": r["label"],
             **{c: r[c] for c in adapter.feature_columns}}
            for r in supervised_rows
        ]
    )
    split_df = assign_chronological_split(
        df, timestamp_col="event_timestamp", id_col="event_id",
        train_frac=config.train_frac, calib_frac=config.calib_frac, test_frac=config.test_frac,
        purge_gap=timedelta(seconds=config.purge_gap_seconds),
    )
    parts = {
        name: split_df.filter(pl.col("split") == name) for name in ("train", "calibration", "test")
    }
    ids = {name: set(part["event_id"].to_list()) for name, part in parts.items()}
    stamps = {name: set(part["event_timestamp"].to_list()) for name, part in parts.items()}
    for a, b in (("train", "calibration"), ("calibration", "test"), ("train", "test")):
        assert not ids[a] & ids[b], f"{a}/{b} share row ids"
        assert not stamps[a] & stamps[b], f"{a}/{b} straddle an equal-timestamp group"
    assert max(stamps["train"]) <= min(stamps["calibration"])
    assert max(stamps["calibration"]) <= min(stamps["test"])


# ============================================================================
# E. Pipeline compatibility
# ============================================================================


def test_every_generated_payload_is_a_valid_debit_card_payload(population):
    for event, _, _ in population:
        assert isinstance(event.channel_payload, DebitCardPayload)
        # Round-trips through pydantic validation exactly as the real
        # persistence path does (channel_payload JSONB -> payload_class).
        assert DebitCardPayload(**event.channel_payload.model_dump(mode="json")) == event.channel_payload
        assert event.amount_minor_units > 0  # channel_events CHECK constraint


def test_feature_vector_width_and_ordering_are_unchanged():
    adapter = get_channel_adapter(CHANNEL)
    # ChannelAdapter.feature_columns is a tuple (frozen ordering); compare
    # as a sequence of names rather than relying on the container type.
    assert list(adapter.feature_columns) == SHARED_FEATURE_COLUMNS + [
        "card_not_present_flag", "cross_border_new_device_combo_flag", "distinct_merchant_count_1h",
    ]
    assert len(adapter.feature_columns) == 15
    assert adapter.feature_schema_version == "v1"


def test_training_and_scoring_build_the_same_feature_vector(population):
    """The supervised-population builder and the pure scoring path must
    agree column-for-column for the same event and the same history --
    otherwise a model trained on one is scored on another."""
    from src.fraud_intel.features.core import FeatureComputationContext
    from src.fraud_intel.features.history import (
        select_customer_historical_events,
        select_customer_historical_source_alerts,
    )

    adapter = get_channel_adapter(CHANNEL)
    events = [e for e, _, _ in population]
    alerts = [sa for _, sa, _ in population if sa is not None]
    labels = [l for _, _, l in population]
    rows = _build_supervised_population(
        adapter=adapter, channel_events=events[:600], source_alerts=[a for a in alerts if any(str(a.event_id) == str(e.event_id) for e in events[:600])],
        synthetic_labels=labels[:600], history_events=events[:600],
        history_source_alerts=[a for a in alerts if any(str(a.event_id) == str(e.event_id) for e in events[:600])],
    )
    assert rows, "expected at least one supervised row in the sampled slice"
    by_id = {str(e.event_id): e for e in events}
    customer_ids = {e.event_id: e.customer_id for e in events[:600]}
    for row in rows[:5]:
        event = by_id[row["event_id"]]
        ctx = FeatureComputationContext(
            current_event=event,
            historical_events=select_customer_historical_events(
                events[:600], customer_id=event.customer_id, as_of_time=event.event_timestamp,
                exclude_event_id=event.event_id,
            ),
            source_alert_history=select_customer_historical_source_alerts(
                [a for a in alerts if any(str(a.event_id) == str(e.event_id) for e in events[:600])],
                event_customer_ids=customer_ids, customer_id=event.customer_id,
                as_of_time=event.event_timestamp, exclude_event_id=event.event_id,
            ),
            as_of_time=event.event_timestamp,
        )
        scoring_features = adapter.compute_features(ctx)
        for column in adapter.feature_columns:
            assert scoring_features[column] == row[column], f"{column} differs between training and scoring"


def test_corrected_population_passes_the_real_minimum_row_and_class_gates(supervised_rows):
    adapter = get_channel_adapter(CHANNEL)
    config = ChannelTrainingRunConfig(channel=CHANNEL)
    df = pl.DataFrame(
        [
            {"event_id": r["event_id"], "event_timestamp": r["event_timestamp"], "label": r["label"],
             **{c: r[c] for c in adapter.feature_columns}}
            for r in supervised_rows
        ]
    )
    split_df = assign_chronological_split(
        df, timestamp_col="event_timestamp", id_col="event_id",
        train_frac=config.train_frac, calib_frac=config.calib_frac, test_frac=config.test_frac,
        purge_gap=timedelta(seconds=config.purge_gap_seconds),
    )
    for name, minimum in (
        ("train", config.min_train_rows), ("calibration", config.min_calib_rows), ("test", config.min_test_rows),
    ):
        part = split_df.filter(pl.col("split") == name)
        _validate_partition(part, name, min_rows=minimum)  # raises on failure
        counts = _class_counts(part)
        # The cold-start promotion gate's own stricter floor.
        assert counts["fraud"] >= 5 and counts["legitimate"] >= 5, (name, counts)
    assert realized_split_fractions(split_df)["train"] > 0.5
