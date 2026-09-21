"""Shared feature-history selection contract (Phase 7B corrective pass).

The ONE leakage-safe, customer-scoped, cross-channel, deterministically-
ordered selection rule used identically by training's supervised-
population construction (src.fraud_intel.models.training), real scoring
(src.fraud_intel.scoring.dispatch._PostgresScoringDataAccess.list_pending),
and shadow-candidate evaluation context loading
(src.fraud_intel.cli_data_access.load_resolved_alert_scoring_contexts).

Guide section 9's as-of-time contract, restated in section 15: "a feature
computed for an event at time T may only read records whose
event_timestamp is strictly earlier than T (ties broken deterministically
by (event_timestamp, event_id) ordering, never by insertion order)."
Section 15 explicitly builds the entity graph "from recent channel_events
history" -- the whole table, never filtered to one channel -- and
customer_id/account_id/device_id/ip_address are channel-agnostic identity
fields (section 6). This selection rule is therefore customer-scoped
ACROSS every channel, never restricted to the target event's own channel.
Channel-specific feature groups never call this module at all -- they
read only the target event's own channel_payload, never history.

Phase 7B corrective pass: this module replaces what were three
independently-written, subtly inconsistent implementations of "select
this customer's historical events/source-alerts as of time T" --
src.fraud_intel.models.training._build_supervised_population() (which was
channel-scoped only, discovered as a real bug against real ACH data:
2 of 441 alerts crossed a priority-band boundary between the channel-
scoped training/diagnostic reconstruction and real, cross-channel-scoped
scoring), and the two real Postgres readers above (which were already
correctly cross-channel, independently of each other and of training).

No database, Docker, or network access anywhere in this module -- it is
pure selection logic over an already-fetched candidate pool. Real
Postgres callers fetch a candidate pool (bounded by customer_id, cheaply
pre-filtered by event_timestamp where convenient) and then call these
functions for the authoritative filter/sort/limit/dedupe step, so every
caller's final selection is governed by this one implementation.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Mapping, Optional, Sequence

from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.source_alert_context import SourceAlertContext

DEFAULT_MAX_HISTORICAL_EVENTS = 1000
DEFAULT_MAX_SOURCE_ALERT_HISTORY = 200


def select_customer_historical_events(
    candidate_events: Sequence[FraudEvent],
    *,
    customer_id: str,
    as_of_time: datetime,
    exclude_event_id: Optional[uuid.UUID] = None,
    limit: int = DEFAULT_MAX_HISTORICAL_EVENTS,
) -> tuple[FraudEvent, ...]:
    """Every `candidate_events` row belonging to `customer_id` (ANY
    channel -- callers pass a cross-channel candidate pool, never a
    single-channel one) with `event_timestamp` strictly earlier than
    `as_of_time`. A row at exactly `as_of_time` is excluded, per the
    guide's own "strictly earlier than T" wording -- never included via a
    same-timestamp tie-break. `exclude_event_id` additionally drops the
    target event itself if present in `candidate_events` (a caller-
    provided safety net; not load-bearing in the normal case, since the
    strict timestamp filter already excludes it whenever the target's own
    row is present with its own, non-earlier timestamp). Another
    customer's events never match `customer_id` and are always excluded.
    Deterministic ordering: `event_timestamp` DESC, then `event_id` DESC
    as an explicit tiebreak among same-timestamp historical rows --
    exactly the "(event_timestamp, event_id) ordering, never insertion
    order" the guide requires -- truncated to the most recent `limit`
    qualifying events. De-dupes by `event_id` if the same event appears
    more than once across whatever pool(s) a caller assembled."""
    seen: dict[uuid.UUID, FraudEvent] = {}
    for event in candidate_events:
        if event.customer_id != customer_id:
            continue
        if event.event_id == exclude_event_id:
            continue
        if event.event_timestamp >= as_of_time:
            continue
        seen[event.event_id] = event
    ordered = sorted(seen.values(), key=lambda e: (e.event_timestamp, str(e.event_id)), reverse=True)
    return tuple(ordered[:limit])


def select_customer_historical_source_alerts(
    candidate_source_alerts: Sequence[SourceAlertContext],
    *,
    event_customer_ids: Mapping[uuid.UUID, str],
    customer_id: str,
    as_of_time: datetime,
    exclude_event_id: Optional[uuid.UUID] = None,
    limit: int = DEFAULT_MAX_SOURCE_ALERT_HISTORY,
) -> tuple[SourceAlertContext, ...]:
    """Same rule as select_customer_historical_events(), applied to
    source_alerts. `SourceAlertContext` itself carries no `customer_id`
    field, so a source alert's owning event's customer is resolved via
    `event_customer_ids` (event_id -> customer_id, built by the caller
    from the same candidate event pool). Ordered by
    `source_alert_created_at` DESC, then `source_alert_id` DESC."""
    seen: dict[uuid.UUID, SourceAlertContext] = {}
    for alert in candidate_source_alerts:
        if alert.event_id == exclude_event_id:
            continue
        if event_customer_ids.get(alert.event_id) != customer_id:
            continue
        if alert.source_alert_created_at >= as_of_time:
            continue
        seen[alert.source_alert_id] = alert
    ordered = sorted(seen.values(), key=lambda a: (a.source_alert_created_at, str(a.source_alert_id)), reverse=True)
    return tuple(ordered[:limit])
