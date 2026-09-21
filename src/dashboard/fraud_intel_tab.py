"""Phase 8: read-only "Fraud Intelligence" dashboard tab for the v1.3
seven-channel platform (`src/fraud_intel/`) -- entirely separate from the
legacy v1.1 "Fraud Analysis"/"Model Performance" tabs in `src/dashboard/
app.py`, which cover the original single-channel `transactions`/
`fraud_decisions` demo and are untouched by this module.

Strictly read-only: every function here only SELECTs. Nothing in this
module calls `record_disposition_and_update_status()`,
`score_and_record_alert()`, `create_alert_if_new()`, `record_evidence()`,
`RunLifecycle.begin()`, or any training/promotion/label-assessment
function -- disposition capture and every other write remains exclusively
`aidp alerts disposition` (CLI-only). A structural test
(`tests/smoke/test_fraud_intel_dashboard.py`) verifies no write-shaped
call appears anywhere in this file's source.

Current-state semantics (Phase 7B corrective pass, same contract as `aidp
alerts list`/`aidp alerts show`): "current" always means the alert's
LATEST `AlertEvidenceRecord` -- `list_alerts_with_current_state()` and
`get_latest_evidence()` (`src.fraud_intel.alerts.queue`) are the single
source of truth for this, both already ordering by
`scored_at DESC, evidence_id DESC` (`LATEST_EVIDENCE_ORDER_SQL`). This
module never reads `fraud_alerts.initial_*` and presents it as current --
those fields are shown separately, always labeled "initial" / "first
scoring".

Never exposes `scenario_id`, `synthetic_scenario_label`, or any other
synthetic-generator ground truth -- `AlertListItem`/`AlertEvidenceRecord`/
`FraudAlertRecord` structurally have no such fields (see
`src.fraud_intel.alerts.queue`), so there is nothing to accidentally leak
by passing those objects' own fields through.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from src.common.config import get_settings
from src.common.db import get_connection
from src.fraud_intel.alerts.queue import (
    AlertEvidenceRecord,
    AlertListItem,
    AnalystDispositionRecord,
    FraudAlertRecord,
    create_default_alert_queue_store,
)

# This dashboard tab is scoped to the local synthetic POC database only --
# never `aidp` -- matching every `aidp fraud-intel`/`aidp alerts` CLI
# command's own required (never defaulted) `--database` contract. Phase 8
# corrective pass: sourced from the SAME existing, already-configured
# `postgres_test_db` setting `src/dashboard/data.py` and
# `tests/integration`/`tests/smoke` use (aidp_test by default,
# overridable via .env's POSTGRES_TEST_DB) -- not an independent literal
# that could drift from the rest of the dashboard's own database choice.
DATABASE = get_settings().postgres_test_db

FRAUD_INTEL_CHANNELS = ("online_banking", "mobile_deposit", "ach", "wire", "atm", "debit_card", "p2p")
PRIORITY_BANDS = ("LOW", "MEDIUM", "HIGH")

# Upper bound on how many current-state rows one queue fetch pulls before
# ranking/paginating in memory -- generous enough to cover any single
# channel's full population in this POC (largest channel so far: 488) and
# a full cross-channel view (3,134 total fraud_alerts as of Phase 7B
# completion), while still being an explicit, bounded query rather than
# an unbounded one. If the true matching count exceeds this, the UI says
# so rather than silently truncating without comment.
MAX_QUEUE_FETCH = 5000
DEFAULT_PAGE_SIZE = 50


def load_queue(
    *, channel: Optional[str], priority_band: Optional[str], page: int, page_size: int
) -> tuple[list[AlertListItem], int, bool]:
    """Read-only: fetches up to MAX_QUEUE_FETCH current-state rows via the
    real store (no hand-written SQL here at all), ranks them by
    `current_operational_priority_score` descending with a fully
    deterministic tie-break, then returns exactly one page.

    Tie-break order: score desc, then `current_scored_at` ascending
    (earlier-scored alerts surface first among score ties -- same
    "earlier wins ties" convention `_rank()`/`LATEST_EVIDENCE_ORDER_SQL`
    use elsewhere), then `alert_id` ascending as the final deterministic
    fallback. Alerts with no evidence at all (`current_operational_
    priority_score is None`) sort last, never crash the sort and never
    silently vanish -- they still appear in the queue if they match the
    channel/band filters (a `None` band never matches an explicit band
    filter, matching `list_alerts_with_current_state()`'s own SQL
    semantics).

    Returns (page_items, total_matching, fetch_was_capped) -- `total_
    matching` is the count actually fetched (post channel/band filter,
    pre-pagination); `fetch_was_capped` is True when that count hit
    MAX_QUEUE_FETCH, meaning the true total could be larger and the UI
    should say so rather than imply completeness.
    """
    store = create_default_alert_queue_store(DATABASE)
    items = store.list_alerts_with_current_state(
        channel=channel, current_priority_band=priority_band, limit=MAX_QUEUE_FETCH
    )

    def sort_key(item: AlertListItem):
        score = item.current_operational_priority_score
        has_score = score is not None
        scored_at = item.current_scored_at
        return (
            0 if has_score else 1,  # scored alerts first
            -(score or 0.0),
            scored_at or item.created_at,
            item.alert_id,
        )

    ranked = sorted(items, key=sort_key)
    total_matching = len(ranked)
    fetch_was_capped = total_matching >= MAX_QUEUE_FETCH

    start = page * page_size
    page_items = ranked[start : start + page_size]
    return page_items, total_matching, fetch_was_capped


def _load_bundle_version(bundle_id: Optional[int]) -> Optional[int]:
    """Read-only single-row lookup: the pinned `channel_model_bundles.
    bundle_version` for one evidence row's `channel_model_bundle_id` --
    a tiny, alert-detail-only convenience the evidence row itself doesn't
    carry (it only has the bundle ID). Not a list/queue query, so this
    does not create N+1 behavior for the ranked queue above."""
    if bundle_id is None:
        return None
    conn = get_connection(DATABASE)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT bundle_version FROM channel_model_bundles WHERE bundle_id = %s", (bundle_id,))
                row = cur.fetchone()
                return row[0] if row else None
    finally:
        conn.close()


def load_alert_detail(alert_id: int) -> dict[str, Any]:
    """Read-only: the alert's immutable identity/initial audit fields,
    its current (latest) evidence, its full disposition history, and the
    pinned bundle_version for the evidence's bundle -- everything Part C
    of the Phase 8 spec requires, and nothing more. Raises LookupError if
    the alert_id doesn't exist (surfaced by the caller as an error
    state, never a silent empty page)."""
    store = create_default_alert_queue_store(DATABASE)
    try:
        alert: FraudAlertRecord = store.get_alert(alert_id)
    except (LookupError, TypeError) as exc:
        # The real store's get_alert() raises TypeError (unpacking None)
        # rather than a LookupError when the row doesn't exist -- normalized
        # to LookupError here so callers have one exception type to handle.
        raise LookupError(f"no fraud_alerts row for alert_id={alert_id}") from exc
    evidence: Optional[AlertEvidenceRecord] = store.get_latest_evidence(alert_id)
    dispositions: list[AnalystDispositionRecord] = store.list_dispositions(alert_id)
    bundle_version = _load_bundle_version(evidence.channel_model_bundle_id) if evidence else None
    return {"alert": alert, "evidence": evidence, "dispositions": dispositions, "bundle_version": bundle_version}


def _queue_dataframe(items: list[AlertListItem]) -> pd.DataFrame:
    rows = [
        {
            "alert_id": i.alert_id,
            "channel": i.channel,
            "status": i.status,
            "current_priority_band": i.current_priority_band or "(no evidence)",
            "current_operational_priority_score": i.current_operational_priority_score,
            "current_scored_at": i.current_scored_at,
            "initial_priority_band (historical)": i.initial_priority_band,
            "created_at": i.created_at,
        }
        for i in items
    ]
    return pd.DataFrame(rows)


def render() -> None:
    """Called from within `with tab_fraud_intel:` in src/dashboard/app.py.
    Matches the existing tabs' convention (top-level st.* calls, a
    `Refresh` button, explicit empty/error states) rather than
    introducing a new dashboard-composition pattern."""
    st.button("Refresh", key="refresh_fraud_intel")

    st.subheader("Ranked Alert Queue — Current Operational State")
    st.caption(
        "Ranked by CURRENT operational priority score (latest evidence), not the frozen first-scoring "
        "snapshot. LOW-band alerts remain visible by default -- this is not a triage-only view."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    channel = filter_col1.selectbox("Channel", options=["(all)"] + list(FRAUD_INTEL_CHANNELS), index=0)
    band = filter_col2.selectbox("Current priority band", options=["(all)"] + list(PRIORITY_BANDS), index=0)
    page_size = filter_col3.selectbox("Rows per page", options=[25, 50, 100, 200], index=1)

    page = st.number_input("Page (0-indexed)", min_value=0, value=0, step=1, key="fraud_intel_queue_page")

    channel_arg = None if channel == "(all)" else channel
    band_arg = None if band == "(all)" else band

    try:
        page_items, total_matching, capped = load_queue(
            channel=channel_arg, priority_band=band_arg, page=int(page), page_size=int(page_size)
        )
    except Exception as exc:  # pragma: no cover - defensive, matches existing tabs' no-crash convention
        st.error(f"Could not load the alert queue: {type(exc).__name__}: {exc}")
        return

    if total_matching == 0:
        st.info("No alerts match the current filters.")
        return

    st.caption(
        f"{total_matching} alert(s) match{' (capped at ' + str(MAX_QUEUE_FETCH) + ' -- narrow the channel filter for a complete count)' if capped else ''}. "
        f"Showing rows {page * page_size + 1}–{min((page + 1) * page_size, total_matching)}."
    )
    df = _queue_dataframe(page_items)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Alert Detail")
    selected_alert_id = st.number_input("Alert ID", min_value=1, value=int(page_items[0].alert_id), step=1, key="fraud_intel_selected_alert")

    try:
        detail = load_alert_detail(int(selected_alert_id))
    except LookupError:
        st.error(f"No alert found with alert_id={int(selected_alert_id)}.")
        return
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load alert detail: {type(exc).__name__}: {exc}")
        return

    alert: FraudAlertRecord = detail["alert"]
    evidence: Optional[AlertEvidenceRecord] = detail["evidence"]
    dispositions: list[AnalystDispositionRecord] = detail["dispositions"]
    bundle_version = detail["bundle_version"]

    st.markdown(f"**Alert {alert.alert_id}** — channel `{alert.channel}` — status `{alert.status}`")
    st.caption(f"Source alert `{alert.source_alert_id}` (`{alert.source_system}`) — event `{alert.event_id}`")

    if evidence is None:
        st.warning("This alert has no scoring evidence yet (current state is unavailable).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current priority band", evidence.priority_band or "N/A")
        c2.metric("Current operational score", f"{evidence.operational_priority_score:.6f}" if evidence.operational_priority_score is not None else "N/A")
        c3.metric("Degraded", "Yes" if evidence.degraded else "No")
        c4.metric("Evidence scored at", str(evidence.scored_at))

        st.caption(f"score_execution_id: `{evidence.score_execution_id}` — event_time: `{evidence.event_time}`")

        st.markdown("**Bundle / policy provenance (pinned on this evidence row)**")
        prov_df = pd.DataFrame(
            [
                {"field": "operational_bundle_id", "value": str(evidence.channel_model_bundle_id)},
                {"field": "operational_bundle_version", "value": str(bundle_version)},
                {"field": "gbm_model_version", "value": str(evidence.gbm_model_version)},
                {"field": "lr_model_version (shadow)", "value": str(evidence.lr_model_version)},
                {"field": "anomaly_model_version", "value": str(evidence.anomaly_model_version)},
                {"field": "preprocessing_artifact_version", "value": str(evidence.preprocessing_artifact_version)},
                {"field": "feature_schema_version", "value": str(evidence.feature_schema_version)},
                {"field": "rule_set_version", "value": str(evidence.rule_set_version)},
                {"field": "graph_policy_version", "value": str(evidence.graph_policy_version)},
                {"field": "ensemble_policy_version", "value": str(evidence.ensemble_policy_version)},
                {"field": "reason_code_version", "value": str(evidence.reason_code_version)},
            ]
        )
        st.dataframe(prov_df, use_container_width=True, hide_index=True)

        st.markdown("**Component scores and statuses**")
        comp_df = pd.DataFrame(
            [
                {"component": "rule", "score_contribution": float(evidence.rule_result.get("score_contribution") or 0.0), "status": str(evidence.component_statuses.get("rules", {}).get("status"))},
                {"component": "gbm (used for score/band)", "score_contribution": evidence.gbm_probability, "status": str(evidence.component_statuses.get("gbm", {}).get("status"))},
                {"component": "lr-shadow (logged only, never scores/bands)", "score_contribution": evidence.lr_probability, "status": "N/A (shadow-only)"},
                {"component": "anomaly", "score_contribution": evidence.anomaly_score, "status": str(evidence.component_statuses.get("anomaly", {}).get("status"))},
                {"component": "graph", "score_contribution": evidence.graph_risk_score, "status": str(evidence.component_statuses.get("graph", {}).get("status"))},
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fired_rules = evidence.rule_result.get("fired_rule_ids") or []
        if fired_rules:
            st.caption(f"Fired rules: {', '.join(fired_rules)}")

        if evidence.reason_codes:
            st.markdown("**Reason codes**")
            st.dataframe(pd.DataFrame(evidence.reason_codes), use_container_width=True, hide_index=True)

    st.markdown("**Initial (first-scoring, historical) values — frozen, never current**")
    st.caption(
        "These reflect ONLY the alert's very first scoring pass and are NEVER updated by a later rescore "
        "(Phase 6 decision 2) -- shown here purely as an audit trail, clearly separate from the current "
        "state above."
    )
    initial_df = pd.DataFrame(
        [
            {"field": "initial_priority_band", "value": str(alert.initial_priority_band)},
            {"field": "initial_operational_priority_score", "value": str(alert.initial_operational_priority_score)},
            {"field": "initial_ensemble_policy_version", "value": str(alert.initial_ensemble_policy_version)},
            {"field": "created_at (first-scoring time)", "value": str(alert.created_at)},
        ]
    )
    st.dataframe(initial_df, use_container_width=True, hide_index=True)

    st.markdown("**Analyst disposition history** (captured only via `aidp alerts disposition` — CLI-only)")
    st.caption("Separate from synthetic ground truth: this platform never displays a scenario's synthetic fraud label anywhere.")
    if not dispositions:
        st.caption("No analyst disposition recorded yet.")
    else:
        disp_df = pd.DataFrame(
            [
                {"disposed_at": d.disposed_at, "analyst_id": d.analyst_id, "disposition": d.disposition, "notes": d.notes}
                for d in dispositions
            ]
        )
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
