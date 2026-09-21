# AiDP v1.3 — Fraud Intelligence Demo Script

**POC-only.** Every command below uses synthetic, locally-generated data
against `aidp_test`. Nothing here is executed automatically by this
script — it documents the exact, real commands to run and what to expect;
**you** run them, in order, in your own terminal. None of this constitutes
Citizens Bank production or regulatory evidence.

Every command below is a real, currently-implemented CLI invocation —
copy/paste-able as written, with `<...>` placeholders for values a real
run generates for you (IDs are deterministic given the same seed/count/
reference-date, but this script never hardcodes a transient run's own
`run_id` or timestamp).

## 0. Setup

```bash
cd local-aidp-poc
source .venv/bin/activate
```

## 1. Verify infrastructure and `aidp_test` identity

```bash
./scripts/healthcheck.sh
```

Expect `[OK]` for PostgreSQL, MinIO, Redpanda, MLflow (FastAPI/Streamlit
`[OK]` only if you've also started those processes — not required for the
CLI steps below).

```bash
python -m src.cli.__main__ config validate --json
```

Confirm every fraud-intel command below is given `--database aidp_test`
explicitly — there is no default, and it is never `aidp`.

## 2. Deterministic generation — all seven channels

```bash
SEED=42
REF_DATE=2026-09-21

for CHANNEL in online_banking mobile_deposit ach wire atm debit_card p2p; do
  python -m src.cli.__main__ fraud-intel generate \
    --channel "$CHANNEL" --count 5000 --seed "$SEED" \
    --reference-date "$REF_DATE" --database aidp_test --json
done
```

Each returns `generation_run_id`/`dataset_version` — deterministic SHA-256
fingerprints of `(channel, count, seed, reference_date)`. Re-running the
identical loop is a safe no-op retry: `inserted_event_count=0,
existing_event_count=5000` on the second pass, same IDs both times.

Capture each channel's `generation_run_id` from the first-run JSON output
(referred to below as `<GENRUN_channel>`) — every later command needs it.

## 3. Training and explicit manual whole-bundle promotion

**Generate all seven channels before training any of them** — cross-channel
feature history is customer-scoped across the whole platform, so training
one channel before a later-generated channel exists would compute that
row's history against an incomplete pool. Train only after Step 2
completes for every channel.

```bash
python -m src.cli.__main__ fraud-intel train \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --database aidp_test --json
```

Returns a new `CANDIDATE` bundle (`bundle_id`, `bundle_version`) — never
`OPERATIONAL` on its own. Repeat per channel.

Before promoting, optionally inspect the cold-start promotion gate
(no `OPERATIONAL` bundle yet ⇒ this routes automatically):

```bash
python -m src.cli.__main__ fraud-intel evaluate \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --candidate-bundle-version 1 \
  --capacity-mode count --capacity-value 50 --recall-target 0.8 \
  --database aidp_test --json
```

Then promote — always a manual, explicit, operator-attributed action:

```bash
python -m src.cli.__main__ fraud-intel promote \
  --channel <CHANNEL> --bundle-version 1 --promoted-by <your-operator-id> \
  --database aidp_test --json
```

Repeat training → (optional gate check) → promote for each of the seven
channels. `"status": "OPERATIONAL"` and a populated `promoted_by`/
`promoted_at` confirm success.

## 4. Scoring and zero-pending retry

```bash
python -m src.cli.__main__ fraud-intel score \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --database aidp_test --json
```

Then run the **identical** command again to demonstrate idempotency:

```bash
python -m src.cli.__main__ fraud-intel score \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --database aidp_test --json
```

Expect the retry: `"pending_count": 0, "records_processed": 0,
"records_rejected": 0, "alerts": []` — no duplicate `fraud_alerts`/
`alert_evidence` row is created. Repeat per channel.

## 5. Open the Fraud Intelligence dashboard

```bash
./scripts/run_dashboard.sh
```

Open http://127.0.0.1:8501 and select the **Fraud Intelligence** tab
(entirely separate from the legacy v1.1 tabs).

## 6. Filter/rank the queue, retaining LOW alerts

In the tab: leave **Channel** = `(all)` and **Current priority band** =
`(all)` to see the full, real cross-channel queue ranked by current
operational priority score (highest first, deterministic tie-break).
Then set **Current priority band** = `LOW` to confirm LOW-band alerts are
genuinely visible and queryable — this is not a triage-only, HIGH/MEDIUM-only
view. Try a single **Channel** filter (e.g. `ach`) to confirm the count
matches that channel's real scored population.

## 7. Open one alert — reason codes, component evidence, versions, `score_execution_id`

Pick any `alert_id` from the queue table and enter it under **Alert
Detail**. Confirm the page shows: current priority band and operational
score, `score_execution_id`, evidence `scored_at`, the pinned operational
bundle ID/version, GBM/LR-shadow/anomaly model versions, preprocessing/
feature-schema/rule-set/graph-policy/ensemble-policy/reason-code versions,
per-component scores and OK/ERROR statuses, fired rule IDs, and the full
reason-code list. Confirm the separate "Initial (first-scoring,
historical)" section is clearly distinguished from the current-state
section above it, and that no `scenario_id`/synthetic label ever appears
anywhere on the page.

## 8. The CLI disposition command (documented demo step — not auto-executed)

```bash
python -m src.cli.__main__ alerts disposition <alert_id> \
  --analyst-id demo-analyst --disposition NEEDS_MORE_INFO \
  --notes "Reviewed in Phase 8 demo" --database aidp_test --json
```

This is the **only** write path for analyst review state in this
subsystem — the dashboard itself never writes a disposition (Step 5–7 are
read-only). Re-open the same alert in **Alert Detail** (Step 7) afterward
to see the new row appear under **Analyst disposition history**.

## 9. Analyst disposition is separate from synthetic truth

Point out explicitly: the disposition just recorded (`NEEDS_MORE_INFO`,
an analyst's real judgment) is a completely independent signal from the
synthetic generator's own ground truth used to build the demo population.
Nothing in the dashboard or the `alerts` commands ever displays
`scenario_id` or `synthetic_scenario_label` — those exist only inside
`synthetic_event_labels`, read only by the label-assessment and generator
code paths, never by any analyst-facing surface.

## 10. Label assessment and idempotency

```bash
python -m src.cli.__main__ fraud-intel labels assess \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --database aidp_test --json
```

Run it again immediately to show the retry: `"assessments_inserted": 0,
"assessments_unchanged": <total>` — same population, no duplicate rows.

## 11. Generation/bundle-scoped evaluation and the rules-only baseline

```bash
python -m src.cli.__main__ fraud-intel evaluate \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --capacity-mode count --capacity-value 50 --recall-target 0.8 \
  --database aidp_test --json
```

With the channel's bundle now `OPERATIONAL` (post Step 3) and labels
resolved (post Step 10), this automatically routes to
`"evaluation_mode": "live_resolved_alerts"`. Point out the
`rules_only_baseline` block in the output — the same
`compute_operational_priority_score()` mechanism with GBM/anomaly/graph
weights zeroed, computed from the real rule signal alone, never a
separately-implemented formula. Compare its `pr_auc`/`roc_auc` against the
full ensemble's own — for most channels the ensemble meaningfully exceeds
the rules-only baseline; for P2P, the rules-only baseline itself reaches
1.0 (its `P2P_NEW_RECIPIENT_NO_MEMO` rule alone perfectly separates that
channel's synthetic fraud scenario — a synthetic-data artifact, not a
claim about real rule quality).

## 12. Cross-channel / capacity metrics

```bash
python -m src.cli.__main__ fraud-intel evaluate \
  --channel <CHANNEL> --generation-run-id <GENRUN_channel> \
  --capacity-mode count --capacity-value 50 --recall-target 0.8 \
  --database aidp_test --json
```

(The same real command as Step 11 — there is no separate "cross-channel"
CLI invocation implemented; each channel's evaluation is independent.)
Point out, from the JSON: `precision_at_capacity`/`recall_at_capacity`
(top-50-ranked-alert view — the capacity-ranked operating model this POC
recommends over a fixed threshold, especially for ATM's weak
fixed-threshold recall), `minimum_alerts_required_for_recall_target`, and
`workload_reduction_at_recall_target`. To compare channels side by side,
run this command once per channel and diff the JSON outputs — there is no
single "run for all channels at once" flag.

## 13. Cleanup

```bash
./scripts/stop.sh    # docker compose down — containers removed, named volumes (data) preserved
```

**Do not use `scripts/reset_demo.sh` here** — it clears `aidp`'s legacy
v1.1 demo data (`transactions`/`fraud_decisions`/seed & output files), not
`aidp_test`'s v1.3 fraud-intelligence tables, and it is never a migration
mechanism (see TROUBLESHOOTING.md). This platform has no analogous
"reset the fraud-intel demo" script; if you need a clean `aidp_test` for a
fresh run, that is a separate, explicit, deliberate action outside this
script's scope — do not improvise a destructive command to achieve it.

---

**Disclaimer, repeated for the transcript**: all data, scores, and metrics
in this demo are synthetic and local. This is a proof-of-concept
demonstrating architectural patterns (generation-scoped training/scoring,
shared cross-channel feature history, immutable-identity/versioned-evidence
alert state, manual whole-bundle promotion, read-only analyst tooling) —
not a claim of production readiness, regulatory approval, or real-world
model accuracy for Citizens Bank or any other institution.
