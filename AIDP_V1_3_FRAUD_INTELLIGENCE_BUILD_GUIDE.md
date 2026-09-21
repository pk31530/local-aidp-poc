# AiDP v1.3 — Fraud-Alert Triage POC — Build Guide

## Purpose

AiDP v1.1 proved one end-to-end fraud-detection flow (batch + real-time,
one channel, one model). AiDP v1.2 added a control plane — typed run
configuration, run lifecycle/provenance, a unified CLI — on top of that
flow, without touching fraud logic.

**AiDP v1.3 is not a new alert-generation system.** Channel and rule
systems already generate fraud alerts today, and analysts already review
nearly all of them. The problem is not too few alerts — it's that every
alert is reviewed with equal priority, burying the small number of
genuinely high-risk ones in noise. v1.3 is **analyst decision support**
that sits *after* existing alert generation: it retains every alert,
enriches it with additional signals, ranks it, and explains it — so
analysts spend review capacity where it matters most. It does not decide
fraud, does not close an alert, and does not remove an alert from the
reviewable population.

This remains a local, synthetic-data-only proof of concept. It validates
architecture and pipeline correctness, not real-world detection accuracy,
and makes no legal, compliance, or regulatory-approval claim anywhere in
this document or the system it describes.

This guide is written for Claude Code in VS Code, the same way the v1.2
guide was. Build one phase at a time. Do not paste all phase prompts
together. Do not skip the stop-and-review point at the end of each phase.

---

## 1. V1.2 baseline and reusable interfaces

v1.3 does not replace anything in v1.1/v1.2 — it adds a parallel,
higher-layer capability on top of the same platform. Every reusable
interface below keeps its existing behavior unchanged; v1.3 depends on it,
never forks it. One exception, called out explicitly: `RunLifecycle`'s
`PIPELINE_NAMES` set is *extended* (additively) in Phase 6 — see §11.

| v1.2/v1.1 pattern | File | What v1.3 reuses it for |
|---|---|---|
| One shared feature core, thin per-source adapters (fix C2) | `src/common/features.py` | Template for "shared feature core + channel adapter," generalized to 7 channels |
| Online feature/profile store, live Postgres queries (fix C1) | `src/common/feature_store.py` | Same live-lookup pattern for channel entities |
| Config-driven thresholds + reason codes | `src/decisioning/engine.py`, `config/fraud_rules.yaml` | Direct ancestor of the v1.3 `RuleProvider` |
| Pydantic schema, bounds validation (fix M4) | `src/common/schemas.py` | Base pattern for the strengthened event contract (§6) |
| Deterministic, seeded synthetic generation (fix C3) | `src/generator/` | Template for the 7 channel generators |
| Typed run configuration, redaction, hashing | `src/control_plane/config.py` | v1.3 run configs subclass the same `RunConfig` base |
| Central run lifecycle/provenance, state machine | `src/control_plane/runs.py` (`RunLifecycle`) | Provenance for every v1.3 run — **extended**, not forked, in Phase 6 |
| Model registry, flavor-aware loading | `src/common/mlflow_setup.py`, `src/ml/train.py` | Template for bundle-component registration — **candidate-only**, see §22 |
| CLI command tree | `src/cli/` | Home for `aidp fraud-intel ...` / `aidp alerts ...` |
| Chronological-leakage lesson | `src/common/splits.py`, `HARDENING_LOG.md` Fix 5 | Applied again, time-based, with an explicit as-of-time contract (§12) |
| Idempotent inserts (`ON CONFLICT`) | migration `001` | Template for idempotent alert/evidence writes (§20) |
| Migration discipline (fresh schema + numbered idempotent migration, `aidp_test`-first, `ON_ERROR_STOP=1`) | migrations `001`, `002` | Unchanged process, continued numbering (`003`, `004`) |
| Credential/error redaction | `provenance.py` (`redact_credentials`, `truncate_text`) | Reused directly, not reimplemented |
| Dashboard smoke-test classification | `tests/smoke/test_dashboard.py` (v1.2 correction) | Same classification applied to the new fraud-intel dashboard tab (§19) |

Nothing in `src/decisioning/`, `src/common/features.py`,
`src/common/scoring.py`, `config/fraud_rules.yaml`, or the v1.1/v1.2
single-channel serving path is modified by this guide.

---

## 2. V1.3 business problem and measurable success criteria

**Problem:** across seven payment channels, existing rule/channel systems
already generate a large volume of fraud alerts. Analysts review nearly
all of them today because there is no reliable way to distinguish the
small number of genuinely high-risk alerts from the rest without
enrichment.

**Objective:** reduce the number of *already-existing* alerts that require
full analyst review, without reducing recall of confirmed-fraud cases,
by ranking every alert and explaining why.

**Human-review framing (precise, not a claim of universal review):**
- Every source alert remains reviewable and auditable — none is deleted.
- The ranked queue determines which alerts receive full review within
  available analyst capacity, not that every alert receives it today.
- LOW-priority alerts are retained and may be sampled or reviewed; they
  are never removed from the reviewable population.
- No alert is ever automatically closed or automatically declared
  legitimate by this POC.

**Measurable success criteria**, evaluated primarily on the **source-alert
population** (§4) — that is the actual analyst workload being reduced —
against synthetic data only:

1. At a fixed analyst review capacity, recall of confirmed-fraud
   scenarios among source alerts is at or above the rules-only baseline.
2. Precision at that same capacity is higher than the rules-only
   baseline.
3. Dollar-weighted recall is at or above the rules-only baseline.
4. Every source alert remains queryable; none are auto-closed or
   suppressed.
5. Every alert has at least one reason code, traceable to a specific
   layer.
6. Every scoring attempt is idempotent — a retried or duplicated scoring
   run never creates duplicate evidence (§20).

These are synthetic-data validation targets, not production SLAs —
restated in §7, §23, §27.

---

## 3. Scope and explicit exclusions

**In scope:** triage/enrichment of alerts across all seven channels (ACH,
Wire, Mobile Check Deposit, Online/Mobile Banking, ATM, Debit Card, P2P),
the full layered-scoring stack (§4), a local analyst review queue and
dashboard tab, synthetic data generation, local Docker-stack-only
infrastructure, CI (unit tests only, no infrastructure).

**Explicitly excluded** — not partial, not stubbed:

- Generating alerts from nothing — the POC's `LocalYamlRuleProvider`
  *simulates* the existing upstream alert-generation system that a real
  bank already has (§4); v1.3 does not claim to replace it.
- Automatic fraud declaration, automatic alert closure, or automatic
  removal of an alert from review.
- Automatic model/bundle promotion (§22) — v1.1's unconditional
  champion-alias behavior is not copied.
- Silently adding model-only high-scoring, non-alerted events to the
  analyst queue (§4) — these may be logged as shadow candidates only.
- Computer-vision check processing or any Orbograph integration (§26) —
  mobile-deposit fields are structured placeholders only (§9).
- A graph database — the entity graph (§15) is in-process only.
- Real customer data, production authentication/authorization, production
  deployment, or regulatory certification (§17/§23, §26).
- Blending the LR challenger into the operational score (§13).
- Using raw, immature analyst dispositions as graph or model ground
  truth (§6).
- OFAC/sanctions screening and BSA/AML monitoring — separate upstream
  systems, not fraud-triage rules implemented here (§10, §23, §26).
- A production FastAPI surface — v1.3 exposes CLI and a read-mostly
  dashboard tab only (§21).

---

## 4. Target architecture

```
7 channel event sources (synthetic generators, §7)
        |
        v
Channel event validated against the strengthened contract (§6)
        |
        v
LocalYamlRuleProvider evaluation (§10)
  — in this local POC, this step SIMULATES the bank's existing
    upstream alert-generation systems. It is not new alert
    generation invented by v1.3; it stands in for what already exists.
        |
        +---------------------------------------------+
        |                                               |
  >= 1 rule fired                                 zero rules fired
        |                                               |
        v                                               v
  SourceAlertContext created (§6)              NOT an alert. Retained only
  (source_alert_id, source_system,              as behavioral/graph history
   source_rule_ids, source_rule_version,        (§9, §15). Never enters the
   source_alert_score, source_alert_             triage/scoring path below,
   reason_codes)                                 except as optional
        |                                        shadow-candidate scoring
        |                                        for monitoring only (never
        |                                        queued, §4 below).
        v
V1.3 TRIAGE PIPELINE (retains and enriches EVERY source alert —
never silently discards one):
        |
        v
  Feature adapter (shared core + channel adapter, §9)
        |
        +--------------------+------------------------+
        |                    |                         |
        v                    v                         v
  Operational GBM     Anomaly layer (§14)      Entity graph layer (§15)
  (calibrated, from                            (proximity-to-fraud uses only
  the promoted bundle,                          MATURE, training-eligible
  §13, §22)                                     resolved labels — §6, §15)
  + LR shadow (logged,
  never blended, §13)
        |                    |                         |
        +--------------------+------------------------+
                              v
        Scoring orchestrator (§8) sequences: context validation ->
        feature adapter -> rule enrichment -> GBM -> anomaly -> graph ->
        ensemble policy -> priority band -> reason codes -> evidence
                              |
                              v
                Versioned ensemble policy (§16)
                              |
                              v
                Explainable reason codes (§17)
                              |
                              v
     Idempotent alert upsert + immutable evidence snapshot (§18, §20)
                              |
                              v
       Analyst review queue (CLI + dashboard tab, §19, §21)
                              |
                              v
     Disposition capture -> append-only label assessments (§8, §10 [new])
                              |
                              v
     Manual, RunLifecycle-tracked bundle promotion (§22)
```

Every "run" (training, scoring batch, label eligibility, evaluation,
promotion) goes through `RunLifecycle`, exactly like v1.2's batch/train/
stream — no second provenance mechanism. See §11 for the four new
`pipeline_name` values this requires.

---

## 5. Shared core versus channel adapters

| Shared (built once) | Channel adapter (built per channel) |
|---|---|
| Strengthened common event contract (§6) | Discriminated channel payload model |
| `LocalYamlRuleProvider` engine, fixed operator allowlist (§10) | Channel rule-set YAML (versioned) |
| Feature-core utilities | Channel feature group |
| Shared training framework, chronological split, preprocessing/calibration (§11, §12) | Channel GBM/LR/anomaly artifacts + preprocessing artifact |
| Scoring orchestrator (§8) | — (channel is a parameter) |
| Ensemble policy engine (loads a config) | Channel ensemble-policy YAML |
| Reason-code builder | Channel-specific reason text/IDs |
| Entity graph engine | Channel-specific entity types feeding it |
| Alert queue, evidence, disposition, label-assessment schema | — (channel-agnostic) |
| Channel model bundle registry + promotion (§22) | Channel's own bundle versions |
| CLI/dashboard surfaces | `--channel` flag selects the adapter |

---

## 6. Common event contract, source-alert context, and seven channel contracts

**`scenario_id` is now `Optional[str] = None`** on the reusable event
contract — it is populated only by the synthetic generator (§7) and is
never a required field of the contract itself, since a real event source
would never supply it.

**Strengthened `FraudEvent` base** (mirrors, and tightens,
`src/common/schemas.py`'s `Transaction`):

| Field | Type | Notes |
|---|---|---|
| `event_id` | `uuid.UUID` | Application-generated (`uuid4()`), not DB-generated; the idempotency key |
| `channel` | `Literal[...]` enum | One of the 7 channel names; also the discriminator (below) |
| `customer_id` | str | Synthetic/tokenized only |
| `account_id` | str | Synthetic/tokenized only |
| `event_timestamp` | `datetime`, UTC, timezone-aware | A validator rejects naive or non-UTC datetimes |
| `amount_minor_units` | `int`, `gt=0` | USD minor units (cents) — **not** a float; avoids binary-float precision issues entirely |
| `direction` | enum | `debit` \| `credit` |
| `device_id` | Optional[str] | Not every channel has one |
| `ip_address` | Optional[str] | Synthetic, never a real IP |
| `scenario_id` | Optional[str] | **New:** synthetic-only, never required (see above) |
| `schema_version` | int | Enforced against a `SUPPORTED_SCHEMA_VERSIONS` set, same pattern as fix M3 |
| `channel_payload` | discriminated union | See below — **not** a raw dict |

**`channel_payload` is a discriminated Pydantic union, not an unvalidated
dict:**
```python
ChannelPayload = Annotated[
    Union[ACHPayload, WirePayload, MobileDepositPayload, OnlineBankingPayload,
          ATMPayload, DebitCardPayload, P2PPayload],
    Field(discriminator="channel"),
]
```
Each `<Channel>Payload` model declares `channel: Literal["ach"]` (etc.) as
its own discriminator value. (A validated channel-model registry —
`dict[str, type[BaseModel]]` used to parse/validate before storage — is an
acceptable equivalent if a discriminated union proves awkward in practice;
either satisfies "no unvalidated dict-only payload.") Serialization is
deterministic: canonical JSON with sorted keys, the same technique
`RunConfig.config_hash()` already uses in v1.2 — not reinvented.

**The channel invariant is enforced, not assumed.** Pydantic's
discriminated union validates `channel_payload` against whichever payload
type its own internal `channel` literal selects — it does **not**, by
itself, verify that the outer `FraudEvent.channel` field agrees with
`channel_payload.channel`. These are two independently-settable fields
that could otherwise disagree. `FraudEvent` therefore carries an explicit
model validator:

```python
@model_validator(mode="after")
def _channel_matches_payload(self) -> "FraudEvent":
    if self.channel != self.channel_payload.channel:
        raise ValueError(
            f"FraudEvent.channel={self.channel!r} does not match "
            f"channel_payload.channel={self.channel_payload.channel!r}"
        )
    return self
```
A Phase 1 test constructs a `FraudEvent` with a deliberately mismatched
`channel`/`channel_payload` pair and asserts it is rejected.

**Channel payload fields** (unchanged content from the prior draft,
carried over exactly):

| Channel | Key fields |
|---|---|
| ACH | `sec_code`, `originating_routing_number` (synthetic), `receiving_routing_number` (synthetic), `batch_id`, `effective_entry_date`, `company_id` |
| Wire | `wire_type` (`domestic`\|`international`), `beneficiary_bank_id` (synthetic), `beneficiary_account` (tokenized), `purpose_code`, `originator_to_beneficiary_info` |
| Mobile Check Deposit | `duplicate_image_hash_flag`, `car_lar_mismatch_flag`, `signature_verification_flag`, `endorsement_present_flag`, `micr_consistency_flag`, `image_quality_score` — structured placeholders only, §7/§26 |
| Online/Mobile Banking | `session_id`, `login_method`, `mfa_used_flag`, `transaction_type`, `target_account` |
| ATM | `atm_id`, `atm_geo_bucket`, `transaction_type`, `card_present_flag` |
| Debit Card | `merchant_id`, `mcc_code`, `pos_entry_mode`, `card_present_flag`, `cross_border_flag` |
| P2P | `recipient_handle` (tokenized), `network` (synthetic name), `memo_present_flag`, `recipient_is_new_flag` |

**New: `SourceAlertContext` — persisted as its own `source_alerts` table,
not an ephemeral computation result.** Created only when
`LocalYamlRuleProvider` fires at least one rule for an event (§4, §10),
and written to `source_alerts` at that point (Phase 1 schema; see §18 for
how `fraud_alerts` references it):

| Field | Meaning |
|---|---|
| `source_alert_id` | Primary key. Distinct from `event_id` — mirrors a real bank's alert ID being distinct from the transaction ID |
| `source_system` | `"LocalYamlRuleProvider (simulated upstream)"` for this POC |
| `event_id` | FK to `channel_events` — **not unique**; see below |
| `source_alert_created_at` | When the (simulated) upstream system generated the alert |
| `source_rule_ids` | List of fired rule IDs from that evaluation |
| `source_rule_version` | The `rule_set_version` that produced it |
| `source_alert_score` | Optional — only if the upstream rule config assigns one |
| `source_alert_reason_codes` | The upstream reason codes, kept distinct from v1.3's own merged reason codes (§17) |
| `generation_run_id` | Synthetic-only provenance — which generator invocation produced this row (§7, Phase 7B cleanup) |
| `dataset_version` | Synthetic-only provenance, same purpose |
| `created_at` | Row-insert timestamp |

**`source_alerts.event_id` is intentionally not unique — one event can
have more than one source alert.** A real bank's upstream systems can
independently re-evaluate the same transaction (a newer `rule_set_version`
re-running historically, or two distinct rule categories firing at
different times), producing two distinct, independently-reviewable alert
records for the same underlying event. `source_alerts` is modeled as a
one-to-many child of `channel_events` for exactly this reason. This is why
`fraud_alerts` (§18) is keyed on the source alert's own identity, never on
`event_id` alone.

Persisting `source_alerts` as its own table (rather than deriving it
on-the-fly from rule evaluation) also keeps the seam correct for a future
real integration: `LocalYamlRuleProvider` only *simulates* an upstream
system for this POC, but a real adapter (§26) would need to ingest a real
system's alerts directly into `source_alerts` without ever running local
rules — the table's identity has to stand on its own, independent of how
it was populated.

**Three separate synthetic output contracts** (never merged into one
object, and never stored in the same place — see §7, §8):
1. **Channel event** (`FraudEvent` above) — the scoring input.
2. **`SourceAlertContext`** (above) — the alert-generation record.
3. **`SyntheticGroundTruthLabel`** — `event_id`, `scenario_id`,
   `synthetic_scenario_label` (bool), `scenario_type`, `label_source =
   "SYNTHETIC_GENERATOR"`, `generated_at`. Stored in its own table (§ below,
   Phase 1), never inside `channel_payload`, never loaded by any scoring
   code path.

**Leakage-prevention tests, mandatory in Phase 1 and re-asserted in Phase
2/4/5:** an explicit test proves none of `synthetic_scenario_label`,
`scenario_id`, `analyst_disposition`, `outcome_status`, or
`training_eligible` can appear as a column in any model feature matrix —
by construction (the feature-building function never reads from the
label/disposition tables at all) and by an explicit assertion
(`assert_no_label_leakage(feature_columns)`) run against every channel's
feature-column list.

**Analyst-facing display rule:** `scenario_id` and any synthetic-label
field are **hidden** from `aidp alerts show`/`list` and the dashboard tab
— they appear only in evaluation/debug output (`aidp fraud-intel
evaluate`, or a `--debug` flag), specifically to prevent analyst bias from
seeing which alerts are "supposed to be" fraud.

---

## 7. Synthetic-data generation strategy and limitations

Deterministic by `(seed, n, reference_date)`, `numpy.random.default_rng` +
`Faker.seed()` — no wall-clock randomness, following `src/generator/`'s
existing pattern exactly.

- Reuse/extend `generate_customers()` for the shared population.
- One generator per channel producing normal events, source-alert-firing
  events, and labeled fraud scenarios — each carrying a `scenario_id`
  for traceability, stored **only** in the separate
  `SyntheticGroundTruthLabel` output (§6), never in the event itself
  beyond the optional field.
- Realistic low fraud prevalence, including scenarios below 5%.
- Deliberate hard false positives — legitimate-but-unusual behavior that
  resembles a fraud pattern and also fires a source alert, so the triage
  layer (not just the upstream rule) is meaningfully tested.
- Channel-specific fraud typologies (final set decided in Phase 1):
  ACH account-takeover batch fraud, Wire BEC one-off, mobile-deposit
  duplicate/altered check, online-banking credential-stuffing + transfer,
  ATM skimming + cash-out, card-not-present testing, P2P mule fan-out.

**Every generated row is tagged for safe, exact cleanup — not a wildcard
pattern.** `channel_events`, `source_alerts`, and `synthetic_event_labels`
each carry `generation_run_id` (one per generator invocation) and
`dataset_version` columns (Phase 1 schema). This is the only mechanism
used to identify and remove synthetic data in `aidp_test` (§25 Phase 7B) —
there is no broad `scenario_id LIKE '...'` deletion anywhere in this
guide; a specific `generation_run_id` is always the deletion key.

**Explicit limitation, restated in §17/§23/§27:** synthetic data and
synthetic evaluation results validate that the *architecture and pipeline*
work correctly end to end. They are not a measure of real-world detection
accuracy, are not representative of actual Citizens Bank fraud patterns,
and are not suitable evidence for any regulatory or production-readiness
claim. No real Citizens Bank information of any kind is used anywhere.

---

## 8. Fraud label and label-maturity policy

Five separate concepts, never collapsed:

| Field | Meaning | Source |
|---|---|---|
| `synthetic_scenario_label` | Generator's ground truth | `SyntheticGroundTruthLabel` table (§6), never `channel_payload` |
| `analyst_disposition` | Latest analyst action | Append-only `analyst_dispositions` (§18) |
| `outcome_status` | `RESOLVED_FRAUD`\|`RESOLVED_LEGITIMATE`\|`UNRESOLVED` | Derived by the versioned policy below |
| `label_source` | `SYNTHETIC_GENERATOR` \| `ANALYST_DISPOSITION` \| `EXTERNAL_CONFIRMATION` (deferred, §26) | Recorded per assessment |
| `maturity_status` | `IMMATURE` \| `MATURE` — mature only after a configurable observation window past `event_timestamp` | Computed |
| `training_eligible` | Boolean, from a **versioned** eligibility policy function | Computed |
| `training_eligibility_policy_version` | Versions the eligibility rule itself | Recorded |

**Critical rule, enforced in code: `analyst_disposition` never
automatically becomes model training ground truth, and never
automatically becomes graph "confirmed fraud" ground truth either (§15).**
For this MVP (no real external confirmation feed exists), training labels
and graph proximity-to-fraud features are derived from
`synthetic_scenario_label` under the versioned policy, restricted to
**mature, training-eligible, resolved** rows only. `analyst_disposition`
is captured and compared (a human-vs-model agreement metric) but never
silently substituted in. This mirrors the LR-shadow "logged, not blended"
philosophy (§13) — a deliberate MVP safety choice.

**Assessments are append-only, not a single mutable row** — see §18's
`label_assessments` table: every re-evaluation under a (possibly new)
policy version produces a new row; the latest is queried, prior policy
decisions remain permanently auditable.

---

## 9. Shared and channel-specific features

**Shared feature group:** velocity/count in configurable windows (reuses
the `WINDOW_10M`/`1H`/`24H` pattern), amount-vs-entity-average and
z-score (computed on `amount_minor_units`, converted consistently), new-
device/new-geo flags, night-transaction flag, counterparty risk score
(train-window-only, per fix Finding 1's lesson), failed-attempt counts,
entity tenure, prior-alert count.

**As-of-time contract — applies to every feature function, restated in
full in §12:** a feature computed for an event at time `T` may only read
records whose `event_timestamp` is strictly earlier than `T` (ties broken
deterministically by `(event_timestamp, event_id)` ordering, never by
insertion order). This applies identically to graph construction (§15).

**Channel-specific feature groups** (unchanged content from the prior
draft):

| Channel | Distinguishing features |
|---|---|
| ACH | batch-size anomaly, same-day mixed SEC codes, first-time receiving routing number, historical return-code rate |
| Wire | beneficiary-country risk score, first-time-beneficiary flag, amount vs. historical max wire, proximity to cutoff time |
| Mobile Check Deposit | duplicate-image-hash flag, CAR/LAR mismatch flag, amount vs. typical deposit ratio, image-quality score, rapid-resubmission flag |
| Online/Mobile Banking | new-device-then-high-value-action combo, MFA-bypass flag, profile-change-then-transfer sequence, session velocity |
| ATM | geo-velocity, ATM-cluster risk score, cash-out ratio, PIN-retry count |
| Debit Card | MCC risk score, card-not-present + high-amount combo, cross-border authorization flag, merchant velocity |
| P2P | new-recipient flag, recipient fan-in/fan-out (graph, §15), rapid-sequential-transfer flag, round-dollar-amount flag |

Every channel's model bundle (§22) carries its **ordered feature-schema
version and feature-name list** alongside its preprocessing artifact
(fitted encoders/imputers/scalers, training-window only) — scoring fails
safely (`FeatureSchemaMismatchError`, no silent misalignment) if the
bundle's schema doesn't match the current feature computation's output.

---

## 10. RuleProvider interface and failure behavior

```python
class RuleEvaluationResult(BaseModel):
    provider_name: str
    provider_version: str
    rule_set_version: str
    fired_rule_ids: list[str]
    rule_categories: dict[str, str]     # rule_id -> MANDATORY_REVIEW | SCORE_CONTRIBUTING | INFORMATIONAL
    reason_codes: list[str]
    score_contribution: float            # sum of SCORE_CONTRIBUTING rules only
    minimum_priority_band: Optional[str] # NEW — see failure behavior below
    evaluated_at: datetime
    latency_ms: float
    provider_status: str                 # OK | UNAVAILABLE | ERROR
    provider_error_code: Optional[str]   # NEW — structured, never a free-text stand-in

class RuleProvider(Protocol):
    def evaluate(self, event: FraudEvent, features: dict) -> RuleEvaluationResult: ...
```

**Rule categories:** `MANDATORY_REVIEW` (forces `priority_band = HIGH`
regardless of ensemble score), `SCORE_CONTRIBUTING` (adds to
`score_contribution`, feeds the ensemble, §16), `INFORMATIONAL` (reason
codes only).

**Example `MANDATORY_REVIEW` rule (fraud-domain, not a sanctions
determination):** a confirmed-compromised-device or known-duplicate-check
match — e.g. `device_id` matches a device already linked to a
`RESOLVED_FRAUD` assessment (§8), or `duplicate_image_hash_flag` is true
for a mobile deposit. **OFAC/sanctions screening and BSA/AML transaction
monitoring are separate upstream bank systems, not fraud-triage rules
implemented by this POC — see §23 and §26.** This guide previously used a
"sanctioned-country beneficiary" example, which has been removed as
exactly the kind of rule this POC does not implement.

**Local implementation:** one channel-agnostic `LocalYamlRuleProvider`
engine, one versioned YAML file per channel
(`config/fraud_intel/rules_<channel>.yaml`). **Rules are evaluated only
through a fixed allowlist of operators — never `eval()`/`exec()`/arbitrary
expressions:**

```yaml
rules:
  - id: COMPROMISED_DEVICE_MATCH
    category: MANDATORY_REVIEW
    when:
      - {field: device_linked_to_resolved_fraud, op: is_true}
    reason_code: COMPROMISED_DEVICE_MATCH
  - id: NEW_DEVICE_HIGH_AMOUNT
    category: SCORE_CONTRIBUTING
    when:
      - {field: new_device_flag, op: is_true}
      - {field: amount_minor_units, op: gt, value: 500000}
    score_contribution: 0.15
    reason_code: NEW_DEVICE_HIGH_AMOUNT
```
Allowlisted `op` values: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`,
`not_in`, `is_true`, `is_false`. Each condition is
`{field, op, value?}`; a rule's `when` list is AND-ed. A single Python
dispatch table (`op` string → a safe comparator function) evaluates every
condition — there is no code path that executes rule content as code.

**Failure behavior — corrected, no score manipulation:** on an internal
exception or timeout, the provider returns
`provider_status="UNAVAILABLE"` (or `"ERROR"`), a structured
`provider_error_code`, a `RULE_PROVIDER_UNAVAILABLE` reason code, and
`minimum_priority_band = "MEDIUM"`. `band_priority()` (§16) explicitly
enforces `minimum_priority_band` whenever `provider_status` is
`UNAVAILABLE`/`ERROR`, as its own clean step — `score_contribution` is
**never** manipulated to simulate this; the floor is a distinct,
explicit, auditable mechanism.

`RuleProvider` is a `Protocol` — the exact seam a real bank's existing
rule/alert-generation system (§26) would implement against.

---

## 11. Per-channel model training framework

One shared training framework parameterized by `channel`. Mirrors
`src/ml/train.py`'s shape, wrapped in `RunLifecycle`:

```
ChannelTrainingRunConfig(channel="ach", features_path=..., ...)
        |
        v
train_channel_configured(config, trigger_source=...)
  -> lifecycle.begin("train", ...)  # pipeline_name stays "train" — a bundle
                                     # is a training-time artifact, not a
                                     # new pipeline_name
  -> chronological split with purge gap (§12)
  -> fit preprocessing artifact (encoders/imputers/scalers) on the
     TRAINING window only
  -> train GBM (primary), calibrate on the calibration window only
  -> train LR (shadow), same features/preprocessing
  -> train anomaly detector (§14)
  -> evaluate all three on the untouched test window
  -> register GBM/LR/anomaly + preprocessing artifact + feature-schema
     version as one CANDIDATE channel_model_bundle (§22) — no alias/
     promotion here
  -> lifecycle.succeed(...) with model_version(s), dataset_version, artifacts
```

**Supervised training population, stated explicitly (this was ambiguous in
an earlier draft and is now resolved):** the GBM/LR/anomaly training
population is restricted to events that have **at least one associated
`source_alerts` row** — i.e., events `LocalYamlRuleProvider` actually
fired on — never the full synthetic event population. This is a
deliberate choice, not an oversight: the orchestrator only ever scores
source-alerted events at inference time (§4), so training on the full
event population (including never-alerted events as implicit negatives)
would create a train/serve population mismatch. Non-alerted events are
still generated and retained — they feed velocity/behavioral features and
graph history as ordinary prior activity (§9, §15) — but are never used as
positive or negative training examples for the supervised GBM/LR
classifiers. The feature-matrix loader joins through `source_alerts` (on
`event_id`) to enforce this, and Phase 4 adds an explicit test asserting
every row in the built training feature matrix has a corresponding
`source_alerts` entry.

**New `pipeline_name` values this guide requires**
(`fraud_score`, `label_eligibility`, `model_promotion`,
`fraud_evaluation`), used starting Phase 6/7 — `RunLifecycle.PIPELINE_NAMES`
(currently `{"batch", "train", "stream"}` in `src/control_plane/runs.py`)
is **extended additively** to include these four, together with the
matching `pipeline_runs.pipeline_name` `CHECK` constraint widening
(migration `004`, same `DROP CONSTRAINT IF EXISTS` / `ADD CONSTRAINT`
pattern as `002`). **No phase may call `lifecycle.begin("label_eligibility",
...)` (or any of the other three) until both the code and the applied
schema permit it** — Phase 6 does the extension and its first real use
together, in the same commit, exactly as `pipeline_name="train"` was
already valid before this guide and needed no change.

---

## 12. Chronological train/calibration/test split — full as-of-time contract

A new, additive function, coexisting with (not replacing) v1.2's
`assign_split()`:

```python
def assign_chronological_split(
    df: pl.DataFrame, timestamp_col: str, id_col: str,
    train_frac: float, calib_frac: float, test_frac: float,
    purge_gap: timedelta = timedelta(0),
) -> pl.DataFrame:
    """Sorts by (timestamp_col, id_col) for a fully deterministic order,
    cuts by TIME, and drops a purge_gap-wide buffer of rows at each
    boundary from every split (prevents near-boundary leakage from
    velocity/window features). Adds a "split" column."""
```

**The complete as-of-time contract, enforced everywhere, not just in the
split function:**
- A feature for an event at time `T` may use only records with timestamp
  strictly earlier than `T` (ties broken by `(timestamp, event_id)`).
- Graph construction/lookup for `T` may use only prior events (§15).
- Risk/counterparty lookups are fit on the training window's rows only.
- Encoders, imputers, scalers, and any feature selection are **fit on the
  training window only**, then applied (never re-fit) to calibration and
  test.
- Calibration (isotonic/Platt) is fit on the calibration window only.
- Ensemble policy weights/thresholds (§16) are tuned on the calibration
  window only — tuning code must not read the test window at all.
- The final test window is completely untouched until evaluation (§21).
- Same-timestamp events never split unpredictably — the `(timestamp,
  id_col)` sort makes the boundary deterministic for a fixed dataset.
- A configurable `purge_gap` creates an explicit buffer between windows
  when the data warrants it (default zero, tunable per channel).

This is the same leakage class as `HARDENING_LOG.md` Fix 5, applied here
to calibration, preprocessing, and ensemble tuning simultaneously, not
just to a risk lookup.

---

## 13. GBM primary and LR shadow challenger

**GBM (primary, operational):** XGBoost, same pattern as
`src/ml/train.py`, trained per channel, calibrated on the calibration
window, feeding the ensemble as `calibrated_gbm_probability`.

**LR (shadow challenger):** scikit-learn `LogisticRegression`, same
feature set and preprocessing artifact, trained/evaluated in the same
framework run, logged as `lr_probability` on every alert's evidence for
comparison — never blended into `operational_priority_score`, never
independently gates priority.

**Structural, not just policy, separation** (unchanged from the prior
draft — this remains the correct design and is explicitly tested, §8
Phase 5):

```python
def compute_operational_priority_score(
    *, rule_score: float, calibrated_gbm_probability: float,
    anomaly_score: float, graph_risk_score: float, policy: EnsemblePolicy,
) -> float: ...
```
No `lr_probability` parameter exists on this function — a code-level
fact, not a convention. `lr_probability` is part of the promoted bundle
(§22) but is never read by the scoring orchestrator's ensemble step.

---

## 14. Anomaly-detection design

One unsupervised model per channel (`IsolationForest`, already in
`requirements.txt` — no new dependency), trained on the training window,
label-agnostic, producing `anomaly_score` (0–1), retrained alongside the
channel's GBM/LR (§11), registered as part of the same candidate bundle.

---

## 15. In-process entity-graph design

A local, in-process graph (proposed library: `networkx` — the **one new
dependency** this guide introduces; confirm before adding it to
`requirements.txt` in Phase 5, or use the documented zero-dependency
adjacency-dict fallback) built per scoring run from recent
`channel_events` history, respecting the as-of-time contract (§12): graph
construction for an event at `T` uses only events with timestamp earlier
than `T`.

**Nodes:** customer, account, device, IP address, beneficiary, recipient,
card, ATM, check/payee. **Edges:** "participated together in an event."

**Graph-risk features:**
- shared-device-across-distinct-customers count
- beneficiary/recipient fan-in count (money-mule signal)
- sender fan-out count
- shortest path to any entity linked to a **mature, training-eligible,
  `RESOLVED_FRAUD`** assessment (§8) — **never** a raw
  `analyst_disposition = CONFIRMED_FRAUD`, which is comparison/feedback
  data only until it matures under the versioned label policy (§8). This
  is a deliberate correction: an unreviewed or freshly-disposed
  "confirmed fraud" tag must not immediately alter graph-based risk for
  every connected entity before the disposition itself has been validated
  by the maturity window.

No graph database — rebuilt or incrementally updated from a bounded
recent-activity window, entirely in-process.

---

## 16. Versioned ensemble policy

```python
class EnsemblePolicy(BaseModel):
    channel: str
    policy_version: str
    weight_rule: float
    weight_gbm: float
    weight_anomaly: float
    weight_graph: float
    high_threshold: float
    medium_threshold: float

def compute_operational_priority_score(*, rule_score, calibrated_gbm_probability,
                                        anomaly_score, graph_risk_score, policy) -> float: ...

def band_priority(
    score: float, policy: EnsemblePolicy, *,
    mandatory_review_hit: bool,
    provider_status: str, minimum_priority_band: Optional[str],
) -> str:
    """HIGH if mandatory_review_hit. Otherwise, if provider_status is
    UNAVAILABLE/ERROR, the band is floored at minimum_priority_band
    (never lower). Otherwise, thresholds from policy. LOW is a real,
    retained band — never dropped (§18)."""
```

Weights/thresholds live in versioned, config-driven YAML
(`config/fraud_intel/ensemble_policy_<channel>.yaml`), tuned on the
calibration window only (§12). Every alert's evidence records which
`policy_version` produced it. **Synthetic calibration does not represent
Citizens Bank production calibration** — restated in §7/§23/§27.

---

## 17. Explainable reason-code contract

Every alert's reason codes merge:
1. Rule-triggered codes (from `RuleEvaluationResult`, both the upstream
   `source_alert_reason_codes` and the triage-layer evaluation, kept
   distinguishable in the merged list).
2. Top-N GBM feature contributions via XGBoost's built-in
   `Booster.predict(pred_contribs=True)` — no new dependency.
3. An anomaly-flag code if `anomaly_score` exceeds a configured
   threshold.
4. Graph-flag codes (e.g. `SHARED_DEVICE_RING`, `RAPID_RECIPIENT_FAN_IN`)
   if graph thresholds are exceeded.

Each reason code: `code`, `text`, `layer` (`rule`\|`gbm`\|`anomaly`\|
`graph`), `severity` (`informational`\|`contributing`\|`mandatory`).
`scenario_id`/synthetic-label fields never appear in reason codes shown to
an analyst (§6).

**Reason codes are explanatory evidence, not causal findings, adverse-action
reasons, or regulatory conclusions.** A reason code states which signal
contributed to a score (e.g. "GBM feature X was high," "shared-device ring
detected") — it is not a determination that the signal *caused* fraud, not
an adverse-action reason under any regulation, and not a compliance or
legal conclusion of any kind. This distinction is restated in §23.

---

## 18. Alert queue, evidence and analyst-disposition data model

**`fraud_alerts`** — one row **per source alert**, not per event (mutable
`status` only; everything else set once at creation from the triggering
`source_alerts` row): `alert_id` (PK), `event_id` (FK to `channel_events`
— **not unique**; the same event can have multiple `fraud_alerts` rows,
one per distinct source alert), `source_alert_id` (FK to `source_alerts`),
`source_system`, `channel`, `customer_id`, `account_id`, `dollar_amount`,
`operational_priority_score`, `priority_band`, `ensemble_policy_version`,
`status` (`OPEN`\|`IN_REVIEW`\|`CLOSED`), `created_at`. The full upstream
alert detail (`source_alert_created_at`, `source_rule_ids`,
`source_rule_version`, `source_alert_score`, `source_alert_reason_codes`)
lives on `source_alerts` itself (§6) and is read via a join — it is not
duplicated onto `fraud_alerts`.

**Idempotency key: `UNIQUE (source_system, source_alert_id)` on
`fraud_alerts` — not `event_id`.** `event_id` alone cannot be the
idempotency key because one event legitimately produces more than one
`fraud_alerts` row when it has more than one source alert (§6). The
compound key is denormalized directly onto `fraud_alerts` (rather than
relying solely on `source_alerts.source_alert_id`'s own uniqueness) so the
constraint is self-evident from `fraud_alerts`'s own columns and remains
correct if a future multi-system deployment reuses alert-id numbering
across different upstream systems.

**Idempotent creation, with safe `alert_id` retrieval on conflict:**
```sql
INSERT INTO fraud_alerts (event_id, source_alert_id, source_system, ...) VALUES (%s, %s, %s, ...)
ON CONFLICT (source_system, source_alert_id) DO NOTHING
RETURNING alert_id;
-- if no row returned (conflict happened):
SELECT alert_id FROM fraud_alerts WHERE source_system = %s AND source_alert_id = %s;
```
Both statements run inside one transaction, matching the existing
`with conn: with conn.cursor()` pattern used throughout this codebase.

**`alert_evidence`** — insert-only, immutable, with **scoring-execution
idempotency**: `evidence_id` (PK, UUID), `alert_id` (FK),
`score_execution_id` (UUID — one per *logical* scoring execution; a
transient-failure retry of that same logical execution reuses the same
`score_execution_id`, while a deliberate, separate re-score is a new
logical execution and always generates a new one), `rule_result`
(JSONB, the triage-layer `RuleEvaluationResult`), `gbm_probability`,
`lr_probability`, `anomaly_score`, `graph_risk_score`, `reason_codes`
(JSONB), `channel_model_bundle_version`, `feature_schema_version`,
`config_hash`, `git_sha`, `event_time`, `scored_at`. **Unique constraint
`(alert_id, score_execution_id)`.** A retry of the exact same scoring
attempt (same `score_execution_id`, e.g. after a transient DB error) is a
safe no-op via `ON CONFLICT (alert_id, score_execution_id) DO NOTHING`. A
deliberate re-score generates a fresh `score_execution_id` and always
produces a new, additional evidence row — history is never overwritten.

**`analyst_dispositions`** — one row per analyst action (already
append-only by nature): `disposition_id` (PK), `alert_id` (FK),
`analyst_id`, `disposition`
(`CONFIRMED_FRAUD`\|`CONFIRMED_LEGITIMATE`\|`NEEDS_MORE_INFO`\|`ESCALATED`),
`notes` (bounded + redacted via `provenance.truncate_text`/
`redact_credentials`), `disposed_at`.

**`label_assessments`** (replaces a single mutable `alert_labels` row) —
**append-only**: `assessment_id` (PK), `alert_id` (FK), `policy_version`,
`evaluated_at`, `maturity_status`, `eligibility_result`, `resolved_label`,
`resolved_label_source`. The latest assessment is queried
(`ORDER BY evaluated_at DESC LIMIT 1`); every prior policy decision
remains permanently auditable — never deleted or updated in place.

**Model-only, non-alerted high scores (§4):** if the triage layer is run
in a monitoring/evaluation context against events that never produced a
source alert, results are written to a separate `shadow_candidate_scores`
log (Phase 7 evaluation tooling), never to `fraud_alerts` — this MVP does
not add them to the analyst queue.

---

## 19. Feedback and retraining eligibility

`analyst_dispositions` feed a dedicated, `RunLifecycle`-tracked batch job
(`pipeline_name="label_eligibility"`, Phase 6) that applies the versioned
eligibility policy (§8) and appends a new `label_assessments` row per
alert. Retraining is never automatic — it stays a manual
`aidp fraud-intel train --channel ach` invocation (§21), exactly like
`aidp train run` today. Eligibility marks data as *usable*; it never
triggers a training run.

---

## 20. Idempotency, auditability, privacy and provenance

- Every event carries a unique `event_id` (UUID); alert creation is
  idempotent per source alert (`(source_system, source_alert_id)`,
  correctly allowing multiple `fraud_alerts` rows for the same
  `event_id` when it has multiple source alerts), and evidence writes are
  idempotent at the scoring-execution level (`score_execution_id`) — §18.
- `alert_evidence` and `label_assessments` are insert-only — full history
  is always reconstructable.
- Both `event_time` and `scored_at` are stored on every evidence row.
- Every evidence row carries `channel_model_bundle_version`,
  `feature_schema_version`, `config_hash`, `git_sha`,
  `ensemble_policy_version`, and `rule_set_version` (inside
  `rule_result`) — full reconstructability of "what exactly produced this
  alert."
- All identifiers are synthetic/tokenized — there is no real PII in this
  POC, and no code path introduces any.
- No credentials are stored anywhere.
- Stored notes/errors are bounded and redacted via the *existing*
  `provenance.truncate_text()`/`redact_credentials()` — not reimplemented.

---

## 21. Evaluation and monitoring metrics

Computed **primarily on the source-alert population** (§2, §4 — that is
the actual analyst workload), per channel, per scenario, and in aggregate.
Never overall accuracy as a headline metric.

- Precision at analyst capacity (top-K source alerts by review budget)
- Recall at analyst capacity (of confirmed-fraud **source alerts**)
- Alerts reviewed per confirmed fraud
- False-positive reduction vs. the rules-only baseline (weights forced to
  `weight_gbm = weight_anomaly = weight_graph = 0`, same versioned
  `EnsemblePolicy` mechanism — not a separate code path)
- Analyst workload reduction at matched recall
- PR-AUC, calibration error (test window only)
- Dollar-weighted recall
- Per-channel and per-scenario breakdowns
- Alert-volume and score-distribution drift indicators
- **Secondary, supplementary only:** shadow-candidate analysis (model-only
  high scores on non-alerted events, §18) — reported separately, never
  blended into the primary source-alert metrics above.

---

## 22. Model governance and manual promotion — channel bundles

**Deliberate difference from v1.1:** `src/ml/train.py`'s `train()`
unconditionally registers and aliases `champion` on every run — not
appropriate here.

**Channel bundle, not individual model, promotion.** A
`channel_model_bundles` table row identifies one *atomic, compatible* set:
calibrated GBM version, LR shadow version, anomaly version, preprocessing
artifact version, ordered feature-schema version, the training
`RunLifecycle` run id, dataset version, and an evaluation-report
reference. Training (§11) registers a bundle with `status = CANDIDATE`.
`status` is one of `CANDIDATE`\|`OPERATIONAL`\|`RETIRED` — promoting a new
bundle retires the previously-operational one (below); nothing is ever
deleted.

```
aidp fraud-intel promote --channel ach --bundle-version 4 --database aidp_test
```

Promotion validates that every referenced component version actually
exists and belongs to the same training run (bundle compatibility), then
promotes the **whole bundle** to `status = OPERATIONAL` as one action —
never an individual model in isolation. LR remains shadow-only for
scoring purposes even though it is part of the promoted bundle — bundle
membership is a governance/versioning concept, not a scoring-input
change (§13 still holds structurally). Promotion is always manual,
`RunLifecycle`-tracked (`pipeline_name="model_promotion"`), and never
invoked automatically by training.

**Promotion is concurrency-safe at the database level, not just by
convention.** Two safeguards, both enforced in Phase 6's migration and
promotion code:
- A **partial unique index** —
  `CREATE UNIQUE INDEX uq_one_operational_bundle_per_channel ON
  channel_model_bundles (channel) WHERE status = 'OPERATIONAL';` —
  guarantees at most one `OPERATIONAL` bundle per channel can exist at
  the database level, independent of application logic.
- The promotion itself runs as **one transaction**: it locks the
  channel's current `OPERATIONAL` row (`SELECT ... FOR UPDATE`, if one
  exists), demotes it to `RETIRED`, then promotes the candidate to
  `OPERATIONAL`, all before commit. If any step fails — including a
  would-be violation of the partial unique index from a concurrent
  promotion racing on the same channel — the entire transaction rolls
  back and the previously-operational bundle is left untouched and still
  `OPERATIONAL`. Two concurrent `promote` invocations for the same
  channel can never both succeed.

---

## 23. US banking design-control considerations

Every statement below is a **design consideration requiring Citizens
Bank legal, compliance, and model-risk validation** — none of it is a
compliance claim, and none of it has been reviewed by counsel.

| Area | Design consideration (not a compliance claim) |
|---|---|
| Model-risk governance | Documented model purpose/limitations, an independent challenger model (§13), versioned validation and calibration (§12, §16) — engineering analogs of model-risk-management practice; not a model-risk-management program |
| Regulation E (consumer EFT / error resolution) | Fraud-alert triage prioritization is distinct from, and does not alter, Reg E's consumer error-resolution timelines and obligations — Citizens Bank compliance must confirm no interaction with error-resolution workflows |
| UCC Article 4A (commercial wire transfers) | The Wire channel's fraud triage does not implement or substitute for Article 4A's commercial-funds-transfer risk allocation rules — flagged for legal review before any Wire-channel production use |
| NACHA operating rules (ACH) | The ACH channel's rules/features (§9, §10) are fraud-signal heuristics, not an implementation of NACHA's operating-rule obligations — requires NACHA-rules-compliance review separately |
| GLBA data safeguarding | Not applicable to this synthetic-only POC; would become applicable immediately if real customer data were ever used, requiring a full safeguards review at that time |
| OFAC/sanctions screening | A separate upstream control at Citizens Bank, not implemented by this POC's fraud-triage rules (§10, §26) |
| BSA/AML transaction monitoring | A separate compliance program at Citizens Bank, distinct from fraud-alert triage; this POC does not implement BSA/AML monitoring (§26) |
| Record retention, access control, audit | Immutable evidence and append-only assessments (§18, §20) are engineering patterns that could *support* a retention/audit program — they do not constitute one |
| Reason-code explainability | Reason codes (§17) are explanatory evidence of which signal contributed to a score — not causal findings, not adverse-action reasons under any regulation, and not a compliance or legal conclusion; any use of them in a regulated context requires Citizens Bank legal/compliance review |

**ECOA/adverse-action discussion removed from this guide entirely** — this
POC is fraud-alert triage, not a credit-decisioning system, so ECOA's
adverse-action-notice requirements are not applicable here and including
that discussion would misrepresent the system's purpose.

Any named regulation, rule-making body, or standards organization above
(Federal Reserve/OCC/FDIC model-risk guidance, CFPB for Reg E, the Uniform
Commercial Code for Article 4A, NACHA for ACH rules, OFAC/FinCEN for
sanctions and BSA/AML) is referenced by name only, as the owner of the
relevant framework — **Citizens Bank counsel and compliance must determine
actual applicability**; nothing here is legal advice or a compliance
determination.

---

## 24. One-week delivery schedule

**This is an aggressive target, dependent on no infrastructure or
dependency blockers arising — not a committed production timeline.** No
test, migration, model layer, or audit control may be skipped merely to
hit Day 7. If a checkpoint fails, delivery extends; the architecture is
never weakened to preserve the schedule.

| Day | Work | Exit criteria |
|---|---|---|
| 1 | Phase 0 + Phase 1 | Guide-commit-only branch state confirmed (§25 Phase 0); all 7 schemas + generators + migration `003` reviewed; `pytest tests/unit -q` passes |
| 2 | Phase 2 | Shared feature core + reference-channel adapter fully unit-tested |
| 3 | Phase 3 | All 7 channels' rule sets pass fixture tests; failure-path (`minimum_priority_band`) tested |
| 4 | Phase 4 | Chronological split + purge gap proven leak-free; GBM+LR+preprocessing trained on fully-mocked fixtures; candidate bundle registered |
| 5 | Phase 5 | Orchestrator (§8) proven end to end on the reference channel via one integration-style unit test, not components alone; LR structurally excluded from the ensemble (tested) |
| 6 | Phase 6 | Idempotent alert/evidence writes tested (including retry-safety); append-only label assessments; CLI surfaces for reference channel; migration `004` reviewed |
| 7 | Phase 7A + 7B (separately approved) + Phase 8 | Remaining 6 channels unit-tested (7A); **real** `aidp_test` migration application, real deterministic generation, real local training of all 7 bundles, explicit promotion, real scoring, integration/smoke verification (7B, only after your separate approval); documentation/demo only after 7B succeeds |

**Definition of "runnable POC complete":** all 7 channels score real,
locally-generated synthetic data through the full pipeline against
`aidp_test`, with at least one promoted bundle per channel, alerts visible
in both the CLI and the dashboard tab, and the cross-channel evaluation
report computed from real (not mocked) scoring output. Anything short of
that is "unit-tested but not yet runtime-verified," and must be reported
as such, not as complete.

---

## 25. Phase-by-phase build instructions

Same discipline as v1.2: fresh-install schema **plus** a numbered,
idempotent migration for every schema change; migration files reviewed
before any execution; applied first, and only, to `aidp_test`, after
separate explicit approval; **never** `reset_demo.sh` as a migration
mechanism; no database operation hidden inside a unit test; unit tests run
without Docker; integration/smoke tests clearly separate; no automatic
model/bundle promotion; no Git push or PR without explicit approval; no AI
attribution or Co-Authored-By lines.

Every migration/database command, when its checkpoint is reached, is
presented **before execution** with: exact target (database name), exact
read/write behavior, expected data changes, and rollback/recovery cost —
the same format already used for migration `002`'s `aidp_test` checkpoint
in this repository's history. No command ever targets `aidp` without a
separate approval.

### Phase 0 — Baseline, architecture and interface documentation

**Objective:** verify branch state and produce an integration map — no
runtime code.

**Corrected preflight** (this guide's own commit will legitimately exist
on this branch by the time Phase 0 runs): confirm the approved guide
commit is present on `feat/aidp-v1-3-fraud-intelligence`; confirm no
*additional*, unexpected commits exist beyond it; confirm the working tree
is clean. Do **not** require "no unique commits beyond origin/main" — that
was correct only before the guide itself was committed.

**Scope:** inspect `src/control_plane/`, `src/common/features.py`,
`src/common/scoring.py`, `src/decisioning/engine.py`,
`config/fraud_rules.yaml`, `infrastructure/postgres/lib/schema.sql`; add a
v1.3 "in progress" section to `ARCHITECTURE.md`.

**Expected files:** `ARCHITECTURE.md` only.

**Copy-paste prompt:**
```text
You are working in the local-aidp-poc repository on branch
feat/aidp-v1-3-fraud-intelligence.

This is Phase 0 of the AiDP v1.3 fraud-intelligence work. Do not implement
any runtime code.

Confirm: the approved AIDP_V1_3_FRAUD_INTELLIGENCE_BUILD_GUIDE.md commit is
present on this branch, no additional unexpected commits exist beyond it,
and the working tree is clean.

Inspect: src/control_plane/, src/common/features.py, src/common/scoring.py,
src/decisioning/engine.py, config/fraud_rules.yaml,
infrastructure/postgres/lib/schema.sql, and the existing test suite.

Add a new "v1.3 fraud intelligence (in progress)" section to
ARCHITECTURE.md summarizing the target design (source-alert triage, not
new alert generation) and explicitly marking it as not yet implemented.

Constraints: no runtime code, no schema changes, no new dependencies, no
database or Docker operations, no Co-authored-by or AI attribution, no
push.

Commit locally with: docs: prepare v1.3 fraud-intelligence work

Report files changed and final git status.
```

**Explicit exclusions:** no channel schemas, no generators, no models.

**Tests:** existing suite unchanged.

**Infra/DB checkpoint:** none.

**Acceptance criteria:** `ARCHITECTURE.md` updated only; branch state
matches the corrected preflight above.

**Commit message:** `docs: prepare v1.3 fraud-intelligence work`

**Stop and review** before Phase 1.

---

### Phase 1 — Seven channel schemas, source-alert context, synthetic-label storage, generators, migration `003`

**Objective:** define every channel's strengthened event contract, the
`SourceAlertContext` contract, separate synthetic-label storage, and
produce synthetic events for all seven channels — no scoring yet.

**Scope:** `src/fraud_intel/events/` (strengthened base + `SourceAlertContext`
+ 7 discriminated channel payloads + the channel-invariant model
validator); `src/fraud_intel/generator/` (7 generators); fresh-install
`channel_events`, `source_alerts`, `synthetic_event_labels`, and
`channel_model_bundles` tables in `schema.sql`; migration
`003_channel_events_and_labels.sql`.

**Expected files:**
```
src/fraud_intel/__init__.py
src/fraud_intel/events/{base,source_alert_context,ach,wire,mobile_deposit,online_banking,atm,debit_card,p2p}.py
src/fraud_intel/generator/{customers,ach,wire,mobile_deposit,online_banking,atm,debit_card,p2p}.py
infrastructure/postgres/lib/schema.sql (channel_events, source_alerts, synthetic_event_labels, channel_model_bundles)
infrastructure/postgres/migrations/003_channel_events_and_labels.sql
tests/unit/test_fraud_intel_events.py
tests/unit/test_fraud_intel_generators.py
tests/unit/test_fraud_intel_label_isolation.py
```

**Copy-paste prompt:**
```text
Implement only Phase 1 of AiDP v1.3: strengthened channel event contracts,
the SourceAlertContext contract, separate synthetic-label storage, and
synthetic generators for all seven channels.

Read the Phase 0 integration map first.

FraudEvent base: event_id (uuid.UUID), channel (Literal enum),
customer_id, account_id, event_timestamp (UTC, timezone-aware, validated),
amount_minor_units (int, gt=0 — NOT a float), direction, device_id
(optional), ip_address (optional), scenario_id (Optional[str] = None —
synthetic-only, never required), schema_version (enforced against a
SUPPORTED_SCHEMA_VERSIONS set), channel_payload (a discriminated Union of
the seven channel payload models, discriminated on a "channel" literal
field on each — not a raw dict). Serialize deterministically (canonical,
sorted-key JSON, reusing the same technique RunConfig.config_hash() uses).

Define the seven channel payload models exactly per the guide's section 6
table.

Add a model validator to FraudEvent that rejects a mismatch between
FraudEvent.channel and channel_payload.channel (see the guide's section 6
code snippet) — Pydantic's discriminated union alone does not enforce
this, since it only validates channel_payload against its own internal
discriminator, not against the outer FraudEvent.channel field.

Define SourceAlertContext per the guide's section 6 (source_alert_id,
source_system, event_id, source_alert_created_at, source_rule_ids,
source_rule_version, source_alert_score [optional],
source_alert_reason_codes, generation_run_id, dataset_version,
created_at). event_id is a foreign key, NOT unique — the same event_id can
appear on more than one source_alerts row.

Define SyntheticGroundTruthLabel (event_id, scenario_id, is_fraud_scenario
[bool], scenario_type, label_source="SYNTHETIC_GENERATOR", generated_at) as
its own model — this is never embedded in channel_payload and never
consumed by any scoring code.

Write one deterministic, seeded generator per channel producing normal
events, source-alert-firing events, and labeled fraud scenarios (low
prevalence including below 5%, deliberate hard-false-positive cases).
Generators emit (FraudEvent, SourceAlertContext | None,
SyntheticGroundTruthLabel) tuples — the label is always produced
separately, never merged into the event.

Schema: add channel_events (event_id UUID PK, channel, customer_id,
account_id, event_timestamp, amount_minor_units, direction, device_id,
ip_address, channel_payload JSONB, scenario_id, schema_version,
generation_run_id, dataset_version, created_at); source_alerts
(source_alert_id UUID PK, source_system, event_id UUID FK to
channel_events [NOT unique — one event can have multiple source alerts],
source_alert_created_at, source_rule_ids JSONB, source_rule_version,
source_alert_score, source_alert_reason_codes JSONB, generation_run_id,
dataset_version, created_at); synthetic_event_labels (event_id UUID unique
FK, scenario_id, is_fraud_scenario, scenario_type, label_source,
generation_run_id, dataset_version, generated_at); and
channel_model_bundles (bundle_id PK, channel, bundle_version,
gbm_model_version, lr_model_version, anomaly_model_version,
preprocessing_artifact_version, feature_schema_version, training_run_id,
dataset_version, evaluation_report_ref, status
[CANDIDATE|OPERATIONAL|RETIRED], created_at, promoted_at, promoted_by),
plus a partial unique index
uq_one_operational_bundle_per_channel ON channel_model_bundles (channel)
WHERE status = 'OPERATIONAL' (see guide section 22 — defined now even
though promotion code lands in Phase 6/22) to
infrastructure/postgres/lib/schema.sql — channel_model_bundles is not
populated until Phase 4, but its shape is defined now.
synthetic_event_labels.event_id stays unique (one ground-truth label per
event — this is unrelated to source_alerts' one-to-many relationship).
Add the matching migration
infrastructure/postgres/migrations/003_channel_events_and_labels.sql,
idempotent, reviewed but NOT applied to any database in this phase.

Constraints: no scoring, no rules, no models, no new dependencies, no
database or Docker operations, no Co-authored-by or AI attribution, no
push.

Tests must cover: each channel schema rejects invalid input (including a
naive or non-UTC timestamp, and a float amount where an int is required);
the discriminated union correctly routes each channel's payload; a
deliberately mismatched FraudEvent.channel / channel_payload.channel pair
is rejected by the model validator; each generator is deterministic for a
fixed seed; fraud prevalence is within the configured low range; an
explicit test proves synthetic_scenario_label and scenario_id are never
present anywhere inside a FraudEvent's channel_payload or top-level
scoring-relevant fields (only the optional scenario_id field itself, never
a label) — i.e. that SyntheticGroundTruthLabel and FraudEvent are
genuinely separate objects that could be serialized/stored independently;
a test proves the same event_id can appear on more than one source_alerts
row without violating any constraint.

Run pytest tests/unit -q. Review the diff, including both SQL files.
Commit locally with:
feat(fraud-intel): add channel event contracts, source-alert context, and synthetic generators

Report files changed, tests/counts, commit SHA, final git status.
```

**Explicit exclusions:** no features, no rules, no models, migration `003`
not applied anywhere.

**Tests:** schema validation (including the new UUID/int/UTC constraints),
generator determinism, label-isolation proof — no infrastructure.

**Infra/DB checkpoint:** migration `003` reviewed only; not applied. (See
§25 Phase 1 checkpoint below, for after Phase 1's code is approved.)

**Acceptance criteria:** all 7 channels have a strengthened schema +
generator; the channel invariant is enforced by a model validator, not
assumed; `source_alerts` correctly supports multiple alerts per event;
`SourceAlertContext` and `SyntheticGroundTruthLabel` are provably separate
from the scored event; fresh schema and migration agree.

**Commit message:** `feat(fraud-intel): add channel event contracts, source-alert context, and synthetic generators`

**Migration `003` checkpoint (optional, your call):** after this phase's
code is reviewed and approved, you may separately approve applying
migration `003` to `aidp_test` only, verified the same way migration `002`
was — pre-check (`current_database()`, existing schema, row count),
apply with `ON_ERROR_STOP=1`, post-check (new tables/columns present, row
count unchanged). This is optional at this point since nothing writes to
these tables until later phases; it is required before Phase 7B.

**Stop and review** before Phase 2.

---

### Phase 2 — Shared feature core, all adapter interfaces, reference channel adapter

**Objective:** build the shared feature-computation core, respecting the
as-of-time contract, and prove the pattern on the reference channel
(Online/Mobile Banking).

**Scope:** `src/fraud_intel/features/core.py` (shared, pure, as-of-time
correct); `src/fraud_intel/features/channels/online_banking.py`
(reference adapter); interface stubs for the other 6.

**Expected files:**
```
src/fraud_intel/features/core.py
src/fraud_intel/features/channels/online_banking.py
src/fraud_intel/features/channels/{ach,wire,mobile_deposit,atm,debit_card,p2p}.py  (stubs)
tests/unit/test_fraud_intel_features_core.py
tests/unit/test_fraud_intel_features_online_banking.py
tests/unit/test_fraud_intel_feature_leakage.py
```

**Copy-paste prompt:**
```text
Implement only Phase 2 of AiDP v1.3: the shared feature core and the
Online/Mobile Banking reference channel's feature adapter.

Read Phase 1's event/label contracts first.

src/fraud_intel/features/core.py: pure functions per the guide's section 9
shared feature group, reusing the WINDOW_10M/1H/24H and RiskLookups
patterns from src/common/features.py. Every function must accept an
explicit "as_of" timestamp parameter and only read history strictly
earlier than it (ties broken by (event_timestamp, event_id)) — never rely
on caller-side pre-filtering alone; assert it internally wherever
practical.

src/fraud_intel/features/channels/online_banking.py: the reference
channel-specific feature group from section 9, built on the shared core.

Add signature-only stub adapters (NotImplementedError) for the remaining
six channels.

Constraints: no rules, no models, no database writes, no new dependencies,
no Docker operations, no Co-authored-by or AI attribution, no push.

Tests must cover: every shared core function against hand-computed
expected values, including an explicit as-of-time boundary test (a record
exactly at T must NOT be used for a feature computed as-of T); the
online-banking adapter against a constructed fixture; the six stubs raise
NotImplementedError predictably; a dedicated leakage test asserting that
none of the feature functions' output column names include
synthetic_scenario_label, scenario_id, analyst_disposition,
outcome_status, or training_eligible, and that none of these functions
accept the label/disposition tables as an input at all (structural, via
inspect.signature).

Run pytest tests/unit -q. Commit locally with:
feat(fraud-intel): add shared feature core and reference channel adapter

Report files changed, tests/counts, commit SHA, final git status.
```

**Explicit exclusions:** the other 6 channels' real feature logic.

**Tests:** pure-function unit tests, as-of-time boundary test, structural
leakage test — no infrastructure.

**Infra/DB checkpoint:** none.

**Acceptance criteria:** shared core respects as-of-time strictly; label
leakage is structurally impossible, not just untested.

**Commit message:** `feat(fraud-intel): add shared feature core and reference channel adapter`

**Stop and review** before Phase 3.

---

### Phase 3 — RuleProvider, fixed operator allowlist, and versioned rules for all seven channels

**Objective:** implement `RuleProvider`, its local YAML engine with a
fixed operator allowlist, and one versioned rule set per channel.

**Scope:** `src/fraud_intel/rules/provider.py`;
`config/fraud_intel/rules_<channel>.yaml` × 7.

**Expected files:**
```
src/fraud_intel/rules/provider.py
config/fraud_intel/rules_{ach,wire,mobile_deposit,online_banking,atm,debit_card,p2p}.yaml
tests/unit/test_fraud_intel_rules.py
```

**Copy-paste prompt:**
```text
Implement only Phase 3 of AiDP v1.3: the RuleProvider interface, its local
YAML-driven implementation with a fixed operator allowlist, and versioned
rule sets for all seven channels.

Read Phase 2's feature core first.

Define RuleEvaluationResult and RuleProvider exactly per the guide's
section 10 contract, including minimum_priority_band and
provider_error_code.

Implement LocalYamlRuleProvider: one engine, loads
config/fraud_intel/rules_<channel>.yaml per channel, each with its own
rule_set_version. Rule conditions use ONLY the fixed operator allowlist
(eq, ne, gt, gte, lt, lte, in, not_in, is_true, is_false) via a Python
dispatch table — never eval(), never exec(), never a general expression
parser. Use "confirmed compromised device" (device_linked_to_resolved_fraud)
as the MANDATORY_REVIEW example, not a sanctions-related example — this
POC does not implement OFAC/sanctions screening or BSA/AML monitoring;
those are separate upstream systems (say so in a code comment).

On an internal exception or timeout, return provider_status="ERROR" (or
"UNAVAILABLE"), a provider_error_code, a RULE_PROVIDER_UNAVAILABLE reason
code, and minimum_priority_band="MEDIUM" — do NOT alter
score_contribution to simulate this.

Constraints: no models, no ensemble, no database writes, no new
dependencies, no Docker operations, no Co-authored-by or AI attribution,
no push.

Tests must cover: each channel's rule set against constructed feature
fixtures; every allowlisted operator; that a malformed rule YAML (e.g. an
operator not on the allowlist) fails validation at load time rather than
silently passing through; provider failure produces provider_status=ERROR
with minimum_priority_band="MEDIUM" and unchanged score_contribution
(explicitly assert score_contribution is not touched by the failure path).

Run pytest tests/unit -q. Commit locally with:
feat(fraud-intel): add rule-provider interface and versioned channel rules

Report files changed, tests/counts, commit SHA, final git status.
```

**Explicit exclusions:** no models, no ensemble combination.

**Tests:** all 7 rule sets, operator allowlist enforcement, corrected
failure-path test — no infrastructure.

**Infra/DB checkpoint:** none.

**Acceptance criteria:** rules are never evaluated as arbitrary code;
failure behavior uses `minimum_priority_band`, never score manipulation.

**Commit message:** `feat(fraud-intel): add rule-provider interface and versioned channel rules`

**Stop and review** before Phase 4.

---

### Phase 4 — Shared training framework, chronological split with purge gap, calibrated GBM, LR shadow, candidate bundle registration

**Objective:** prove the full training path — chronological split with an
explicit purge gap, training-window-only preprocessing, calibrated GBM,
shadow LR — on the reference channel, registering one candidate bundle.

**Scope:** `src/common/splits.py` gets `assign_chronological_split()`
(additive); `src/fraud_intel/config.py` (`ChannelTrainingRunConfig`);
`src/fraud_intel/models/{training,calibration,preprocessing}.py`; bundle
registration into `channel_model_bundles` (`status=CANDIDATE`).

**Expected files:**
```
src/common/splits.py (add assign_chronological_split, additive)
src/fraud_intel/config.py
src/fraud_intel/models/{training,calibration,preprocessing,bundle}.py
tests/unit/test_fraud_intel_chronological_split.py
tests/unit/test_fraud_intel_training.py
tests/unit/test_fraud_intel_training_population.py
tests/unit/test_fraud_intel_bundle_registration.py
```

**Copy-paste prompt:**
```text
Implement only Phase 4 of AiDP v1.3: the shared per-channel training
framework with a chronological split (including a purge gap), training-
window-only preprocessing, calibrated GBM primary, LR shadow challenger,
and candidate channel-bundle registration — proven on the Online/Mobile
Banking reference channel only.

Read Phases 1-3 and re-read src/ml/train.py, src/control_plane/runs.py,
src/control_plane/config.py, and HARDENING_LOG.md Fix 5 before writing
anything.

Add assign_chronological_split(df, timestamp_col, id_col, train_frac,
calib_frac, test_frac, purge_gap=timedelta(0)) to src/common/splits.py as
a new, additive function — do not modify or remove assign_split(). Sort by
(timestamp_col, id_col) for full determinism; drop a purge_gap-wide buffer
of rows at each boundary from every split.

Add ChannelTrainingRunConfig (subclasses RunConfig) to
src/fraud_intel/config.py, following BatchRunConfig/TrainingRunConfig's
exact shape.

src/fraud_intel/models/preprocessing.py: a ChannelPreprocessor that fits
(encoders/imputers/scalers, as applicable) on the TRAINING window only and
applies (never re-fits) to calibration/test; versioned
(preprocessing_artifact_version) and serializable.

Implement train_channel_configured(config, *, trigger_source="legacy") in
src/fraud_intel/models/training.py: begin() a RunLifecycle run
(pipeline_name="train", channel in config_snapshot) -> load features,
joined through source_alerts on event_id so the training population is
restricted to events with at least one source_alerts row (guide section
11 — never the full synthetic event population) -> assign_chronological_split with a configurable purge gap -> fit
preprocessing on the training window -> train GBM (XGBoost) -> calibrate
on the calibration window only (src/fraud_intel/models/calibration.py,
isotonic or Platt) -> train LR (scikit-learn) -> evaluate both on the
untouched test window -> register GBM/LR + preprocessing artifact +
ordered feature-schema version to MLflow (registration only, no alias) ->
write one channel_model_bundles row with status=CANDIDATE referencing all
of the above plus this training run's id -> succeed() with
dataset_version, model_version(s), artifacts. Any exception ->
fail_from_exception() -> re-raise.

Constraints: reference channel only. No MLflow alias anywhere. Calibration
must never touch the test window — add a runtime assertion, not just a
docstring, that raises if it's attempted. No database schema changes
beyond what Phase 1 already defined. No Docker/infrastructure operations,
no real large-scale training — small synthetic fixtures only, fully
mocked MLflow/training internals exactly like
tests/unit/test_train.py's existing pattern. No Co-authored-by or AI
attribution, no push.

Tests must cover: assign_chronological_split with a nonzero purge_gap
produces non-overlapping windows with the expected gap; calibration
fitting raises if given test-window rows (test this directly, not just by
absence); the training framework runs end to end on a tiny synthetic
fixture with GBM/LR/MLflow all faked (no real training, no real MLflow);
a channel_model_bundles row is written with status=CANDIDATE and
references a consistent set of versions; a failure mid-training records
FAILED via fail_from_exception and re-raises; no MLflow alias is ever set;
and — in tests/unit/test_fraud_intel_training_population.py — an explicit
test asserting that every row in the built training feature matrix has a
corresponding source_alerts entry, and that a fixture event with NO
source_alerts row is correctly excluded from the training population even
though it has a synthetic_event_labels row.

Run pytest tests/unit -q. Review the diff. Commit locally with:
feat(fraud-intel): add chronological training framework with candidate bundle registration

Report files changed, tests/counts, commit SHA, final git status.
```

**Explicit exclusions:** anomaly/graph/ensemble/reason codes (Phase 5);
other 6 channels (Phase 7); no alias/promotion anywhere.

**Tests:** split correctness with purge gap, calibration-leakage
assertion (tested, not just documented), fully mocked training run,
bundle-registration shape, failure path — no infrastructure, no real
training.

**Infra/DB checkpoint:** none (bundle table already exists from Phase 1's
reviewed-but-unapplied migration).

**Acceptance criteria:** chronological split proven leak-free including at
the purge-gap boundary; preprocessing fit only on the training window;
GBM calibrated only on the calibration window; training population is
provably restricted to source-alerted events only; one candidate bundle
registered with full provenance.

**Commit message:** `feat(fraud-intel): add chronological training framework with candidate bundle registration`

**Stop and review** before Phase 5.

---

### Phase 5 — Anomaly, graph, ensemble, reason codes, and the scoring orchestrator (reference channel)

**Objective:** complete the full layered-scoring pipeline **through one
concrete orchestrator function** — this is the phase where the
architecture is proven working together for the first time, not just as
separate components.

**Scope:** anomaly detector added to Phase 4's framework; in-process
entity graph (mature-labels-only proximity feature); versioned ensemble
policy with the corrected `band_priority()`; reason-code builder; **the
scoring orchestrator** (`src/fraud_intel/scoring/orchestrator.py`). New
dependency decision point: `networkx`.

**Expected files:**
```
src/fraud_intel/models/anomaly.py
src/fraud_intel/graph/entity_graph.py
src/fraud_intel/ensemble/policy.py
config/fraud_intel/ensemble_policy_online_banking.yaml
src/fraud_intel/reason_codes/builder.py
src/fraud_intel/scoring/orchestrator.py
requirements.txt (± networkx, only if approved)
tests/unit/test_fraud_intel_anomaly.py
tests/unit/test_fraud_intel_graph.py
tests/unit/test_fraud_intel_ensemble.py
tests/unit/test_fraud_intel_reason_codes.py
tests/unit/test_fraud_intel_orchestrator.py
```

**Copy-paste prompt:**
```text
Implement only Phase 5 of AiDP v1.3: anomaly detection, the in-process
entity graph, the versioned ensemble policy with corrected failure-band
handling, the reason-code builder, and — critically — a concrete scoring
orchestrator that sequences all of it, for the Online/Mobile Banking
reference channel, completing the full pipeline end to end for the first
time.

Read Phase 4's training framework and Phase 3's corrected RuleProvider
failure contract first.

Anomaly: IsolationForest per channel (scikit-learn, already a dependency),
added to Phase 4's training framework, trained on the training window,
registered as part of the same candidate bundle (no separate alias
mechanism).

Graph: src/fraud_intel/graph/entity_graph.py, in-process only. Before
writing code, stop and tell me whether you'll use networkx (new
dependency, needs my approval) or a hand-rolled adjacency-dict. Respect
the as-of-time contract from Phase 2. The "shortest path to a
fraud-linked entity" feature must query ONLY mature, training-eligible,
RESOLVED_FRAUD label_assessments rows (Phase 6 introduces this table; for
this phase, accept a resolved-fraud entity list as an explicit parameter
so the graph logic itself doesn't need the not-yet-built table) — it must
NEVER read raw analyst_disposition values.

Ensemble: src/fraud_intel/ensemble/policy.py per section 16.
compute_operational_priority_score has no lr_probability parameter
anywhere in its signature (verify with an explicit structural test).
band_priority() enforces minimum_priority_band when provider_status is
UNAVAILABLE/ERROR, in addition to the existing mandatory-review-hit and
threshold logic. Add config/fraud_intel/ensemble_policy_online_banking.yaml.

Reason codes: src/fraud_intel/reason_codes/builder.py, merging rule +
GBM-contribution (Booster.predict(pred_contribs=True), no new dependency)
+ anomaly-flag + graph-flag codes per section 17, excluding
scenario_id/synthetic-label content entirely.

Orchestrator: src/fraud_intel/scoring/orchestrator.py,
score_source_alert(event, source_alert_context, bundle, policy,
resolved_fraud_entities) -> ScoredAlert, sequencing exactly: context
validation -> feature adapter -> rule enrichment -> GBM (from the bundle)
-> anomaly -> graph -> ensemble policy -> priority band -> reason codes ->
an evidence-ready result object. This is THE integration point — no other
code path may duplicate this sequencing logic.

Constraints: reference channel only. lr_probability must never reach the
ensemble function. No database writes yet — the orchestrator returns a
result object; persistence is Phase 6. No Docker/infrastructure
operations. No Co-authored-by or AI attribution, no push.

Tests must cover: anomaly detector on a small fixture; graph
fan-in/fan-out/shared-device/shortest-path-to-resolved-fraud-only logic
(with a test proving a raw analyst_disposition entity is NOT treated as
fraud-linked); the ensemble function's signature has no lr_probability
parameter; a MANDATORY_REVIEW rule hit forces HIGH; a provider failure
enforces minimum_priority_band via band_priority(); reason codes merge all
four sources deterministically; and — the key test for this phase — a
single test that runs score_source_alert() end to end on a constructed
reference-channel fixture and asserts the final ScoredAlert's band, score,
and reason codes are correct, proving the orchestrator (not just its
components) works.

Run pytest tests/unit -q. Review the diff (including requirements.txt if
changed). Commit locally with:
feat(fraud-intel): add anomaly, graph, ensemble, reason codes, and the scoring orchestrator

Report files changed, tests/counts, whether networkx was added (and why),
commit SHA, final git status.
```

**Explicit exclusions:** other 6 channels; alert/evidence persistence
(Phase 6).

**Tests:** all four new layers plus **one full orchestrator-level test**,
not components alone — this satisfies the "component tests alone are not
sufficient" requirement directly.

**Infra/DB checkpoint:** none.

**Acceptance criteria:** `score_source_alert()` exists and is exercised
end to end by at least one test; LR is provably shadow-only; graph
fraud-proximity uses only mature resolved labels, never raw dispositions;
rule-provider failure floors the band via `minimum_priority_band`, never
via score manipulation.

**Commit message:** `feat(fraud-intel): add anomaly, graph, ensemble, reason codes, and the scoring orchestrator`

**Stop and review** before Phase 6.

---

### Phase 6 — Analyst queue, immutable/idempotent evidence, append-only assessments, lifecycle extension, CLI surfaces, migration `004`

**Objective:** persist alerts and evidence idempotently, capture analyst
review as append-only history, extend `RunLifecycle` for the new
pipeline names, and expose the reference channel through the CLI. **No
FastAPI/API surface is implemented in this phase** — CLI only.

**Scope:** `fraud_alerts` / `alert_evidence` / `analyst_dispositions` /
`label_assessments` tables (fresh schema + migration `004`, which also
widens `pipeline_runs.pipeline_name`); `RunLifecycle.PIPELINE_NAMES`
extension in `src/control_plane/runs.py`; alert-creation + evidence
(idempotent at both the `event_id` and `score_execution_id` level);
disposition capture; label eligibility batch job; `aidp fraud-intel
score/train/evaluate/promote/model show` and `aidp alerts
list/show/disposition` CLI commands, each with an explicit, non-implicit
database target.

**Expected files:**
```
infrastructure/postgres/lib/schema.sql (4 new tables + pipeline_name CHECK widened)
infrastructure/postgres/migrations/004_fraud_alerts_evidence_and_lifecycle.sql
src/control_plane/runs.py (PIPELINE_NAMES extended, additive)
src/fraud_intel/alerts/queue.py
src/fraud_intel/labels/eligibility.py
src/cli/__main__.py (extended: fraud-intel score/train/evaluate/promote/model-show, alerts list/show/disposition)
tests/unit/test_fraud_intel_alert_queue.py
tests/unit/test_fraud_intel_evidence_idempotency.py
tests/unit/test_fraud_intel_label_eligibility.py
tests/unit/test_control_plane_runs.py (extended: new pipeline_name values)
tests/unit/test_cli_fraud_intel.py
```

**Copy-paste prompt:**
```text
Implement only Phase 6 of AiDP v1.3: the alert queue, idempotent evidence
at both the event and scoring-attempt level, append-only label
assessments, the RunLifecycle pipeline_name extension, and CLI surfaces —
reference channel only for actual scoring. Do not implement a FastAPI
endpoint anywhere in this phase; CLI only.

Read Phase 5's orchestrator output shape, src/control_plane/runs.py, and
infrastructure/postgres/migrations/002 first.

Extend src/control_plane/runs.py's PIPELINE_NAMES set additively to
include "fraud_score", "label_eligibility", "model_promotion", and
"fraud_evaluation", alongside the existing "batch"/"train"/"stream". Do
not remove or rename any existing value. Update existing
tests/unit/test_control_plane_runs.py tests only if they assert the exact
set membership (add the four new values to that assertion; do not touch
anything else in that file).

Schema: add the four tables from section 18 (fraud_alerts, alert_evidence
[insert-only, document as such], analyst_dispositions, label_assessments
[append-only, replaces any single mutable label row]), with fraud_alerts
referencing source_alerts (Phase 1) via source_alert_id, a unique
constraint on fraud_alerts(source_system, source_alert_id) — NOT
event_id, since one event can have multiple source alerts and therefore
multiple fraud_alerts rows (guide section 18) — and a unique constraint on
alert_evidence(alert_id, score_execution_id). In the SAME migration, widen
pipeline_runs.pipeline_name's CHECK constraint (DROP CONSTRAINT IF
EXISTS / ADD CONSTRAINT, same pattern as migration 002) to include the
four new values — schema and code must land together. Add migration
004_fraud_alerts_evidence_and_lifecycle.sql, idempotent, reviewed but NOT
applied to any database in this phase.

src/fraud_intel/alerts/queue.py: create_alert_if_new(event,
source_alert_context) using the INSERT ... ON CONFLICT (source_system,
source_alert_id) DO NOTHING RETURNING alert_id, falling back to a SELECT
on (source_system, source_alert_id) on conflict, exactly per section 18's
documented pattern — a second, distinct source alert for the SAME
event_id must create a second, separate fraud_alerts row, never be
treated as a duplicate; record_evidence(alert_id,
score_execution_id, scored_result) using INSERT ... ON CONFLICT (alert_id,
score_execution_id) DO NOTHING — a same-attempt retry is a no-op, a new
score_execution_id always inserts a new row; record_disposition(alert_id,
analyst_id, disposition, notes) with notes passed through
provenance.truncate_text()/redact_credentials().

src/fraud_intel/labels/eligibility.py: applies the versioned eligibility
policy (section 8) and APPENDS a new label_assessments row (never updates
an existing one), run as its own RunLifecycle job with
pipeline_name="label_eligibility" — only after the PIPELINE_NAMES
extension above is in place.

Extend src/cli/__main__.py with, following src/cli/output.py's existing
CLIUserError/exit-code/stdout-stderr conventions exactly (do not duplicate
that logic): `aidp fraud-intel generate --channel X --count N --database
DB --json`; `aidp fraud-intel train --channel X --database DB --json`;
`aidp fraud-intel score --channel X --database DB --json`; `aidp
fraud-intel evaluate --channel X --database DB --json` (may show
scenario_id/labels — this is the debug/evaluation surface, per section 6);
`aidp fraud-intel promote --channel X --bundle-version N --database DB
--json`; `aidp model show --channel X --database DB --json`; `aidp alerts
list/show/disposition --database DB` (never show scenario_id/synthetic
labels here — the analyst-facing surface). EVERY database-touching
command — every one listed above, with no exception — requires an
explicit --database argument or a typed config field; no command may
implicitly default to the production "aidp" database; require it to be
passed or fail with a clear CLIUserError if omitted.

Constraints: migration 004 is reviewed, not applied, in this phase. No
automatic retraining or promotion triggered by anything in this phase. No
Docker/infrastructure operations beyond the existing fully-mocked CLI test
pattern. No Co-authored-by or AI attribution, no push.

Tests must cover: a duplicate (source_system, source_alert_id) does not
create a second alert; a SECOND, DISTINCT source alert for the SAME
event_id DOES create a second, separate fraud_alerts row (this is
correct behavior, not a bug — assert both directions); a retry with the
SAME score_execution_id does not create duplicate evidence
(test this with a fake store simulating a retry); a genuine re-score with
a NEW score_execution_id creates an additional evidence row; disposition
notes are redacted/bounded; label eligibility always APPENDS, never
updates, and a query for "latest assessment" returns the most recent row
while older ones remain queryable; every new CLI command is tested with a
fully mocked RunLifecycle/alert-store/bundle, zero real database contact;
a command invoked without an explicit database target where one is
required raises CLIUserError rather than silently defaulting to a
production database name.

Run pytest tests/unit -q. Review both SQL files and the full diff. Commit
locally with:
feat(fraud-intel): add analyst review queue, idempotent evidence, append-only assessments, and CLI surfaces

Report files changed, migration filename, tests/counts, commit SHA, final
git status.
```

**Explicit exclusions:** migration `004` not applied anywhere; no
automatic retraining/promotion; no FastAPI/API implementation.

**Tests:** idempotency at both levels (event + scoring-attempt),
append-only assessment history, redaction, explicit-database-target
enforcement, fully mocked CLI — no infrastructure.

**Infra/DB checkpoint:** migration `004` reviewed only, not applied — see
the Phase 6 migration checkpoint below.

**Acceptance criteria:** alert creation is idempotent per source alert
(keyed on `(source_system, source_alert_id)`, correctly allowing multiple
alerts per event), and evidence writes are idempotent at the
scoring-execution level; label history is append-only and auditable;
`RunLifecycle` accepts the four new pipeline names only after the schema
permits it; every database-touching CLI command requires an explicit
target.

**Commit message:** `feat(fraud-intel): add analyst review queue, idempotent evidence, append-only assessments, and CLI surfaces`

**Migration `003` + `004` checkpoint (present here explicitly, both
migrations, as required):**
```
# Pre-check (aidp_test only)
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "SELECT current_database();"
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "\d channel_events"
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "SELECT count(*) FROM pipeline_runs;"

# Apply, in order, only after your separate explicit approval of each:
docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test -f - \
  < infrastructure/postgres/migrations/003_channel_events_and_labels.sql
docker exec -i aidp-postgres psql -v ON_ERROR_STOP=1 -U aidp -d aidp_test -f - \
  < infrastructure/postgres/migrations/004_fraud_alerts_evidence_and_lifecycle.sql

# Post-check
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "\d fraud_alerts"
docker exec -i aidp-postgres psql -U aidp -d aidp_test -c "SELECT count(*) FROM pipeline_runs;"
```
**Expected data changes:** new tables/columns only — zero existing rows
modified, `pipeline_runs` row count unchanged. **Rollback cost:** the same
class of cost as migration `002` — no built-in undo; a reverse script
would need to drop the new tables/columns and narrow `pipeline_name`'s
CHECK back, which would fail if any `fraud_score`/`label_eligibility`/
`model_promotion`/`fraud_evaluation` rows already exist by then. **Target:
`aidp_test` only, never `aidp`, in this phase.**

**Stop and review** before Phase 7A.

---

### Phase 7A — Remaining six channel adapters, unit-tested only

**Objective:** fill in the six remaining channel adapters/rules/ensemble
configs using the proven pattern — unit-tested with mocks, exactly like
the reference channel was through Phase 6. **This phase alone does not
constitute a runnable seven-channel POC** — that requires Phase 7B.

**Scope:** ACH, Wire, Mobile Check Deposit, ATM, Debit Card, P2P feature
adapters (filling Phase 2's stubs); their training/scoring via the
existing framework, unmodified; per-channel ensemble policy YAML;
cross-channel evaluation code (against mocked/fixture data only in this
phase).

**Expected files:**
```
src/fraud_intel/features/channels/{ach,wire,mobile_deposit,atm,debit_card,p2p}.py (filled in)
config/fraud_intel/ensemble_policy_{ach,wire,mobile_deposit,atm,debit_card,p2p}.yaml
src/fraud_intel/evaluation/cross_channel.py
src/fraud_intel/monitoring/drift.py
tests/unit/test_fraud_intel_features_{ach,wire,mobile_deposit,atm,debit_card,p2p}.py
tests/unit/test_fraud_intel_cross_channel_evaluation.py
```

**Copy-paste prompt:**
```text
Implement only Phase 7A of AiDP v1.3: the remaining six channel feature
adapters, their training/scoring using the existing framework/orchestrator
unmodified, and cross-channel evaluation + drift monitoring code — unit-
tested against fixtures only. This phase does not run real training or
touch any database; that is Phase 7B, separately approved.

Read Phases 2-6 in full first. This phase must not introduce any new
architectural pattern — only fill in the proven one six more times.
If you find yourself writing channel-specific code outside a feature
adapter or its rule/ensemble YAML, stop and flag it rather than
proceeding.

Fill in the six stub adapters per section 9. Add one
ensemble_policy_<channel>.yaml per channel. Route each channel's
training/scoring through the exact same train_channel_configured()/
score_source_alert() code from Phases 4-6.

src/fraud_intel/evaluation/cross_channel.py: every section-21 metric per
channel and in aggregate, computed PRIMARILY over the source-alert
population, from a set of scored+disposed synthetic alerts, including the
rules-only-baseline comparison and a clearly-separate, secondary
shadow-candidate summary.

src/fraud_intel/monitoring/drift.py: alert-volume and score-distribution
drift indicators from stored alert_evidence history.

Constraints: no new architectural layer, no schema changes, no real
training, no database or Docker operations, no Co-authored-by or AI
attribution, no push.

Tests must cover: each of the six channels' feature adapters against
fixtures; that training/scoring for every channel goes through the
identical shared code path (assert no channel-specific branch exists
outside config/adapter files); cross-channel evaluation against a
constructed fixture, including the rules-only-baseline comparison and the
source-alert-population framing; drift indicators against a constructed
time series.

Run pytest tests/unit -q. Commit locally with:
feat(fraud-intel): add remaining channel adapters and cross-channel evaluation (unit-tested)

Report files changed, tests/counts, commit SHA, final git status.
```

**Explicit exclusions:** no real training, no database contact, no
scoring against real data.

**Tests:** all 6 remaining adapters, shared-code-path assertion,
cross-channel evaluation, drift — fixtures/mocks only.

**Infra/DB checkpoint:** none.

**Acceptance criteria:** all 7 channels' code paths are unit-tested end to
end via mocks; **explicitly not yet claimed as a runnable seven-channel
POC** — that claim requires Phase 7B.

**Commit message:** `feat(fraud-intel): add remaining channel adapters and cross-channel evaluation (unit-tested)`

**Stop and review** before requesting Phase 7B's separate approval.

---

### Phase 7B — Runtime verification: real migrations, real generation, real training, real promotion, real scoring (separately approved)

**Objective:** actually prove the system runs, against real local
infrastructure and `aidp_test` — this is the phase that turns "unit-tested"
into "runnable POC," and it does not happen without your separate,
explicit approval of each step below.

**This phase's steps, each shown before execution with exact target,
read/write behavior, expected data changes, and rollback/recovery cost —
none run without your separate approval, and none ever target `aidp`:**

1. **Confirm migrations `003`/`004` already applied to `aidp_test`** (from
   Phase 6's checkpoint) — if not yet applied, apply now using the exact
   commands shown in Phase 6, with your separate approval.
2. **Deterministic synthetic generation** — `aidp fraud-intel generate
   --channel <each of 7> --database aidp_test --json`, writing to
   `channel_events`/`source_alerts`/`synthetic_event_labels` in
   `aidp_test` only, each row tagged with this invocation's
   `generation_run_id` and `dataset_version` (§7).
   Read/write: writes new rows only, to `aidp_test`. Rollback cost: an
   exact, dependency-order-safe delete scoped to this run's
   `generation_run_id` — never a wildcard `scenario_id LIKE` pattern:
   ```sql
   BEGIN;
   DELETE FROM alert_evidence WHERE alert_id IN (
     SELECT alert_id FROM fraud_alerts WHERE event_id IN (
       SELECT event_id FROM channel_events WHERE generation_run_id = %(run_id)s
     )
   );
   DELETE FROM label_assessments WHERE alert_id IN (
     SELECT alert_id FROM fraud_alerts WHERE event_id IN (
       SELECT event_id FROM channel_events WHERE generation_run_id = %(run_id)s
     )
   );
   DELETE FROM analyst_dispositions WHERE alert_id IN (
     SELECT alert_id FROM fraud_alerts WHERE event_id IN (
       SELECT event_id FROM channel_events WHERE generation_run_id = %(run_id)s
     )
   );
   DELETE FROM fraud_alerts WHERE event_id IN (
     SELECT event_id FROM channel_events WHERE generation_run_id = %(run_id)s
   );
   DELETE FROM source_alerts WHERE generation_run_id = %(run_id)s;
   DELETE FROM synthetic_event_labels WHERE event_id IN (
     SELECT event_id FROM channel_events WHERE generation_run_id = %(run_id)s
   );
   DELETE FROM channel_events WHERE generation_run_id = %(run_id)s;
   COMMIT;
   ```
   Children are deleted before parents (evidence/assessments/dispositions
   before `fraud_alerts`, `fraud_alerts` before `source_alerts`, everything
   before `channel_events`), the whole delete is one transaction so a
   failure at any step rolls back cleanly, and the target is always
   `aidp_test` — never `aidp` — passed explicitly, never defaulted.
3. **Real local training** — `aidp fraud-intel train --channel <each of
   7> --database aidp_test --json`, actually fitting GBM/LR/anomaly on the
   data generated in step 2 above (restricted to source-alerted events
   only, per guide section 11), registering 7 candidate bundles to the
   local MLflow instance. Read: `aidp_test`. Write: MLflow registry
   (local), `channel_model_bundles` rows in `aidp_test`. Rollback cost:
   MLflow model versions are not deleted by this guide; bundle rows can be
   deleted from `aidp_test`.
4. **Explicit promotion** — `aidp fraud-intel promote --channel <each of
   7> --bundle-version N --database aidp_test --json`, one at a time,
   after you review each candidate bundle's evaluation report. Write:
   demotes the channel's current `OPERATIONAL` bundle (if any) to
   `RETIRED` and promotes the candidate to `OPERATIONAL`, both in one
   transaction, in `aidp_test` only (§22).
5. **Real scoring** — `aidp fraud-intel score --channel <each of 7>
   --database aidp_test --json`, running the real orchestrator against the
   real generated events, writing real `fraud_alerts`/`alert_evidence` rows
   to `aidp_test`.
6. **Affected integration/smoke tests** — run only the fraud-intel
   integration/smoke tests added in this phase (clearly separated from
   `tests/unit`, following the existing `tests/integration`/`tests/smoke`
   convention), against `aidp_test` only.

**Expected files:** `tests/integration/test_fraud_intel_*.py`,
`tests/smoke/test_fraud_intel_end_to_end.py` (if warranted) — test code
only; steps 1–6 above are commands, not code changes.

**Explicit exclusions:** nothing in this phase ever targets `aidp`; no
migration is applied without a separate approval per migration; no
`reset_demo.sh` use; no production deployment of any kind.

**Acceptance criteria:** all 7 channels have an `OPERATIONAL` bundle in
`aidp_test`; at least one real, non-mocked alert with full evidence exists
per channel; the cross-channel evaluation report (Phase 7A's code) run
against this real data, not fixtures, produces the section-21 metrics.

**Commit message** (test code only, if any was added):
`test(fraud-intel): add integration/smoke coverage for real scoring against aidp_test`

**Stop and review.** Documentation/demo packaging (Phase 8) begins only
after this phase succeeds — the guide does not claim a runnable
seven-channel POC based on mocked unit tests alone.

---

### Phase 8 — Dashboard tab, CI, documentation, demo packaging, final verification

**Objective:** add the minimal analyst-facing dashboard surface, extend CI
(still no infrastructure), document v1.3 against what Phase 7B actually
proved, and run a final read-only-first verification.

**Scope:** a new "Fraud Intelligence" Streamlit tab (read-mostly: ranked
queue, filters, reason codes, layer-by-layer evidence, versions,
disposition history — disposition-writing stays CLI-driven); its
execution tests classified as **smoke tests** (`tests/smoke/`, requiring
infrastructure), following the exact correction already applied to
`test_dashboard.py` in v1.2; `.github/workflows/ci.yml` unchanged in spirit
(still `pytest tests/unit`, still no secrets/Docker); documentation
updates; `DEMO_SCRIPT_V1_3.md`; final verification report.

**Expected files:**
```
src/dashboard/fraud_intel_tab.py (or equivalent, wired into app.py)
tests/smoke/test_fraud_intel_dashboard.py
ARCHITECTURE.md, RUNBOOK.md, TROUBLESHOOTING.md (updated)
DEMO_SCRIPT_V1_3.md (new)
```

**Copy-paste prompt:**
```text
Implement only Phase 8 of AiDP v1.3: the dashboard tab, documentation, and
final verification. Do not add Docker services to CI. Do not change fraud
logic. Do not proceed if Phase 7B has not succeeded — this phase documents
and demos what Phase 7B actually proved, not an aspirational system.

Add a "Fraud Intelligence" tab to the Streamlit dashboard (read-mostly):
priority-ranked alert queue with channel/priority filters, reason codes,
layer-by-layer evidence (rule/GBM/anomaly/graph contributions and
versions), model/rule/ensemble-policy versions, and disposition history.
Never display scenario_id or synthetic-label fields here (section 6).
Disposition capture may remain CLI-only. Classify this tab's execution
tests as smoke tests (tests/smoke/test_fraud_intel_dashboard.py, following
the exact pattern and reasoning already used for tests/smoke/test_dashboard.py
in v1.2 — requires the local stack, not part of tests/unit or CI).

Update ARCHITECTURE.md's v1.3 section from "in progress" to a real
description of what Phase 7B actually verified running. Update RUNBOOK.md
with the full aidp fraud-intel / aidp alerts command reference, and
migrations 003+004's aidp_test-first procedure. Update TROUBLESHOOTING.md
with failure modes this guide's own phases surfaced (rule-provider
unavailable, chronological-split/purge-gap misconfiguration, feature-
schema mismatch on scoring, bundle-compatibility validation failure on
promotion).

Write DEMO_SCRIPT_V1_3.md using ONLY commands actually implemented in
Phases 6-7B: generate (Phase 6 CLI) -> train -> promote -> score (all
Phase 6/7B) -> show the ranked queue in the dashboard tab (this phase) ->
show one alert's full evidence trail via `aidp alerts show` -> capture a
disposition via CLI -> show the cross-channel evaluation report (Phase 7A
code, run against Phase 7B's real data).

Do not claim production readiness or regulatory approval anywhere. Do not
hardcode a specific test count.

Validation: pytest tests/unit -q, python -m compileall src, python -m
src.cli --help, git diff --check. Manually check all changed docs for
secrets, absolute paths, and stale/unsupported claims.

Do not push. Do not open a PR. Do not apply any further migration beyond
what Phase 7B already had separately approved. Do not modify aidp.

Commit locally with: ci: validate fraud-intelligence changes

Report validation results, files changed, commit SHA, final git status.
```

**Explicit exclusions:** no CI infrastructure services added; no further
schema changes; no write-enabled dashboard controls beyond what's
explicitly scoped.

**Tests:** `pytest tests/unit -q`, `compileall`, CLI smoke checks — no
infrastructure for the unit suite; the new dashboard test is itself
classified as a smoke test, not part of this count.

**Infra/DB checkpoint:** none new — this phase only documents/demos what
Phase 7B already verified.

**Acceptance criteria:** documentation reflects what actually ran in
Phase 7B, not aspiration; `DEMO_SCRIPT_V1_3.md` uses only implemented
commands; dashboard tab is visually demoable against real `aidp_test`
data; final verification report produced (Passed/Warning/Blocker/
Deferred, mirroring v1.2's Phase 6 format).

**Commit message:** `ci: validate fraud-intelligence changes`

**Stop and review.** This is the last phase — push/PR decisions remain
yours, separately, exactly like v1.2's.

---

## 26. Deferred Citizens Bank enterprise integrations

| Deferred integration | Local placeholder here | Real integration point (not built) |
|---|---|---|
| Citizens Bank production rule/alert-generation engines | `LocalYamlRuleProvider`, explicitly *simulating* this role for the POC (§4, §10) | A real `RuleProvider` implementation calling the bank's actual upstream alert-generation systems |
| OFAC/sanctions screening | Not implemented as a fraud-triage rule anywhere (§10, §23) | Citizens Bank's separate sanctions-screening system |
| BSA/AML transaction monitoring | Not implemented anywhere (§23) | Citizens Bank's separate BSA/AML program |
| Orbograph (check processing) | Structured placeholder fields only (§6, §9) | Real image analysis / check-processing API |
| Mainframe ingestion feeds | Synthetic generators (§7) | Real core-banking/mainframe event feeds |
| Production Kafka | Local Redpanda | Production Kafka cluster |
| AWS Bedrock | None used | Any managed LLM/AI service |
| Aurora | Local Postgres in Docker | AWS Aurora |
| Snowflake | Local Parquet/Polars/DuckDB | Snowflake |
| TensorFlow/PyTorch enterprise model endpoints | XGBoost/scikit-learn, local MLflow registry | Enterprise model-serving infrastructure |
| Real customer data | 100% synthetic (§7) | Real, governed customer data under a real data-access process |
| Production authentication/authorization | None — local, single-operator CLI | Real identity/access management |
| A production FastAPI/API surface | CLI + read-mostly dashboard tab only (§21, §25 Phase 6) | A real, authenticated API surface, if ever built |
| Automatic case closure | Never implemented — every alert stays reviewable (§2, §18) | A real case-management system's closure workflow, if ever built |
| Production deployment | Local Docker Compose only | Real infrastructure, HA, monitoring, on-call |
| Regulatory certification | Design considerations requiring legal/compliance/model-risk review only (§23) | Actual legal/compliance/regulatory review |

---

## 27. Final demo and acceptance criteria

**Demo walkthrough** (`DEMO_SCRIPT_V1_3.md`, Phase 8, using only
Phase 6/7B-implemented commands against real Phase 7B data):
1. Generate synthetic events for all 7 channels into `aidp_test`.
2. Train and promote one bundle per channel.
3. Score real events through the real orchestrator, creating real alerts.
4. Show the ranked queue (CLI and dashboard tab) — HIGH/MEDIUM/LOW, with
   LOW alerts still visible.
5. Show one alert's full reason-code list and layer-by-layer evidence
   trail, including its `score_execution_id`.
6. Capture an analyst disposition; show it stored separately from the
   synthetic label, and show that a repeat disposition/re-score never
   overwrites prior history.
7. Show the cross-channel evaluation report computed from this real data:
   precision/recall at capacity on the **source-alert population**,
   dollar-weighted recall, PR-AUC, calibration error, vs. the rules-only
   baseline, plus the GBM-vs-LR-shadow comparison.

**Acceptance criteria:**
- Every mandatory section (§1–§27) is reflected in the actual
  implementation by the end of Phase 8, and Phase 7B has actually run
  (this is not claimed on the basis of mocked unit tests alone).
- **Every source alert is retained** — none was ever silently discarded,
  auto-closed, or removed from the reviewable population, at any phase.
- **Every scoring attempt is idempotent** — a retried or duplicated
  scoring execution never creates duplicate evidence (tested in Phase 6,
  verified for real in Phase 7B).
- `pytest tests/unit -q` passes with zero infrastructure contact.
- `tests/integration`/`tests/smoke` (including the new fraud-intel ones)
  are clearly separated and require the local stack.
- Migrations `003` and `004` exist in source control, reviewed, and were
  applied to `aidp_test` only, each after separate explicit approval —
  `aidp` was never migrated as part of this guide's own instructions.
- The LR challenger's score never reached `operational_priority_score` in
  any run — structurally guaranteed, not just by convention.
- No bundle was ever promoted except via the explicit, manual,
  `RunLifecycle`-tracked promotion action, and always as a whole
  compatible bundle, never an individual model.
- No document produced under this guide claims regulatory approval,
  production readiness, or automatic model governance.
- No Git push or pull request happened without your separate, explicit
  approval, exactly like every phase in v1.2.

---

## Stop conditions

Stop and request review if any phase proposes: applying a migration to
`aidp` without a separate approval step (`aidp_test` always requires its
own separate approval too); blending the LR challenger into the
operational score; auto-closing, suppressing, or silently discarding a
source alert; automatically promoting a bundle or an individual model;
using raw, immature `analyst_disposition` as graph or training ground
truth; using `eval()`/`exec()`/an unbounded expression evaluator for
rules; introducing a graph database, production Kafka, AWS Bedrock,
Aurora, or Snowflake; implementing OFAC/sanctions screening or BSA/AML
monitoring as a fraud-triage rule; claiming regulatory approval or
production readiness anywhere; using `reset_demo.sh` as a migration
mechanism; combining multiple phases into one commit; claiming a runnable
seven-channel POC based on mocked unit tests alone (before Phase 7B);
adding a new dependency not explicitly called out and approved (this
guide names exactly one candidate: `networkx`, §15); implementing a
production FastAPI endpoint; or force-pushing.
