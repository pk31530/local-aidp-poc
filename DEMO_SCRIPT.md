# Demo Script

A repeatable walkthrough matching the guide's worked scenario (section 55).
All commands below were actually run against this build — see BUILD_LOG.md
Phase 6/10 for the verified output.

## 0. Setup (once)

```bash
./scripts/start.sh
./scripts/healthcheck.sh     # confirm all 6 [OK]
./scripts/run_api.sh &        # leave running
./scripts/run_dashboard.sh &  # leave running
```

Open the dashboard: http://127.0.0.1:8501
Open Swagger: http://127.0.0.1:8000/docs

## 1. Normal transactions (should all APPROVE)

```bash
curl -s -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"C101","amount":4500,"merchant":"Grocery","country":"India","device_id":"DEV100001","payment_method":"CARD"}'
```

Expected: `"decision":"APPROVE"`, low `fraud_probability`, empty `reason_codes`.

Customer `C101` is guaranteed to exist (fix C3) with a baseline typical
spend of ₹8,000 — matching the guide's example customer exactly.

## 2. The suspicious transaction

```bash
curl -s -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"C101","amount":82000,"merchant":"Electronics","country":"Singapore","device_id":"DEV998","payment_method":"CARD"}'
```

Expected (matches guide section 17's worked example):

```json
{
  "fraud_probability": 0.85,
  "risk_level": "HIGH",
  "decision": "REVIEW",
  "reason_codes": ["NEW_DEVICE", "NEW_COUNTRY", "HIGH_AMOUNT_VS_AVERAGE"]
}
```

Why: ₹82,000 vs. C101's ₹8,000 average (10x), a device never seen before,
a country never seen before, all computed live from `customer_profiles`
(fix C1) by the exact same feature logic the batch pipeline uses (fix C2).

## 3. See it on the dashboard

Refresh the **Fraud Analysis** tab — the suspicious transaction appears in
"Highest-Risk Transactions", at or near the top. The **Live Transaction
Feed** shows both transactions from steps 1-2.

## 4. Velocity — rapid repeated transactions

```bash
for i in 1 2 3 4; do
  curl -s -X POST http://127.0.0.1:8000/score \
    -H "Content-Type: application/json" \
    -d '{"customer_id":"C1001","amount":300,"merchant":"Grocery","country":"UAE","device_id":"DEV561903","payment_method":"CARD"}' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['decision'], d['reason_codes'])"
done
```

The 4th call should show `HIGH_VELOCITY` in `reason_codes`
(`transactions_last_10m >= 3`) — proof the online feature store (fix C1)
is genuinely live, not a snapshot.

## 5. Real-time streaming demo

In one terminal:
```bash
./scripts/run_consumer.sh
```

In another:
```bash
./scripts/run_stream.sh --rate 10 --duration 30 --fraud-ratio 0.1
```

Watch the consumer's log stream `transaction_scored` events in real time,
and the **Executive Overview** / **Live Transaction Feed** dashboard tabs
update as you refresh.

## 6. Batch pipeline + model training (offline story)

```bash
python -m src.processing.pipeline    # RAW -> CLEAN -> CURATED -> FEATURES
./scripts/train_model.sh              # trains, registers, prints precision/recall/F1/ROC-AUC
```

Open MLflow (http://127.0.0.1:5001) → Models → `fraud-detection-model` to
see the version history and the `champion` alias.

## 7. Platform health

```bash
./scripts/healthcheck.sh
```

or the dashboard's **Platform Health** tab — both should show all green.
