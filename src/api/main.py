"""Phase 6: FastAPI real-time scoring service.

Wires together, for the first time, the pieces built in earlier phases:
the real-time feature-store adapter (src/common/feature_store.py, fix C1),
the shared feature module (src/common/features.py, fix C2), the trained
model (Phase 5), and the decisioning engine (src/decisioning/engine.py).

Run with: uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from psycopg2.pool import ThreadedConnectionPool

from src.common.config import PROJECT_ROOT, get_settings
from src.common.features import RiskLookups
from src.common.logging import bind_transaction_id, clear_transaction_id, configure_logging, get_logger
from src.common.mlflow_setup import configure_mlflow, load_champion_model
from src.common.scoring import score_and_persist

from src.api.schemas import (
    DecisionRecord,
    HealthStatus,
    MetricsSummary,
    ModelInfo,
    ScoreRequest,
    ScoreResponse,
    TransactionRecord,
)

configure_logging("api")
log = get_logger(__name__)
settings = get_settings()

RISK_LOOKUPS_PATH = PROJECT_ROOT / "data" / "models" / "risk_lookups.json"

state: dict = {"model": None, "model_version": None, "risk_lookups": None, "pool": None}


def _load_model() -> None:
    try:
        configure_mlflow()
        model, version = load_champion_model(settings.mlflow_model_name)
        state["model"] = model
        state["model_version"] = version
        log.info("model_loaded", model_version=version)
    except Exception:
        log.error("model_load_failed", exc_info=True)
        state["model"] = None
        state["model_version"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pool"] = ThreadedConnectionPool(1, 10, dsn=settings.postgres_dsn)

    _load_model()

    if RISK_LOOKUPS_PATH.exists():
        state["risk_lookups"] = RiskLookups.load(RISK_LOOKUPS_PATH)
    else:
        log.warning("risk_lookups_missing_using_neutral_fallback")  # fix C3
        state["risk_lookups"] = RiskLookups.empty()

    yield

    if state["pool"]:
        state["pool"].closeall()


app = FastAPI(title="AiDP Fraud Detection API", lifespan=lifespan)


def _checkout_validated_connection():
    """Borrows a connection from the pool and pings it before handing it
    out. A pooled connection can go stale if Postgres restarts underneath
    it (the pool doesn't know); discard a stale one and get a fresh one
    instead of returning something broken to every caller from then on."""
    conn = state["pool"].getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn
    except Exception:
        state["pool"].putconn(conn, close=True)
        conn = state["pool"].getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return conn


def get_conn():
    try:
        conn = _checkout_validated_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        yield conn
    finally:
        state["pool"].putconn(conn)


def require_model() -> None:
    # fix H2: 503, not 500, when the model isn't loaded.
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.get("/health", response_model=HealthStatus)
def health(conn=Depends(get_conn)):
    model_loaded = state["model"] is not None
    postgres_ok = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        postgres_ok = False

    body = HealthStatus(
        status="ok" if (model_loaded and postgres_ok) else "degraded",
        model_loaded=model_loaded,
        postgres_connected=postgres_ok,
        model_version=str(state["model_version"]) if state["model_version"] else None,
    )
    status_code = 200 if (model_loaded and postgres_ok) else 503
    return JSONResponse(content=body.model_dump(), status_code=status_code)


@app.get("/model", response_model=ModelInfo)
def model_info(_: None = Depends(require_model), conn=Depends(get_conn)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT model_version, precision_score, recall_score, f1_score, roc_auc_score
            FROM model_versions
            WHERE model_name = %s AND is_active = true
            ORDER BY registered_at DESC LIMIT 1
            """,
            (settings.mlflow_model_name,),
        )
        row = cur.fetchone()

    return ModelInfo(
        model_name=settings.mlflow_model_name,
        model_version=str(state["model_version"]),
        alias="champion",
        precision=float(row["precision_score"]) if row else None,
        recall=float(row["recall_score"]) if row else None,
        f1=float(row["f1_score"]) if row else None,
        roc_auc=float(row["roc_auc_score"]) if row else None,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest, _: None = Depends(require_model), conn=Depends(get_conn)):
    transaction_id = f"TXA{uuid.uuid4().hex[:12].upper()}"
    bind_transaction_id(transaction_id)
    try:
        now = datetime.now(timezone.utc)

        result = score_and_persist(
            conn,
            state["model"],
            str(state["model_version"]),
            state["risk_lookups"],
            transaction_id=transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            merchant=request.merchant,
            country=request.country,
            device_id=request.device_id,
            payment_method=request.payment_method,
            transaction_timestamp=now,
            source="api",
        )

        log.info(
            "transaction_scored",
            customer_id=request.customer_id,
            fraud_probability=round(result.fraud_probability, 4),
            risk_level=result.risk_level,
            decision=result.decision,
        )

        return ScoreResponse(
            transaction_id=result.transaction_id,
            fraud_probability=round(result.fraud_probability, 6),
            risk_level=result.risk_level,
            decision=result.decision,
            model_version=str(state["model_version"]),
            reason_codes=result.reason_codes,
        )
    finally:
        clear_transaction_id()


@app.get("/transactions/{transaction_id}", response_model=TransactionRecord)
def get_transaction(transaction_id: str, conn=Depends(get_conn)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM transactions WHERE transaction_id = %s", (transaction_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    return TransactionRecord(**row)


@app.get("/decisions/{transaction_id}", response_model=DecisionRecord)
def get_decision(transaction_id: str, conn=Depends(get_conn)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM fraud_decisions WHERE transaction_id = %s", (transaction_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="decision not found")
    return DecisionRecord(**row)


@app.get("/metrics", response_model=MetricsSummary)
def metrics(conn=Depends(get_conn)):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT decision, count(*) AS n, avg(fraud_probability) AS avg_prob
            FROM fraud_decisions
            GROUP BY decision
            """
        )
        rows = cur.fetchall()

    decision_counts = {r["decision"]: r["n"] for r in rows}
    total = sum(decision_counts.values())
    high_risk = decision_counts.get("REVIEW", 0) + decision_counts.get("BLOCK", 0)
    weighted_prob_sum = sum(r["n"] * float(r["avg_prob"]) for r in rows)

    return MetricsSummary(
        total_scored=total,
        decision_counts=decision_counts,
        high_risk_rate=round(high_risk / total, 4) if total else 0.0,
        average_fraud_probability=round(weighted_prob_sum / total, 4) if total else 0.0,
    )
