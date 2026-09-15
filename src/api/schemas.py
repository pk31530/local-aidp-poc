from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.common.schemas import MAX_REASONABLE_AMOUNT


class _Base(BaseModel):
    # Several fields below are legitimately named model_* (model_version,
    # model_name); disable pydantic's "model_" protected-namespace warning
    # rather than rename them away from their natural, schema-matching names.
    model_config = ConfigDict(protected_namespaces=())


class ScoreRequest(_Base):
    customer_id: str
    amount: float = Field(gt=0, le=MAX_REASONABLE_AMOUNT)  # fix M4
    merchant: str
    country: str
    device_id: str
    payment_method: str


class ScoreResponse(_Base):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    model_version: str
    reason_codes: list[str]


class TransactionRecord(_Base):
    transaction_id: str
    customer_id: str
    amount: float
    merchant: str
    country: str
    device_id: str
    payment_method: str
    transaction_timestamp: datetime
    source: str
    created_at: datetime


class DecisionRecord(_Base):
    transaction_id: str
    customer_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    reason_codes: list[str]
    model_version: str
    created_at: datetime


class ModelInfo(_Base):
    model_name: str
    model_version: str
    alias: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None


class HealthStatus(_Base):
    status: str
    model_loaded: bool
    postgres_connected: bool
    model_version: Optional[str] = None


class MetricsSummary(_Base):
    total_scored: int
    decision_counts: dict
    high_risk_rate: float
    average_fraud_probability: float
