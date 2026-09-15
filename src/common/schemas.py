"""Shared Pydantic schemas.

One definition of a "transaction" used by the generator, the batch pipeline,
the streaming consumer, and the API — so type + bounds validation (fix M4) is
identical everywhere instead of re-implemented per layer.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# Sane upper bound for this POC's synthetic INR amounts (fix M4: bounds
# validation on top of type validation). Not a real payment-network limit.
MAX_REASONABLE_AMOUNT = 10_000_000.0


class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    transaction_timestamp: datetime
    amount: float = Field(gt=0, le=MAX_REASONABLE_AMOUNT)
    merchant: str
    country: str
    device_id: str
    payment_method: str
    schema_version: int = 1
    # Training/demo ground-truth label only. Never used as a model input
    # feature — see src/common/features.py.
    is_fraud: Optional[bool] = None
