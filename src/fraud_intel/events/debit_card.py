"""Debit Card channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class DebitCardPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["debit_card"] = "debit_card"
    merchant_id: str
    mcc_code: str
    pos_entry_mode: str
    card_present_flag: bool
    cross_border_flag: bool
    # Phase 7A additive correction: a deterministic, synthetic token
    # identifying "the same card" across events for graph fan-in purposes
    # (guide's approved CARD entity type) -- NEVER a real PAN. Optional so
    # every payload constructed before this field existed remains valid.
    card_token: Optional[str] = None
