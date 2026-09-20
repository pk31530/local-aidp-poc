"""Debit Card channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DebitCardPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["debit_card"] = "debit_card"
    merchant_id: str
    mcc_code: str
    pos_entry_mode: str
    card_present_flag: bool
    cross_border_flag: bool
