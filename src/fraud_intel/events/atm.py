"""ATM channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ATMPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["atm"] = "atm"
    atm_id: str
    atm_geo_bucket: str
    transaction_type: str
    card_present_flag: bool
