"""ACH channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ACHPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["ach"] = "ach"
    sec_code: str
    originating_routing_number: str
    receiving_routing_number: str
    batch_id: str
    effective_entry_date: date
    company_id: str
