"""Wire channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class WirePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["wire"] = "wire"
    wire_type: Literal["domestic", "international"]
    beneficiary_bank_id: str
    beneficiary_account: str
    purpose_code: str
    originator_to_beneficiary_info: Optional[str] = None
