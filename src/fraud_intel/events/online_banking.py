"""Online/Mobile Banking channel payload (guide section 6's channel-payload
table)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class OnlineBankingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["online_banking"] = "online_banking"
    session_id: str
    login_method: str
    mfa_used_flag: bool
    transaction_type: str
    target_account: str
