"""P2P channel payload (guide section 6's channel-payload table)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class P2PPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["p2p"] = "p2p"
    recipient_handle: str
    network: str
    memo_present_flag: bool
    recipient_is_new_flag: bool
