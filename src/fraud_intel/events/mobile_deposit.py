"""Mobile Check Deposit channel payload (guide section 6's channel-payload
table). Fields are structured placeholders only, per the guide's explicit
scope exclusion of real image analysis (section 3, section 26) -- no
computer-vision or Orbograph integration anywhere in this repository.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class MobileDepositPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: Literal["mobile_deposit"] = "mobile_deposit"
    duplicate_image_hash_flag: bool
    car_lar_mismatch_flag: bool
    signature_verification_flag: bool
    endorsement_present_flag: bool
    micr_consistency_flag: bool
    image_quality_score: float = Field(ge=0.0, le=1.0)
    # Phase 7A additive correction: a deterministic, synthetic token
    # identifying "the same payee" across events for graph fan-in purposes
    # (guide's approved CHECK_PAYEE entity type) -- NEVER a real payee name
    # or account number. Optional so every payload constructed before this
    # field existed remains valid.
    check_payee_token: Optional[str] = None
