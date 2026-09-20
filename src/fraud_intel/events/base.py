"""Strengthened, channel-agnostic event contract (guide section 6).

Mirrors, and tightens, src/common/schemas.py's Transaction: a UUID
event_id, an integer minor-unit amount (never a float -- StrictInt, so
Pydantic's normal float-to-int coercion for whole numbers does not apply),
UTC-aware timestamps, an enforced schema_version, and a discriminated
channel_payload union instead of an unvalidated dict.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.wire import WirePayload

SUPPORTED_SCHEMA_VERSIONS = {1}

Channel = Literal["ach", "wire", "mobile_deposit", "online_banking", "atm", "debit_card", "p2p"]
Direction = Literal["debit", "credit"]

ChannelPayload = Annotated[
    Union[
        ACHPayload,
        WirePayload,
        MobileDepositPayload,
        OnlineBankingPayload,
        ATMPayload,
        DebitCardPayload,
        P2PPayload,
    ],
    Field(discriminator="channel"),
]


def _reject_naive_or_non_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("must be UTC")
    return value


class FraudEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    channel: Channel
    customer_id: str
    account_id: str
    event_timestamp: datetime
    amount_minor_units: StrictInt = Field(gt=0)
    direction: Direction
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    scenario_id: Optional[str] = None
    schema_version: int = 1
    channel_payload: ChannelPayload

    @field_validator("event_timestamp")
    @classmethod
    def _validate_event_timestamp(cls, value: datetime) -> datetime:
        return _reject_naive_or_non_utc(value)

    @field_validator("schema_version")
    @classmethod
    def _reject_unsupported_schema_version(cls, value: int) -> int:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version {value!r}; supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return value

    @model_validator(mode="after")
    def _channel_matches_payload(self) -> "FraudEvent":
        """Pydantic's discriminated union validates channel_payload against
        its OWN internal discriminator only -- it does not, by itself,
        verify that the outer `channel` field agrees with
        `channel_payload.channel`. These are two independently-settable
        fields that could otherwise disagree (guide section 6)."""
        if self.channel != self.channel_payload.channel:
            raise ValueError(
                f"FraudEvent.channel={self.channel!r} does not match "
                f"channel_payload.channel={self.channel_payload.channel!r}"
            )
        return self
