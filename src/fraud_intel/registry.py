"""The shared channel-adapter registry (Phase 7A). One typed mapping from
channel -> {feature computation, entity extraction, payload class}, so
src.fraud_intel.scoring.orchestrator.score_source_alert() and
src.fraud_intel.models.training.train_channel_configured() stay the
ONLY two orchestration functions in this codebase -- no per-channel
scoring/training pipeline, no long channel-specific if/elif chain
anywhere outside this module.

Every entry's `compute_features`/`extract_entities` is one of the plain
functions already defined in src.fraud_intel.features.channels.* (or, for
online_banking, src.fraud_intel.graph.entity_graph's own default
extractor) -- this module does not itself contain any channel-specific
logic, only the mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from pydantic import BaseModel

from src.fraud_intel.events.ach import ACHPayload
from src.fraud_intel.events.atm import ATMPayload
from src.fraud_intel.events.base import FraudEvent
from src.fraud_intel.events.debit_card import DebitCardPayload
from src.fraud_intel.events.mobile_deposit import MobileDepositPayload
from src.fraud_intel.events.online_banking import OnlineBankingPayload
from src.fraud_intel.events.p2p import P2PPayload
from src.fraud_intel.events.wire import WirePayload
from src.fraud_intel.features.channels import ach as _ach
from src.fraud_intel.features.channels import atm as _atm
from src.fraud_intel.features.channels import debit_card as _debit_card
from src.fraud_intel.features.channels import mobile_deposit as _mobile_deposit
from src.fraud_intel.features.channels import p2p as _p2p
from src.fraud_intel.features.channels import wire as _wire
from src.fraud_intel.features.channels.online_banking import (
    ONLINE_BANKING_FEATURE_COLUMNS,
    ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
    compute_online_banking_features,
)
from src.fraud_intel.features.core import FeatureComputationContext
from src.fraud_intel.graph.entity_graph import EntityKey, extract_entities_online_banking


class UnknownChannelError(ValueError):
    """`channel` is not a registered channel adapter -- raised by
    get_channel_adapter(), never a bare KeyError."""


@dataclass(frozen=True)
class ChannelAdapter:
    channel: str
    feature_columns: tuple[str, ...]
    feature_schema_version: str
    payload_class: Type[BaseModel]
    compute_features: Callable[[FeatureComputationContext], dict]
    extract_entities: Callable[[FraudEvent], list[EntityKey]]


CHANNEL_ADAPTERS: dict[str, ChannelAdapter] = {
    "online_banking": ChannelAdapter(
        channel="online_banking",
        feature_columns=tuple(ONLINE_BANKING_FEATURE_COLUMNS),
        feature_schema_version=ONLINE_BANKING_FEATURE_SCHEMA_VERSION,
        payload_class=OnlineBankingPayload,
        compute_features=compute_online_banking_features,
        extract_entities=extract_entities_online_banking,
    ),
    "ach": ChannelAdapter(
        channel="ach",
        feature_columns=tuple(_ach.ACH_FEATURE_COLUMNS),
        feature_schema_version=_ach.ACH_FEATURE_SCHEMA_VERSION,
        payload_class=ACHPayload,
        compute_features=_ach.compute_ach_features,
        extract_entities=_ach.extract_entities,
    ),
    "wire": ChannelAdapter(
        channel="wire",
        feature_columns=tuple(_wire.WIRE_FEATURE_COLUMNS),
        feature_schema_version=_wire.WIRE_FEATURE_SCHEMA_VERSION,
        payload_class=WirePayload,
        compute_features=_wire.compute_wire_features,
        extract_entities=_wire.extract_entities,
    ),
    "mobile_deposit": ChannelAdapter(
        channel="mobile_deposit",
        feature_columns=tuple(_mobile_deposit.MOBILE_DEPOSIT_FEATURE_COLUMNS),
        feature_schema_version=_mobile_deposit.MOBILE_DEPOSIT_FEATURE_SCHEMA_VERSION,
        payload_class=MobileDepositPayload,
        compute_features=_mobile_deposit.compute_mobile_deposit_features,
        extract_entities=_mobile_deposit.extract_entities,
    ),
    "atm": ChannelAdapter(
        channel="atm",
        feature_columns=tuple(_atm.ATM_FEATURE_COLUMNS),
        feature_schema_version=_atm.ATM_FEATURE_SCHEMA_VERSION,
        payload_class=ATMPayload,
        compute_features=_atm.compute_atm_features,
        extract_entities=_atm.extract_entities,
    ),
    "debit_card": ChannelAdapter(
        channel="debit_card",
        feature_columns=tuple(_debit_card.DEBIT_CARD_FEATURE_COLUMNS),
        feature_schema_version=_debit_card.DEBIT_CARD_FEATURE_SCHEMA_VERSION,
        payload_class=DebitCardPayload,
        compute_features=_debit_card.compute_debit_card_features,
        extract_entities=_debit_card.extract_entities,
    ),
    "p2p": ChannelAdapter(
        channel="p2p",
        feature_columns=tuple(_p2p.P2P_FEATURE_COLUMNS),
        feature_schema_version=_p2p.P2P_FEATURE_SCHEMA_VERSION,
        payload_class=P2PPayload,
        compute_features=_p2p.compute_p2p_features,
        extract_entities=_p2p.extract_entities,
    ),
}


def get_channel_adapter(channel: str) -> ChannelAdapter:
    try:
        return CHANNEL_ADAPTERS[channel]
    except KeyError:
        raise UnknownChannelError(
            f"no channel adapter registered for {channel!r}; expected one of {sorted(CHANNEL_ADAPTERS)}"
        ) from None
