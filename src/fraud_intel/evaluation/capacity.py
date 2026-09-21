"""Analyst review capacity (Phase 7A decision 6). Exactly one of a fixed
alert COUNT or a FRACTION of the population -- enforced at the type
level via a discriminated union, not by a runtime "only one may be set"
check on a single model. There is no implicit default anywhere in this
module: every caller (Phase 7A's own unit tests, and Phase 7B's real
dispatch) must construct and pass an explicit AnalystCapacityConfig.
"""
from __future__ import annotations

import math
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class CountCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["count"] = "count"
    value: int = Field(gt=0)


class FractionCapacity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["fraction"] = "fraction"
    value: float = Field(gt=0, le=1)


# The typed config itself: a caller constructs CountCapacity(value=...) or
# FractionCapacity(value=...) directly -- the discriminated union is what
# makes "exactly one of count or fraction" a type-level fact (pydantic
# rejects any object that isn't unambiguously one or the other), not just
# a documented convention.
AnalystCapacityConfig = Annotated[Union[CountCapacity, FractionCapacity], Field(discriminator="mode")]


def resolve_capacity_count(capacity: Union[CountCapacity, FractionCapacity], total_alerts: int) -> int:
    """The number of alerts an analyst reviews under `capacity`, given a
    channel with `total_alerts` source-alerted rows. Always capped at
    `total_alerts` -- a count or fraction larger than the population never
    produces a request for more reviews than exist."""
    if total_alerts <= 0:
        return 0
    if capacity.mode == "count":
        return min(capacity.value, total_alerts)
    return min(total_alerts, max(1, math.ceil(capacity.value * total_alerts)))
