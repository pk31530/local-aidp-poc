"""ATM channel feature adapter -- stub only. Real feature logic is
implemented in Phase 7A (guide section 25), following the pattern proven
on the Online/Mobile Banking reference channel in Phase 2."""
from __future__ import annotations

from src.fraud_intel.features.core import FeatureComputationContext


def compute_atm_features(ctx: FeatureComputationContext, **kwargs) -> dict:
    raise NotImplementedError("ATM channel features are implemented in Phase 7A, not Phase 2.")
