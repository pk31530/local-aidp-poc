"""Canonical MLflow registered-model naming for fraud-intelligence channel
models (Phase 7B Stage 5 corrective pass).

Before this module existed, the same name was independently reconstructed
in three places: src.fraud_intel.models.training's private
_registered_model_name_prefix() (a prefix only, suffix appended inline at
each mlflow.*.log_model() call), src.fraud_intel.scoring.dispatch's public
mlflow_model_name_prefix() (kept "identical" to training's only by a
cross-file unit test), and src.fraud_intel.models.promotion's own ad hoc
f"{bundle.channel}-{label}" construction -- which used the wrong channel
separator, omitted the "fraud-detection-model-fraud-intel-" prefix
entirely, and used "lr" instead of "lr-shadow". That last one made every
real promotion's component verification fail, discovered only on the
first real promotion attempt against real MLflow. registered_model_name()
below is now the ONLY place this string is ever built -- training,
scoring's artifact loader, and promotion's verifier all call it.
"""
from __future__ import annotations

from typing import Literal

from src.common.config import get_settings

ModelComponent = Literal["gbm", "lr-shadow", "anomaly"]
MODEL_COMPONENTS: tuple[ModelComponent, ...] = ("gbm", "lr-shadow", "anomaly")


class UnknownModelComponentError(ValueError):
    """`component` is not one of MODEL_COMPONENTS -- never silently
    guessed or defaulted."""


def registered_model_name(channel: str, component: str) -> str:
    """The exact MLflow registered-model name a channel's GBM/LR-shadow/
    anomaly model is logged and looked up under, e.g.
    "fraud-detection-model-fraud-intel-online-banking-gbm". This function
    is the single source of truth for that string -- no caller may
    reconstruct it independently."""
    if component not in MODEL_COMPONENTS:
        raise UnknownModelComponentError(f"unknown model component {component!r}; expected one of {MODEL_COMPONENTS}")
    return f"{get_settings().mlflow_model_name}-fraud-intel-{channel.replace('_', '-')}-{component}"
