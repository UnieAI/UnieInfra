# SPDX-License-Identifier: MIT
# Copyright (c) 2026 UnieInfra contributors
"""UnieInfra's out-of-tree scheduling policy for vLLM.

The policy itself lives in :mod:`unieinfra_scheduler.policy` and imports
nothing but the stdlib, so it is unit-testable without vLLM or a GPU.
:mod:`unieinfra_scheduler.vllm_scheduler` is the only module that touches vLLM;
it is imported lazily by ``--scheduler-cls`` and never at package import time,
so installing this distribution alongside a stock vLLM changes nothing until
the flag is passed.
"""

from unieinfra_scheduler.policy import (
    AdmissionCap,
    GateConfig,
    GateDecision,
    PrefillGate,
    ResourceCost,
    derive_admission_cap,
    gate_config_for,
)

__all__ = [
    "AdmissionCap",
    "GateConfig",
    "GateDecision",
    "PrefillGate",
    "ResourceCost",
    "derive_admission_cap",
    "gate_config_for",
]

__version__ = "0.1.0"
