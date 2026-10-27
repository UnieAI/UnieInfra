"""Preconditions for tests that launch a real server or engine.

Tests that load full checkpoints must not start downloading hundreds of GB or
spin up schedulers on fewer GPUs than their mesh needs; on a box that cannot
satisfy them they should skip with a reason instead of hanging.

``TOKENSPEED_TEST_ALLOW_DOWNLOAD=1`` lifts the offline-cache requirement for
machines that are expected to fetch checkpoints on demand (e.g. CI).
"""

from __future__ import annotations

import json
import os
from typing import Optional


def visible_gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _downloads_allowed() -> bool:
    return os.environ.get("TOKENSPEED_TEST_ALLOW_DOWNLOAD", "") not in ("", "0")


def _cached_file(repo: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo, filename)
    return path if isinstance(path, str) and os.path.isfile(path) else None


def _weights_complete(lookup) -> bool:
    """True when every weight shard named by the checkpoint is present."""
    index = lookup("model.safetensors.index.json")
    if index is not None:
        with open(index) as f:
            shards = set(json.load(f).get("weight_map", {}).values())
        return bool(shards) and all(lookup(s) is not None for s in shards)
    return lookup("model.safetensors") is not None


def model_is_available(model: str) -> bool:
    """A local checkpoint directory, or a fully cached Hub snapshot."""
    if os.path.isdir(model):

        def lookup(name):
            path = os.path.join(model, name)
            return path if os.path.isfile(path) else None

    else:

        def lookup(name):
            return _cached_file(model, name)

    return lookup("config.json") is not None and _weights_complete(lookup)


def launch_skip_reason(*models: str, min_gpus: int = 1) -> Optional[str]:
    """Why a launch test cannot run here, or None when it can."""
    have = visible_gpu_count()
    if have < min_gpus:
        return f"needs {min_gpus} visible GPU(s), found {have}"
    if not _downloads_allowed():
        missing = [m for m in models if m and not model_is_available(m)]
        if missing:
            return (
                f"checkpoint not available offline: {', '.join(missing)} "
                "(set TOKENSPEED_TEST_ALLOW_DOWNLOAD=1 to fetch)"
            )
    return None


def role_gpu_skip_reason(role_gpu_vars: dict) -> Optional[str]:
    """Serve scripts pin roles to physical GPU ids via CUDA_VISIBLE_DEVICES.

    ``role_gpu_vars`` maps each env var to its default id list. When the
    caller already restricted CUDA_VISIBLE_DEVICES, the defaults would escape
    that restriction, so every role must be pinned explicitly.
    """
    ids = set()
    for var, default in role_gpu_vars.items():
        ids.update(x for x in os.environ.get(var, default).split(",") if x)
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        unset = [v for v in role_gpu_vars if v not in os.environ]
        if unset:
            return (
                "CUDA_VISIBLE_DEVICES is restricted but the serve script would "
                f"pin default physical GPUs; set {', '.join(unset)} explicitly"
            )
        return None
    have = visible_gpu_count()
    if have < len(ids):
        return f"needs {len(ids)} GPUs for its roles, found {have}"
    return None
