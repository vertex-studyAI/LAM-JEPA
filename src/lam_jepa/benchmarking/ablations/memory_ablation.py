from __future__ import annotations
from dataclasses import replace

from ...model import LAMJEPAConfig
from ..runner import build_variant_config


def no_memory(cfg: LAMJEPAConfig) -> LAMJEPAConfig:
    out = replace(cfg)
    out.use_memory = False
    return out
