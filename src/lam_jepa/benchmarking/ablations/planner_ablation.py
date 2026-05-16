from __future__ import annotations
from dataclasses import replace

from ...model import LAMJEPAConfig


def no_planner(cfg: LAMJEPAConfig) -> LAMJEPAConfig:
    out = replace(cfg)
    out.use_planner = False
    return out
