
from __future__ import annotations

import torch
from pkgutil import extend_path
from pathlib import Path

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

__path__ = extend_path(__path__, __name__)
_src_pkg = Path(__file__).resolve().parents[1] / "src" / "lam_jepa"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))
