from __future__ import annotations

from pathlib import Path
import yaml
import json

ROOT = Path(__file__).resolve().parents[2]
meta = yaml.safe_load((ROOT / "datasets" / "metadata" / "curriculum_graph.yaml").read_text())
(ROOT / "datasets" / "metadata").mkdir(parents=True, exist_ok=True)
with open(ROOT / "datasets" / "metadata" / "curriculum_graph.json", "w") as f:
    json.dump(meta, f, indent=2)
