from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional
import json
import time


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict):
        record = dict(record)
        record.setdefault("timestamp", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
