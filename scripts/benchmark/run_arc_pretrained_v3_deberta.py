from __future__ import annotations

import json
import sys
from pathlib import Path

import run_arc_pretrained_v2_deberta as v2

PROTOCOL_ID = "lam-jepa-arc-challenge-v3"
TOKENIZER_MAX_LENGTH = 96


def output_path(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--out") + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("protocol-v3 DeBERTa runner requires --out") from exc


def require_max_length(argv: list[str]) -> None:
    try:
        value = int(argv[argv.index("--max-length") + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"protocol v3 requires explicit --max-length {TOKENIZER_MAX_LENGTH}") from exc
    if value != TOKENIZER_MAX_LENGTH:
        raise SystemExit(f"protocol v3 requires --max-length {TOKENIZER_MAX_LENGTH}, got {value}")


def bind_runtime_metadata(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise SystemExit("pretrained baseline output is missing protocol metadata")
    protocol["frozen_protocol_id"] = PROTOCOL_ID
    protocol["tokenizer_max_length"] = TOKENIZER_MAX_LENGTH
    protocol["tokenizer_use_fast"] = False
    protocol["trust_remote_code"] = False
    protocol["checkpoint_format"] = "pytorch_bin"
    protocol["minimum_torch_version"] = "2.6.0"
    protocol["weights_only"] = True
    payload["protocol"] = protocol
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    require_max_length(sys.argv)
    path = output_path(sys.argv)
    v2.main()
    bind_runtime_metadata(path)


if __name__ == "__main__":
    main()
