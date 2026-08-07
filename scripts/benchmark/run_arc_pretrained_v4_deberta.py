from __future__ import annotations

import json
import sys
from pathlib import Path

import run_arc_pretrained_v2_deberta as v2

PROTOCOL_ID = "lam-jepa-arc-challenge-v4"
TOKENIZER_MAX_LENGTH = 96


def value(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"protocol v4 requires explicit {flag}") from exc


def main() -> None:
    max_length = int(value(sys.argv, "--max-length"))
    if max_length != TOKENIZER_MAX_LENGTH:
        raise SystemExit(f"protocol v4 requires --max-length {TOKENIZER_MAX_LENGTH}, got {max_length}")
    output = Path(value(sys.argv, "--out"))

    v2.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise SystemExit("pretrained baseline output missing protocol metadata")
    protocol.update(
        {
            "frozen_protocol_id": PROTOCOL_ID,
            "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
            "tokenizer_use_fast": False,
            "trust_remote_code": False,
            "checkpoint_format": "pytorch_bin",
            "minimum_torch_version": "2.6.0",
            "weights_only": True,
        }
    )
    payload["protocol"] = protocol
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
