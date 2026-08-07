from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROTOCOL_ID = "lam-jepa-arc-challenge-v3"
MODEL_ID = "microsoft/deberta-v3-xsmall"
MODEL_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
MODEL_LICENSE = "MIT"
TRANSFORMERS_VERSION = "4.57.6"
SENTENCEPIECE_VERSION = "0.2.2"
TOKENIZER_MAX_LENGTH = 128


def load_implementation():
    implementation_path = Path(__file__).with_name("run_arc_pretrained_baseline.py")
    spec = importlib.util.spec_from_file_location("lam_jepa_arc_pretrained_impl", implementation_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load pretrained baseline implementation: {implementation_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_ID = MODEL_ID
    module.MODEL_REVISION = MODEL_REVISION
    module.MODEL_LICENSE = MODEL_LICENSE
    module.PINNED_TRANSFORMERS_VERSION = TRANSFORMERS_VERSION
    return module


def requested_output_path(argv: list[str]) -> Path:
    try:
        index = argv.index("--out")
        return Path(argv[index + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("protocol-v3 runner requires --out") from exc


def enforce_protocol_cli(argv: list[str]) -> None:
    if "--max-length" in argv:
        index = argv.index("--max-length")
        try:
            value = int(argv[index + 1])
        except (IndexError, ValueError) as exc:
            raise SystemExit("invalid --max-length") from exc
        if value != TOKENIZER_MAX_LENGTH:
            raise SystemExit(
                f"protocol v3 requires --max-length {TOKENIZER_MAX_LENGTH}, got {value}"
            )


def bind_protocol_metadata(output_path: Path) -> None:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise SystemExit("pretrained baseline output is missing protocol metadata")
    protocol["frozen_protocol_id"] = PROTOCOL_ID
    protocol["pretrained_model_id"] = MODEL_ID
    protocol["pretrained_model_revision"] = MODEL_REVISION
    protocol["pretrained_model_license"] = MODEL_LICENSE
    protocol["transformers_version_pin"] = TRANSFORMERS_VERSION
    protocol["sentencepiece_version_pin"] = SENTENCEPIECE_VERSION
    protocol["max_length"] = TOKENIZER_MAX_LENGTH
    payload["protocol"] = protocol
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    enforce_protocol_cli(sys.argv)
    output_path = requested_output_path(sys.argv)
    implementation = load_implementation()
    implementation.main()
    bind_protocol_metadata(output_path)


if __name__ == "__main__":
    main()
