from __future__ import annotations

from packaging.version import Version
import torch
import transformers
from transformers import AutoModelForMultipleChoice, AutoTokenizer

import run_arc_pretrained_baseline as base


MODEL_ID = "microsoft/deberta-v3-xsmall"
MODEL_REVISION = "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
MODEL_LICENSE = "MIT"
MIN_SAFE_TORCH = Version("2.6.0")


def load_deberta(device: str):
    if transformers.__version__ != base.PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "transformers version mismatch: "
            f"expected={base.PINNED_TRANSFORMERS_VERSION} actual={transformers.__version__}"
        )
    torch_version = Version(torch.__version__.split("+", 1)[0])
    if torch_version < MIN_SAFE_TORCH:
        raise RuntimeError(
            "protocol-v2 DeBERTa uses an immutable PyTorch .bin checkpoint; "
            f"torch>={MIN_SAFE_TORCH} is required for the hardened weights-only loader, got {torch.__version__}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        use_fast=False,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        use_safetensors=False,
        weights_only=True,
    ).to(device)
    config_revision = getattr(model.config, "_commit_hash", None)
    if config_revision is not None and str(config_revision) != MODEL_REVISION:
        raise RuntimeError(f"loaded config revision mismatch: {config_revision}")
    return tokenizer, model


def main() -> None:
    # Reuse the already verified ARC pairing/training/robustness implementation while binding
    # its immutable model globals to the exact strong baseline frozen in protocol v2.
    base.MODEL_ID = MODEL_ID
    base.MODEL_REVISION = MODEL_REVISION
    base.MODEL_LICENSE = MODEL_LICENSE
    base.load_pretrained = load_deberta
    base.main()


if __name__ == "__main__":
    main()
