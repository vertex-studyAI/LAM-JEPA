from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read(path: Path) -> dict:
    require(path.is_file(), f"missing protocol: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"protocol must be an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that ARC protocol v2 changed capacity accounting without changing the frozen pretrained comparator contract."
    )
    parser.add_argument("--v1", type=Path, default=Path("protocols/arc_challenge_v1.json"))
    parser.add_argument("--v2", type=Path, default=Path("protocols/arc_challenge_v2.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    v1 = read(args.v1)
    v2 = read(args.v2)
    require(v1.get("protocol_id") == "lam-jepa-arc-challenge-v1", "unexpected v1 id")
    require(v2.get("protocol_id") == "lam-jepa-arc-challenge-v2", "unexpected v2 id")
    require(v2.get("supersedes") == v1.get("protocol_id"), "v2 does not explicitly supersede v1")
    require(v1.get("status") == v2.get("status") == "FROZEN_BEFORE_CONFIRMATORY_TEST", "protocol freeze status changed")

    unchanged_paths = {
        "scientific_question": (v1.get("scientific_question"), v2.get("scientific_question")),
        "claim_scope": (v1.get("claim_scope"), v2.get("claim_scope")),
        "dataset": (v1.get("dataset"), v2.get("dataset")),
        "training_budget": (v1.get("training_budget"), v2.get("training_budget")),
        "strong_pretrained_baseline": (
            v1.get("models", {}).get("strong_pretrained_baseline"),
            v2.get("models", {}).get("strong_pretrained_baseline"),
        ),
        "metrics": (v1.get("metrics"), v2.get("metrics")),
        "robustness": (v1.get("robustness"), v2.get("robustness")),
        "negative_control": (v1.get("negative_control"), v2.get("negative_control")),
        "ablations": (v1.get("ablations"), v2.get("ablations")),
        "claim_gate": (v1.get("claim_gate"), v2.get("claim_gate")),
    }
    for label, (left, right) in unchanged_paths.items():
        require(left == right, f"pretrained/scientific contract drifted between v1 and v2: {label}")

    v1_matched = v1.get("models", {}).get("matched_capacity_supervised_baseline", {})
    v2_matched = v2.get("models", {}).get("matched_capacity_supervised_baseline", {})
    require(v1_matched != v2_matched, "v2 must contain an explicit matched-capacity correction")
    require(
        "gradient-active" in str(v2_matched.get("parameter_accounting", "")),
        "v2 matched-capacity correction must use gradient-active accounting",
    )
    require(v1.get("models", {}).get("lam_jepa") == v2.get("models", {}).get("lam_jepa"), "LAM ARC variant changed")
    require(v1.get("models", {}).get("majority_reference") == v2.get("models", {}).get("majority_reference"), "majority reference changed")

    pretrained = v2["models"]["strong_pretrained_baseline"]
    require(pretrained.get("model") == "microsoft/deberta-v3-xsmall", "frozen DeBERTa model changed")
    require(pretrained.get("revision") == "14809e4f1fe1895fcba8b258271a940c6ca45ec4", "frozen DeBERTa revision changed")
    require(pretrained.get("license") == "MIT", "frozen DeBERTa license changed")
    require(float(v2["training_budget"]["pretrained_baseline_learning_rate"]) == 2e-5, "frozen DeBERTa LR changed")
    require(v2["training_budget"]["training_seeds"] == [1, 2, 3, 4, 5], "confirmatory seed set changed")
    require(v2["training_budget"]["epochs"] == 20, "confirmatory epochs changed")
    require(v2["training_budget"]["batch_size"] == 32, "confirmatory batch size changed")

    report = {
        "status": "passed",
        "v1": v1["protocol_id"],
        "v2": v2["protocol_id"],
        "v2_supersedes_v1": True,
        "only_scientifically_relevant_changed_contract": "matched_capacity_supervised_baseline accounting",
        "pretrained_contract_unchanged": True,
        "frozen_pretrained_model": pretrained,
        "test_access_authorized": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
