from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).resolve().parent)
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lam_jepa.benchmarking.arc_challenge import dataset_digest, id_digest, load_arc_split


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def ordered_id_digest(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for item_id in ids:
        digest.update(item_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently verify ARC protocol-v3 eligibility evidence.")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("protocols/arc_challenge_v3.json"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol.get("protocol_id") == "lam-jepa-arc-challenge-v3", "wrong protocol")
    eligibility = (protocol.get("dataset") or {}).get("eligibility") or {}
    required_choice_count = int(eligibility.get("required_choice_count", -1))
    require(required_choice_count == 4, "v3 required choice count drift")
    require(audit.get("status") == "passed", "eligibility audit did not pass")
    require(audit.get("protocol_id") == protocol["protocol_id"], "audit protocol mismatch")
    require(audit.get("test_split_accessed") is False, "eligibility audit touched locked test")
    require(audit.get("eligibility_rule") == eligibility.get("rule"), "audit eligibility rule mismatch")

    verified_splits: dict[str, dict[str, object]] = {}
    for split, path in (("train", args.train), ("validation", args.validation)):
        examples = load_arc_split(path)
        eligible = [example for example in examples if len(example.choices) == required_choice_count]
        excluded = [example for example in examples if len(example.choices) != required_choice_count]
        distribution = Counter(len(example.choices) for example in examples)
        block = (audit.get("splits") or {}).get(split)
        require(isinstance(block, dict), f"audit block missing: {split}")
        require(int(block.get("source_rows", -1)) == len(examples), f"{split}: source row count mismatch")
        require(int(block.get("eligible_rows", -1)) == len(eligible), f"{split}: eligible row count mismatch")
        require(int(block.get("excluded_rows_count", -1)) == len(excluded), f"{split}: excluded row count mismatch")
        require(len(examples) == len(eligible) + len(excluded), f"{split}: partition does not cover source")
        require(block.get("source_dataset_digest") == dataset_digest(examples), f"{split}: source dataset digest mismatch")
        require(block.get("source_id_digest") == id_digest(examples), f"{split}: source ID digest mismatch")
        require(block.get("eligible_dataset_digest") == dataset_digest(eligible), f"{split}: eligible dataset digest mismatch")
        require(block.get("eligible_id_digest") == id_digest(eligible), f"{split}: eligible ID digest mismatch")
        require(block.get("excluded_id_digest") == id_digest(excluded), f"{split}: excluded ID digest mismatch")
        declared_distribution = {int(key): int(value) for key, value in (block.get("choice_count_distribution") or {}).items()}
        require(declared_distribution == dict(sorted(distribution.items())), f"{split}: choice-count distribution mismatch")
        require(sum(declared_distribution.values()) == len(examples), f"{split}: distribution total mismatch")
        require(declared_distribution.get(4, 0) == len(eligible), f"{split}: four-choice count/eligibility mismatch")

        declared_excluded = block.get("excluded_rows")
        require(isinstance(declared_excluded, list) and len(declared_excluded) == len(excluded), f"{split}: excluded evidence length mismatch")
        expected_excluded = [
            {"id": example.item_id, "choice_count": len(example.choices)}
            for example in excluded
        ]
        require(declared_excluded == expected_excluded, f"{split}: excluded row evidence mismatch")
        require(all(int(row["choice_count"]) != 4 for row in declared_excluded), f"{split}: eligible row listed as excluded")
        require(
            ordered_id_digest([str(row["id"]) for row in declared_excluded]) == block.get("excluded_id_digest"),
            f"{split}: independently derived excluded-ID digest mismatch",
        )

        verified_splits[split] = {
            "source_rows": len(examples),
            "choice_count_distribution": dict(sorted(distribution.items())),
            "eligible_rows": len(eligible),
            "excluded_rows": len(excluded),
            "eligible_id_digest": id_digest(eligible),
            "excluded_id_digest": id_digest(excluded),
            "excluded": expected_excluded,
        }

    require(any(block["excluded_rows"] > 0 for block in verified_splits.values()), "v3 correction is not exercised: no non-four-choice rows observed")
    report = {
        "verdict": "ARC_PROTOCOL_V3_ELIGIBILITY_VERIFIED_ONLY",
        "protocol_id": protocol["protocol_id"],
        "required_choice_count": required_choice_count,
        "test_split_accessed": False,
        "verified_splits": verified_splits,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
