from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _max_probability_range(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    width = len(rows[0]["probabilities"])
    maxima = [-math.inf] * width
    minima = [math.inf] * width
    for row in rows:
        probs = row["probabilities"]
        if len(probs) != width:
            raise ValueError("inconsistent probability vector width")
        for i, value in enumerate(probs):
            value = float(value)
            maxima[i] = max(maxima[i], value)
            minima[i] = min(minima[i], value)
    return max(high - low for high, low in zip(maxima, minima, strict=True))


def _summarize_records(records: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        rows = record["predictions"]
        predictions = [int(row["prediction"]) for row in rows]
        labels = [int(row["label"]) for row in rows]
        counts = Counter(predictions)
        correct = sum(int(p == y) for p, y in zip(predictions, labels, strict=True))
        prob_range = _max_probability_range(rows)
        unique = sorted(counts)
        reversal = record.get("choice_reversal_predictions")
        reversal_summary = None
        if reversal:
            rev_preds = [int(row["prediction"]) for row in reversal]
            max_reversal_prob_delta = max(
                abs(float(a) - float(b))
                for original, reversed_row in zip(rows, reversal, strict=True)
                for a, b in zip(original["probabilities"], reversed_row["probabilities"], strict=True)
            )
            reversal_summary = {
                "argmax_same_rate": sum(int(a == b) for a, b in zip(predictions, rev_preds, strict=True)) / len(rows),
                "max_probability_delta_original_vs_reversed": max_reversal_prob_delta,
            }
        out.append(
            {
                "seed": int(record["seed"]),
                "n": len(rows),
                "accuracy": correct / len(rows),
                "unique_prediction_count": len(unique),
                "predicted_classes": unique,
                "prediction_counts": {str(k): counts[k] for k in sorted(counts)},
                "max_probability_range_across_examples": prob_range,
                "constant_argmax": len(unique) == 1,
                "nearly_input_invariant_probabilities": prob_range <= tolerance,
                "choice_reversal": reversal_summary,
            }
        )
    return out


def analyze(payload: dict[str, Any], tolerance: float = 1e-6) -> dict[str, Any]:
    variants = payload["variants"]
    if "full" not in variants:
        raise ValueError("results payload lacks full variant")
    reference_rows = variants["full"]["records"][0]["predictions"]
    labels = [int(row["label"]) for row in reference_rows]
    label_counts = Counter(labels)
    n = len(labels)
    label_frequencies = {str(k): label_counts.get(k, 0) / n for k in range(4)}

    variant_summary = {
        name: _summarize_records(data["records"], tolerance)
        for name, data in variants.items()
    }
    negative_summary = _summarize_records(payload["negative_control"]["records"], tolerance)

    constant_runs = []
    for group_name, group in list(variant_summary.items()) + [("negative_control", negative_summary)]:
        for run in group:
            if run["constant_argmax"]:
                chosen = int(run["predicted_classes"][0])
                expected = label_frequencies[str(chosen)]
                constant_runs.append(
                    {
                        "group": group_name,
                        "seed": run["seed"],
                        "chosen_class": chosen,
                        "observed_accuracy": run["accuracy"],
                        "constant_class_reference_accuracy": expected,
                        "exact_accuracy_match": math.isclose(run["accuracy"], expected, rel_tol=0.0, abs_tol=1e-12),
                    }
                )

    mechanism_explanation = {}
    full_by_seed = {run["seed"]: run for run in variant_summary["full"]}
    for ablation in ("no_planner", "no_target"):
        if ablation not in variant_summary:
            continue
        rows = []
        for abl_run in variant_summary[ablation]:
            seed = abl_run["seed"]
            full_run = full_by_seed[seed]
            obs = full_run["accuracy"] - abl_run["accuracy"]
            if full_run["constant_argmax"] and abl_run["constant_argmax"]:
                fc = int(full_run["predicted_classes"][0])
                ac = int(abl_run["predicted_classes"][0])
                class_shift = label_frequencies[str(fc)] - label_frequencies[str(ac)]
            else:
                fc = ac = None
                class_shift = None
            rows.append(
                {
                    "seed": seed,
                    "full_class": fc,
                    "ablation_class": ac,
                    "observed_accuracy_delta": obs,
                    "delta_explained_by_constant_class_frequency_shift": class_shift,
                    "exact_explanation": class_shift is not None and math.isclose(obs, class_shift, rel_tol=0.0, abs_tol=1e-12),
                }
            )
        mechanism_explanation[ablation] = rows

    all_runs = [r for group in variant_summary.values() for r in group] + negative_summary
    conclusion = {
        "all_runs_constant_argmax": all(r["constant_argmax"] for r in all_runs),
        "all_runs_nearly_input_invariant_probabilities": all(r["nearly_input_invariant_probabilities"] for r in all_runs),
        "all_available_choice_reversals_preserve_argmax": all(
            r["choice_reversal"] is None or r["choice_reversal"]["argmax_same_rate"] == 1.0 for r in all_runs
        ),
        "all_available_choice_reversals_nearly_preserve_probabilities": all(
            r["choice_reversal"] is None or r["choice_reversal"]["max_probability_delta_original_vs_reversed"] <= tolerance
            for r in all_runs
        ),
        "all_constant_run_accuracies_match_constant_class_reference": all(r["exact_accuracy_match"] for r in constant_runs),
        "all_mechanism_deltas_explained_by_constant_class_frequency_shifts": all(
            row["exact_explanation"] for rows in mechanism_explanation.values() for row in rows
        ),
        "interpretation": (
            "Under this retained validation artifact, each seed/variant emits one argmax class for every validation item, "
            "with probability vectors nearly invariant across inputs at the configured tolerance. Reported accuracies and "
            "paired mechanism deltas therefore reduce to validation-label class frequencies and seed-dependent constant-class selection. "
            "For variants with retained choice-reversal predictions, reversing the choices also preserves every argmax and changes "
            "probabilities only below the configured tolerance."
        ),
    }

    return {
        "tolerance": tolerance,
        "validation_n": n,
        "validation_label_counts": {str(k): label_counts.get(k, 0) for k in range(4)},
        "validation_label_frequencies": label_frequencies,
        "variants": variant_summary,
        "negative_control": negative_summary,
        "constant_run_accuracy_checks": constant_runs,
        "mechanism_delta_checks": mechanism_explanation,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose input-invariant / constant-class prediction collapse in retained ARC controls results.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    report = analyze(payload, tolerance=args.tolerance)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
