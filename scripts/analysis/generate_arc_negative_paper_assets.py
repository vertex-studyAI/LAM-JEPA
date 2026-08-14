from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import statistics
from pathlib import Path
from typing import Sequence


BOOTSTRAP_SEED = 20260807
MAX_FLOAT32_ACCURACY_DRIFT = 2e-8


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(x: float) -> str:
    return f"{x:.10f}"


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def prediction_accuracy(rows: Sequence[dict]) -> float:
    if not rows:
        raise ValueError("cannot score empty prediction rows")
    correct = sum(int(row["prediction"]) == int(row["label"]) for row in rows)
    return correct / len(rows)


def paired_bootstrap_ci(deltas: Sequence[float], *, seed: int, samples: int = 10000) -> tuple[float, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires at least one delta")
    rng = random.Random(seed)
    n = len(deltas)
    boot = [float(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n))) for _ in range(samples)]
    boot.sort()
    return boot[int(0.025 * (samples - 1))], boot[int(0.975 * (samples - 1))]


def require_close(actual: float, expected: float, *, tol: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{label}: {actual} != {expected} within tolerance {tol}")


def collect_variant(full: dict, key: str, drifts: list[float]) -> tuple[dict, dict]:
    metric_values: list[float] = []
    direct_values: list[float] = []
    for record in full["variants"][key]["records"]:
        metric_accuracy = float(record["metrics"]["accuracy"])
        direct_accuracy = prediction_accuracy(record["predictions"])
        drift = abs(metric_accuracy - direct_accuracy)
        drifts.append(drift)
        if drift > MAX_FLOAT32_ACCURACY_DRIFT:
            raise RuntimeError(f"{key}/seed={record['seed']}: prediction/metric accuracy drift {drift} too large")
        metric_values.append(metric_accuracy)
        direct_values.append(direct_accuracy)

    metric_summary = summarize(metric_values)
    direct_summary = summarize(direct_values)
    stored = full["variants"][key]["accuracy"]
    require_close(metric_summary["mean"], stored["mean"], tol=1e-15, label=f"{key}.mean")
    require_close(metric_summary["std"], stored["std"], tol=1e-15, label=f"{key}.std")
    if metric_summary["n"] != stored["n"]:
        raise RuntimeError(f"{key}.n mismatch")
    return metric_summary, direct_summary


def collect_arm(records: Sequence[dict], arm: str, drifts: list[float]) -> tuple[list[float], list[float]]:
    metric_values: list[float] = []
    direct_values: list[float] = []
    for record in records:
        metric_accuracy = float(record[arm]["metrics"]["accuracy"])
        direct_accuracy = prediction_accuracy(record[arm]["predictions"])
        drift = abs(metric_accuracy - direct_accuracy)
        drifts.append(drift)
        if drift > MAX_FLOAT32_ACCURACY_DRIFT:
            raise RuntimeError(f"{arm}/seed={record['seed']}: prediction/metric accuracy drift {drift} too large")
        metric_values.append(metric_accuracy)
        direct_values.append(direct_accuracy)
    return metric_values, direct_values


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LAM-JEPA negative-result paper assets from retained raw JSON artifacts.")
    p.add_argument("--full-controls", required=True, type=Path)
    p.add_argument("--matched", required=True, type=Path)
    p.add_argument("--pretrained", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    full = load(args.full_controls)
    matched = load(args.matched)
    pretrained = load(args.pretrained)
    drifts: list[float] = []

    protocol = full["protocol"]
    assert protocol["protocol_id"] == "lam-jepa-arc-challenge-v3"
    assert protocol["seeds"] == [1, 2, 3, 4, 5]
    assert protocol["epochs"] == 20
    assert protocol["batch_size"] == 32
    assert protocol["learning_rate"] == 0.0003
    assert protocol["model_steps"] == 1
    assert protocol["validation_eligibility"]["used_rows"] == 295
    assert protocol["train_eligibility"]["used_rows"] == 1117
    assert "not downloaded or evaluated" in protocol["test_split_policy"]

    full_metric, _ = collect_variant(full, "full", drifts)
    no_planner_metric, _ = collect_variant(full, "no_planner", drifts)
    no_target_metric, _ = collect_variant(full, "no_target", drifts)

    negative_values: list[float] = []
    for record in full["negative_control"]["records"]:
        metric_accuracy = float(record["metrics"]["accuracy"])
        direct_accuracy = prediction_accuracy(record["predictions"])
        drift = abs(metric_accuracy - direct_accuracy)
        drifts.append(drift)
        if drift > MAX_FLOAT32_ACCURACY_DRIFT:
            raise RuntimeError(f"negative-control/seed={record['seed']}: prediction/metric drift {drift} too large")
        negative_values.append(metric_accuracy)
    negative_metric = summarize(negative_values)
    require_close(negative_metric["mean"], full["negative_control"]["accuracy"]["mean"], tol=1e-15, label="negative.mean")
    require_close(negative_metric["std"], full["negative_control"]["accuracy"]["std"], tol=1e-15, label="negative.std")
    if not full["negative_control"]["pass"]:
        raise RuntimeError("frozen shuffled-label control did not pass its recorded ceiling")

    full_per_seed = [float(record["metrics"]["accuracy"]) for record in full["variants"]["full"]["records"]]
    np_per_seed = [float(record["metrics"]["accuracy"]) for record in full["variants"]["no_planner"]["records"]]
    nt_per_seed = [float(record["metrics"]["accuracy"]) for record in full["variants"]["no_target"]["records"]]
    np_deltas = [a - b for a, b in zip(full_per_seed, np_per_seed, strict=True)]
    nt_deltas = [a - b for a, b in zip(full_per_seed, nt_per_seed, strict=True)]
    np_effect = summarize(np_deltas)
    nt_effect = summarize(nt_deltas)
    np_ci = paired_bootstrap_ci(np_deltas, seed=BOOTSTRAP_SEED)
    nt_ci = paired_bootstrap_ci(nt_deltas, seed=BOOTSTRAP_SEED + 1)
    stored_np = full["paired_effects"]["no_planner"]
    stored_nt = full["paired_effects"]["no_target"]
    require_close(np_effect["mean"], stored_np["mean_full_minus_ablation"], tol=1e-15, label="no_planner.effect.mean")
    require_close(np_effect["std"], stored_np["std_paired_difference"], tol=1e-15, label="no_planner.effect.std")
    require_close(np_ci[0], stored_np["paired_bootstrap_ci95_low"], tol=1e-15, label="no_planner.ci.low")
    require_close(np_ci[1], stored_np["paired_bootstrap_ci95_high"], tol=1e-15, label="no_planner.ci.high")
    require_close(nt_effect["mean"], stored_nt["mean_full_minus_ablation"], tol=1e-15, label="no_target.effect.mean")
    require_close(nt_effect["std"], stored_nt["std_paired_difference"], tol=1e-15, label="no_target.effect.std")
    require_close(nt_ci[0], stored_nt["paired_bootstrap_ci95_low"], tol=1e-15, label="no_target.ci.low")
    require_close(nt_ci[1], stored_nt["paired_bootstrap_ci95_high"], tol=1e-15, label="no_target.ci.high")

    _, matched_lam_direct = collect_arm(matched["records"], "lam_jepa", drifts)
    _, matched_sup_direct = collect_arm(matched["records"], "matched_supervised", drifts)
    matched_lam = summarize(matched_lam_direct)
    matched_sup = summarize(matched_sup_direct)
    matched_effect = summarize([a - b for a, b in zip(matched_lam_direct, matched_sup_direct, strict=True)])

    _, pretrained_lam_direct = collect_arm(pretrained["records"], "lam_jepa", drifts)
    _, pretrained_base_direct = collect_arm(pretrained["records"], "pretrained_baseline", drifts)
    pretrained_lam = summarize(pretrained_lam_direct)
    pretrained_base = summarize(pretrained_base_direct)
    pretrained_effect = summarize([a - b for a, b in zip(pretrained_lam_direct, pretrained_base_direct, strict=True)])
    pre_protocol = pretrained["protocol"]
    assert pre_protocol["seeds"] == [1, 2]
    assert pre_protocol["train_examples"] == 8
    assert pre_protocol["validation_examples"] == 16
    assert pre_protocol["max_train_steps"] == 1
    assert pre_protocol["pretrained_model_id"] == "microsoft/deberta-v3-xsmall"
    assert pre_protocol["resolved_pretrained_revision"] == "14809e4f1fe1895fcba8b258271a940c6ca45ec4"
    assert "not downloaded or evaluated" in pre_protocol["test_split_policy"]

    rows = [
        ("LAM-JEPA full", full_metric["mean"], full_metric["std"], full_metric["n"], "retained float32 per-seed metric"),
        ("No planner", no_planner_metric["mean"], no_planner_metric["std"], no_planner_metric["n"], "retained float32 per-seed metric"),
        ("No target", no_target_metric["mean"], no_target_metric["std"], no_target_metric["n"], "retained float32 per-seed metric"),
        ("Shuffled labels", negative_metric["mean"], negative_metric["std"], negative_metric["n"], "retained float32 per-seed metric"),
        ("Matched supervised", matched_sup["mean"], matched_sup["std"], matched_sup["n"], "exact per-example prediction count"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "arc_validation_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "mean_accuracy", "sample_std", "n", "numeric_basis"])
        for row in rows:
            w.writerow([row[0], fmt(row[1]), fmt(row[2]), row[3], row[4]])

    table = [
        "# ARC negative-result paper tables",
        "",
        "Generated only from the three retained JSON inputs named in the command; no test split is used.",
        "",
        "## Validation accuracy",
        "",
        "| System | Mean | Sample SD | n |",
        "|---|---:|---:|---:|",
    ]
    for label, mean, sd, n, _ in rows:
        table.append(f"| {label} | {fmt(mean)} | {fmt(sd)} | {n} |")
    table += [
        "",
        "## Paired effects",
        "",
        "| Contrast | Mean paired difference | 95% bootstrap CI | n |",
        "|---|---:|---:|---:|",
        f"| Full − no planner | {fmt(np_effect['mean'])} | [{fmt(np_ci[0])}, {fmt(np_ci[1])}] | 5 |",
        f"| Full − no target | {fmt(nt_effect['mean'])} | [{fmt(nt_ci[0])}, {fmt(nt_ci[1])}] | 5 |",
        f"| Full − matched supervised | {fmt(matched_effect['mean'])} | not applicable to frozen mechanism gate | {matched_effect['n']} |",
        "",
        "## Bounded pretrained characterization",
        "",
        f"LAM mean `{fmt(pretrained_lam['mean'])}` vs pinned pretrained mean `{fmt(pretrained_base['mean'])}`; paired LAM − pretrained `{fmt(pretrained_effect['mean'])}`; n={pretrained_lam['n']}. This is a development characterization, not a final superiority/inferiority claim.",
        "",
        f"Maximum absolute exact-count vs retained float32 accuracy drift: `{max(drifts):.3e}`. This changes no frozen scientific conclusion or criterion.",
        "",
    ]
    (args.out_dir / "ARC_NEGATIVE_RESULT_TABLES.generated.md").write_text("\n".join(table), encoding="utf-8")

    width, height = 900, 500
    left, right, top, bottom = 90, 30, 45, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.20, 0.30

    def y(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    bar_w = 90
    gap = plot_w / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Frozen ARC validation accuracy (mean ± sample SD)</text>',
    ]
    for tick in [0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        yy = y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.2f}</text>')
    for i, (label, mean, sd, _, _) in enumerate(rows):
        cx = left + gap * (i + 0.5)
        yy = y(mean)
        base = y(y_min)
        parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{yy:.1f}" width="{bar_w}" height="{base-yy:.1f}" fill="#555555"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{y(mean+sd):.1f}" x2="{cx:.1f}" y2="{y(mean-sd):.1f}" stroke="black" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-9:.1f}" y1="{y(mean+sd):.1f}" x2="{cx+9:.1f}" y2="{y(mean+sd):.1f}" stroke="black" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-9:.1f}" y1="{y(mean-sd):.1f}" x2="{cx+9:.1f}" y2="{y(mean-sd):.1f}" stroke="black" stroke-width="2"/>')
        safe = html.escape(label)
        parts.append(f'<text x="{cx:.1f}" y="{height-68}" transform="rotate(-25 {cx:.1f} {height-68})" text-anchor="end" font-family="sans-serif" font-size="12">{safe}</text>')
    parts += [
        f'<line x1="{left}" y1="{y(y_min):.1f}" x2="{width-right}" y2="{y(y_min):.1f}" stroke="black"/>',
        "</svg>",
    ]
    (args.out_dir / "arc_validation_accuracy.generated.svg").write_text("\n".join(parts), encoding="utf-8")

    manifest = {
        "generator": "scripts/analysis/generate_arc_negative_paper_assets.py",
        "inputs": {
            "full_controls": {"path": str(args.full_controls), "sha256": sha256(args.full_controls)},
            "matched": {"path": str(args.matched), "sha256": sha256(args.matched)},
            "pretrained": {"path": str(args.pretrained), "sha256": sha256(args.pretrained)},
        },
        "outputs": [
            "arc_validation_accuracy.csv",
            "ARC_NEGATIVE_RESULT_TABLES.generated.md",
            "arc_validation_accuracy.generated.svg",
        ],
        "numeric_basis": {
            "full_controls_table": "reaggregated retained float32 per-seed metrics; each checked against direct per-example counts",
            "mechanism_effects": "independently recomputed from retained per-seed metrics, including frozen bootstrap",
            "matched_table": "independently recomputed from retained per-example predictions",
            "pretrained_characterization": "independently recomputed from retained per-example predictions",
        },
        "max_exact_count_vs_float32_accuracy_abs_diff": max(drifts),
        "claim_boundary": "Validation/development evidence only; no locked ARC test data; SVG error bars are sample SD, not confidence intervals.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
