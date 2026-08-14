from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float) -> str:
    return f"{x:.10f}"


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

    protocol = full["protocol"]
    assert protocol["protocol_id"] == "lam-jepa-arc-challenge-v3"
    assert protocol["seeds"] == [1, 2, 3, 4, 5]
    assert protocol["epochs"] == 20
    assert protocol["batch_size"] == 32
    assert protocol["learning_rate"] == 0.0003
    assert protocol["model_steps"] == 1
    assert protocol["validation_eligibility"]["used_rows"] == 295
    assert protocol["train_eligibility"]["used_rows"] == 1117

    rows = []
    for key, label in [("full", "LAM-JEPA full"), ("no_planner", "No planner"), ("no_target", "No target")]:
        a = full["variants"][key]["accuracy"]
        rows.append((label, a["mean"], a["std"], a["n"], "full-controls artifact"))
    a = full["negative_control"]["accuracy"]
    rows.append(("Shuffled labels", a["mean"], a["std"], a["n"], "full-controls artifact"))
    a = matched["summary"]["matched_supervised_accuracy"]
    rows.append(("Matched supervised", a["mean"], a["std"], a["n"], "matched-baseline artifact"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "arc_validation_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "mean_accuracy", "sample_std", "n", "source"])
        for row in rows:
            w.writerow([row[0], fmt(row[1]), fmt(row[2]), row[3], row[4]])

    np_eff = full["paired_effects"]["no_planner"]
    nt_eff = full["paired_effects"]["no_target"]
    pm = matched["summary"]["paired_accuracy_delta_lam_minus_matched"]
    pre = pretrained["summary"]

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
        f"| Full − no planner | {fmt(np_eff['mean_full_minus_ablation'])} | [{fmt(np_eff['paired_bootstrap_ci95_low'])}, {fmt(np_eff['paired_bootstrap_ci95_high'])}] | 5 |",
        f"| Full − no target | {fmt(nt_eff['mean_full_minus_ablation'])} | [{fmt(nt_eff['paired_bootstrap_ci95_low'])}, {fmt(nt_eff['paired_bootstrap_ci95_high'])}] | 5 |",
        f"| Full − matched supervised | {fmt(pm['mean'])} | not recomputed here | {pm['n']} |",
        "",
        "## Bounded pretrained characterization",
        "",
        f"LAM mean `{fmt(pre['lam_accuracy']['mean'])}` vs pinned pretrained mean `{fmt(pre['pretrained_accuracy']['mean'])}`; paired LAM − pretrained `{fmt(pre['paired_accuracy_delta_lam_minus_pretrained']['mean'])}`; n={pre['lam_accuracy']['n']}. This is a development characterization, not a final superiority/inferiority claim.",
        "",
    ]
    (args.out_dir / "ARC_NEGATIVE_RESULT_TABLES.generated.md").write_text("\n".join(table), encoding="utf-8")

    # Small dependency-free SVG. Error bars are sample SD, explicitly labeled.
    width, height = 900, 500
    left, right, top, bottom = 90, 30, 45, 100
    plot_w, plot_h = width-left-right, height-top-bottom
    y_min, y_max = 0.20, 0.30
    def y(v: float) -> float:
        return top + (y_max-v)/(y_max-y_min)*plot_h
    bar_w = 90
    gap = plot_w / len(rows)
    colors = ["#555555"] * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="450" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Frozen ARC validation accuracy (mean ± sample SD)</text>']
    for tick in [0.20,0.22,0.24,0.26,0.28,0.30]:
        yy=y(tick)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.2f}</text>')
    for i,(label,mean,sd,n,_) in enumerate(rows):
        cx=left+gap*(i+0.5)
        yy=y(mean); base=y(y_min)
        parts.append(f'<rect x="{cx-bar_w/2:.1f}" y="{yy:.1f}" width="{bar_w}" height="{base-yy:.1f}" fill="{colors[i]}"/>')
        parts.append(f'<line x1="{cx:.1f}" y1="{y(mean+sd):.1f}" x2="{cx:.1f}" y2="{y(mean-sd):.1f}" stroke="black" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-9:.1f}" y1="{y(mean+sd):.1f}" x2="{cx+9:.1f}" y2="{y(mean+sd):.1f}" stroke="black" stroke-width="2"/>')
        parts.append(f'<line x1="{cx-9:.1f}" y1="{y(mean-sd):.1f}" x2="{cx+9:.1f}" y2="{y(mean-sd):.1f}" stroke="black" stroke-width="2"/>')
        safe=html.escape(label)
        parts.append(f'<text x="{cx:.1f}" y="{height-68}" transform="rotate(-25 {cx:.1f} {height-68})" text-anchor="end" font-family="sans-serif" font-size="12">{safe}</text>')
    parts += [f'<line x1="{left}" y1="{y(y_min):.1f}" x2="{width-right}" y2="{y(y_min):.1f}" stroke="black"/>', '</svg>']
    (args.out_dir / "arc_validation_accuracy.generated.svg").write_text("\n".join(parts), encoding="utf-8")

    manifest = {
        "generator": "scripts/analysis/generate_arc_negative_paper_assets.py",
        "inputs": {
            "full_controls": str(args.full_controls),
            "matched": str(args.matched),
            "pretrained": str(args.pretrained),
        },
        "outputs": [
            "arc_validation_accuracy.csv",
            "ARC_NEGATIVE_RESULT_TABLES.generated.md",
            "arc_validation_accuracy.generated.svg",
        ],
        "claim_boundary": "Validation/development evidence only; no locked ARC test data; SVG error bars are sample SD, not confidence intervals.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
