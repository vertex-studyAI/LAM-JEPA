from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_primary_table(controls: dict[str, Any], matched: dict[str, Any], out: Path) -> None:
    variants = controls["variants"]
    negative = controls["negative_control"]["accuracy"]
    matched_summary = matched["summary"]
    rows = [
        ("LAM-JEPA (matched-baseline run)", matched_summary["lam_accuracy"]),
        ("Matched supervised", matched_summary["matched_supervised_accuracy"]),
        ("Full LAM-JEPA (controls run)", variants["full"]["accuracy"]),
        ("no_planner", variants["no_planner"]["accuracy"]),
        ("no_target", variants["no_target"]["accuracy"]),
        ("Shuffled-label control", negative),
    ]
    lines = ["| System | n | Mean accuracy | Sample SD |", "|---|---:|---:|---:|"]
    for name, summary in rows:
        lines.append(
            f"| {name} | {int(summary['n'])} | {float(summary['mean']):.10f} | {float(summary['std']):.10f} |"
        )
    lines += [
        "",
        "Source: machine-generated directly from the retained controls and matched-baseline JSON artifacts.",
        "The two LAM-JEPA rows come from distinct retained experiment artifacts and are not averaged together.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_effect_csv(controls: dict[str, Any], out: Path) -> list[dict[str, float | str]]:
    effects = controls["paired_effects"]
    rows: list[dict[str, float | str]] = []
    for key in ("no_planner", "no_target"):
        item = effects[key]
        rows.append(
            {
                "ablation": key,
                "mean_full_minus_ablation": float(item["mean_full_minus_ablation"]),
                "ci95_low": float(item["paired_bootstrap_ci95_low"]),
                "ci95_high": float(item["paired_bootstrap_ci95_high"]),
            }
        )
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_effect_svg(rows: list[dict[str, float | str]], out: Path) -> None:
    width, height = 760, 320
    left, right, top, bottom = 130, 40, 42, 66
    plot_w = width - left - right
    values = [0.0]
    for row in rows:
        values += [float(row["mean_full_minus_ablation"]), float(row["ci95_low"]), float(row["ci95_high"])]
    span = max(abs(min(values)), abs(max(values)), 0.015) * 1.25
    x_min, x_max = -span, span

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    ys = [105, 185]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Frozen ARC mechanism effects</text>',
        f'<line x1="{sx(0):.2f}" y1="{top}" x2="{sx(0):.2f}" y2="{height-bottom}" stroke="black" stroke-width="1" stroke-dasharray="4,4"/>',
    ]
    for row, y in zip(rows, ys, strict=True):
        label = str(row["ablation"])
        mean = float(row["mean_full_minus_ablation"])
        lo = float(row["ci95_low"])
        hi = float(row["ci95_high"])
        parts += [
            f'<text x="{left-14}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="15">{escape_xml(label)}</text>',
            f'<line x1="{sx(lo):.2f}" y1="{y}" x2="{sx(hi):.2f}" y2="{y}" stroke="black" stroke-width="2"/>',
            f'<line x1="{sx(lo):.2f}" y1="{y-7}" x2="{sx(lo):.2f}" y2="{y+7}" stroke="black" stroke-width="2"/>',
            f'<line x1="{sx(hi):.2f}" y1="{y-7}" x2="{sx(hi):.2f}" y2="{y+7}" stroke="black" stroke-width="2"/>',
            f'<circle cx="{sx(mean):.2f}" cy="{y}" r="5" fill="black"/>',
        ]
    axis_y = height - bottom
    parts.append(f'<line x1="{left}" y1="{axis_y}" x2="{width-right}" y2="{axis_y}" stroke="black"/>')
    for value in (x_min, x_min / 2, 0.0, x_max / 2, x_max):
        x = sx(value)
        parts += [
            f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y+6}" stroke="black"/>',
            f'<text x="{x:.2f}" y="{axis_y+24}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:+.3f}</text>',
        ]
    parts += [
        f'<text x="{left + plot_w/2:.2f}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="14">Full minus ablation accuracy</text>',
        '<text x="380" y="286" text-anchor="middle" font-family="sans-serif" font-size="11">Bars: retained paired bootstrap 95% intervals. Descriptive; not a significance claim.</text>',
        "</svg>",
    ]
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LAM-JEPA negative-paper tables/figures from retained JSON artifacts.")
    parser.add_argument("--controls-json", type=Path, required=True)
    parser.add_argument("--matched-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    controls = load_json(args.controls_json)
    matched = load_json(args.matched_json)

    require(controls["protocol"]["protocol_id"] == "lam-jepa-arc-challenge-v3", "unexpected controls protocol")
    require(controls["protocol"]["seeds"] == [1, 2, 3, 4, 5], "controls must use frozen five seeds")
    require(controls["protocol"]["epochs"] == 20, "controls must use frozen 20 epochs")
    require(set(controls["variants"]) >= {"full", "no_planner", "no_target"}, "required control variants missing")
    require(matched["protocol"]["seeds"] == [1, 2, 3, 4, 5], "matched comparison must use frozen five seeds")
    require(matched["protocol"]["epochs"] == 20, "matched comparison must use frozen 20 epochs")
    require(matched["protocol"]["primary_metric"] == "multiple-choice accuracy", "unexpected matched primary metric")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "arc_primary_results.md"
    csv_path = args.out_dir / "arc_mechanism_effects.csv"
    svg_path = args.out_dir / "arc_mechanism_effects.svg"
    manifest_path = args.out_dir / "arc_paper_artifact_manifest.json"

    write_primary_table(controls, matched, table_path)
    rows = write_effect_csv(controls, csv_path)
    write_effect_svg(rows, svg_path)

    manifest = {
        "generator": str(Path(__file__).as_posix()),
        "inputs": {
            "controls_json": {"path": str(args.controls_json), "sha256": sha256(args.controls_json)},
            "matched_json": {"path": str(args.matched_json), "sha256": sha256(args.matched_json)},
        },
        "outputs": {},
        "claim_boundary": "Generated only from retained frozen artifacts; no metric, seed, threshold, or result is recomputed from manuscript prose.",
    }
    for path in (table_path, csv_path, svg_path):
        manifest["outputs"][path.name] = {"sha256": sha256(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
