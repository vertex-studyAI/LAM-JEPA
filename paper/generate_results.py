from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_task_accuracy(payload: dict[str, Any]) -> dict[str, float]:
    out = {}
    if not payload:
        return out
    scores = payload.get("scores") or payload.get("tasks") or {}
    for task, val in scores.items():
        if isinstance(val, dict) and "accuracy" in val:
            out[task] = float(val["accuracy"])
        elif isinstance(val, (float, int)):
            out[task] = float(val)
    return out


def _bar_plot(values: dict[str, float], title: str, out_path: Path) -> None:
    if not values:
        return
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, vals)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_table(values: dict[str, float], out_path: Path) -> None:
    lines = ["task,accuracy"]
    for k, v in sorted(values.items(), key=lambda kv: kv[0]):
        lines.append(f"{k},{v:.4f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def generate_results(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    outputs = root / "outputs"
    results_dir = root / "results"
    paper_dir = root / "paper"
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _load_json(outputs / "benchmark_results.json") or _load_json(outputs / "eval_all.json") or {}
    ablation = _load_json(outputs / "ablation_results.json") or {}
    extended = _load_json(root / "results" / "extended_results.json") or {}

    benchmark_acc = _collect_task_accuracy(benchmark)
    ablation_acc = {}
    for name, payload in ablation.items() if isinstance(ablation, dict) else []:
        if isinstance(payload, dict):
            task_scores = payload.get("scores", {})
            if isinstance(task_scores, dict):
                vals = [float(v.get("accuracy", v)) for v in task_scores.values() if isinstance(v, (dict, float, int))]
                if vals:
                    ablation_acc[name] = float(np.mean(vals))

    if benchmark_acc:
        _bar_plot(benchmark_acc, "Benchmark accuracy", results_dir / "benchmark_accuracy.png")
        _write_table(benchmark_acc, results_dir / "benchmark_accuracy.csv")
    if ablation_acc:
        _bar_plot(ablation_acc, "Ablation accuracy", results_dir / "ablation_accuracy.png")
        _write_table(ablation_acc, results_dir / "ablation_accuracy.csv")

    report = {
        "benchmark_tasks": benchmark_acc,
        "ablation_summary": ablation_acc,
        "source_files": [str(p) for p in [outputs / "benchmark_results.json", outputs / "ablation_results.json", root / "results" / "extended_results.json"] if p.exists()],
    }
    (paper_dir / "generated_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = generate_results(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
