from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

HF_TASKS = [
    ("gsm8k", "gsm8k", {"train": "train", "test": "test"}, DATASETS / "external" / "math" / "gsm8k"),
    ("sciq", "sciq", {"train": "train", "test": "validation"}, DATASETS / "external" / "science" / "sciq"),
    ("openbookqa", "openbookqa", {"train": "train", "test": "test", "validation": "validation"}, DATASETS / "external" / "science" / "openbookqa"),
    ("race", "race", {"train": "train", "validation": "validation", "test": "test"}, DATASETS / "external" / "reading" / "race"),
    ("drop", "drop", {"train": "train", "validation": "validation", "test": "test"}, DATASETS / "external" / "reading" / "drop"),
    ("strategyqa", "strategy_qa", {"train": "train", "validation": "validation", "test": "test"}, DATASETS / "external" / "reasoning" / "strategyqa"),
    ("proofwriter", "proofwriter", {"train": "train", "validation": "validation", "test": "test"}, DATASETS / "external" / "reasoning" / "proofwriter"),
]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_packages() -> None:
    try:
        import datasets  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "datasets", "pyyaml", "pandas", "numpy", "tqdm", "networkx", "scikit-learn"])


def download_hf_dataset(name: str, config: str, splits: dict[str, str], out_dir: Path) -> None:
    from datasets import load_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for out_split, hf_split in splits.items():
        ds = load_dataset(name, config, split=hf_split)
        ds.save_to_disk(str(out_dir / out_split))
        print(f"saved {name}:{hf_split} -> {out_dir / out_split}")


def main() -> None:
    ensure_packages()

    for name, config, splits, out_dir in HF_TASKS:
        try:
            download_hf_dataset(name, config, splits, out_dir)
        except Exception as e:
            print(f"[warn] failed {name}/{config}: {e}")

    # Official repos / manual pulls
    repos = [
        ("https://github.com/openai/grade-school-math.git", DATASETS / "external" / "math" / "gsm8k_repo"),
        ("https://github.com/hendrycks/math.git", DATASETS / "external" / "math" / "MATH_repo"),
        ("https://github.com/allenai/OpenBookQA.git", DATASETS / "external" / "science" / "openbookqa_repo"),
        ("https://github.com/allenai/proofwriter.git", DATASETS / "external" / "reasoning" / "proofwriter_repo"),
        ("https://github.com/eladsegal/strategyqa.git", DATASETS / "external" / "reasoning" / "strategyqa_repo"),
    ]

    for url, dest in repos:
        if dest.exists():
            continue
        try:
            run(["git", "clone", "--depth", "1", url, str(dest)])
        except Exception as e:
            print(f"[warn] clone failed {url}: {e}")

    # Manual note directories
    (DATASETS / "external" / "tutoring" / "assistments" / "MANUAL_DOWNLOAD_REQUIRED.txt").write_text(
        "Download ASSISTments from https://sites.google.com/site/assistmentsdata/ and place the files here.
"
    )
    (DATASETS / "external" / "tutoring" / "ednet" / "MANUAL_DOWNLOAD_REQUIRED.txt").write_text(
        "Download EdNet from its official source and place the files here.
"
    )

if __name__ == "__main__":
    main()
