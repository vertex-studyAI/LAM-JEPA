from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets" / "student_traces"
OUT.mkdir(parents=True, exist_ok=True)


def build_from_assistments_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    for student_id, group in df.groupby("user_id"):
        group = group.sort_values(group.columns.intersection(["start_time", "timestamp"]).tolist()[0] if any(c in group.columns for c in ["start_time", "timestamp"]) else group.index.name)
        payload = {
            "student_id": str(student_id),
            "sessions": [],
        }
        for _, row in group.iterrows():
            payload["sessions"].append({
                "skill_id": row.get("skill_id"),
                "correct": int(row.get("correct", 0)),
                "response_time": float(row.get("ms_first_response", row.get("response_time", 0.0)) or 0.0),
                "hint_count": int(row.get("hint_count", 0) or 0),
                "timestamp": str(row.get("start_time", row.get("timestamp", ""))),
            })
        with open(OUT / f"assistments_{student_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

if __name__ == "__main__":
    print("Populate this script with the exact ASSISTments CSV path after download.")
