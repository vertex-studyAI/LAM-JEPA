# Roadmap

## Completed in this repo update

- reproducible single-run training entrypoint
- benchmark runner with seed sweep support
- educational task generators for multiple domains
- student-state model
- misconception diagnosis and tutoring policy
- curriculum engine and mastery tracker
- bootstrap and permutation-based statistical summaries
- paper-ready result generation
- populated dataset, docs, outputs, and experiments folders

## Next research moves

1. Replace synthetic task generators with real licensed external datasets.
2. Extend the student model to ingest full interaction logs, response times, and text explanations.
3. Add stronger OOD splits and benchmark leaderboards.
4. Export plots and tables directly into the paper build.
5. Compare against BKT, DKT, GRU4Rec-style baselines, and ablation variants.
6. Add a small web demo for tutor policy inspection and explanation rendering.

## Run order

1. `python scripts/train/train_single.py`
2. `python scripts/eval/eval_all.py`
3. `python scripts/bench/run_benchmarks.py`
4. `python scripts/analysis/aggregate_seeds.py`
5. `python scripts/paper/generate_results.py`
