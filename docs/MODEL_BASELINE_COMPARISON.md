# Paired Model-to-Baseline Comparison Contract

LAM-JEPA model accuracy may be compared with the repository's label-distribution references only when both evaluators consumed the exact same ordered input-and-label rows.

## Run the paired protocol

Use the same seed, batch size, batch count, task registry, and vocabulary configuration:

```bash
python scripts/eval/eval_all.py \
  --checkpoint experiments/seed_1/final.pt \
  --device cpu \
  --batch-size 64 \
  --batches 8 \
  --seed 7 \
  --out outputs/eval_all.json

python scripts/eval/eval_baselines.py \
  --batch-size 64 \
  --batches 8 \
  --seed 7 \
  --out outputs/eval_baselines.json

python scripts/analysis/compare_model_to_baselines.py \
  --model outputs/eval_all.json \
  --baselines outputs/eval_baselines.json \
  --out outputs/model_baseline_comparison.json
```

Each task result contains a 256-bit digest binding its ordered input fingerprints to its ordered target labels. The comparison command fails closed unless model and baseline digests match exactly. It also rejects differences in seed, batch size, batch count, task order, sample count, or target semantics.

The model evaluator resets the benchmark sampler immediately before evaluation and preserves the sampler RNG state around model inference. This prevents model construction or forward execution from silently changing later benchmark rows.

## Reported references

For each task, the comparison artifact records:

- model accuracy;
- the sampled-label majority-frequency reference;
- expected accuracy from guessing uniformly over observed sampled labels;
- expected accuracy from guessing uniformly over the configured output vocabulary;
- descriptive accuracy deltas against each reference;
- whether model accuracy is above, equal to, or below the majority-frequency reference.

## Interpretation boundary

These values are descriptive for one exactly paired sampled evaluation. A positive delta is not a confidence interval, a significance test, held-out generalization evidence, or proof of educational effectiveness. It does not validate the synthetic generator, establish benchmark quality, prove novelty, or demonstrate superiority over another model.

The `target_semantics` field remains authoritative. For `gsm8k`, `reading`, `tutoring`, and `reasoning`, the current target is a generated concept-proxy label. Accuracy and baseline deltas on those tasks must not be described as natural-language answer correctness.

The majority value is an oracle frequency computed from sampled labels, not a deployable predictor. The digest establishes row identity only; it is not a privacy guarantee and does not make the underlying benchmark valid.

## CI evidence

The CPU reproducibility workflow generates model evaluation, baseline evaluation, the paired comparison, and an independent verification report. The verifier recomputes every delta, checks exact sample digests and semantics, validates summary counts, and fails if the claim boundary is weakened.
