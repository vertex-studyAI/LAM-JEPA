# Label-Distribution Baseline Contract

LAM-JEPA evaluation reports must be interpreted against transparent references derived from the exact same task sampler. Run:

```bash
python scripts/eval/eval_baselines.py \
  --batch-size 64 \
  --batches 8 \
  --seed 7 \
  --out outputs/eval_baselines.json
```

The command evaluates every task in `EDTECH_TASKS` without loading or executing a model. It records sample counts, input and prompt diversity, label support, target semantics, and three reference accuracies:

- `majority_accuracy`: the frequency of the most common label in the sampled evaluation rows;
- `uniform_observed_label_accuracy`: expected accuracy when guessing uniformly over labels observed in those rows;
- `uniform_full_vocab_accuracy`: expected accuracy when guessing uniformly over the configured output vocabulary.

## Interpretation boundary

The majority value is an **oracle class-frequency reference** because it is calculated from evaluation labels. It is not a deployable predictor and must not be described as held-out model performance. The uniform values describe expected guessing accuracy under their stated label spaces.

These references expose class imbalance and output-space difficulty. They do not validate the synthetic generators, establish educational effectiveness, or prove that LAM-JEPA learned natural-language reasoning. The existing `target_semantics` field remains authoritative: several generated tasks use concept-proxy labels rather than answer-correctness targets.

## Reproducibility evidence

CI runs the baseline command with a fixed seed and verifies:

1. exact coverage of every declared benchmark task;
2. exact sample counts;
3. finite values bounded to `[0, 1]`;
4. majority accuracy consistency with its label count;
5. uniform accuracies consistency with observed support and vocabulary size;
6. valid label ranges;
7. unchanged target and baseline semantics.

The JSON output and structured verification report are retained beside the training checkpoint and model-evaluation evidence. Future result tables should include at least one relevant baseline column and should never compare proxy-label accuracy to answer correctness without an explicit semantic distinction.
