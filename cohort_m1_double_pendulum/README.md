# JEPA Cohort M1 — deterministic double-pendulum artifact

This directory is a **standalone cohort milestone artifact**. It does not amend, overwrite, or strengthen any frozen LAM-JEPA claim, manuscript result, or release decision in the parent repository.

## M1 acceptance criteria

The artifact freezes the following before any JEPA tuning:

1. deterministic double-pendulum simulator using fixed-step RK4;
2. deterministic SVG renderer for trajectory inspection;
3. trajectory-level train/validation/test assignment using a stable hash of initial conditions;
4. frozen evaluation contract and split policy;
5. two trivial one-step baselines: persistence and constant-velocity extrapolation;
6. a single reproduction command that emits split manifests, baseline metrics, a sample held-out trajectory, and an SVG visualization.

The held-out test split must not be used for model selection or hyperparameter tuning. Any later JEPA model must be added as a new method evaluated against this frozen contract.

## Reproduce

From the repository root:

```bash
python -m cohort_m1_double_pendulum.run --out artifacts/cohort_m1 --steps 400 --dt 0.02
```

Run the deterministic self-check:

```bash
python -m cohort_m1_double_pendulum.selfcheck
```

The reproduction command writes:

- `split_manifest.csv` — trajectory IDs, split labels, and initial conditions;
- `baseline_metrics.json` — one-step RMSE for persistence and constant-velocity baselines by split;
- `sample_test_trajectory.csv` — one immutable held-out trajectory sample;
- `sample_test_trajectory.svg` — deterministic visualization of that held-out trajectory;
- `run_manifest.json` — contract version, simulator settings, split counts, and SHA-256 hashes of generated artifacts.

## Frozen dataset grid

Initial conditions are the Cartesian product of:

- `theta1 ∈ {-1.2, -0.8, -0.4, 0.4, 0.8, 1.2}` radians
- `theta2 ∈ {-1.0, -0.5, 0.5, 1.0}` radians
- `omega1 ∈ {-0.5, 0.0, 0.5}` rad/s
- `omega2 ∈ {-0.5, 0.0, 0.5}` rad/s

This yields 216 trajectories. Split assignment is a pure SHA-256 function of the canonicalized initial-condition tuple:

- buckets `0–6`: train
- bucket `7`: validation
- buckets `8–9`: test

Changing the grid, bucket rule, simulator equations, integration method, metric definition, or evaluation horizon requires a new contract version rather than silently mutating M1.

## Frozen metric

For each trajectory and each transition `state_t → state_(t+1)`, evaluate one-step prediction over the four-dimensional state
`[theta1, omega1, theta2, omega2]`.

Angular residuals are wrapped into `[-π, π)`. Angular-velocity residuals are ordinary differences. The reported score is root mean squared error across all four components and all eligible transitions in a split.

This metric is intentionally simple. It is a baseline gate, not evidence that a learned representation is scientifically useful.

## Research-integrity boundary

M1 exists to make later experimentation harder to move after seeing results. Do not:

- tune on the test split;
- alter split assignment because a seed is inconvenient;
- report only favorable initial conditions;
- change the metric after looking at JEPA outcomes without versioning the contract;
- treat beating these trivial baselines as sufficient evidence for a publishable claim.

Null, mixed, unstable, and negative results remain valid cohort outcomes.