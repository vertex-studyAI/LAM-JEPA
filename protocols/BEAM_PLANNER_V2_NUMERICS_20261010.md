# Opt-in planner v2 norm-penalty numerics

This maintenance change applies only to `beam_plan_v2`. The frozen planner,
closed ARC-v5 negative study and unopened locked confirmatory test remain at
their original boundary in `FINAL_STATUS_2026-09-30.md`.

The declared score penalty is 0.01 times the latent Euclidean norm. Computing
the norm first can overflow through squaring finite coordinates, and float16
can overflow the unscaled norm even when the scaled penalty is representable.
For m=max(abs(z)), the code now computes (0.01 m)*norm(z/m), using divisor one
at m=0. This implements the same real-valued objective and applies the reduction
factor before restoring magnitude. Truly unrepresentable states/scores still
raise through the existing admission checks. The initial inference state is
explicitly detached, including a zero-depth search, which no_grad alone does
not guarantee for views.

Six new cases use independent math.hypot penalties for float16 60,000,
float32 1e30 and float64 1e300 coordinates, exact zero states, deterministic
ancestry and zero-depth detachment. Four fail on the exact parent; two zero
controls pass. All 20 planner-v2 tests pass after repair. This is bounded
software verification, not evidence of planner capability or ARC superiority.

The existing planner-v2 development contract is this branch's successor record.
PR #203 separately adds the repository canonical state; its integration should
index this record without erasing the closed predecessor or that PR's statistics
history. No competing canonical research-state file is introduced here.

## Verification identity and reproduction

Exact source parent: `05ec9a9c0dc69a8f7aee9e837bcb6e95a4bfec11`. Revision: `lam.beam-planner-v2.norm-penalty.1`.
The source/test SHA-256 map and test environment are retained in
`research/verification/maintenance_20261010/receipt.json`. Failed constructed witnesses and any JUnit files
remain in the same directory. These are engineering artifacts, not scientific
results or a research-completion checkpoint.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts='' -q tests/test_planner_v2.py tests/test_planner_v2_extreme_states.py
```
