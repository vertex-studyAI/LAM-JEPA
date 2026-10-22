# GitHub integration contract — 2026-10-22

Integrate the already proposed engineering and frozen-artifact work in PRs191,
200–206 while retaining their Git parents, negative ARC validation disposition,
and unopened confirmatory test. No new scientific campaign is authorized.

The observed integration defect is that `predict` selects the argmax after
float32 softmax rounding, whereas `explain`, rollout logits and classification
v2 select the logits argmax. A constructed finite pair `[0, 1e-9]` discriminates
these decisions. The correction must select the logits argmax while returning
the same probability tensor and preserving the caller's module training modes.

Verification budget: one constructed regression, one unchanged-source baseline
command, then one affected unit/planner/statistics/classification/CKA suite with
at most300 cases and120 CPU wall seconds. Use existing local dependencies,
single-thread CPU, no downloads or protected data. Preserve all failed outputs
and any fixture gaps. No training campaign or optimizer step is needed locally.
The existing hosted reproducibility workflow includes its bounded tiny fitting
smokes on ARC train/validation and generated data; that scope must be reported
accurately and does not authorize the locked confirmatory test.

Review source and artifact identities before creating a combined commit. Keep
the parent canonical-state snapshots and append a new integration receipt.
The ordinary unweakened checks on the final combined head must pass before merge.
