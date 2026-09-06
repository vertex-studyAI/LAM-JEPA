# ARC successor v1 pre-outcome execution lock

**Status:** PREOUTCOME EXECUTION CONTRACT ONLY — NOT AUTHORIZED  
**Stack base:** `f74b8ea99778fc7e8199d9f3df7379e20f51f584` (successor evidence-schema PR #166)  
**Scientific outcome access:** prohibited

## What this closes

This package freezes the *shape* of the final execution boundary before any successor result exists. It makes source identity, runtime/environment identity, authorization evidence, dataset-byte identity, and the exact command surface mandatory inputs to any future scientific runner.

The current checked-in entrypoint has one successful mode only:

```text
python tools/arc_successor_locked_entrypoint.py --mode verify-preoutcome
```

It reads control-plane JSON only. It does not import model, dataset, training, or evaluation code.

An `execute` request is recognized only so the repository can prove that it fails closed. In the present state it must refuse because the successor protocol is still `DRAFT_NOT_FROZEN`, scientific execution and outcome access remain unauthorized, hard blockers remain, no scientific runner identity is bound, and no independent authorization receipt or execution manifest is supplied.

## Frozen identities retained

Primary systems remain `B0`, `B1`, `T1`, and `T2`. Seeds remain `11, 23, 37, 53, 71`. The old ARC-v5 locked confirmatory test remains prohibited. The pre-outcome decision-rule, context-target, encoder, and retained-evidence contracts remain the inherited scientific/control-plane dependencies.

## What this does not close

This does **not** resolve any currently outstanding successor hard blocker:

- independent `DATA_FRESHNESS_AUDIT` review;
- `CONFIRMATORY_DATASET` approval and exact bytes;
- `COLLAPSE_THRESHOLDS`;
- `PARAMETER_MATCH_TOLERANCE`;
- `MAX_COMPUTE_RATIO`;
- the final scientific `EXACT_REPRODUCE_COMMAND`.

It also does not bind a scientific implementation, environment versions, host, dataset bytes, authorization receipt, or output package. Those values must be frozen in a separate evidence-backed review before the execution entrypoint can be changed to permit a scientific run.

## Claim boundary

The only supported publication-facing statement from this artifact is:

> A fail-closed source/runtime/authorization execution contract was frozen before successor outcome access.

This artifact is not evidence that the successor works, improves reasoning, beats supervised learning, confirms a mechanism, has external validation, or is authorized to execute.
